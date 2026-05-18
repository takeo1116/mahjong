"""Regression tests for the ModelPolicyAgent <-> PPO log_prob consistency
(combined logits formulation).

ISSUE-0013 Session A follow-up: ``ModelPolicyAgent`` samples actions from a
``[discard_logits (34), candidate_scores (Cmax)]`` combined softmax and saves
the combined ``log_prob`` to ``AgentDecision.extras["log_prob"]``. The PPO
``compute_ppo_loss`` must reconstruct the **same combined distribution** so
that ``ratio = exp(new_log_prob - old_log_prob) = 1`` and
``approx_kl_mean ≈ 0`` immediately after rollout (before any model update).

These tests guard against future regressions where the two paths diverge.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import riichienv
import torch
import torch.nn.functional as F  # noqa: N812

from mahjong_agent.actions.convert import legal_actions_to_model_set
from mahjong_agent.actions.types import (
    ActionFamily,
    ActionKey,
    LegalActionSet,
    ModelAction,
)
from mahjong_agent.agents import ModelPolicyAgent, ModelPolicyConfig
from mahjong_agent.data import (
    SCHEMA_VERSION,
    DecisionSample,
    collate_decision_samples,
)
from mahjong_agent.encoders import PublicObservationEncoder
from mahjong_agent.evaluation import (
    SeatAgents,
    SelfPlayConfig,
    SelfPlayRunner,
)
from mahjong_agent.models import Stage03Model, Stage03ModelConfig
from mahjong_agent.targets.yaku import NUM_YAKU
from mahjong_agent.training import (
    PPOBatch,
    PPOConfig,
    compute_ppo_loss,
    compute_returns_and_advantages,
)


def _build_encoder_model() -> tuple[PublicObservationEncoder, Stage03Model]:
    encoder = PublicObservationEncoder()
    cfg = Stage03ModelConfig.from_encoder_metadata(
        encoder.metadata(), hidden_dim=32, trunk_layers=1, candidate_hidden_dim=16,
    )
    model = Stage03Model(cfg).eval()
    return encoder, model


# ----------------------------------------------------------------------
# Single-decision ratio=1 regression
# ----------------------------------------------------------------------


def test_model_rollout_then_ppo_has_zero_kl_and_no_clip():
    """Agent rollout で取った old_log_prob と PPO 側 new_log_prob が一致し、
    更新前 ratio=1 / approx_kl≈0 / clip_fraction=0 になる。"""
    torch.manual_seed(0)
    encoder, model = _build_encoder_model()
    agent = ModelPolicyAgent(
        model, encoder, ModelPolicyConfig(greedy=False), seed=0,
    )

    env = riichienv.RiichiEnv(riichienv.GameType.YON_IKKYOKU)
    env.reset(seed=42)
    cp = int(env.current_player)
    obs = env.get_observation(cp)
    legal_set = legal_actions_to_model_set(
        list(obs.legal_actions()),
        actor=cp,
        num_players=int(env.num_players),
        env_for_riichi=env,
    )

    rng = random.Random(0)
    # collect a few decisions to make a small batch
    samples: list[DecisionSample] = []
    for step in range(8):
        dec = agent.select_action(legal_set, observation=obs, rng=rng)
        # PPO 非対象 (shortcut) はテスト目的ではスキップ
        if dec.extras.get("ppo_exclude"):
            continue
        family = dec.action.family
        if family == ActionFamily.NORMAL_DISCARD:
            sel_tt = int(dec.action.tile_type)
            sel_idx = -1
        else:
            sel_tt = -1
            sel_idx = next(
                i for i, c in enumerate(legal_set.candidates)
                if c.key == dec.action.key
            )
        cand_feat = encoder.encode_candidates(legal_set)
        samples.append(
            DecisionSample(
                schema_version=SCHEMA_VERSION,
                episode_id="ep",
                round_id=0,
                step_id=step,
                player_id=cp,
                decision_family=family.value,
                actor_type="policy",
                observation=encoder.encode_observation(obs),
                discard_mask=encoder.discard_legal_mask(legal_set),
                candidate_features=cand_feat,
                selected_discard_tile_type=sel_tt,
                selected_candidate_index=sel_idx,
                old_log_prob=float(dec.extras["log_prob"]),
                value=float(dec.extras["value"]),
                yaku_target=np.zeros(NUM_YAKU, dtype=np.float32),
            )
        )

    assert samples, "no non-shortcut samples collected; bump the loop above"

    cfg = PPOConfig(target_kl_enabled=False, advantage_normalize=False)
    data = compute_returns_and_advantages(samples, cfg)
    batch = collate_decision_samples(samples)
    ppo_batch = PPOBatch(
        batch=batch,
        returns=torch.from_numpy(data.returns),
        advantages=torch.zeros_like(torch.from_numpy(data.returns)),
        eligible=torch.from_numpy(data.eligible),
    )
    _, metrics = compute_ppo_loss(model, ppo_batch, cfg)
    # 同一 model + 同一 old_log_prob → ratio=1, KL≈0, clip=0
    assert metrics.approx_kl_mean == pytest.approx(0.0, abs=1e-5), (
        f"approx_kl_mean={metrics.approx_kl_mean} (expected ~0). "
        f"old/new log_prob mismatch?"
    )
    assert metrics.approx_kl_max == pytest.approx(0.0, abs=1e-5)
    assert metrics.clip_fraction == pytest.approx(0.0, abs=1e-6)


# ----------------------------------------------------------------------
# Mixed-family batch regression
# ----------------------------------------------------------------------


def _normal_discard_sample(
    *, tile_type: int, step_id: int, num_cands: int = 0, obs_dim: int = 8,
) -> DecisionSample:
    cand = np.zeros((num_cands, 4), dtype=np.float32) if num_cands > 0 else (
        np.zeros((0, 4), dtype=np.float32)
    )
    dmask = np.zeros(34, dtype=np.float32)
    dmask[tile_type] = 1.0
    dmask[(tile_type + 1) % 34] = 1.0  # extra legal tile to avoid only 1 option
    return DecisionSample(
        schema_version=SCHEMA_VERSION,
        episode_id="syn",
        round_id=0, step_id=step_id, player_id=0,
        decision_family=ActionFamily.NORMAL_DISCARD.value,
        actor_type="policy",
        observation=np.ones(obs_dim, dtype=np.float32),
        discard_mask=dmask,
        candidate_features=cand,
        selected_discard_tile_type=tile_type,
        selected_candidate_index=-1,
        yaku_target=np.zeros(NUM_YAKU, dtype=np.float32),
    )


def _candidate_sample(
    *, selected_index: int, num_cands: int, step_id: int, obs_dim: int = 8,
) -> DecisionSample:
    cand = np.zeros((num_cands, 4), dtype=np.float32)
    for i in range(num_cands):
        cand[i, i % 4] = 1.0
    return DecisionSample(
        schema_version=SCHEMA_VERSION,
        episode_id="syn",
        round_id=0, step_id=step_id, player_id=0,
        decision_family=ActionFamily.PASS.value,
        actor_type="policy",
        observation=np.ones(obs_dim, dtype=np.float32) * 0.5,
        discard_mask=np.zeros(34, dtype=np.float32),
        candidate_features=cand,
        selected_discard_tile_type=-1,
        selected_candidate_index=selected_index,
        yaku_target=np.zeros(NUM_YAKU, dtype=np.float32),
    )


def _set_combined_old_log_probs(
    model: Stage03Model, samples: list[DecisionSample]
) -> None:
    """combined log_prob で old_log_prob を埋める helper (test 専用)。"""
    model.eval()
    with torch.no_grad():
        batch = collate_decision_samples(samples)
        obs = batch.observation.float()
        dmask = batch.discard_mask.float()
        fwd = model(obs, discard_mask=dmask)
        cand_out = model.score_candidates(obs, batch.candidate_features.float())
        cand_mask = batch.candidate_mask.float()
        cmax = int(cand_out.candidate_scores.size(1))
        if cmax > 0:
            masked_cand = cand_out.candidate_scores + (1.0 - cand_mask) * -1e9
        else:
            masked_cand = cand_out.candidate_scores
        combined = torch.cat([fwd.discard_logits, masked_cand], dim=-1)
        log_softmax = F.log_softmax(combined, dim=-1)
        for i, s in enumerate(samples):
            if s.decision_family == ActionFamily.NORMAL_DISCARD.value:
                idx = int(s.selected_discard_tile_type)
            else:
                idx = 34 + int(s.selected_candidate_index)
            s.old_log_prob = float(log_softmax[i, idx].item())
    model.train()


def test_mixed_discard_and_candidate_batch_ratio_one():
    """discard sample と candidate sample が同 batch に居ても、combined
    formulation で old/new log_prob が一致して ratio=1。"""
    torch.manual_seed(0)
    cfg_model = Stage03ModelConfig(
        observation_dim=8, candidate_dim=4,
        hidden_dim=16, trunk_layers=1, candidate_hidden_dim=8,
    )
    model = Stage03Model(cfg_model)
    samples = [
        _normal_discard_sample(tile_type=5, step_id=0),
        _normal_discard_sample(tile_type=10, step_id=1),
        _candidate_sample(selected_index=1, num_cands=3, step_id=2),
        _candidate_sample(selected_index=0, num_cands=2, step_id=3),
        _normal_discard_sample(tile_type=7, step_id=4),
        _candidate_sample(selected_index=2, num_cands=4, step_id=5),
    ]
    _set_combined_old_log_probs(model, samples)

    cfg = PPOConfig(target_kl_enabled=False, advantage_normalize=False)
    data = compute_returns_and_advantages(samples, cfg)
    batch = collate_decision_samples(samples)
    ppo_batch = PPOBatch(
        batch=batch,
        returns=torch.from_numpy(data.returns),
        advantages=torch.zeros_like(torch.from_numpy(data.returns)),
        eligible=torch.from_numpy(data.eligible),
    )
    _, metrics = compute_ppo_loss(model, ppo_batch, cfg)
    assert metrics.approx_kl_mean == pytest.approx(0.0, abs=1e-5)
    assert metrics.approx_kl_max == pytest.approx(0.0, abs=1e-5)
    assert metrics.clip_fraction == pytest.approx(0.0, abs=1e-6)
    # discard / candidate 両 family が PPO 対象に含まれている
    assert metrics.discard_count > 0
    assert metrics.candidate_count > 0


# ----------------------------------------------------------------------
# Shortcut sample → PPO excluded
# ----------------------------------------------------------------------


def test_model_policy_shortcut_marks_ppo_exclude_true():
    torch.manual_seed(0)
    encoder, model = _build_encoder_model()
    agent = ModelPolicyAgent(
        model, encoder, ModelPolicyConfig(greedy=True), seed=0,
    )
    tsumo = ModelAction(key=ActionKey(ActionFamily.TSUMO), actor=0)
    legal_set = LegalActionSet(decision_player=0, candidates=(tsumo,))
    env = riichienv.RiichiEnv(riichienv.GameType.YON_IKKYOKU)
    env.reset(seed=0)
    obs = env.get_observation(0)
    dec = agent.select_action(legal_set, observation=obs)
    assert dec.action.family == ActionFamily.TSUMO
    assert dec.extras.get("ppo_exclude") is True
    assert dec.extras["log_prob"] == 0.0


def test_shortcut_sample_excluded_from_ppo_eligibility():
    """``metadata.ppo_exclude=True`` の sample は eligible にならない。"""
    s_excluded = DecisionSample(
        schema_version=SCHEMA_VERSION,
        episode_id="ep", round_id=0, step_id=0, player_id=0,
        decision_family=ActionFamily.TSUMO.value,
        actor_type="policy",
        observation=np.ones(8, dtype=np.float32),
        discard_mask=np.zeros(34, dtype=np.float32),
        candidate_features=np.zeros((1, 4), dtype=np.float32),
        selected_discard_tile_type=-1,
        selected_candidate_index=0,
        metadata={"ppo_exclude": True},
        yaku_target=np.zeros(NUM_YAKU, dtype=np.float32),
    )
    s_normal = DecisionSample(
        schema_version=SCHEMA_VERSION,
        episode_id="ep", round_id=0, step_id=1, player_id=0,
        decision_family=ActionFamily.NORMAL_DISCARD.value,
        actor_type="policy",
        observation=np.ones(8, dtype=np.float32),
        discard_mask=np.ones(34, dtype=np.float32),
        candidate_features=np.zeros((0, 4), dtype=np.float32),
        selected_discard_tile_type=5,
        selected_candidate_index=-1,
        yaku_target=np.zeros(NUM_YAKU, dtype=np.float32),
    )
    data = compute_returns_and_advantages([s_excluded, s_normal], PPOConfig())
    assert bool(data.eligible[0]) is False, "shortcut sample should be excluded"
    assert bool(data.eligible[1]) is True


def test_self_play_runner_persists_ppo_exclude_in_sample_metadata():
    """SelfPlayRunner で ModelPolicyAgent rollout を走らせると shortcut
    sample の metadata に ``ppo_exclude=True`` が記録される。"""
    torch.manual_seed(0)
    encoder, model = _build_encoder_model()
    agent = ModelPolicyAgent(model, encoder, ModelPolicyConfig(greedy=False), seed=0)
    sa = SeatAgents.homogeneous(agent, actor_type="policy", num_players=4)
    runner = SelfPlayRunner(
        config=SelfPlayConfig(
            game_type=riichienv.GameType.YON_IKKYOKU,
            max_steps_per_game=4000,
        )
    )
    found_shortcut = False
    # 数 seed 試して shortcut sample が拾える episode を探す
    for seed in range(8):
        res = runner.run_episode(sa, seed=seed)
        assert res.crash_context is None
        shortcuts = [
            s for s in res.samples
            if s.decision_family in (
                ActionFamily.TSUMO.value,
                ActionFamily.RON.value,
                ActionFamily.KYUSHU_KYUHAI.value,
            )
        ]
        if shortcuts:
            for s in shortcuts:
                assert s.metadata.get("ppo_exclude") is True, (
                    f"shortcut family={s.decision_family} missing ppo_exclude "
                    f"in metadata={s.metadata}"
                )
            found_shortcut = True
            break
    # shortcut が発生しない seed しか無い場合でも、少なくとも残りの sample
    # では ppo_exclude が立っていないことを確認 (= 通常 sample の互換性)。
    if not found_shortcut:
        res = runner.run_episode(sa, seed=0)
        for s in res.samples:
            assert s.metadata.get("ppo_exclude") is not True


def test_random_agent_samples_do_not_get_ppo_exclude():
    """random / rule_based agent は extras に ppo_exclude を入れないので、
    sample.metadata に False (= 未設定) が維持される。"""
    from mahjong_agent.agents import RandomAgent

    sa = SeatAgents.homogeneous(
        RandomAgent(seed=0), actor_type="random", num_players=4,
    )
    runner = SelfPlayRunner(
        config=SelfPlayConfig(
            game_type=riichienv.GameType.YON_IKKYOKU,
            max_steps_per_game=4000,
        )
    )
    res = runner.run_episode(sa, seed=0)
    for s in res.samples:
        assert s.metadata.get("ppo_exclude") is not True


# ----------------------------------------------------------------------
# greedy / temperature -> ppo_exclude
# ----------------------------------------------------------------------


def _agent_with(greedy: bool, temperature: float):
    encoder, model = _build_encoder_model()
    cfg = ModelPolicyConfig(greedy=greedy, temperature=temperature)
    return ModelPolicyAgent(model, encoder, cfg, seed=0), encoder, model


def _real_legal_and_obs(seed: int = 42):
    env = riichienv.RiichiEnv(riichienv.GameType.YON_IKKYOKU)
    env.reset(seed=seed)
    cp = int(env.current_player)
    obs = env.get_observation(cp)
    legal_set = legal_actions_to_model_set(
        list(obs.legal_actions()),
        actor=cp,
        num_players=int(env.num_players),
        env_for_riichi=env,
    )
    return env, obs, legal_set, cp


def test_model_policy_greedy_true_marks_ppo_exclude():
    torch.manual_seed(0)
    agent, _, _ = _agent_with(greedy=True, temperature=1.0)
    _, obs, legal_set, _ = _real_legal_and_obs(seed=42)
    dec = agent.select_action(legal_set, observation=obs)
    # win-shortcut が来ないケースを assert (= 通常 sampling 経路で
    # ppo_exclude が立っている)。
    if dec.action.family in (
        ActionFamily.TSUMO, ActionFamily.RON, ActionFamily.KYUSHU_KYUHAI,
    ):
        # shortcut でも ppo_exclude は True (本テストの本筋とは別経路)
        assert dec.extras.get("ppo_exclude") is True
        return
    assert dec.extras.get("ppo_exclude") is True, (
        "greedy=True で sampling 経路に居るのに ppo_exclude が立っていない"
    )
    # log_prob は diagnostics 用に保存される
    assert "log_prob" in dec.extras


def test_model_policy_temperature_non_one_marks_ppo_exclude():
    torch.manual_seed(0)
    agent, _, _ = _agent_with(greedy=False, temperature=2.0)
    _, obs, legal_set, _ = _real_legal_and_obs(seed=43)
    dec = agent.select_action(legal_set, observation=obs)
    if dec.action.family in (
        ActionFamily.TSUMO, ActionFamily.RON, ActionFamily.KYUSHU_KYUHAI,
    ):
        assert dec.extras.get("ppo_exclude") is True
        return
    assert dec.extras.get("ppo_exclude") is True
    assert "log_prob" in dec.extras


def test_model_policy_low_temperature_marks_ppo_exclude():
    """temperature < 1 (e.g., 0.5) でも除外される。"""
    torch.manual_seed(0)
    agent, _, _ = _agent_with(greedy=False, temperature=0.5)
    _, obs, legal_set, _ = _real_legal_and_obs(seed=44)
    dec = agent.select_action(legal_set, observation=obs)
    if dec.action.family in (
        ActionFamily.TSUMO, ActionFamily.RON, ActionFamily.KYUSHU_KYUHAI,
    ):
        assert dec.extras.get("ppo_exclude") is True
        return
    assert dec.extras.get("ppo_exclude") is True


def test_model_policy_default_does_not_mark_ppo_exclude():
    """default (greedy=False, temperature=1.0) では sampling 経路で
    ppo_exclude が **立たない**。"""
    torch.manual_seed(0)
    agent, _, _ = _agent_with(greedy=False, temperature=1.0)
    _, obs, legal_set, _ = _real_legal_and_obs(seed=45)
    dec = agent.select_action(legal_set, observation=obs)
    if dec.action.family in (
        ActionFamily.TSUMO, ActionFamily.RON, ActionFamily.KYUSHU_KYUHAI,
    ):
        # shortcut は別経路で除外される。本テスト本筋ではないので skip。
        return
    assert dec.extras.get("ppo_exclude") is not True, (
        f"default config で sampling 経路に居るのに ppo_exclude が立っている "
        f"(extras={dec.extras})"
    )


def test_self_play_runner_persists_temperature_ppo_exclude():
    """SelfPlayRunner で temperature != 1.0 の agent を 4 席に置いた rollout
    では、policy actor の sample がほぼ全て ppo_exclude=True となり、PPO
    eligible にならない。"""
    torch.manual_seed(0)
    agent, _, _ = _agent_with(greedy=False, temperature=1.5)
    sa = SeatAgents.homogeneous(agent, actor_type="policy", num_players=4)
    runner = SelfPlayRunner(
        config=SelfPlayConfig(
            game_type=riichienv.GameType.YON_IKKYOKU,
            max_steps_per_game=4000,
        )
    )
    res = runner.run_episode(sa, seed=0)
    assert res.crash_context is None
    assert res.samples
    # 全 policy sample が ppo_exclude=True (= shortcut + temperature ≠ 1 経路
    # の両方で True が立つ)
    for s in res.samples:
        assert s.metadata.get("ppo_exclude") is True, (
            f"sample={s.decision_family} should be ppo_exclude=True under "
            f"temperature=1.5"
        )
    # ratio 計算側でも eligible が完全に 0 になる
    data = compute_returns_and_advantages(res.samples, PPOConfig())
    assert int(data.eligible.sum()) == 0


def test_self_play_runner_persists_greedy_ppo_exclude():
    """SelfPlayRunner で greedy=True の agent を 4 席に置いた rollout でも
    policy sample が全て ppo_exclude=True となる。"""
    torch.manual_seed(0)
    agent, _, _ = _agent_with(greedy=True, temperature=1.0)
    sa = SeatAgents.homogeneous(agent, actor_type="policy", num_players=4)
    runner = SelfPlayRunner(
        config=SelfPlayConfig(
            game_type=riichienv.GameType.YON_IKKYOKU,
            max_steps_per_game=4000,
        )
    )
    res = runner.run_episode(sa, seed=0)
    assert res.crash_context is None
    assert res.samples
    for s in res.samples:
        assert s.metadata.get("ppo_exclude") is True
    data = compute_returns_and_advantages(res.samples, PPOConfig())
    assert int(data.eligible.sum()) == 0


def test_self_play_runner_default_temperature_keeps_eligible():
    """default config (greedy=False, temperature=1.0) では policy sample の
    多数が eligible=True になる (shortcut sample 以外)。"""
    torch.manual_seed(0)
    agent, _, _ = _agent_with(greedy=False, temperature=1.0)
    sa = SeatAgents.homogeneous(agent, actor_type="policy", num_players=4)
    runner = SelfPlayRunner(
        config=SelfPlayConfig(
            game_type=riichienv.GameType.YON_IKKYOKU,
            max_steps_per_game=4000,
        )
    )
    res = runner.run_episode(sa, seed=0)
    assert res.crash_context is None
    data = compute_returns_and_advantages(res.samples, PPOConfig())
    n_eligible = int(data.eligible.sum())
    assert n_eligible > 0, (
        "default config なのに eligible sample が 0; sampling rollout 経路が"
        " ratio=1 で渡せていない可能性"
    )
    # shortcut sample は metadata.ppo_exclude=True が立つので、eligible
    # と shortcut sample の union が全 policy sample にほぼ一致する。
    n_shortcut = sum(
        1 for s in res.samples
        if s.metadata.get("ppo_exclude") is True
    )
    assert n_eligible + n_shortcut == len(res.samples), (
        f"eligible={n_eligible} + shortcut={n_shortcut} != total="
        f"{len(res.samples)} (mismatch implies stray exclusion)"
    )


# ----------------------------------------------------------------------
# Sanity: per-family decomposition still works under combined formulation
# ----------------------------------------------------------------------


def test_per_family_diagnostics_sum_to_total_policy_loss():
    """combined formulation でも family ごとの policy_loss が
    eligible 全体の policy_loss と整合する (= weighted mean が一致)。"""
    torch.manual_seed(0)
    cfg_model = Stage03ModelConfig(
        observation_dim=8, candidate_dim=4,
        hidden_dim=16, trunk_layers=1, candidate_hidden_dim=8,
    )
    model = Stage03Model(cfg_model)
    samples = [
        _normal_discard_sample(tile_type=5, step_id=0),
        _normal_discard_sample(tile_type=10, step_id=1),
        _candidate_sample(selected_index=1, num_cands=3, step_id=2),
        _candidate_sample(selected_index=0, num_cands=2, step_id=3),
    ]
    # わざと old_log_prob を 0 にして ratio ≠ 1 を作る
    for s in samples:
        s.old_log_prob = -5.0
    cfg = PPOConfig(
        target_kl_enabled=False, advantage_normalize=False,
        clip_epsilon=0.2,
    )
    data = compute_returns_and_advantages(samples, cfg)
    batch = collate_decision_samples(samples)
    advs = np.ones(len(samples), dtype=np.float32)  # constant advantage
    ppo_batch = PPOBatch(
        batch=batch,
        returns=torch.from_numpy(data.returns),
        advantages=torch.from_numpy(advs),
        eligible=torch.from_numpy(data.eligible),
    )
    _, m = compute_ppo_loss(model, ppo_batch, cfg)
    # weighted-mean check
    d = m.discard_count
    c = m.candidate_count
    if d + c > 0:
        weighted = (
            m.discard_policy_loss * d + m.candidate_policy_loss * c
        ) / (d + c)
        assert m.policy_loss == pytest.approx(weighted, rel=1e-5, abs=1e-5)
