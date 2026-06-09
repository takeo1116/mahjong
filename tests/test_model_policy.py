"""Tests for ``ModelPolicyAgent`` (Batch 2)."""
from __future__ import annotations

import math
import random

import pytest
import riichienv
import torch

from mahjong_agent.actions.convert import legal_actions_to_model_set
from mahjong_agent.actions.resolver import resolve
from mahjong_agent.actions.types import (
    ActionFamily,
    ActionKey,
    LegalActionSet,
    ModelAction,
)
from mahjong_agent.agents import (
    ModelPolicyAgent,
    ModelPolicyConfig,
    RuleBasedBaselineAgent,
)
from mahjong_agent.encoders import PublicObservationEncoder
from mahjong_agent.evaluation import (
    SeatAgents,
    SelfPlayConfig,
    SelfPlayRunner,
)
from mahjong_agent.models import Stage03Model, Stage03ModelConfig
from mahjong_agent.training import (
    PPOConfig,
    compute_returns_and_advantages,
    fit_ppo,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _build_encoder_model() -> tuple[PublicObservationEncoder, Stage03Model]:
    encoder = PublicObservationEncoder(num_players=4)
    cfg = Stage03ModelConfig.from_encoder_metadata(
        encoder.metadata(), hidden_dim=32, trunk_layers=1,
        candidate_hidden_dim=16,
    )
    model = Stage03Model(cfg).eval()
    return encoder, model


def _ikkyoku_seat_agents(seed: int = 0):
    encoder, model = _build_encoder_model()
    agent = ModelPolicyAgent(model, encoder, ModelPolicyConfig(greedy=False), seed=seed)
    return SeatAgents.homogeneous(agent, actor_type="policy", num_players=4), encoder, model


# ----------------------------------------------------------------------
# Standalone behavior
# ----------------------------------------------------------------------


def test_model_policy_picks_legal_discard_only():
    """legal な normal discard tile_type のみ選ぶ。"""
    torch.manual_seed(0)
    encoder, model = _build_encoder_model()
    agent = ModelPolicyAgent(
        model, encoder, ModelPolicyConfig(greedy=True), seed=0,
    )
    # 実 env から legal set を取る
    env = riichienv.RiichiEnv(riichienv.GameType.YON_IKKYOKU)
    env.reset(seed=42)
    cp = int(env.current_player)
    obs = env.get_observation(cp)
    la = list(obs.legal_actions())
    legal_set = legal_actions_to_model_set(
        la, actor=cp, num_players=int(env.num_players), env_for_riichi=env,
    )

    dec = agent.select_action(legal_set, observation=obs)
    assert isinstance(dec.action, ModelAction)
    family = dec.action.family
    if family == ActionFamily.NORMAL_DISCARD:
        assert dec.action.tile_type in legal_set.normal_discard
    else:
        assert dec.action in legal_set.candidates


def test_model_policy_emits_log_prob_and_value_in_extras():
    torch.manual_seed(0)
    encoder, model = _build_encoder_model()
    agent = ModelPolicyAgent(model, encoder, ModelPolicyConfig(), seed=0)
    env = riichienv.RiichiEnv(riichienv.GameType.YON_IKKYOKU)
    env.reset(seed=11)
    cp = int(env.current_player)
    obs = env.get_observation(cp)
    legal_set = legal_actions_to_model_set(
        list(obs.legal_actions()),
        actor=cp,
        num_players=int(env.num_players),
        env_for_riichi=env,
    )
    dec = agent.select_action(legal_set, observation=obs)
    assert "log_prob" in dec.extras
    assert "value" in dec.extras
    assert "family" in dec.extras
    # log_prob は finite (= 大きな負値 or 0 周辺)
    assert math.isfinite(float(dec.extras["log_prob"]))
    assert math.isfinite(float(dec.extras["value"]))
    # log_prob <= 0 (= log of softmax probability)
    assert float(dec.extras["log_prob"]) <= 1e-6


def test_model_policy_greedy_deterministic():
    """greedy=True なら同じ state で常に同じ action。"""
    torch.manual_seed(0)
    encoder, model = _build_encoder_model()
    agent = ModelPolicyAgent(model, encoder, ModelPolicyConfig(greedy=True), seed=0)
    env = riichienv.RiichiEnv(riichienv.GameType.YON_IKKYOKU)
    env.reset(seed=7)
    cp = int(env.current_player)
    obs = env.get_observation(cp)
    legal_set = legal_actions_to_model_set(
        list(obs.legal_actions()),
        actor=cp,
        num_players=int(env.num_players),
        env_for_riichi=env,
    )
    dec1 = agent.select_action(legal_set, observation=obs)
    dec2 = agent.select_action(legal_set, observation=obs)
    assert dec1.action.key == dec2.action.key


def test_model_policy_sampling_uses_rng():
    """sampling mode で per-call rng を渡せば deterministic に振る舞う。"""
    torch.manual_seed(0)
    encoder, model = _build_encoder_model()
    agent = ModelPolicyAgent(model, encoder, ModelPolicyConfig(greedy=False), seed=0)
    env = riichienv.RiichiEnv(riichienv.GameType.YON_IKKYOKU)
    env.reset(seed=13)
    cp = int(env.current_player)
    obs = env.get_observation(cp)
    legal_set = legal_actions_to_model_set(
        list(obs.legal_actions()),
        actor=cp,
        num_players=int(env.num_players),
        env_for_riichi=env,
    )
    rng_a = random.Random(99)
    rng_b = random.Random(99)
    dec_a = agent.select_action(legal_set, observation=obs, rng=rng_a)
    dec_b = agent.select_action(legal_set, observation=obs, rng=rng_b)
    assert dec_a.action.key == dec_b.action.key


def test_model_policy_resolves_to_raw_action():
    """選択した action が resolver で raw action に解決できる。"""
    torch.manual_seed(0)
    encoder, model = _build_encoder_model()
    agent = ModelPolicyAgent(model, encoder, ModelPolicyConfig(greedy=True), seed=0)
    env = riichienv.RiichiEnv(riichienv.GameType.YON_IKKYOKU)
    env.reset(seed=9)
    cp = int(env.current_player)
    obs = env.get_observation(cp)
    legal_set = legal_actions_to_model_set(
        list(obs.legal_actions()),
        actor=cp,
        num_players=int(env.num_players),
        env_for_riichi=env,
    )
    dec = agent.select_action(legal_set, observation=obs)
    raw_seq = resolve(legal_set, dec.action)
    assert len(raw_seq) >= 1


def test_model_policy_requires_observation():
    encoder, model = _build_encoder_model()
    agent = ModelPolicyAgent(model, encoder, ModelPolicyConfig(), seed=0)
    legal_set = LegalActionSet(
        decision_player=0,
        normal_discard={
            5: ModelAction(
                key=ActionKey(ActionFamily.NORMAL_DISCARD, tile_type=5),
                actor=0,
            )
        },
    )
    with pytest.raises(ValueError):
        agent.select_action(legal_set, observation=None)


def test_model_policy_win_shortcut_for_tsumo():
    """TSUMO が legal なら model を無視して取る (= log_prob=0)。"""
    torch.manual_seed(0)
    encoder, model = _build_encoder_model()
    agent = ModelPolicyAgent(model, encoder, ModelPolicyConfig(greedy=True), seed=0)

    # 簡易: candidate に TSUMO だけが入った LegalActionSet
    tsumo = ModelAction(key=ActionKey(ActionFamily.TSUMO), actor=0)
    legal_set = LegalActionSet(
        decision_player=0, candidates=(tsumo,),
    )
    # observation は何でもよいが、encoder forward が動くダミー obs が要る。
    env = riichienv.RiichiEnv(riichienv.GameType.YON_IKKYOKU)
    env.reset(seed=21)
    obs = env.get_observation(0)
    dec = agent.select_action(legal_set, observation=obs)
    assert dec.action.family == ActionFamily.TSUMO
    assert dec.extras["log_prob"] == 0.0


# ----------------------------------------------------------------------
# SelfPlayRunner integration
# ----------------------------------------------------------------------


def test_self_play_runner_records_log_prob_and_value_for_model_agent():
    """ModelPolicyAgent rollout → DecisionSample.old_log_prob/value が埋まる。"""
    torch.manual_seed(0)
    sa, _, _ = _ikkyoku_seat_agents(seed=0)
    runner = SelfPlayRunner(
        config=SelfPlayConfig(
            game_type=riichienv.GameType.YON_IKKYOKU,
            max_steps_per_game=4000,
        )
    )
    res = runner.run_episode(sa, seed=5)
    assert res.crash_context is None
    assert len(res.samples) > 0
    # ModelPolicyAgent の rollout sample は policy actor_type
    assert all(s.actor_type == "policy" for s in res.samples)

    # win shortcut 以外の sample は log_prob が non-zero (= < 0)
    non_win = [
        s for s in res.samples
        if s.decision_family not in (
            ActionFamily.TSUMO.value,
            ActionFamily.RON.value,
            ActionFamily.KYUSHU_KYUHAI.value,
        )
    ]
    assert non_win
    assert any(s.old_log_prob < 0.0 for s in non_win)
    # value は all finite
    assert all(math.isfinite(float(s.value)) for s in res.samples)


def test_self_play_runner_ikkyoku_smoke_no_crash():
    """ModelPolicyAgent 4 席で YON_IKKYOKU を 1 episode 完走させる smoke。"""
    torch.manual_seed(0)
    sa, _, _ = _ikkyoku_seat_agents(seed=1)
    runner = SelfPlayRunner(
        config=SelfPlayConfig(
            game_type=riichienv.GameType.YON_IKKYOKU,
            max_steps_per_game=4000,
        )
    )
    res = runner.run_episode(sa, seed=3)
    assert res.crash_context is None
    assert res.num_rounds >= 1
    assert sorted(res.final_ranks) == [1, 2, 3, 4]


# ----------------------------------------------------------------------
# PPO eligibility integration
# ----------------------------------------------------------------------


def test_model_rollout_samples_are_ppo_eligible():
    """policy actor_type + old_log_prob 付き sample が PPO eligible。"""
    torch.manual_seed(0)
    sa, _, _ = _ikkyoku_seat_agents(seed=2)
    runner = SelfPlayRunner(
        config=SelfPlayConfig(
            game_type=riichienv.GameType.YON_IKKYOKU,
            max_steps_per_game=4000,
        )
    )
    res = runner.run_episode(sa, seed=4)
    assert res.crash_context is None

    data = compute_returns_and_advantages(res.samples, PPOConfig())
    n_eligible = int(data.eligible.sum())
    assert n_eligible > 0, "model rollout sample が 1 件も PPO eligible になっていない"


def test_ppo_fit_runs_on_model_rollout_samples():
    """ModelPolicyAgent rollout → PPO 1 epoch 完走する smoke。"""
    torch.manual_seed(0)
    sa, _, model = _ikkyoku_seat_agents(seed=7)
    runner = SelfPlayRunner(
        config=SelfPlayConfig(
            game_type=riichienv.GameType.YON_IKKYOKU,
            max_steps_per_game=4000,
        )
    )
    res = runner.run_episode(sa, seed=8)
    assert res.samples
    cfg = PPOConfig(
        learning_rate=1e-3, batch_size=64, num_epochs=1,
        target_kl_enabled=False,
    )
    # 同じ model を使い続けると ratio ≈ 1 で動作する (rollout 時に log_prob を
    # 取った後、PPO update 前なので current policy も同じ logits を返す)。
    result = fit_ppo(model, res.samples, cfg)
    assert result.final is not None
    assert result.final.num_samples == len(res.samples)
    # 少なくとも 1 batch は applied されている
    assert result.final.target_kl_applied_minibatches >= 1


# ----------------------------------------------------------------------
# Existing baselines unchanged
# ----------------------------------------------------------------------


def test_rule_based_baseline_still_works_alongside_model_agent():
    """rule_based agent と model agent が混在する rollout が動く。"""
    torch.manual_seed(0)
    encoder, model = _build_encoder_model()
    rule = RuleBasedBaselineAgent()
    model_agent = ModelPolicyAgent(model, encoder, ModelPolicyConfig(greedy=False), seed=0)
    sa = SeatAgents.from_pairs({
        0: (model_agent, "policy"),
        1: (rule, "rule_based"),
        2: (model_agent, "policy"),
        3: (rule, "rule_based"),
    })
    runner = SelfPlayRunner(
        config=SelfPlayConfig(
            game_type=riichienv.GameType.YON_IKKYOKU,
            max_steps_per_game=4000,
        )
    )
    res = runner.run_episode(sa, seed=15)
    assert res.crash_context is None
    actor_types = {s.actor_type for s in res.samples}
    assert "policy" in actor_types
    assert "rule_based" in actor_types
    # rule_based actor の sample は old_log_prob=0 のまま
    rule_samples = [s for s in res.samples if s.actor_type == "rule_based"]
    assert rule_samples
    assert all(s.old_log_prob == 0.0 for s in rule_samples)
    # policy actor の sample は log_prob non-zero がある
    policy_samples = [s for s in res.samples if s.actor_type == "policy"]
    assert policy_samples
    assert any(s.old_log_prob < 0.0 for s in policy_samples)


# ----------------------------------------------------------------------
# Encode reuse (double-encode 解消) regression tests
# ----------------------------------------------------------------------


from contextlib import contextmanager  # noqa: E402

import numpy as np  # noqa: E402

from mahjong_agent.agents import RandomAgent  # noqa: E402


@contextmanager
def _count_encode_calls():
    """``PublicObservationEncoder.encode_observation`` の呼び出し回数を
    一時的に数える test 用 context manager。終了時に元の method を戻す。"""
    orig = PublicObservationEncoder.encode_observation
    counter = {"n": 0}

    def wrapped(self, obs):
        counter["n"] += 1
        return orig(self, obs)

    PublicObservationEncoder.encode_observation = wrapped
    try:
        yield counter
    finally:
        PublicObservationEncoder.encode_observation = orig


def test_model_policy_extras_include_observation_feat():
    torch.manual_seed(0)
    encoder, model = _build_encoder_model()
    agent = ModelPolicyAgent(model, encoder, ModelPolicyConfig(), seed=0)
    env = riichienv.RiichiEnv(riichienv.GameType.YON_IKKYOKU)
    env.reset(seed=11)
    cp = int(env.current_player)
    obs = env.get_observation(cp)
    legal_set = legal_actions_to_model_set(
        list(obs.legal_actions()),
        actor=cp,
        num_players=int(env.num_players),
        env_for_riichi=env,
    )
    dec = agent.select_action(legal_set, observation=obs)
    feat = dec.extras.get("observation_feat")
    assert isinstance(feat, np.ndarray)
    assert feat.dtype == np.float32
    expected_dim = encoder.metadata().observation_dim
    assert feat.shape == (expected_dim,)


def test_model_policy_rollout_encode_calls_per_sample_is_one():
    """二重 encode 解消後: ModelPolicyAgent rollout の
    ``encode_observation`` 呼び出し回数が `samples` と同数程度になる。"""
    torch.manual_seed(0)
    sa, _, _ = _ikkyoku_seat_agents(seed=0)
    runner = SelfPlayRunner(
        config=SelfPlayConfig(
            game_type=riichienv.GameType.YON_IKKYOKU,
            max_steps_per_game=4000,
        )
    )
    with _count_encode_calls() as counter:
        res = runner.run_episode(sa, seed=4)
    assert res.crash_context is None
    assert len(res.samples) > 0
    n_encode = counter["n"]
    n_samples = len(res.samples)
    # encode/sample が 1 を上回らない (= shortcut 含めても 1 件 1 encode)。
    # 余裕を持って 1.1 上限を許容。
    ratio = n_encode / max(1, n_samples)
    assert ratio <= 1.1, (
        f"encode/sample ratio = {ratio:.3f} (encode={n_encode}, "
        f"samples={n_samples}); expected <= 1.1 after double-encode fix"
    )


def test_random_agent_rollout_still_uses_fallback_encode():
    """RandomAgent は extras に observation_feat を入れないので、Runner 側
    fallback で encode が走り、sample が問題なく作られる。"""
    sa = SeatAgents.homogeneous(
        RandomAgent(seed=0), actor_type="random", num_players=4,
    )
    runner = SelfPlayRunner(
        config=SelfPlayConfig(
            game_type=riichienv.GameType.YON_IKKYOKU,
            max_steps_per_game=4000,
        )
    )
    with _count_encode_calls() as counter:
        res = runner.run_episode(sa, seed=5)
    assert res.crash_context is None
    assert len(res.samples) > 0
    # fallback 経路 → 1 encode per sample
    assert counter["n"] == len(res.samples)
    # 形が壊れていないこと
    enc = PublicObservationEncoder()
    expected_dim = enc.metadata().observation_dim
    for s in res.samples[:5]:
        assert s.observation.shape == (expected_dim,)


def test_rule_based_agent_rollout_still_uses_fallback_encode():
    sa = SeatAgents.homogeneous(
        RuleBasedBaselineAgent(), actor_type="rule_based", num_players=4,
    )
    runner = SelfPlayRunner(
        config=SelfPlayConfig(
            game_type=riichienv.GameType.YON_IKKYOKU,
            max_steps_per_game=4000,
        )
    )
    with _count_encode_calls() as counter:
        res = runner.run_episode(sa, seed=6)
    assert res.crash_context is None
    assert len(res.samples) > 0
    assert counter["n"] == len(res.samples)


def test_self_play_runner_rejects_observation_feat_shape_mismatch():
    """extras['observation_feat'] が encoder.observation_dim と一致しないとき
    Runner は ValueError で fail-fast する。"""
    from mahjong_agent.agents.base import AgentDecision

    class _BadAgent:
        """observation_feat に間違った shape を返す pathological agent。"""

        def __init__(self, encoder):
            self._encoder = encoder
            self._rule = RuleBasedBaselineAgent()

        def select_action(self, legal_set, *, rng=None, observation=None):
            dec = self._rule.select_action(
                legal_set, rng=rng, observation=observation,
            )
            bad_feat = np.zeros(
                self._encoder.metadata().observation_dim + 7,
                dtype=np.float32,
            )
            return AgentDecision(
                action=dec.action,
                rationale=dec.rationale,
                extras={**dict(dec.extras), "observation_feat": bad_feat},
            )

    encoder, _ = _build_encoder_model()
    bad = _BadAgent(encoder)
    sa = SeatAgents.homogeneous(bad, actor_type="bad", num_players=4)
    runner = SelfPlayRunner(
        config=SelfPlayConfig(
            game_type=riichienv.GameType.YON_IKKYOKU,
            max_steps_per_game=4000,
        )
    )
    res = runner.run_episode(sa, seed=0)
    # crash_context に shape mismatch が記録される (= silent corruption ではない)
    assert res.crash_context is not None
    assert "observation_feat" in str(res.crash_context.get("error_message", ""))


def test_self_play_runner_rejects_observation_feat_wrong_type():
    """observation_feat が np.ndarray でなければ TypeError で fail-fast。"""
    from mahjong_agent.agents.base import AgentDecision

    class _BadAgent:
        def __init__(self):
            self._rule = RuleBasedBaselineAgent()

        def select_action(self, legal_set, *, rng=None, observation=None):
            dec = self._rule.select_action(
                legal_set, rng=rng, observation=observation,
            )
            return AgentDecision(
                action=dec.action,
                rationale=dec.rationale,
                # list は ndarray ではない
                extras={**dict(dec.extras), "observation_feat": [0.0, 1.0]},
            )

    bad = _BadAgent()
    sa = SeatAgents.homogeneous(bad, actor_type="bad", num_players=4)
    runner = SelfPlayRunner(
        config=SelfPlayConfig(
            game_type=riichienv.GameType.YON_IKKYOKU,
            max_steps_per_game=4000,
        )
    )
    res = runner.run_episode(sa, seed=0)
    assert res.crash_context is not None
    err = str(res.crash_context.get("error_message", ""))
    assert "observation_feat" in err


# ----------------------------------------------------------------------
# sampling fallback illegal-discard regression (ISSUE-0021)
# ----------------------------------------------------------------------


def _nd_action(tile_type: int, actor: int = 0) -> ModelAction:
    return ModelAction(
        key=ActionKey(family=ActionFamily.NORMAL_DISCARD, tile_type=tile_type),
        actor=actor,
        _raw_actions=(),
    )


class _HighRng:
    """常に 1.0 を返す rng。cumulative sampling の fallback 経路を強制する。"""

    def random(self) -> float:
        return 1.0


def test_sampling_fallback_picks_legal_index_not_illegal_last():
    """legal discard が {5,7} (idx 33 は illegal) のとき、rng が cumsum を
    超えても fallback が illegal な末尾 index (33) ではなく legal index を選ぶ。
    旧実装の `idx = len(probs)-1` バグの regression。"""
    encoder, model = _build_encoder_model()
    agent = ModelPolicyAgent(
        model, encoder, ModelPolicyConfig(greedy=False), seed=0
    )
    env = riichienv.RiichiEnv(riichienv.GameType.YON_IKKYOKU)
    env.reset(seed=0)
    cp = env.current_player
    obs = env.get_observation(cp)
    # idx 33 を含まない legal discard set (no candidates → combined 末尾=33 が illegal)
    legal_set = LegalActionSet(
        decision_player=cp,
        normal_discard={5: _nd_action(5, cp), 7: _nd_action(7, cp)},
        candidates=(),
    )
    # rng が 1.0 を返し cumsum を超える → fallback 経路
    dec = agent.select_action(
        legal_set, observation=obs, rng=_HighRng()
    )
    assert dec.family == ActionFamily.NORMAL_DISCARD
    assert dec.tile_type in {5, 7}  # legal のみ、33 ではない
    assert dec.action.key.tile_type != 33


def test_sampling_only_picks_legal_across_many_rng_values():
    """様々な rng 値で sampling しても、常に legal discard のみ選ばれる
    (illegal index を踏まない)。"""
    encoder, model = _build_encoder_model()
    agent = ModelPolicyAgent(
        model, encoder, ModelPolicyConfig(greedy=False), seed=0
    )
    env = riichienv.RiichiEnv(riichienv.GameType.YON_IKKYOKU)
    env.reset(seed=0)
    cp = env.current_player
    obs = env.get_observation(cp)
    legal_tts = {3, 5, 7, 11}
    legal_set = LegalActionSet(
        decision_player=cp,
        normal_discard={tt: _nd_action(tt, cp) for tt in legal_tts},
        candidates=(),
    )

    class _FixedRng:
        def __init__(self, v: float):
            self._v = v

        def random(self) -> float:
            return self._v

    # 0.0, ~1.0, 境界付近を含む様々な r で legal のみ選ばれること
    for v in (0.0, 0.25, 0.5, 0.75, 0.999999, 1.0):
        dec = agent.select_action(legal_set, observation=obs, rng=_FixedRng(v))
        assert dec.family == ActionFamily.NORMAL_DISCARD
        assert dec.tile_type in legal_tts


def test_sampling_degenerate_distribution_fail_fast():
    """combined_mask が全 0 相当 (legal action 無し) は fail-fast する。"""
    encoder, model = _build_encoder_model()
    agent = ModelPolicyAgent(
        model, encoder, ModelPolicyConfig(greedy=False), seed=0
    )
    env = riichienv.RiichiEnv(riichienv.GameType.YON_IKKYOKU)
    env.reset(seed=0)
    cp = env.current_player
    obs = env.get_observation(cp)
    legal_set = LegalActionSet(
        decision_player=cp, normal_discard={}, candidates=()
    )
    with pytest.raises(ValueError):
        agent.select_action(legal_set, observation=obs)
