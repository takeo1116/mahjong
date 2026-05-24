"""Tests for ISSUE-0015 learner stabilizers.

カバー対象:

- ``optimizer_groups.build_lr_grouped_optimizer`` の default 互換性と opt-in 分離。
- ``sample_weighting.compute_per_player_round_weights`` の数値検証。
- PPO + imitation の per-player-round weighting opt-in。
- ``exclude_post_riichi_discards`` の policy / value 両 branch からの除外。
- ``value_loss_includes_excluded`` opt-in (shortcut sample が value だけに入る)。
- ``Stage03Model.semantic_summary_in_policy`` の forward shape / detach。
- training source の hidden info / local docs guard。
"""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from mahjong_agent.actions.types import ActionFamily
from mahjong_agent.data import SCHEMA_VERSION, DecisionSample, collate_decision_samples
from mahjong_agent.models import Stage03Model, Stage03ModelConfig
from mahjong_agent.targets.yaku import NUM_YAKU
from mahjong_agent.training import (
    ImitationConfig,
    PPOBatch,
    PPOConfig,
    compute_imitation_loss,
    compute_ppo_loss,
    compute_returns_and_advantages,
)
from mahjong_agent.training.optimizer_groups import (
    LRGroupConfig,
    build_lr_grouped_optimizer,
    classify_parameters_by_group,
)
from mahjong_agent.training.ppo import make_ppo_optimizer_with_info
from mahjong_agent.training.sample_weighting import compute_per_player_round_weights

_OBS_DIM = 16
_CAND_DIM = 6


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _build_model() -> Stage03Model:
    cfg = Stage03ModelConfig(
        observation_dim=_OBS_DIM,
        candidate_dim=_CAND_DIM,
        hidden_dim=32,
        trunk_layers=1,
        candidate_hidden_dim=16,
    )
    return Stage03Model(cfg)


def _build_model_with_semantic_summary() -> Stage03Model:
    cfg = Stage03ModelConfig(
        observation_dim=_OBS_DIM,
        candidate_dim=_CAND_DIM,
        hidden_dim=32,
        trunk_layers=1,
        candidate_hidden_dim=16,
        semantic_summary_in_policy=True,
    )
    return Stage03Model(cfg)


def _make_discard_sample(
    *,
    tile_type: int = 5,
    legal_tiles: tuple[int, ...] = (3, 5, 7),
    episode_id: str = "ep",
    round_id: int = 0,
    step_id: int = 0,
    player_id: int = 0,
    actor_type: str = "policy",
    metadata: dict | None = None,
) -> DecisionSample:
    obs = np.full(_OBS_DIM, 1.0, dtype=np.float32)
    mask = np.zeros(34, dtype=np.float32)
    for t in legal_tiles:
        mask[int(t)] = 1.0
    return DecisionSample(
        schema_version=SCHEMA_VERSION,
        episode_id=episode_id,
        round_id=int(round_id),
        step_id=int(step_id),
        player_id=int(player_id),
        decision_family=ActionFamily.NORMAL_DISCARD.value,
        actor_type=actor_type,
        observation=obs,
        discard_mask=mask,
        candidate_features=np.zeros((0, _CAND_DIM), dtype=np.float32),
        selected_discard_tile_type=int(tile_type),
        selected_candidate_index=-1,
        yaku_target=np.zeros(NUM_YAKU, dtype=np.float32),
        metadata=dict(metadata or {}),
    )


def _make_shortcut_sample(
    *,
    step_id: int = 0,
    episode_id: str = "ep",
    round_id: int = 0,
    player_id: int = 0,
) -> DecisionSample:
    """``ppo_exclude=True`` 付きの candidate sample (TSUMO 想定)。"""
    obs = np.full(_OBS_DIM, 1.0, dtype=np.float32)
    cand = np.zeros((1, _CAND_DIM), dtype=np.float32)
    cand[0, 0] = 1.0
    return DecisionSample(
        schema_version=SCHEMA_VERSION,
        episode_id=episode_id,
        round_id=int(round_id),
        step_id=int(step_id),
        player_id=int(player_id),
        decision_family=ActionFamily.TSUMO.value,
        actor_type="policy",
        observation=obs,
        discard_mask=np.zeros(34, dtype=np.float32),
        candidate_features=cand,
        selected_discard_tile_type=-1,
        selected_candidate_index=0,
        yaku_target=np.zeros(NUM_YAKU, dtype=np.float32),
        metadata={"ppo_exclude": True},
    )


# ----------------------------------------------------------------------
# 1. lr groups
# ----------------------------------------------------------------------


def test_lr_groups_default_is_single_group_and_backward_compatible():
    model = _build_model()
    opt, info = build_lr_grouped_optimizer(
        model,
        base_lr=3e-4,
        base_weight_decay=0.0,
        lr_group_config=None,
    )
    assert len(opt.param_groups) == 1
    assert info["enabled"] is False
    assert "all" in info["groups"]
    # JSON serializable
    json.dumps(info)


def test_lr_groups_opt_in_creates_multiple_param_groups():
    model = _build_model()
    cfg = LRGroupConfig(
        enabled=True,
        policy_lr=1e-3,
        value_semantic_lr=2e-4,
        trunk_lr=5e-4,
    )
    opt, info = build_lr_grouped_optimizer(
        model, base_lr=3e-4, lr_group_config=cfg
    )
    # Stage03Model は trunk / discard_head / candidate_scorer / value_head /
    # terminal_head / yaku_head を持つので、3 group (policy / value_semantic /
    # trunk) は全て non-empty で param_group が 3 つになる (default group は空)。
    assert len(opt.param_groups) == 3
    assert info["enabled"] is True
    # 各 group の lr が反映されている
    lrs = {g["name"]: g["lr"] for g in opt.param_groups}
    assert lrs["policy"] == pytest.approx(1e-3)
    assert lrs["value_semantic"] == pytest.approx(2e-4)
    assert lrs["trunk"] == pytest.approx(5e-4)
    # JSON serializable
    s = json.dumps(info)
    assert "policy" in s
    assert "value_semantic" in s
    assert "trunk" in s


def test_lr_groups_param_count_matches_full_model():
    """group 分類後の合計 param 数が model.parameters() の合計と一致する。"""
    model = _build_model()
    expected = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _, info = build_lr_grouped_optimizer(
        model, base_lr=1e-3, lr_group_config=LRGroupConfig(enabled=True)
    )
    actual = sum(g["param_count"] for g in info["groups"].values())
    assert actual == expected


def test_classify_parameters_by_group_covers_all_modules():
    model = _build_model()
    groups = classify_parameters_by_group(model.named_parameters())
    # Stage03 top-level modules: trunk / discard_head / value_head /
    # terminal_head / yaku_head / candidate_scorer。default は空 (forward-compat)。
    assert groups["policy"]  # discard_head + candidate_scorer
    assert groups["value_semantic"]  # value/terminal/yaku
    assert groups["trunk"]  # trunk
    # 重複が無い
    all_names = [
        n for lst in groups.values() for n in lst
    ]
    assert len(all_names) == len(set(all_names))


def test_make_ppo_optimizer_with_info_returns_diagnostics():
    model = _build_model()
    cfg = PPOConfig(
        learning_rate=3e-4,
        lr_group_config=LRGroupConfig(enabled=True, policy_lr=1e-3),
    )
    _, info = make_ppo_optimizer_with_info(model, cfg)
    assert info["enabled"] is True
    assert info["groups"]["policy"]["lr"] == pytest.approx(1e-3)
    json.dumps(info)  # JSON serializable


# ----------------------------------------------------------------------
# 2. per-player-round sample weighting
# ----------------------------------------------------------------------


def test_per_player_round_weights_sum_to_one_per_key():
    eids = ["ep", "ep", "ep", "ep2"]
    rids = [0, 0, 1, 0]
    pids = [0, 0, 0, 0]
    w = compute_per_player_round_weights(eids, rids, pids)
    # ("ep", 0, 0) は 2 sample → 各 0.5
    # ("ep", 1, 0) は 1 sample → 1.0
    # ("ep2", 0, 0) は 1 sample → 1.0
    assert w[0] == pytest.approx(0.5)
    assert w[1] == pytest.approx(0.5)
    assert w[2] == pytest.approx(1.0)
    assert w[3] == pytest.approx(1.0)


def test_per_player_round_weights_empty_input_returns_empty():
    w = compute_per_player_round_weights([], [], [])
    assert w.shape == (0,)
    assert w.dtype == np.float32


def test_imitation_weighted_loss_differs_from_unweighted():
    """同じ batch でも weighting on/off で loss が変わる (= 反映されている)。"""
    torch.manual_seed(0)
    model = _build_model()
    # 同 (episode, round, player) に 4 sample、別 trip に 1 sample
    samples = []
    for i in range(4):
        samples.append(
            _make_discard_sample(
                tile_type=5,
                step_id=i,
                episode_id="A",
                round_id=0,
                player_id=0,
            )
        )
    samples.append(
        _make_discard_sample(
            tile_type=3,
            step_id=10,
            episode_id="B",
            round_id=0,
            player_id=0,
        )
    )
    batch = collate_decision_samples(samples)
    cfg_off = ImitationConfig(terminal_loss_coef=0.0, yaku_loss_coef=0.0)
    cfg_on = ImitationConfig(
        terminal_loss_coef=0.0,
        yaku_loss_coef=0.0,
        per_player_round_weighting=True,
    )
    _, m_off = compute_imitation_loss(model, batch, cfg_off)
    _, m_on = compute_imitation_loss(model, batch, cfg_on)
    # off と on で discard_loss は通常一致しない (weight 分布が違う)
    assert m_off.discard_count == m_on.discard_count == 5
    assert not math.isclose(
        m_off.discard_loss, m_on.discard_loss, abs_tol=1e-9
    ), "expected weighted vs unweighted to differ"


def _weighting_sample_set() -> list[DecisionSample]:
    """同 (episode, round, player) に 4 sample + 別 triple に 1 sample。"""
    samples = [
        _make_discard_sample(
            tile_type=5, step_id=i, episode_id="A", round_id=0, player_id=0
        )
        for i in range(4)
    ]
    samples.append(
        _make_discard_sample(
            tile_type=3, step_id=10, episode_id="B", round_id=0, player_id=0
        )
    )
    return samples


def test_ppo_returns_sample_weight_default_all_ones_even_with_weighting():
    """trajectory-level の ``PPOTrainingData.sample_weight`` は weighting flag に
    関わらず常に全 1.0 (= minibatch-local 計算へ統一したため)。"""
    samples = _weighting_sample_set()
    for flag in (False, True):
        cfg = PPOConfig(
            target_kl_enabled=False,
            advantage_normalize=False,
            per_player_round_weighting=flag,
        )
        data = compute_returns_and_advantages(samples, cfg)
        assert data.sample_weight.shape == (len(samples),)
        assert np.allclose(data.sample_weight, 1.0), (
            f"per_player_round_weighting={flag}: trajectory sample_weight "
            f"should stay all 1.0"
        )


def test_ppo_minibatch_local_weighting_matches_expected():
    """``_iter_minibatches`` が minibatch-local per-player-round weight を作る。

    1 minibatch に全 sample を入れた場合:
      - ("A", 0, 0) は 4 sample → 各 0.25
      - ("B", 0, 0) は 1 sample → 1.0
    """
    import random as _random

    from mahjong_agent.training.ppo import _iter_minibatches

    samples = _weighting_sample_set()
    cfg_on = PPOConfig(
        target_kl_enabled=False,
        advantage_normalize=False,
        per_player_round_weighting=True,
        shuffle=False,
        batch_size=100,  # 全 sample を 1 minibatch に
    )
    data = compute_returns_and_advantages(samples, cfg_on)
    batches = list(
        _iter_minibatches(
            data,
            batch_size=int(cfg_on.batch_size),
            shuffle=False,
            rng=_random.Random(0),
            per_player_round_weighting=True,
        )
    )
    assert len(batches) == 1
    w = batches[0].sample_weight.numpy()
    assert w.tolist() == pytest.approx([0.25, 0.25, 0.25, 0.25, 1.0])

    # flag off → 全 1.0
    batches_off = list(
        _iter_minibatches(
            data,
            batch_size=int(cfg_on.batch_size),
            shuffle=False,
            rng=_random.Random(0),
            per_player_round_weighting=False,
        )
    )
    assert batches_off[0].sample_weight.numpy().tolist() == pytest.approx(
        [1.0, 1.0, 1.0, 1.0, 1.0]
    )


def test_ppo_and_imitation_weighting_semantics_match():
    """同一 minibatch に対し、PPO (_iter_minibatches) と imitation
    (compute_per_player_round_weights on batch) が同じ weight 配列を出す。"""
    import random as _random

    from mahjong_agent.training.ppo import _iter_minibatches

    samples = _weighting_sample_set()
    cfg = PPOConfig(
        target_kl_enabled=False,
        advantage_normalize=False,
        per_player_round_weighting=True,
        shuffle=False,
        batch_size=100,
    )
    data = compute_returns_and_advantages(samples, cfg)
    ppo_batch = next(
        iter(
            _iter_minibatches(
                data,
                batch_size=int(cfg.batch_size),
                shuffle=False,
                rng=_random.Random(0),
                per_player_round_weighting=True,
            )
        )
    )
    ppo_weights = ppo_batch.sample_weight.numpy()

    # imitation 側の経路 (collate 済み batch から weight 計算) を再現
    batch = collate_decision_samples(samples)
    imit_weights = compute_per_player_round_weights(
        episode_ids=batch.episode_id,
        round_ids=[int(x) for x in batch.round_id.tolist()],
        player_ids=[int(x) for x in batch.player_id.tolist()],
    )
    assert ppo_weights.tolist() == pytest.approx(imit_weights.tolist())


# ----------------------------------------------------------------------
# 3. post-riichi exclusion
# ----------------------------------------------------------------------


def test_imitation_post_riichi_exclusion_drops_marked_samples():
    torch.manual_seed(0)
    model = _build_model()
    s1 = _make_discard_sample(tile_type=5, step_id=0)
    s2 = _make_discard_sample(
        tile_type=5,
        step_id=1,
        metadata={"is_post_riichi_discard": True},
    )
    s3 = _make_discard_sample(tile_type=7, step_id=2)
    batch = collate_decision_samples([s1, s2, s3])
    cfg_on = ImitationConfig(
        terminal_loss_coef=0.0,
        yaku_loss_coef=0.0,
        exclude_post_riichi_discards=True,
    )
    _, m_on = compute_imitation_loss(model, batch, cfg_on)
    assert m_on.discard_count == 2
    assert m_on.post_riichi_excluded_count == 1

    cfg_off = ImitationConfig(terminal_loss_coef=0.0, yaku_loss_coef=0.0)
    _, m_off = compute_imitation_loss(model, batch, cfg_off)
    assert m_off.discard_count == 3
    assert m_off.post_riichi_excluded_count == 0


def test_ppo_post_riichi_exclusion_removes_from_both_policy_and_value():
    samples = [
        _make_discard_sample(tile_type=5, step_id=0),
        _make_discard_sample(
            tile_type=5,
            step_id=1,
            metadata={"is_post_riichi_discard": True},
        ),
        _make_discard_sample(tile_type=7, step_id=2),
    ]
    cfg = PPOConfig(
        target_kl_enabled=False,
        advantage_normalize=False,
        exclude_post_riichi_discards=True,
    )
    data = compute_returns_and_advantages(samples, cfg)
    # post-riichi sample は eligible / value_eligible 両方で False
    assert data.eligible.tolist() == [True, False, True]
    assert data.value_eligible.tolist() == [True, False, True]


# ----------------------------------------------------------------------
# 4. shortcut sample value loss opt-in
# ----------------------------------------------------------------------


def test_ppo_shortcut_never_enters_policy_loss():
    """``ppo_exclude=True`` の sample は opt-in に関わらず policy には入らない。"""
    samples = [
        _make_discard_sample(tile_type=5, step_id=0),
        _make_shortcut_sample(step_id=1),
        _make_discard_sample(tile_type=7, step_id=2),
    ]
    cfg_off = PPOConfig(
        target_kl_enabled=False,
        advantage_normalize=False,
        value_loss_includes_excluded=False,
    )
    data_off = compute_returns_and_advantages(samples, cfg_off)
    assert data_off.eligible.tolist() == [True, False, True]
    assert data_off.value_eligible.tolist() == [True, False, True]

    cfg_on = PPOConfig(
        target_kl_enabled=False,
        advantage_normalize=False,
        value_loss_includes_excluded=True,
    )
    data_on = compute_returns_and_advantages(samples, cfg_on)
    # policy 側 eligible は不変
    assert data_on.eligible.tolist() == [True, False, True]
    # value 側だけ shortcut sample が True に追加
    assert data_on.value_eligible.tolist() == [True, True, True]


def test_ppo_value_loss_extra_shortcut_count_diagnostic():
    """opt-in 時の diagnostics ``value_loss_extra_shortcut_count`` が反映される。"""
    torch.manual_seed(0)
    model = _build_model()
    samples = [
        _make_discard_sample(tile_type=5, step_id=0),
        _make_shortcut_sample(step_id=1),
    ]
    cfg = PPOConfig(
        target_kl_enabled=False,
        advantage_normalize=False,
        value_loss_includes_excluded=True,
    )
    data = compute_returns_and_advantages(samples, cfg)
    batch = collate_decision_samples(samples)
    ppo_batch = PPOBatch(
        batch=batch,
        returns=torch.from_numpy(data.returns),
        advantages=torch.from_numpy(np.ones(len(samples), dtype=np.float32)),
        eligible=torch.from_numpy(data.eligible),
        value_eligible=torch.from_numpy(data.value_eligible),
        sample_weight=torch.from_numpy(data.sample_weight),
    )
    _, metrics = compute_ppo_loss(model, ppo_batch, cfg)
    assert metrics.value_loss_extra_shortcut_count == 1


# ----------------------------------------------------------------------
# 5. semantic summary injection
# ----------------------------------------------------------------------


def test_semantic_summary_default_off_forward_shapes_unchanged():
    model = _build_model()
    obs = torch.randn(2, _OBS_DIM)
    out = model(obs)
    assert out.discard_logits.shape == (2, 34)
    assert out.value.shape == (2,)
    # default off では discard_head 入力 dim = hidden_dim と一致
    assert model.discard_head.in_features == model.config.hidden_dim


def test_semantic_summary_opt_in_forward_runs():
    model = _build_model_with_semantic_summary()
    obs = torch.randn(2, _OBS_DIM)
    out = model(obs)
    assert out.discard_logits.shape == (2, 34)
    assert out.terminal_logits.shape == (2, model.config.num_terminal_classes)
    assert out.yaku_logits.shape == (2, NUM_YAKU)
    # discard_head 入力 dim が hidden + summary
    expected_in = model.config.hidden_dim + (
        model.config.num_terminal_classes + NUM_YAKU
    )
    assert model.discard_head.in_features == expected_in
    # candidate scorer も同じ summary を取り込む
    cand = torch.randn(2, 3, _CAND_DIM)
    cscore = model.score_candidates(obs, cand)
    assert cscore.candidate_scores.shape == (2, 3)


def test_semantic_summary_detach_blocks_grad_from_policy_to_heads():
    """policy 側 loss を backward しても terminal_head / yaku_head に grad が
    流れないこと (= summary が detach されている)。"""
    model = _build_model_with_semantic_summary()
    obs = torch.randn(2, _OBS_DIM, requires_grad=False)
    out = model(obs)
    # discard 側 loss だけ計算
    target = torch.tensor([0, 1], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(out.discard_logits, target)
    # zero grad
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss.backward()
    # terminal_head と yaku_head の weight には grad が乗らない (= None or 0)
    for name, p in model.named_parameters():
        if name.startswith("terminal_head") or name.startswith("yaku_head"):
            if p.grad is not None:
                assert float(p.grad.abs().sum().item()) == 0.0, (
                    f"{name} should have zero grad from policy loss "
                    f"(semantic summary is detached)"
                )


def test_semantic_summary_no_detach_does_propagate_gradient():
    """``semantic_summary_detach=False`` (debug 用) では grad が流れる。"""
    cfg = Stage03ModelConfig(
        observation_dim=_OBS_DIM,
        candidate_dim=_CAND_DIM,
        hidden_dim=32,
        trunk_layers=1,
        candidate_hidden_dim=16,
        semantic_summary_in_policy=True,
        semantic_summary_detach=False,
    )
    model = Stage03Model(cfg)
    obs = torch.randn(2, _OBS_DIM)
    out = model(obs)
    target = torch.tensor([0, 1], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(out.discard_logits, target)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss.backward()
    has_grad = False
    for name, p in model.named_parameters():
        if name.startswith(("terminal_head", "yaku_head")):
            if p.grad is not None and float(p.grad.abs().sum().item()) > 0:
                has_grad = True
                break
    assert has_grad, (
        "with detach=False, gradient should flow to terminal/yaku heads"
    )


# ----------------------------------------------------------------------
# 6. hidden info / source guards
# ----------------------------------------------------------------------


def test_training_source_does_not_reference_hidden_state_or_local_docs():
    import mahjong_agent.training as pkg

    pkg_dir = Path(pkg.__file__).parent
    forbidden = (
        "env.hands",
        "env.wall",
        "env.state",
        "full_state",
        "private_hand",
        "PROJECT_RULE.md",
        "ISSUE_BOARD.md",
        "ISSUE-",
        "AGENTS.md",
        "CLAUDE.md",
        "majong-rl",
    )
    for f in pkg_dir.rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        for tok in forbidden:
            assert tok not in text, (
                f"training source {f.name} mentions {tok!r}"
            )


def test_optimizer_groups_signature_does_not_take_hidden_inputs():
    sig = inspect.signature(build_lr_grouped_optimizer)
    forbidden = {"env", "hands", "wall", "state"}
    assert forbidden.isdisjoint(set(sig.parameters.keys()))


def test_sample_weighting_signature_takes_only_identity_fields():
    sig = inspect.signature(compute_per_player_round_weights)
    assert list(sig.parameters.keys()) == [
        "episode_ids",
        "round_ids",
        "player_ids",
    ]
