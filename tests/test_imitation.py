"""Tests for the imitation warm-start trainer (ISSUE-0010)."""
from __future__ import annotations

import copy
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from mahjong_agent.actions.types import ActionFamily
from mahjong_agent.data import (
    SCHEMA_VERSION,
    DecisionSample,
    collate_decision_samples,
    read_decision_shard,
    write_decision_shard,
)
from mahjong_agent.models import Stage03Model, Stage03ModelConfig
from mahjong_agent.targets.terminal import NUM_TERMINAL_CLASSES
from mahjong_agent.targets.yaku import NUM_YAKU
from mahjong_agent.training import (
    ImitationConfig,
    ImitationMetrics,
    compute_imitation_loss,
    fit_imitation,
    make_default_optimizer,
    metrics_to_json,
    train_imitation_epoch,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

_OBS_DIM = 16
_CAND_DIM = 6


def _build_model(*, obs_dim: int = _OBS_DIM, cand_dim: int = _CAND_DIM):
    cfg = Stage03ModelConfig(
        observation_dim=obs_dim,
        candidate_dim=cand_dim,
        hidden_dim=32,
        trunk_layers=1,
        candidate_hidden_dim=16,
    )
    return Stage03Model(cfg)


def _make_discard_sample(
    *,
    tile_type: int = 5,
    legal_tiles: tuple[int, ...] = (3, 5, 7),
    obs_value: float = 1.0,
    episode_id: str = "syn",
    step_id: int = 0,
    actor_type: str = "rule_based",
) -> DecisionSample:
    obs = np.full(_OBS_DIM, float(obs_value), dtype=np.float32)
    mask = np.zeros(34, dtype=np.float32)
    for t in legal_tiles:
        mask[int(t)] = 1.0
    return DecisionSample(
        schema_version=SCHEMA_VERSION,
        episode_id=episode_id,
        round_id=0,
        step_id=step_id,
        player_id=0,
        decision_family=ActionFamily.NORMAL_DISCARD.value,
        actor_type=actor_type,
        observation=obs,
        discard_mask=mask,
        candidate_features=np.zeros((0, _CAND_DIM), dtype=np.float32),
        selected_discard_tile_type=int(tile_type),
        selected_candidate_index=-1,
        yaku_target=np.zeros(NUM_YAKU, dtype=np.float32),
    )


def _make_candidate_sample(
    *,
    selected_index: int = 1,
    num_cands: int = 3,
    obs_value: float = 0.5,
    family: ActionFamily = ActionFamily.PASS,
    step_id: int = 0,
    actor_type: str = "rule_based",
    candidate_features: np.ndarray | None = None,
) -> DecisionSample:
    obs = np.full(_OBS_DIM, float(obs_value), dtype=np.float32)
    if candidate_features is None:
        # 各 candidate を distinct one-hot like vector
        cand = np.zeros((num_cands, _CAND_DIM), dtype=np.float32)
        for i in range(num_cands):
            cand[i, i % _CAND_DIM] = 1.0
    else:
        cand = candidate_features.astype(np.float32)
        num_cands = cand.shape[0]
    return DecisionSample(
        schema_version=SCHEMA_VERSION,
        episode_id="syn",
        round_id=0,
        step_id=step_id,
        player_id=0,
        decision_family=family.value,
        actor_type=actor_type,
        observation=obs,
        discard_mask=np.zeros(34, dtype=np.float32),
        candidate_features=cand,
        selected_discard_tile_type=-1,
        selected_candidate_index=int(selected_index),
        yaku_target=np.zeros(NUM_YAKU, dtype=np.float32),
    )


def _attach_terminal(sample: DecisionSample, terminal_class: int) -> DecisionSample:
    sample.terminal_class = int(terminal_class)
    return sample


def _attach_yaku(
    sample: DecisionSample, yaku_indices: list[int], *, han: int = 2, fu: int = 30
) -> DecisionSample:
    arr = np.zeros(NUM_YAKU, dtype=np.float32)
    for i in yaku_indices:
        arr[int(i)] = 1.0
    sample.yaku_target = arr
    sample.yaku_loss_mask = 1.0
    sample.han = int(han)
    sample.fu = int(fu)
    return sample


# ----------------------------------------------------------------------
# ImitationConfig / metrics
# ----------------------------------------------------------------------


def test_imitation_config_defaults_are_sane():
    cfg = ImitationConfig()
    assert cfg.batch_size > 0
    assert cfg.num_epochs > 0
    assert 0.0 < cfg.learning_rate <= 1.0
    assert cfg.terminal_loss_coef >= 0.0
    assert cfg.yaku_loss_coef >= 0.0
    assert cfg.device == "cpu"
    assert cfg.shuffle is True


def test_metrics_to_dict_is_json_serializable():
    m = ImitationMetrics(
        loss=0.1, policy_loss=0.05, discard_loss=0.03, candidate_loss=0.02,
        terminal_loss=0.01, yaku_loss=0.04,
        discard_count=5, candidate_count=2, terminal_count=7, yaku_count=1,
        num_samples=7, num_batches=1,
        accuracy_discard=0.6, accuracy_candidate=0.5, accuracy_terminal=0.4,
        accuracy_yaku_micro=0.7, grad_norm=1.23,
    )
    s = metrics_to_json(m)
    parsed = json.loads(s)
    assert parsed["loss"] == pytest.approx(0.1)
    assert parsed["discard_count"] == 5
    assert "accuracy_yaku_micro" in parsed
    assert "grad_norm" in parsed


# ----------------------------------------------------------------------
# compute_imitation_loss: branch-specific behavior
# ----------------------------------------------------------------------


def test_compute_loss_discard_branch_finite_and_responds():
    model = _build_model()
    samples = [_make_discard_sample(tile_type=5, step_id=i) for i in range(4)]
    batch = collate_decision_samples(samples)
    loss, metrics = compute_imitation_loss(
        model, batch, ImitationConfig(terminal_loss_coef=0.0, yaku_loss_coef=0.0)
    )
    assert torch.is_tensor(loss)
    assert math.isfinite(float(loss.item()))
    assert metrics.discard_count == 4
    assert metrics.candidate_count == 0
    assert metrics.terminal_count == 0
    assert metrics.yaku_count == 0
    assert metrics.policy_loss == pytest.approx(metrics.discard_loss)
    assert metrics.num_samples == 4


def test_compute_loss_candidate_branch_finite_and_responds():
    model = _build_model()
    samples = [_make_candidate_sample(selected_index=1) for _ in range(4)]
    batch = collate_decision_samples(samples)
    loss, metrics = compute_imitation_loss(
        model, batch, ImitationConfig(terminal_loss_coef=0.0, yaku_loss_coef=0.0)
    )
    assert metrics.discard_count == 0
    assert metrics.candidate_count == 4
    assert math.isfinite(float(loss.item()))


def test_compute_loss_padding_mask_excludes_invalid_candidate_positions():
    """異なる C を持つ samples を 1 batch にしたとき、padding を CE 対象に
    しない (= 結果が C=Cmax での誤った CE と一致しないこと)。"""
    model = _build_model()
    s1 = _make_candidate_sample(selected_index=0, num_cands=2)
    s2 = _make_candidate_sample(selected_index=2, num_cands=4)
    batch = collate_decision_samples([s1, s2])
    assert batch.candidate_features.shape[1] == 4  # Cmax
    loss, metrics = compute_imitation_loss(
        model, batch, ImitationConfig(terminal_loss_coef=0.0, yaku_loss_coef=0.0)
    )
    assert metrics.candidate_count == 2
    # padding position が candidate_mask=0 で塞がれているので loss は finite
    assert math.isfinite(float(loss.item()))


def test_compute_loss_terminal_branch_responds():
    model = _build_model()
    samples = []
    for i in range(6):
        s = _make_discard_sample(tile_type=5, step_id=i)
        s = _attach_terminal(s, terminal_class=i % NUM_TERMINAL_CLASSES)
        samples.append(s)
    batch = collate_decision_samples(samples)
    _, metrics = compute_imitation_loss(
        model, batch, ImitationConfig(terminal_loss_coef=1.0, yaku_loss_coef=0.0)
    )
    assert metrics.terminal_count == 6
    assert metrics.terminal_loss > 0


def test_compute_loss_yaku_branch_winner_only():
    model = _build_model()
    s1 = _make_discard_sample(tile_type=5, step_id=0)  # non-winner
    s2 = _make_discard_sample(tile_type=5, step_id=1)
    s2 = _attach_yaku(s2, [0, 12])  # winner with riichi+tanyao indices
    samples = [s1, s2]
    batch = collate_decision_samples(samples)
    _, metrics = compute_imitation_loss(
        model, batch, ImitationConfig(terminal_loss_coef=0.0, yaku_loss_coef=1.0)
    )
    # winner-only mask → yaku count == 1
    assert metrics.yaku_count == 1
    assert metrics.yaku_loss > 0


def test_compute_loss_empty_branches_do_not_crash():
    """terminal / yaku が 0 sample でも crash しないこと。"""
    model = _build_model()
    samples = [_make_discard_sample(tile_type=5) for _ in range(4)]
    batch = collate_decision_samples(samples)
    _, metrics = compute_imitation_loss(
        model, batch, ImitationConfig()
    )
    assert metrics.terminal_count == 0
    assert metrics.terminal_loss == 0.0
    assert metrics.yaku_count == 0
    assert metrics.yaku_loss == 0.0


def test_compute_loss_all_invalid_returns_zero():
    """全 sample が candidate も discard も無効なら loss=0 で crash しない。"""
    model = _build_model()
    # discard / candidate どちらの target も -1
    s = _make_discard_sample(tile_type=5)
    s.selected_discard_tile_type = -1  # invalid
    batch = collate_decision_samples([s])
    loss, metrics = compute_imitation_loss(model, batch, ImitationConfig())
    assert metrics.discard_count == 0
    assert metrics.candidate_count == 0
    assert float(loss.item()) == 0.0


def test_train_imitation_epoch_fails_fast_on_empty_input():
    model = _build_model()
    opt = make_default_optimizer(model, ImitationConfig())
    with pytest.raises(ValueError):
        train_imitation_epoch(model, [], opt, ImitationConfig())


# ----------------------------------------------------------------------
# Overfit smokes
# ----------------------------------------------------------------------


def test_tiny_overfit_discard():
    torch.manual_seed(0)
    model = _build_model()
    samples = [_make_discard_sample(tile_type=5, step_id=i) for i in range(64)]
    cfg = ImitationConfig(
        learning_rate=1e-2, batch_size=16, num_epochs=20,
        terminal_loss_coef=0.0, yaku_loss_coef=0.0, seed=0,
    )
    result = fit_imitation(model, samples, cfg)
    assert result.epochs[0].discard_loss > result.epochs[-1].discard_loss
    assert result.epochs[-1].accuracy_discard >= 0.95


def test_tiny_overfit_candidate():
    torch.manual_seed(0)
    model = _build_model()
    samples = [_make_candidate_sample(selected_index=1) for _ in range(64)]
    cfg = ImitationConfig(
        learning_rate=1e-2, batch_size=16, num_epochs=20,
        terminal_loss_coef=0.0, yaku_loss_coef=0.0, seed=0,
    )
    result = fit_imitation(model, samples, cfg)
    assert result.epochs[0].candidate_loss > result.epochs[-1].candidate_loss
    assert result.epochs[-1].accuracy_candidate >= 0.95


def test_terminal_overfit_smoke():
    """terminal aux loss が下がること。"""
    torch.manual_seed(0)
    model = _build_model()
    samples = []
    # 同じ observation -> 同じ terminal class (固定 target)
    for i in range(64):
        s = _make_discard_sample(tile_type=5, step_id=i)
        s = _attach_terminal(s, terminal_class=2)
        samples.append(s)
    cfg = ImitationConfig(
        learning_rate=1e-2, batch_size=16, num_epochs=20,
        terminal_loss_coef=1.0, yaku_loss_coef=0.0, seed=0,
    )
    result = fit_imitation(model, samples, cfg)
    assert result.epochs[0].terminal_loss > result.epochs[-1].terminal_loss


def test_yaku_overfit_smoke():
    """yaku aux loss が下がること。"""
    torch.manual_seed(0)
    model = _build_model()
    samples = []
    for i in range(64):
        s = _make_discard_sample(tile_type=5, step_id=i)
        s = _attach_yaku(s, [0, 12, 25])
        samples.append(s)
    cfg = ImitationConfig(
        learning_rate=1e-2, batch_size=16, num_epochs=20,
        terminal_loss_coef=0.0, yaku_loss_coef=1.0, seed=0,
    )
    result = fit_imitation(model, samples, cfg)
    assert result.epochs[0].yaku_loss > result.epochs[-1].yaku_loss


# ----------------------------------------------------------------------
# Optimizer / parameter update behavior
# ----------------------------------------------------------------------


def test_model_parameters_are_updated_after_step():
    torch.manual_seed(0)
    model = _build_model()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    samples = [_make_discard_sample(tile_type=5, step_id=i) for i in range(8)]
    cfg = ImitationConfig(learning_rate=1e-2, batch_size=8, num_epochs=1, seed=0)
    opt = make_default_optimizer(model, cfg)
    train_imitation_epoch(model, samples, opt, cfg)
    # 少なくとも discard_head の重みは動く
    diff_d = (model.discard_head.weight - before["discard_head.weight"]).abs().sum().item()
    assert diff_d > 0.0


def test_max_grad_norm_clips_gradients():
    torch.manual_seed(0)
    model = _build_model()
    samples = [_make_discard_sample(tile_type=5, step_id=i) for i in range(16)]
    cfg = ImitationConfig(
        learning_rate=1e-2, batch_size=16, num_epochs=1,
        max_grad_norm=0.01, seed=0,
    )
    opt = make_default_optimizer(model, cfg)
    metrics = train_imitation_epoch(model, samples, opt, cfg)
    # grad_norm は clip 前の値が記録される (clip 自体は behavior smoke で OK)
    assert metrics.grad_norm > 0.0


# ----------------------------------------------------------------------
# Shard roundtrip integration
# ----------------------------------------------------------------------


def test_shard_roundtrip_then_train_smoke(tmp_path: Path):
    torch.manual_seed(0)
    model = _build_model()
    samples = [_make_discard_sample(tile_type=5, step_id=i) for i in range(32)]
    path = tmp_path / "shard.npz"
    write_decision_shard(path, samples)
    loaded = read_decision_shard(path)
    assert len(loaded) == 32
    cfg = ImitationConfig(
        learning_rate=1e-2, batch_size=16, num_epochs=3,
        terminal_loss_coef=0.0, yaku_loss_coef=0.0, seed=0,
    )
    result = fit_imitation(model, loaded, cfg)
    assert result.final is not None
    assert result.final.num_samples == 32
    # JSON serializable
    json.dumps(result.to_dict())


# ----------------------------------------------------------------------
# Mixed batch
# ----------------------------------------------------------------------


def test_mixed_discard_and_candidate_samples_in_one_batch():
    torch.manual_seed(0)
    model = _build_model()
    samples = []
    for i in range(8):
        samples.append(_make_discard_sample(tile_type=5, step_id=i))
    for i in range(8):
        samples.append(_make_candidate_sample(selected_index=1, step_id=100 + i))
    batch = collate_decision_samples(samples)
    _, metrics = compute_imitation_loss(
        model, batch, ImitationConfig(terminal_loss_coef=0.0, yaku_loss_coef=0.0)
    )
    assert metrics.discard_count == 8
    assert metrics.candidate_count == 8
    assert metrics.policy_loss == pytest.approx(
        metrics.discard_loss + metrics.candidate_loss
    )


# ----------------------------------------------------------------------
# Hidden info / source guards
# ----------------------------------------------------------------------


def test_imitation_trainer_does_not_import_env():
    """trainer モジュールが ``riichienv`` / ``RiichiEnvAdapter`` を import しない。"""
    import mahjong_agent.training.imitation as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    for tok in ("riichienv", "RiichiEnvAdapter", "PublicObservationEncoder"):
        assert tok not in src, (
            f"training/imitation.py references {tok!r} (should not import env)"
        )


def test_training_source_does_not_reference_hidden_state():
    import mahjong_agent.training as pkg

    pkg_dir = Path(pkg.__file__).parent
    forbidden = (
        "env.hands",
        "env.wall",
        "env.state",
        "full_state",
        "private_hand",
    )
    for f in pkg_dir.rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        for tok in forbidden:
            assert tok not in text, (
                f"training source {f.name} references {tok!r}"
            )


def test_training_source_does_not_reference_local_docs():
    import mahjong_agent.training as pkg

    pkg_dir = Path(pkg.__file__).parent
    forbidden = (
        "PROJECT_RULE.md",
        "ISSUE_BOARD.md",
        "ISSUE-",
        "AGENTS.md",
        "CLAUDE.md",
        "majong-rl",
        "CHANGE_QUEUE",
    )
    for f in pkg_dir.rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        for tok in forbidden:
            assert tok not in text, (
                f"training source {f.name} mentions {tok!r}"
            )


def test_compute_imitation_loss_signature_does_not_take_hidden_inputs():
    sig = inspect.signature(compute_imitation_loss)
    params = set(sig.parameters.keys())
    forbidden = {"hands", "wall", "state", "env", "full_state", "private_hand"}
    assert forbidden.isdisjoint(params)


# ----------------------------------------------------------------------
# tie-aware imitation: accuracy / loss semantics
# ----------------------------------------------------------------------


def _force_discard_argmax(model: Stage03Model, tile_type: int) -> None:
    """``discard_head`` を logits = one-hot(tile_type) 相当に固定する helper。

    weight=0, bias は tile_type のみ正、その他は大きく負にすることで、
    任意の trunk 出力に対して argmax が ``tile_type`` になる
    (discard_mask が当該 tile を許可している前提)。
    """
    with torch.no_grad():
        model.discard_head.weight.zero_()
        bias = torch.full_like(model.discard_head.bias, -100.0)
        bias[int(tile_type)] = 100.0
        model.discard_head.bias.copy_(bias)


def _make_discard_sample_with_best_mask(
    *,
    selected_tile: int,
    legal_tiles: tuple[int, ...],
    best_tiles: tuple[int, ...] | None,
    step_id: int = 0,
) -> DecisionSample:
    """best_mask を任意に設定できる discard sample factory。

    best_tiles=None / 空 tuple なら ``teacher_best_mask`` は全 0 (= 未提供扱い)。
    """
    s = _make_discard_sample(
        tile_type=selected_tile, legal_tiles=legal_tiles, step_id=step_id
    )
    if best_tiles:
        mask = np.zeros(34, dtype=np.float32)
        for t in best_tiles:
            mask[int(t)] = 1.0
        s.teacher_best_mask = mask
    return s


def test_tie_aware_accuracy_counts_argmax_in_best_mask_as_correct():
    """tie_aware_discard=True で best_mask 内の別 tile を argmax した場合、
    loss semantics 的に正解として ``accuracy_discard=1.0`` になること。
    """
    torch.manual_seed(0)
    model = _build_model()
    # 切る tile を argmax として固定 (tile_type=3 を argmax にする)
    _force_discard_argmax(model, tile_type=3)
    # target は 5 (best_mask に 3 と 5 を含めるので tie-aware では正解扱い)
    samples = [
        _make_discard_sample_with_best_mask(
            selected_tile=5,
            legal_tiles=(3, 5, 7),
            best_tiles=(3, 5),
            step_id=i,
        )
        for i in range(4)
    ]
    batch = collate_decision_samples(samples)
    _, metrics = compute_imitation_loss(
        model,
        batch,
        ImitationConfig(
            tie_aware_discard=True,
            terminal_loss_coef=0.0,
            yaku_loss_coef=0.0,
        ),
    )
    assert metrics.discard_count == 4
    assert metrics.accuracy_discard == pytest.approx(1.0)


def test_tie_aware_accuracy_falls_back_to_hard_top1_when_best_mask_empty():
    """tie_aware_discard=True でも best_mask 全 0 の sample では hard top-1
    一致でのみ correct (target != argmax → accuracy 0)。"""
    torch.manual_seed(0)
    model = _build_model()
    _force_discard_argmax(model, tile_type=3)
    # target=5, best_mask は未提供 (全 0) → hard top1 fallback。
    # argmax=3 != target=5 なので不一致 → accuracy=0.
    samples_miss = [
        _make_discard_sample_with_best_mask(
            selected_tile=5,
            legal_tiles=(3, 5, 7),
            best_tiles=None,
            step_id=i,
        )
        for i in range(4)
    ]
    batch_miss = collate_decision_samples(samples_miss)
    _, metrics_miss = compute_imitation_loss(
        model,
        batch_miss,
        ImitationConfig(
            tie_aware_discard=True,
            terminal_loss_coef=0.0,
            yaku_loss_coef=0.0,
        ),
    )
    assert metrics_miss.discard_count == 4
    assert metrics_miss.accuracy_discard == pytest.approx(0.0)

    # 同条件で target=3 (= argmax) なら hard top1 一致 → accuracy=1.0
    samples_hit = [
        _make_discard_sample_with_best_mask(
            selected_tile=3,
            legal_tiles=(3, 5, 7),
            best_tiles=None,
            step_id=100 + i,
        )
        for i in range(4)
    ]
    batch_hit = collate_decision_samples(samples_hit)
    _, metrics_hit = compute_imitation_loss(
        model,
        batch_hit,
        ImitationConfig(
            tie_aware_discard=True,
            terminal_loss_coef=0.0,
            yaku_loss_coef=0.0,
        ),
    )
    assert metrics_hit.discard_count == 4
    assert metrics_hit.accuracy_discard == pytest.approx(1.0)


def test_tie_aware_off_uses_hard_top1_even_with_best_mask():
    """tie_aware_discard=False では best_mask があっても hard top-1 のみで
    accuracy を計算する (旧仕様互換)。"""
    torch.manual_seed(0)
    model = _build_model()
    _force_discard_argmax(model, tile_type=3)
    # tie-aware ON なら best_mask に 3 が居るので正解扱いになる条件だが、
    # tie_aware_discard=False なので argmax=3 != target=5 → accuracy=0。
    samples = [
        _make_discard_sample_with_best_mask(
            selected_tile=5,
            legal_tiles=(3, 5, 7),
            best_tiles=(3, 5),
            step_id=i,
        )
        for i in range(4)
    ]
    batch = collate_decision_samples(samples)
    _, metrics = compute_imitation_loss(
        model,
        batch,
        ImitationConfig(
            tie_aware_discard=False,
            terminal_loss_coef=0.0,
            yaku_loss_coef=0.0,
        ),
    )
    assert metrics.discard_count == 4
    assert metrics.accuracy_discard == pytest.approx(0.0)


# ----------------------------------------------------------------------
# Pre-collated batches path
# ----------------------------------------------------------------------


def test_train_imitation_epoch_accepts_pre_collated_batches():
    torch.manual_seed(0)
    model = _build_model()
    s1 = collate_decision_samples(
        [_make_discard_sample(tile_type=5, step_id=i) for i in range(8)]
    )
    s2 = collate_decision_samples(
        [_make_discard_sample(tile_type=5, step_id=100 + i) for i in range(8)]
    )
    cfg = ImitationConfig(learning_rate=1e-2, batch_size=8, num_epochs=1, seed=0)
    opt = make_default_optimizer(model, cfg)
    metrics = train_imitation_epoch(model, [s1, s2], opt, cfg)
    assert metrics.num_batches == 2
    assert metrics.num_samples == 16


# ----------------------------------------------------------------------
# Determinism: same seed gives same epoch metrics
# ----------------------------------------------------------------------


def test_same_seed_gives_same_epoch_metrics():
    def run():
        torch.manual_seed(0)
        model = _build_model()
        samples = [
            _make_discard_sample(tile_type=(i % 7) + 1, step_id=i)
            for i in range(32)
        ]
        # legal tile を target 含むよう調整 (discard_mask が target を含むことを保証)
        for s in samples:
            t = s.selected_discard_tile_type
            s.discard_mask[t] = 1.0
        cfg = ImitationConfig(
            learning_rate=1e-2, batch_size=16, num_epochs=2,
            terminal_loss_coef=0.0, yaku_loss_coef=0.0, seed=42,
        )
        return fit_imitation(model, copy.deepcopy(samples), cfg)

    r1 = run()
    r2 = run()
    assert r1.final.loss == pytest.approx(r2.final.loss, rel=1e-6, abs=1e-6)
    assert r1.final.num_samples == r2.final.num_samples
