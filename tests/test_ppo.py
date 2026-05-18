"""Tests for the PPO learner v1 (ISSUE-0011)."""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from mahjong_agent.actions.types import ActionFamily
from mahjong_agent.data import (
    SCHEMA_VERSION,
    DecisionSample,
    collate_decision_samples,
)
from mahjong_agent.models import Stage03Model, Stage03ModelConfig
from mahjong_agent.targets.terminal import NUM_TERMINAL_CLASSES
from mahjong_agent.targets.yaku import NUM_YAKU
from mahjong_agent.training import (
    PPOBatch,
    PPOConfig,
    PPOMetrics,
    compute_ppo_loss,
    compute_returns_and_advantages,
    fit_ppo,
    make_default_ppo_optimizer,
    ppo_metrics_to_json,
    train_ppo_epoch,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

_OBS_DIM = 8
_CAND_DIM = 4


def _build_model(*, obs_dim: int = _OBS_DIM, cand_dim: int = _CAND_DIM) -> Stage03Model:
    cfg = Stage03ModelConfig(
        observation_dim=obs_dim,
        candidate_dim=cand_dim,
        hidden_dim=16,
        trunk_layers=1,
        candidate_hidden_dim=8,
    )
    return Stage03Model(cfg)


def _make_discard_sample(
    *,
    ep: str = "ep",
    pid: int = 0,
    step: int = 0,
    tile: int = 5,
    legal_tiles: tuple[int, ...] = (3, 5, 7),
    terminated: bool = False,
    round_over: bool = False,
    reward: float = 0.0,
    value: float = 0.0,
    old_log_prob: float = 0.0,
    actor_type: str = "policy",
    terminal_class: int = -1,
    yaku_indices: list[int] | None = None,
) -> DecisionSample:
    obs = np.ones(_OBS_DIM, dtype=np.float32)
    dmask = np.zeros(34, dtype=np.float32)
    for t in legal_tiles:
        dmask[int(t)] = 1.0
    yaku = np.zeros(NUM_YAKU, dtype=np.float32)
    yaku_loss_mask = 0.0
    if yaku_indices is not None:
        for j in yaku_indices:
            yaku[int(j)] = 1.0
        yaku_loss_mask = 1.0
    return DecisionSample(
        schema_version=SCHEMA_VERSION,
        episode_id=ep,
        round_id=0,
        step_id=step,
        player_id=pid,
        decision_family=ActionFamily.NORMAL_DISCARD.value,
        actor_type=actor_type,
        observation=obs,
        discard_mask=dmask,
        candidate_features=np.zeros((0, _CAND_DIM), dtype=np.float32),
        selected_discard_tile_type=int(tile),
        selected_candidate_index=-1,
        old_log_prob=float(old_log_prob),
        value=float(value),
        reward=float(reward),
        terminated=bool(terminated),
        round_over=bool(round_over),
        terminal_class=int(terminal_class),
        yaku_target=yaku,
        yaku_loss_mask=float(yaku_loss_mask),
    )


def _make_candidate_sample(
    *,
    ep: str = "ep",
    pid: int = 0,
    step: int = 0,
    family: ActionFamily = ActionFamily.PASS,
    selected_index: int = 1,
    num_cands: int = 3,
    terminated: bool = False,
    round_over: bool = False,
    reward: float = 0.0,
    value: float = 0.0,
    old_log_prob: float = 0.0,
    actor_type: str = "policy",
) -> DecisionSample:
    obs = np.ones(_OBS_DIM, dtype=np.float32) * 0.5
    cand = np.zeros((num_cands, _CAND_DIM), dtype=np.float32)
    for i in range(num_cands):
        cand[i, i % _CAND_DIM] = 1.0
    return DecisionSample(
        schema_version=SCHEMA_VERSION,
        episode_id=ep,
        round_id=0,
        step_id=step,
        player_id=pid,
        decision_family=family.value,
        actor_type=actor_type,
        observation=obs,
        discard_mask=np.zeros(34, dtype=np.float32),
        candidate_features=cand,
        selected_discard_tile_type=-1,
        selected_candidate_index=int(selected_index),
        old_log_prob=float(old_log_prob),
        value=float(value),
        reward=float(reward),
        terminated=bool(terminated),
        round_over=bool(round_over),
        yaku_target=np.zeros(NUM_YAKU, dtype=np.float32),
    )


def _set_old_log_probs_to_current(model: Stage03Model, samples: list[DecisionSample]):
    """各 sample の ``old_log_prob`` を現在の model の **combined** log_prob
    に揃える (= ratio=1 で始めるテスト用 helper)。

    ``compute_ppo_loss`` は combined logits ``[discard (34), candidate (Cmax)]``
    の softmax から ``new_log_prob`` を取り出すので、old/new の formulation
    を揃えるためにこの helper も combined を使う。
    """
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
                if idx < 0:
                    continue
            else:
                if int(s.selected_candidate_index) < 0:
                    continue
                idx = 34 + int(s.selected_candidate_index)
            s.old_log_prob = float(log_softmax[i, idx].item())
    model.train()


def _make_ppo_batch(
    samples: list[DecisionSample], config: PPOConfig
) -> PPOBatch:
    data = compute_returns_and_advantages(samples, config)
    return PPOBatch(
        batch=collate_decision_samples(samples),
        returns=torch.from_numpy(data.returns),
        advantages=torch.from_numpy(data.advantages),
        eligible=torch.from_numpy(data.eligible),
    )


# ----------------------------------------------------------------------
# Config / metrics
# ----------------------------------------------------------------------


def test_ppo_config_defaults_match_spec():
    cfg = PPOConfig()
    assert cfg.gamma == pytest.approx(0.95)
    assert cfg.gae_lambda == pytest.approx(0.95)
    assert cfg.clip_epsilon == pytest.approx(0.2)
    assert cfg.value_loss_coef == pytest.approx(0.5)
    assert cfg.entropy_coef == pytest.approx(0.01)
    assert cfg.policy_only is False
    assert cfg.separated is True
    assert cfg.target_kl_enabled is True
    assert cfg.gradient_norms_enabled is False
    assert cfg.include_actor_types == ("policy",)


def test_metrics_to_dict_is_json_serializable():
    m = PPOMetrics(
        loss=0.1, policy_loss=0.05, value_loss=0.03, entropy=0.7,
        discard_count=4, candidate_count=2, num_samples=6, num_batches=1,
        ppo_included_count=6,
        decision_family={"normal_discard": {"count": 4.0, "policy_loss": 0.05}},
        actor_type_counts={"policy": 6},
        grad_norms_per_component={"policy": 1.2},
    )
    s = ppo_metrics_to_json(m)
    parsed = json.loads(s)
    assert parsed["entropy"] == pytest.approx(0.7)
    assert parsed["decision_family"]["normal_discard"]["count"] == 4.0
    assert parsed["actor_type_counts"]["policy"] == 6
    assert parsed["grad_norms_per_component"]["policy"] == pytest.approx(1.2)


# ----------------------------------------------------------------------
# Returns / advantages
# ----------------------------------------------------------------------


def test_returns_advantages_simple_trajectory_alignment():
    """手計算で確認できる簡単な trajectory で GAE 計算の整合を確認する。"""
    cfg = PPOConfig(gamma=1.0, gae_lambda=1.0)
    samples = [
        _make_discard_sample(step=0, reward=0.0, value=0.0),
        _make_discard_sample(step=1, reward=0.0, value=0.0),
        _make_discard_sample(step=2, reward=1.0, value=0.0, terminated=True, round_over=True),
    ]
    data = compute_returns_and_advantages(samples, cfg)
    # gamma=1, lambda=1, V=0: returns is cumulative reward from t to end.
    # All rewards are 0 except step 2 = 1.0. So returns[0..2] = 1, 1, 1.
    np.testing.assert_allclose(data.returns, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(data.advantages, [1.0, 1.0, 1.0])


def test_returns_round_over_bootstrap_zero():
    """round_over=True で bootstrap を切ること。"""
    cfg = PPOConfig(gamma=0.5, gae_lambda=0.0)  # lambda=0 → TD(0)
    samples = [
        _make_discard_sample(step=0, reward=0.0, value=10.0, round_over=False),
        _make_discard_sample(step=1, reward=1.0, value=10.0, round_over=True),
        _make_discard_sample(step=2, reward=0.0, value=10.0, terminated=True, round_over=True),
    ]
    data = compute_returns_and_advantages(samples, cfg)
    # TD(0) delta_t = r_t + gamma * V_{t+1} * not_done - V_t
    # step 2: terminated → not_done=0 → delta = 0 - 10 = -10, return = 0
    # step 1: round_over → not_done=0 → delta = 1 - 10 = -9, return = 1
    # step 0: not_done=1 → delta = 0 + 0.5 * 10 - 10 = -5, return = 5
    np.testing.assert_allclose(data.returns, [5.0, 1.0, 0.0], rtol=1e-5)
    np.testing.assert_allclose(data.advantages, [-5.0, -9.0, -10.0], rtol=1e-5)


def test_returns_separate_player_trajectories():
    """別 player の sample は独立 trajectory として GAE される。"""
    cfg = PPOConfig(gamma=1.0, gae_lambda=1.0)
    samples = [
        _make_discard_sample(pid=0, step=0, reward=0.0, value=0.0),
        _make_discard_sample(pid=1, step=1, reward=2.0, value=0.0, terminated=True, round_over=True),
        _make_discard_sample(pid=0, step=2, reward=3.0, value=0.0, terminated=True, round_over=True),
    ]
    data = compute_returns_and_advantages(samples, cfg)
    # player 0 traj: step 0 (reward 0) -> step 2 (reward 3, term) → return0=3
    # player 1 traj: step 1 alone (reward 2, term) → return=2
    assert data.returns[0] == pytest.approx(3.0)
    assert data.returns[1] == pytest.approx(2.0)
    assert data.returns[2] == pytest.approx(3.0)


def test_eligible_excludes_non_policy_actor_types_by_default():
    cfg = PPOConfig()
    samples = [
        _make_discard_sample(step=0, actor_type="policy"),
        _make_discard_sample(step=1, actor_type="rule_based"),
        _make_discard_sample(step=2, actor_type="random"),
    ]
    data = compute_returns_and_advantages(samples, cfg)
    assert data.eligible[0] is np.True_ or data.eligible[0]
    assert not data.eligible[1]
    assert not data.eligible[2]


def test_eligible_includes_actor_types_when_opt_in():
    cfg = PPOConfig(include_actor_types=("policy", "rule_based"))
    samples = [
        _make_discard_sample(step=0, actor_type="policy"),
        _make_discard_sample(step=1, actor_type="rule_based"),
        _make_discard_sample(step=2, actor_type="random"),
    ]
    data = compute_returns_and_advantages(samples, cfg)
    assert bool(data.eligible[0]) is True
    assert bool(data.eligible[1]) is True
    assert bool(data.eligible[2]) is False


# ----------------------------------------------------------------------
# compute_ppo_loss
# ----------------------------------------------------------------------


def test_ppo_loss_discard_branch_finite():
    torch.manual_seed(0)
    model = _build_model()
    samples = [
        _make_discard_sample(step=i, reward=0.0, value=0.0, terminated=(i == 3), round_over=(i == 3))
        for i in range(4)
    ]
    _set_old_log_probs_to_current(model, samples)
    samples[-1].reward = 1.0  # 末端で reward 与える
    cfg = PPOConfig(target_kl_enabled=False, value_loss_coef=0.0, entropy_coef=0.0)
    ppo_batch = _make_ppo_batch(samples, cfg)
    loss, metrics = compute_ppo_loss(model, ppo_batch, cfg)
    assert math.isfinite(float(loss.item()))
    assert metrics.discard_count == 4
    assert metrics.candidate_count == 0
    assert metrics.ppo_included_count == 4


def test_ppo_loss_candidate_branch_finite():
    torch.manual_seed(0)
    model = _build_model()
    samples = [
        _make_candidate_sample(step=i, reward=0.0, value=0.0, terminated=(i == 3), round_over=(i == 3))
        for i in range(4)
    ]
    _set_old_log_probs_to_current(model, samples)
    samples[-1].reward = 1.0
    cfg = PPOConfig(target_kl_enabled=False, value_loss_coef=0.0, entropy_coef=0.0)
    ppo_batch = _make_ppo_batch(samples, cfg)
    loss, metrics = compute_ppo_loss(model, ppo_batch, cfg)
    assert math.isfinite(float(loss.item()))
    assert metrics.discard_count == 0
    assert metrics.candidate_count == 4


def test_ppo_loss_mixed_family_batch():
    torch.manual_seed(0)
    model = _build_model()
    samples = []
    samples.extend(_make_discard_sample(step=i, reward=0.0, value=0.0) for i in range(4))
    samples.extend(_make_candidate_sample(step=10 + i, reward=0.0, value=0.0) for i in range(4))
    samples[-1].terminated = True
    samples[-1].round_over = True
    samples[-1].reward = 1.0
    _set_old_log_probs_to_current(model, samples)
    cfg = PPOConfig(target_kl_enabled=False, value_loss_coef=0.0, entropy_coef=0.0)
    ppo_batch = _make_ppo_batch(samples, cfg)
    loss, metrics = compute_ppo_loss(model, ppo_batch, cfg)
    assert metrics.discard_count == 4
    assert metrics.candidate_count == 4
    assert "normal_discard" in metrics.decision_family
    assert "candidate" in metrics.decision_family
    assert math.isfinite(float(loss.item()))


def test_ppo_loss_excludes_non_policy_actor_types_from_ratio():
    torch.manual_seed(0)
    model = _build_model()
    samples = [
        _make_discard_sample(step=0, actor_type="policy", terminated=True, round_over=True, reward=1.0),
        _make_discard_sample(step=1, actor_type="rule_based", terminated=True, round_over=True, reward=1.0),
        _make_discard_sample(step=2, actor_type="random", terminated=True, round_over=True, reward=1.0),
    ]
    _set_old_log_probs_to_current(model, samples)
    cfg = PPOConfig(target_kl_enabled=False)
    ppo_batch = _make_ppo_batch(samples, cfg)
    _, metrics = compute_ppo_loss(model, ppo_batch, cfg)
    assert metrics.discard_count == 1
    assert metrics.ppo_included_count == 1
    assert metrics.ppo_excluded_count == 2
    assert metrics.ppo_excluded_by_actor_type == 2
    assert "rule_based" in metrics.actor_type_counts
    assert "random" in metrics.actor_type_counts


def test_ppo_loss_old_log_prob_match_gives_ratio_1():
    """old_log_prob を現在 policy に合わせると ratio=1, kl≈0。"""
    torch.manual_seed(0)
    model = _build_model()
    samples = [
        _make_discard_sample(step=i, reward=0.0, value=0.0)
        for i in range(8)
    ]
    samples[-1].terminated = True
    samples[-1].round_over = True
    samples[-1].reward = 1.0
    _set_old_log_probs_to_current(model, samples)
    cfg = PPOConfig(target_kl_enabled=False, value_loss_coef=0.0, entropy_coef=0.0)
    ppo_batch = _make_ppo_batch(samples, cfg)
    _, metrics = compute_ppo_loss(model, ppo_batch, cfg)
    assert metrics.approx_kl_mean == pytest.approx(0.0, abs=1e-5)
    # clip_fraction is 0 because ratio==1
    assert metrics.clip_fraction == pytest.approx(0.0)


# ----------------------------------------------------------------------
# Auxiliary / policy_only
# ----------------------------------------------------------------------


def test_policy_only_mode_does_not_update_value_head():
    torch.manual_seed(0)
    model = _build_model()
    samples = [
        _make_discard_sample(step=i, reward=1.0, value=0.0, terminated=True, round_over=True)
        for i in range(8)
    ]
    _set_old_log_probs_to_current(model, samples)
    before = model.value_head.weight.detach().clone()
    cfg = PPOConfig(
        num_epochs=2, batch_size=8, learning_rate=1e-2,
        policy_only=True, target_kl_enabled=False,
    )
    fit_ppo(model, samples, cfg)
    after = model.value_head.weight.detach()
    # policy_only=True なら value head は更新されない
    assert torch.allclose(before, after)


def test_value_mode_updates_value_head():
    torch.manual_seed(0)
    model = _build_model()
    samples = [
        _make_discard_sample(step=i, reward=1.0, value=0.0, terminated=True, round_over=True)
        for i in range(8)
    ]
    _set_old_log_probs_to_current(model, samples)
    before = model.value_head.weight.detach().clone()
    cfg = PPOConfig(
        num_epochs=2, batch_size=8, learning_rate=1e-2,
        policy_only=False, target_kl_enabled=False,
    )
    fit_ppo(model, samples, cfg)
    after = model.value_head.weight.detach()
    assert not torch.allclose(before, after)


def test_policy_only_skips_terminal_and_yaku_aux():
    torch.manual_seed(0)
    model = _build_model()
    samples = []
    for i in range(8):
        s = _make_discard_sample(step=i, reward=1.0, value=0.0,
                                 terminated=(i == 7), round_over=(i == 7),
                                 terminal_class=2, yaku_indices=[0, 5])
        samples.append(s)
    _set_old_log_probs_to_current(model, samples)
    cfg = PPOConfig(target_kl_enabled=False, policy_only=True)
    ppo_batch = _make_ppo_batch(samples, cfg)
    _, metrics = compute_ppo_loss(model, ppo_batch, cfg)
    assert metrics.terminal_count == 0
    assert metrics.yaku_count == 0
    assert metrics.terminal_loss == 0.0
    assert metrics.yaku_loss == 0.0


def test_value_mode_uses_terminal_and_yaku_aux():
    torch.manual_seed(0)
    model = _build_model()
    samples = []
    for i in range(8):
        s = _make_discard_sample(step=i, reward=1.0, value=0.0,
                                 terminated=(i == 7), round_over=(i == 7),
                                 terminal_class=2,
                                 yaku_indices=[0, 5] if i == 7 else None)
        samples.append(s)
    _set_old_log_probs_to_current(model, samples)
    cfg = PPOConfig(target_kl_enabled=False, policy_only=False)
    ppo_batch = _make_ppo_batch(samples, cfg)
    _, metrics = compute_ppo_loss(model, ppo_batch, cfg)
    assert metrics.terminal_count == 8
    assert metrics.yaku_count == 1
    assert metrics.terminal_loss > 0
    assert metrics.yaku_loss > 0


# ----------------------------------------------------------------------
# Entropy
# ----------------------------------------------------------------------


def test_entropy_bonus_affects_total_loss():
    """entropy_coef を変えると total loss が変わる (符号: -coef*entropy)。"""
    torch.manual_seed(0)
    model = _build_model()
    samples = [
        _make_discard_sample(step=i, reward=0.0, value=0.0,
                             terminated=(i == 3), round_over=(i == 3))
        for i in range(4)
    ]
    samples[-1].reward = 1.0
    _set_old_log_probs_to_current(model, samples)
    cfg_a = PPOConfig(
        target_kl_enabled=False, value_loss_coef=0.0, entropy_coef=0.0,
    )
    cfg_b = PPOConfig(
        target_kl_enabled=False, value_loss_coef=0.0, entropy_coef=0.5,
    )
    pb_a = _make_ppo_batch(samples, cfg_a)
    pb_b = _make_ppo_batch(samples, cfg_b)
    loss_a, m_a = compute_ppo_loss(model, pb_a, cfg_a)
    loss_b, m_b = compute_ppo_loss(model, pb_b, cfg_b)
    # entropy が positive → -coef*entropy<0 → loss_b < loss_a
    assert m_a.entropy > 0
    assert float(loss_b.item()) < float(loss_a.item())


# ----------------------------------------------------------------------
# Target KL
# ----------------------------------------------------------------------


def test_target_kl_skip_increments_skipped_and_not_applied():
    """approx_kl が target を超える場合 skip され、applied に混ぜない。"""
    torch.manual_seed(0)
    model = _build_model()
    samples = [
        _make_discard_sample(step=i, reward=1.0, value=0.0,
                             terminated=True, round_over=True,
                             old_log_prob=-5.0)  # 大きな old_log_prob → ratio 高 → kl 高
        for i in range(4)
    ]
    cfg = PPOConfig(
        num_epochs=1, batch_size=4, learning_rate=1e-3,
        target_kl_enabled=True, target_kl=0.001,
        target_kl_stop_multiplier=1.0,
        target_kl_skip_minibatch_on_exceed=True,
    )
    opt = make_default_ppo_optimizer(model, cfg)
    data = compute_returns_and_advantages(samples, cfg)
    metrics, early_stopped = train_ppo_epoch(model, data, opt, cfg)
    assert metrics.target_kl_checked_minibatches >= 1
    assert metrics.target_kl_skipped_minibatches >= 1
    assert metrics.target_kl_applied_minibatches == 0
    assert metrics.num_updates == 0
    assert early_stopped is True


def test_target_kl_disabled_does_not_skip():
    torch.manual_seed(0)
    model = _build_model()
    samples = [
        _make_discard_sample(step=i, reward=1.0, value=0.0,
                             terminated=True, round_over=True,
                             old_log_prob=-5.0)
        for i in range(4)
    ]
    cfg = PPOConfig(
        num_epochs=1, batch_size=4, learning_rate=1e-3,
        target_kl_enabled=False,
    )
    opt = make_default_ppo_optimizer(model, cfg)
    data = compute_returns_and_advantages(samples, cfg)
    metrics, early_stopped = train_ppo_epoch(model, data, opt, cfg)
    assert metrics.target_kl_skipped_minibatches == 0
    assert metrics.target_kl_applied_minibatches >= 1
    assert metrics.num_updates >= 1
    assert early_stopped is False


# ----------------------------------------------------------------------
# Gradient norm diagnostics
# ----------------------------------------------------------------------


def test_gradient_norm_default_off_does_not_call_autograd_grad(monkeypatch):
    """gradient_norms_enabled=False (default) で torch.autograd.grad が呼ばれない。"""
    torch.manual_seed(0)
    model = _build_model()
    samples = [
        _make_discard_sample(step=i, reward=0.0, value=0.0,
                             terminated=(i == 3), round_over=(i == 3))
        for i in range(4)
    ]
    samples[-1].reward = 1.0
    _set_old_log_probs_to_current(model, samples)
    cfg = PPOConfig(num_epochs=1, batch_size=4, learning_rate=1e-3,
                    target_kl_enabled=False, gradient_norms_enabled=False)
    opt = make_default_ppo_optimizer(model, cfg)
    data = compute_returns_and_advantages(samples, cfg)

    call_count = {"n": 0}
    orig_grad = torch.autograd.grad

    def _spy_grad(*args, **kwargs):
        call_count["n"] += 1
        return orig_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", _spy_grad)
    train_ppo_epoch(model, data, opt, cfg)
    assert call_count["n"] == 0


def test_gradient_norm_enabled_records_component_norms():
    torch.manual_seed(0)
    model = _build_model()
    samples = [
        _make_discard_sample(step=i, reward=0.0, value=0.0,
                             terminated=(i == 3), round_over=(i == 3))
        for i in range(4)
    ]
    samples[-1].reward = 1.0
    _set_old_log_probs_to_current(model, samples)
    cfg = PPOConfig(num_epochs=1, batch_size=4, learning_rate=1e-3,
                    target_kl_enabled=False, gradient_norms_enabled=True,
                    gradient_norms_max_batches_per_epoch=2)
    opt = make_default_ppo_optimizer(model, cfg)
    data = compute_returns_and_advantages(samples, cfg)
    metrics, _ = train_ppo_epoch(model, data, opt, cfg)
    # 少なくとも policy component の grad_norm が記録される
    assert "policy" in metrics.grad_norms_per_component
    assert metrics.grad_norms_per_component["policy"] >= 0.0


# ----------------------------------------------------------------------
# Tiny update smoke
# ----------------------------------------------------------------------


def test_tiny_ppo_update_changes_policy_params():
    torch.manual_seed(0)
    model = _build_model()
    # 異なる reward / value で advantage に variance を持たせる
    samples = [
        _make_discard_sample(step=i, reward=float(i), value=0.0,
                             terminated=True, round_over=True)
        for i in range(8)
    ]
    _set_old_log_probs_to_current(model, samples)
    before = model.discard_head.weight.detach().clone()
    cfg = PPOConfig(
        num_epochs=3, batch_size=8, learning_rate=1e-2,
        target_kl_enabled=False,
    )
    fit_ppo(model, samples, cfg)
    after = model.discard_head.weight.detach()
    assert not torch.allclose(before, after)


def test_fit_ppo_empty_samples_raises():
    model = _build_model()
    with pytest.raises(ValueError):
        fit_ppo(model, [], PPOConfig())


# ----------------------------------------------------------------------
# Diagnostics / JSON serializable
# ----------------------------------------------------------------------


def test_decision_family_diagnostics_json_serializable():
    torch.manual_seed(0)
    model = _build_model()
    samples = []
    samples.extend(
        _make_discard_sample(step=i, reward=0.0, value=0.0,
                             terminated=(i == 3), round_over=(i == 3))
        for i in range(4)
    )
    samples.extend(
        _make_candidate_sample(step=10 + i, reward=0.0, value=0.0,
                               terminated=(i == 3), round_over=(i == 3))
        for i in range(4)
    )
    samples[-1].reward = 1.0
    _set_old_log_probs_to_current(model, samples)
    cfg = PPOConfig(num_epochs=1, batch_size=8, target_kl_enabled=False)
    result = fit_ppo(model, samples, cfg)
    m = result.final
    assert m is not None
    s = json.dumps(m.to_dict())
    parsed = json.loads(s)
    assert "decision_family" in parsed
    # epoch aggregate には両 family が入る
    assert "normal_discard" in parsed["decision_family"]
    assert "candidate" in parsed["decision_family"]


def test_excluded_sample_diagnostics_json_serializable():
    torch.manual_seed(0)
    model = _build_model()
    samples = [
        _make_discard_sample(step=0, actor_type="policy", terminated=True, round_over=True, reward=1.0),
        _make_discard_sample(step=1, actor_type="rule_based", terminated=True, round_over=True, reward=1.0),
        _make_discard_sample(step=2, actor_type="random", terminated=True, round_over=True, reward=1.0),
    ]
    _set_old_log_probs_to_current(model, samples)
    cfg = PPOConfig(num_epochs=1, batch_size=3, target_kl_enabled=False)
    result = fit_ppo(model, samples, cfg)
    m = result.final
    assert m is not None
    parsed = json.loads(json.dumps(m.to_dict()))
    assert parsed["ppo_excluded_by_actor_type"] >= 2
    assert "rule_based" in parsed["actor_type_counts"]


def test_ppo_run_result_json_serializable():
    torch.manual_seed(0)
    model = _build_model()
    samples = [
        _make_discard_sample(step=i, reward=1.0, value=0.0,
                             terminated=True, round_over=True)
        for i in range(8)
    ]
    _set_old_log_probs_to_current(model, samples)
    cfg = PPOConfig(num_epochs=2, batch_size=8, target_kl_enabled=False)
    result = fit_ppo(model, samples, cfg)
    s = json.dumps(result.to_dict())
    parsed = json.loads(s)
    assert "epochs" in parsed
    assert "final" in parsed
    assert "early_stopped" in parsed


# ----------------------------------------------------------------------
# Hidden info / source guards
# ----------------------------------------------------------------------


def test_compute_ppo_loss_signature_does_not_take_hidden_inputs():
    sig = inspect.signature(compute_ppo_loss)
    params = set(sig.parameters.keys())
    forbidden = {"hands", "wall", "state", "env", "full_state", "private_hand"}
    assert forbidden.isdisjoint(params)


def test_ppo_trainer_does_not_import_env():
    import mahjong_agent.training.ppo as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    for tok in ("riichienv", "RiichiEnvAdapter", "PublicObservationEncoder"):
        assert tok not in src, f"ppo.py references {tok!r}"


def test_training_source_does_not_reference_hidden_state():
    import mahjong_agent.training as pkg

    pkg_dir = Path(pkg.__file__).parent
    forbidden = (
        "env.hands", "env.wall", "env.state", "full_state", "private_hand"
    )
    for f in pkg_dir.rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        for tok in forbidden:
            assert tok not in text, f"training source {f.name} references {tok!r}"


def test_training_source_does_not_reference_local_docs():
    import mahjong_agent.training as pkg

    pkg_dir = Path(pkg.__file__).parent
    forbidden = (
        "PROJECT_RULE.md", "ISSUE_BOARD.md", "ISSUE-", "AGENTS.md",
        "CLAUDE.md", "majong-rl", "CHANGE_QUEUE",
    )
    for f in pkg_dir.rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        for tok in forbidden:
            assert tok not in text, f"training source {f.name} mentions {tok!r}"


# ----------------------------------------------------------------------
# Auxiliary: terminal target_class shape consistency
# ----------------------------------------------------------------------


def test_terminal_class_target_in_range():
    """terminal_class >= 0 の sample が CE 入力に正しく入ること。"""
    model = _build_model()
    samples = []
    for i in range(NUM_TERMINAL_CLASSES):
        s = _make_discard_sample(step=i, reward=0.0, value=0.0,
                                 terminated=(i == NUM_TERMINAL_CLASSES - 1),
                                 round_over=(i == NUM_TERMINAL_CLASSES - 1),
                                 terminal_class=i)
        samples.append(s)
    samples[-1].reward = 1.0
    _set_old_log_probs_to_current(model, samples)
    cfg = PPOConfig(target_kl_enabled=False, policy_only=False)
    ppo_batch = _make_ppo_batch(samples, cfg)
    _, metrics = compute_ppo_loss(model, ppo_batch, cfg)
    assert metrics.terminal_count == NUM_TERMINAL_CLASSES
    assert math.isfinite(metrics.terminal_loss)
