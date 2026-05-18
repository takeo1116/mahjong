"""Batch 1 integration tests: rule-based agent + teacher_best_mask schema +
tie-aware imitation loss + SelfPlayRunner wiring.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from mahjong_agent.actions.types import (
    ActionFamily,
    ActionKey,
    LegalActionSet,
    ModelAction,
)
from mahjong_agent.agents import RandomAgent, RuleBasedBaselineAgent
from mahjong_agent.agents.base import AgentDecision
from mahjong_agent.data import (
    SCHEMA_VERSION,
    DecisionSample,
    collate_decision_samples,
    read_decision_shard,
    write_decision_shard,
)
from mahjong_agent.evaluation import (
    SeatAgents,
    SelfPlayConfig,
    SelfPlayRunner,
)
from mahjong_agent.models import Stage03Model, Stage03ModelConfig
from mahjong_agent.targets.yaku import NUM_YAKU
from mahjong_agent.training import (
    ImitationConfig,
    compute_imitation_loss,
)

# ----------------------------------------------------------------------
# Schema version + DecisionSample.teacher_best_mask
# ----------------------------------------------------------------------


def test_schema_version_is_v2():
    assert SCHEMA_VERSION == 2


def test_decision_sample_teacher_best_mask_default_zero():
    s = DecisionSample()
    assert s.teacher_best_mask.shape == (34,)
    assert s.teacher_best_mask.dtype == np.float32
    assert float(s.teacher_best_mask.sum()) == 0.0


def test_teacher_best_mask_collate_roundtrip():
    """collate で teacher_best_mask が (N, 34) tensor として stack される。"""
    s0 = DecisionSample(
        episode_id="e", round_id=0, step_id=0, player_id=0,
        decision_family=ActionFamily.NORMAL_DISCARD.value, actor_type="rule_based",
        observation=np.ones(8, dtype=np.float32),
        discard_mask=np.zeros(34, dtype=np.float32),
        candidate_features=np.zeros((0, 4), dtype=np.float32),
        selected_discard_tile_type=5,
        teacher_best_mask=_mask_of(5, 7),
        teacher_available=True,
    )
    s1 = DecisionSample(
        episode_id="e", round_id=0, step_id=1, player_id=1,
        decision_family=ActionFamily.NORMAL_DISCARD.value, actor_type="rule_based",
        observation=np.ones(8, dtype=np.float32),
        discard_mask=np.zeros(34, dtype=np.float32),
        candidate_features=np.zeros((0, 4), dtype=np.float32),
        selected_discard_tile_type=3,
        teacher_best_mask=_mask_of(3),
        teacher_available=True,
    )
    batch = collate_decision_samples([s0, s1])
    assert batch.teacher_best_mask.shape == (2, 34)
    assert batch.teacher_best_mask.dtype == torch.float32
    assert batch.teacher_best_mask[0, 5] == 1.0
    assert batch.teacher_best_mask[0, 7] == 1.0
    assert batch.teacher_best_mask[1, 3] == 1.0


def test_teacher_best_mask_shard_roundtrip(tmp_path: Path):
    s = DecisionSample(
        episode_id="e",
        observation=np.arange(8, dtype=np.float32),
        discard_mask=np.zeros(34, dtype=np.float32),
        candidate_features=np.zeros((0, 4), dtype=np.float32),
        teacher_best_mask=_mask_of(11, 15, 22),
        teacher_available=True,
    )
    path = tmp_path / "shard.npz"
    write_decision_shard(path, [s])
    loaded = read_decision_shard(path)
    assert len(loaded) == 1
    np.testing.assert_array_equal(
        loaded[0].teacher_best_mask, s.teacher_best_mask
    )
    assert loaded[0].teacher_available is True


def _mask_of(*tile_types: int) -> np.ndarray:
    m = np.zeros(34, dtype=np.float32)
    for t in tile_types:
        m[t] = 1.0
    return m


# ----------------------------------------------------------------------
# RuleBasedBaselineAgent: shanten min + teacher_best_mask
# ----------------------------------------------------------------------


def _normal_discard_legal_set(tile_types: list[int]) -> LegalActionSet:
    nd = {
        t: ModelAction(
            key=ActionKey(ActionFamily.NORMAL_DISCARD, tile_type=t),
            actor=0,
            _raw_actions=(),
        )
        for t in tile_types
    }
    return LegalActionSet(decision_player=0, normal_discard=nd)


class _FakeObs:
    """riichienv.Observation を模す軽量 stub。"""

    def __init__(self, hand: list[int], player_id: int = 0, melds=None):
        self.hand = hand
        self.player_id = player_id
        self.melds = melds if melds is not None else [[], [], [], []]


def test_rule_based_baseline_picks_shanten_min_with_teacher_mask():
    agent = RuleBasedBaselineAgent()
    # 14 牌: 1m×3 2m×3 3m×3 4m×2 9m×2 + 1s (浮き) → 1s を切ると agari に近づく
    # tile_ids: 1m tile_type=0 → tile_id 0..3
    hand_ids = (
        [0, 1, 2]          # 1m × 3 (tile_ids 0..3 of tile_type 0)
        + [4, 5, 6]        # 2m × 3
        + [8, 9, 10]       # 3m × 3
        + [12, 13]         # 4m × 2
        + [32, 33]         # 9m × 2
        + [72]             # 1s = tile_type 18
    )
    obs = _FakeObs(hand_ids, player_id=0)
    legal_set = _normal_discard_legal_set(
        sorted({tid // 4 for tid in hand_ids})
    )
    dec = agent.select_action(legal_set, observation=obs)
    assert dec.action.family == ActionFamily.NORMAL_DISCARD
    # 1s (tile_type 18) を切るのが best
    assert dec.action.tile_type == 18
    # extras に teacher_best_mask と teacher_discard_tile_type が入る
    assert "teacher_best_mask" in dec.extras
    mask = dec.extras["teacher_best_mask"]
    assert mask.shape == (34,)
    assert mask[18] == 1.0
    assert dec.extras["teacher_discard_tile_type"] == 18


def test_rule_based_baseline_falls_back_without_observation():
    """observation=None なら legacy "max tile_type" にフォールバックする。"""
    agent = RuleBasedBaselineAgent()
    legal_set = _normal_discard_legal_set([3, 5, 17])
    dec = agent.select_action(legal_set, observation=None)
    assert dec.action.tile_type == 17  # 最大 tile_type
    # teacher_best_mask は extras に居ない
    assert "teacher_best_mask" not in dec.extras


def test_rule_based_baseline_disabled_shanten_uses_legacy():
    agent = RuleBasedBaselineAgent(use_shanten_discard=False)
    legal_set = _normal_discard_legal_set([3, 5, 17])
    dec = agent.select_action(legal_set, observation=_FakeObs([0]))
    assert dec.action.tile_type == 17  # legacy 最大 tile_type


def test_rule_based_call_yakuhai_pon_in_response_phase():
    agent = RuleBasedBaselineAgent()
    legal = LegalActionSet(
        decision_player=0,
        candidates=(
            ModelAction(
                key=ActionKey(ActionFamily.PASS), actor=0,
            ),
            ModelAction(
                key=ActionKey(
                    ActionFamily.PON, tile_type=31,
                    consume_tile_types=(31, 31),
                    target_rel_seat=3,
                ),
                actor=0,
            ),
        ),
    )
    # 手に 白 × 2 (= ポン可)。tile_id 124..127 が 白 (tile_type 31)。
    obs = _FakeObs([124, 125], player_id=0)
    dec = agent.select_action(legal, observation=obs)
    assert dec.action.family == ActionFamily.PON


# ----------------------------------------------------------------------
# Tie-aware imitation loss
# ----------------------------------------------------------------------


def _build_small_model(obs_dim: int = 8, cand_dim: int = 4) -> Stage03Model:
    cfg = Stage03ModelConfig(
        observation_dim=obs_dim, candidate_dim=cand_dim,
        hidden_dim=16, trunk_layers=1, candidate_hidden_dim=8,
    )
    return Stage03Model(cfg)


def _discard_sample(
    *,
    tile_type: int,
    teacher_best_mask: np.ndarray | None = None,
    obs_value: float = 1.0,
    step_id: int = 0,
) -> DecisionSample:
    return DecisionSample(
        episode_id="e", round_id=0, step_id=step_id, player_id=0,
        decision_family=ActionFamily.NORMAL_DISCARD.value, actor_type="rule_based",
        observation=np.full(8, obs_value, dtype=np.float32),
        discard_mask=np.ones(34, dtype=np.float32),
        candidate_features=np.zeros((0, 4), dtype=np.float32),
        selected_discard_tile_type=tile_type,
        teacher_best_mask=(
            teacher_best_mask
            if teacher_best_mask is not None
            else np.zeros(34, dtype=np.float32)
        ),
        teacher_available=teacher_best_mask is not None,
        yaku_target=np.zeros(NUM_YAKU, dtype=np.float32),
    )


def test_tie_aware_loss_lower_when_target_in_mask():
    """tie-aware: top1 と異なる tile を model が好んでいても、tile が mask
    に入っていれば loss は hard CE より小さくなる。"""
    torch.manual_seed(0)
    model = _build_small_model()
    # mask: tile_type 5 と 12 が tied best。selected (hard target) = 5。
    mask = _mask_of(5, 12)
    samples = [
        _discard_sample(tile_type=5, teacher_best_mask=mask, step_id=i)
        for i in range(4)
    ]
    batch = collate_decision_samples(samples)

    cfg_hard = ImitationConfig(
        tie_aware_discard=False,
        terminal_loss_coef=0.0, yaku_loss_coef=0.0,
    )
    cfg_tie = ImitationConfig(
        tie_aware_discard=True,
        terminal_loss_coef=0.0, yaku_loss_coef=0.0,
    )
    _, m_hard = compute_imitation_loss(model, batch, cfg_hard)
    _, m_tie = compute_imitation_loss(model, batch, cfg_tie)
    # 同じ確率分布上で hard CE は -log(p_5)、tie-aware は -log(p_5 + p_12)
    # → tie-aware loss ≤ hard loss
    assert m_tie.discard_loss <= m_hard.discard_loss + 1e-6
    # 12 に少しでも mass があれば strict <
    # (= 初期化が偏らない限りほぼ確実に成り立つ)


def test_tie_aware_loss_zero_when_softmax_concentrated_on_mask():
    """tie-aware で teacher_best_mask 内の任意 tile に確率を集中させれば
    loss はほぼ 0 に下がる。"""
    torch.manual_seed(0)
    model = _build_small_model()
    mask = _mask_of(5, 12)
    samples = [
        _discard_sample(tile_type=5, teacher_best_mask=mask, step_id=i)
        for i in range(16)
    ]
    cfg = ImitationConfig(
        learning_rate=5e-2, batch_size=16, num_epochs=50,
        tie_aware_discard=True,
        terminal_loss_coef=0.0, yaku_loss_coef=0.0,
    )
    from mahjong_agent.training import fit_imitation

    result = fit_imitation(model, samples, cfg)
    # 学習後の discard_loss は十分小さくなる
    assert result.epochs[-1].discard_loss < result.epochs[0].discard_loss
    assert result.epochs[-1].discard_loss < 0.5


def test_tie_aware_falls_back_to_hard_when_mask_zero():
    """teacher_best_mask が全 0 の sample は tie_aware=True でも hard CE。"""
    torch.manual_seed(0)
    model = _build_small_model()
    # mask は全 0 (= teacher 情報無し sample)
    samples = [
        _discard_sample(tile_type=5, teacher_best_mask=None, step_id=i)
        for i in range(4)
    ]
    batch = collate_decision_samples(samples)
    cfg_hard = ImitationConfig(
        tie_aware_discard=False,
        terminal_loss_coef=0.0, yaku_loss_coef=0.0,
    )
    cfg_tie = ImitationConfig(
        tie_aware_discard=True,
        terminal_loss_coef=0.0, yaku_loss_coef=0.0,
    )
    _, m_hard = compute_imitation_loss(model, batch, cfg_hard)
    _, m_tie = compute_imitation_loss(model, batch, cfg_tie)
    # mask 全 0 ⇒ hard CE に fall back ⇒ 同値
    assert m_tie.discard_loss == pytest.approx(m_hard.discard_loss, rel=1e-6)


# ----------------------------------------------------------------------
# SelfPlayRunner wiring (extras → DecisionSample.teacher_*)
# ----------------------------------------------------------------------


def test_self_play_runner_picks_up_teacher_best_mask():
    """rule_based agent を 4 席に置き、teacher_best_mask が sample に書かれる。"""
    runner = SelfPlayRunner(
        config=SelfPlayConfig(max_steps_per_game=4000)
    )
    sa = SeatAgents.homogeneous(
        RuleBasedBaselineAgent(), actor_type="rule_based", num_players=4
    )
    res = runner.run_episode(sa, seed=3)
    assert res.crash_context is None
    # 通常打牌 sample のうち、teacher_best_mask が non-zero なものが存在する
    nd_samples = [
        s for s in res.samples
        if s.decision_family == ActionFamily.NORMAL_DISCARD.value
    ]
    assert nd_samples
    teacher_samples = [s for s in nd_samples if float(s.teacher_best_mask.sum()) > 0]
    assert len(teacher_samples) > 0, (
        "rule_based agent が discard を選んだ sample で teacher_best_mask が "
        "1 件も attach されていない"
    )
    # selected_discard_tile_type は best_mask の hot 位置のいずれかに入る
    for s in teacher_samples:
        assert s.teacher_best_mask[s.selected_discard_tile_type] == 1.0
        assert s.teacher_available is True
        # teacher_discard_tile_type も同じ値が入っている
        assert s.teacher_discard_tile_type == s.selected_discard_tile_type


def test_self_play_runner_random_agent_leaves_teacher_mask_empty():
    """random agent では teacher_best_mask は空 (teacher 情報無し)。"""
    runner = SelfPlayRunner(
        config=SelfPlayConfig(max_steps_per_game=4000)
    )
    sa = SeatAgents.homogeneous(
        RandomAgent(seed=0), actor_type="random", num_players=4
    )
    res = runner.run_episode(sa, seed=5)
    assert res.crash_context is None
    for s in res.samples:
        assert float(s.teacher_best_mask.sum()) == 0.0


# ----------------------------------------------------------------------
# AgentDecision.extras flow sanity
# ----------------------------------------------------------------------


def test_agent_decision_extras_default_is_empty_dict():
    a = AgentDecision(
        action=ModelAction(
            key=ActionKey(ActionFamily.NORMAL_DISCARD, tile_type=5),
            actor=0,
        ),
    )
    assert a.extras == {}


# ----------------------------------------------------------------------
# Tie-aware soft target sanity (math test)
# ----------------------------------------------------------------------


def test_tie_aware_loss_formula_matches_log_sum_exp():
    """tie-aware: -log(p_5 + p_12) と F.log_softmax をマニュアル計算したもの
    が一致する。"""
    torch.manual_seed(0)
    model = _build_small_model()
    mask = _mask_of(5, 12)
    samples = [_discard_sample(tile_type=5, teacher_best_mask=mask)]
    batch = collate_decision_samples(samples)
    cfg = ImitationConfig(
        tie_aware_discard=True,
        terminal_loss_coef=0.0, yaku_loss_coef=0.0,
    )
    loss_t, m = compute_imitation_loss(model, batch, cfg)

    # マニュアル計算
    with torch.no_grad():
        obs = batch.observation
        dmask = batch.discard_mask
        fwd = model(obs, discard_mask=dmask)
        log_softmax = F.log_softmax(fwd.discard_logits, dim=-1)
        expected_loss = -torch.logsumexp(
            torch.stack([log_softmax[0, 5], log_softmax[0, 12]]),
            dim=0,
        )
    assert m.discard_loss == pytest.approx(float(expected_loss.item()), rel=1e-5)
