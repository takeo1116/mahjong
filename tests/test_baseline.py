"""Tests for ``mahjong_agent.baseline`` (shanten / ukeire / discard / call_policy)。"""
from __future__ import annotations

import numpy as np
import pytest

from mahjong_agent.actions.types import (
    ActionFamily,
    ActionKey,
    LegalActionSet,
    ModelAction,
)
from mahjong_agent.baseline import (
    DiscardSelectResult,
    RuleBasedCallPolicy,
    compute_shanten,
    count_acceptance,
    find_best_discard,
)
from mahjong_agent.baseline.call_policy import (
    extract_hand_counts,
    extract_own_meld_count,
)

# ----------------------------------------------------------------------
# shanten
# ----------------------------------------------------------------------


def _counts(*items: tuple[int, int]) -> list[int]:
    """``(tile_type, count)`` ペアから 34-dim list を作る。"""
    c = [0] * 34
    for t, k in items:
        c[t] = k
    return c


def test_shanten_tenpai_basic():
    # 11122233344455 (= 14 tile) は和了形 → 1m を 1 つ抜くと tenpai。
    # ここは tenpai (= 0) を確認する 13 牌から始める。
    # 11122233344 + 5m6m7m + 9m雀頭 で待ち = tenpai 寄り
    c = _counts((0, 3), (1, 3), (2, 3), (3, 2), (8, 2))  # 13 牌
    sh = compute_shanten(c, 0)
    assert sh <= 0  # tenpai 以下


def test_shanten_complete_hand_returns_negative_one():
    # 完全和了形: 234m 234p 234s 西西 + 5m (= 14 牌)
    # 数牌 234 が 3 mentsu と 西西 雀頭 + 5m 浮き、はテンパイ 0 shanten
    # ここでは確実な agari pattern にする:
    # 11122233344499m (456m 完成形ではないが) は 4 mentsu + 1 pair 形
    c = _counts((0, 3), (1, 3), (2, 3), (3, 3), (8, 2))  # 14 牌
    # 1m刻 2m刻 3m刻 4m刻 9m雀頭 = 4 面子 1 雀頭 → 和了
    assert compute_shanten(c, 0) == -1


def test_shanten_chiitoitsu_priority():
    # 七対子テンパイ: 11m 22m 33m 11p 22p 33p 1s (= 13 牌, 6 pair + 1 isolated)
    c = _counts((0, 2), (1, 2), (2, 2), (9, 2), (10, 2), (11, 2), (18, 1))
    sh = compute_shanten(c, 0)
    assert sh == 0  # tenpai


def test_shanten_kokushi():
    # 国士テンパイ: 1m 9m 1p 9p 1s 9s 東南西北白發中 全種 1 枚 + ペアなし
    # = 13 牌、ペア 0 → 13 - 13 - 0 = 0 (tenpai)
    c = [0] * 34
    for t in (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33):
        c[t] = 1
    assert compute_shanten(c, 0) == 0


def test_shanten_meld_count_reduces_required_mentsu():
    # 副露 1 面子済み (= meld_count=1), 残り 11 牌が tenpai に近い形
    # 234m 567m + 23p ペア待ち = 11 牌 tenpai (1 副露 + 2 mentsu + 1 pair + 1 head wait)
    c = _counts((1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
                (10, 1), (11, 1), (12, 1), (18, 1), (19, 1))
    sh = compute_shanten(c, 1)
    assert sh <= 2


def test_shanten_invalid_length_raises():
    with pytest.raises(ValueError):
        compute_shanten([0] * 33)


# ----------------------------------------------------------------------
# ukeire
# ----------------------------------------------------------------------


def test_ukeire_zero_when_already_complete():
    # 完全和了形なら新規受け入れは無い (shanten=-1)
    c = _counts((0, 3), (1, 3), (2, 3), (3, 3), (8, 2))
    sh = compute_shanten(c, 0)
    assert sh == -1
    # 1 step 下がる tile は無い (= already complete)
    assert count_acceptance(c, sh) == 0


def test_ukeire_positive_at_tenpai():
    # 7 pair tenpai (1s 待ち) → 1s 単騎なら受け入れ = 4 - 1 = 3
    c = _counts((0, 2), (1, 2), (2, 2), (9, 2), (10, 2), (11, 2), (18, 1))
    sh = compute_shanten(c, 0)
    assert sh == 0
    acc = count_acceptance(c, sh)
    assert acc >= 1


# ----------------------------------------------------------------------
# find_best_discard
# ----------------------------------------------------------------------


def test_find_best_discard_prefers_shanten_min():
    # 14 牌: 完全和了形 + 浮き牌 1 枚 → 浮き牌を切れば和了 (shanten=-1)
    # 1m×3 2m×3 3m×3 4m×3 9m×2 はそのまま和了形 14 牌、ここに 1s を追加して 15 牌?
    # find_best_discard は 14 牌前提 (1 discard → 13 牌の shanten 評価)
    # 1m×3 2m×3 3m×3 4m×2 9m×2 + 1s ×1 = 14 牌
    counts = _counts((0, 3), (1, 3), (2, 3), (3, 2), (8, 2), (18, 1))
    mask = np.zeros(34, dtype=np.float32)
    for tt in (0, 1, 2, 3, 8, 18):
        mask[tt] = 1.0
    result = find_best_discard(counts, mask)
    assert isinstance(result, DiscardSelectResult)
    # 1s を切ると 1m刻 2m刻 3m刻 + 4m対子 + 9m対子 = 3面子 2対子 (4m or 9m が雀頭+ターツ)
    # ここでは "1s を切るのが best" になるはず (shanten 最小)
    assert result.best_tile_type == 18
    assert result.best_mask[18] == 1.0
    # 他の tile を切ると shanten が悪化するはず
    assert result.best_mask.sum() == 1.0


def test_find_best_discard_tie_set_contains_multiple():
    # 14 牌: 11m 99m 11p 99p 11s 99s 東 西 (= 6 pair + 2 isolated honors)
    # 規則形では絶望的に shanten が悪い。chiitoi 軸でだけ評価される。
    # 東 を切る / 西 を切る どちらでも chiitoi 0 shanten (= tenpai)、
    # 残りの単独牌で受け、ukeire はどちらも 3 で同点 → tied。
    counts = _counts(
        (0, 2), (8, 2),   # 1m 9m pair
        (9, 2), (17, 2),  # 1p 9p pair
        (18, 2), (26, 2), # 1s 9s pair
        (27, 1), (29, 1), # 東 西 isolated
    )
    mask = np.ones(34, dtype=np.float32)
    result = find_best_discard(counts, mask)
    # 東 (27) と 西 (29) が tied
    assert result.best_mask[27] == 1.0
    assert result.best_mask[29] == 1.0
    # top1 は tile_type 昇順先頭 (= 27, 東)
    assert result.best_tile_type == 27


def test_find_best_discard_no_legal_returns_minus_one():
    counts = _counts((0, 3), (1, 3), (2, 3), (3, 3), (8, 2))
    mask = np.zeros(34, dtype=np.float32)
    result = find_best_discard(counts, mask)
    assert result.best_tile_type == -1


# ----------------------------------------------------------------------
# RuleBasedCallPolicy
# ----------------------------------------------------------------------


def _candidate(
    family: ActionFamily,
    *,
    tile_type: int | None = None,
    consume: tuple[int, ...] = (),
    target_rel_seat: int | None = None,
) -> ModelAction:
    key = ActionKey(
        family=family,
        tile_type=tile_type,
        consume_tile_types=consume,
        target_rel_seat=target_rel_seat,
    )
    return ModelAction(key=key, actor=0, _raw_actions=())


def test_call_policy_yakuhai_pon_preferred_over_pass():
    policy = RuleBasedCallPolicy()
    legal = LegalActionSet(
        decision_player=0,
        candidates=(
            _candidate(ActionFamily.PASS),
            _candidate(
                ActionFamily.PON, tile_type=31, consume=(31, 31),
                target_rel_seat=3,
            ),  # 白
        ),
    )
    hand = _counts((31, 2))  # 自手に白 2 枚 (= ポン可)
    ev = policy.select_call(legal, hand)
    assert ev is not None
    assert ev.candidate.family == ActionFamily.PON
    assert ev.candidate.tile_type == 31
    assert ev.score >= 90  # yakuhai 系列


def test_call_policy_yaochu_pon_falls_back_to_pass():
    policy = RuleBasedCallPolicy()
    legal = LegalActionSet(
        decision_player=0,
        candidates=(
            _candidate(ActionFamily.PASS),
            _candidate(
                ActionFamily.PON, tile_type=0, consume=(0, 0),
                target_rel_seat=3,
            ),  # 1m
        ),
    )
    hand = _counts((0, 2))  # 1m 対子
    ev = policy.select_call(legal, hand)
    assert ev is not None
    assert ev.candidate.family == ActionFamily.PASS


def test_call_policy_tanyao_pon_preferred_over_pass():
    policy = RuleBasedCallPolicy()
    # 中張牌のポン (5m, 数牌) + 手が tanyao 互換 (字牌・1/9 牌なし)
    legal = LegalActionSet(
        decision_player=0,
        candidates=(
            _candidate(ActionFamily.PASS),
            _candidate(
                ActionFamily.PON, tile_type=4, consume=(4, 4),
                target_rel_seat=3,
            ),
        ),
    )
    # 自手: 2m 3m 4m 4m 5m 6m 7m 中張牌のみ
    hand = _counts((1, 1), (2, 1), (3, 1), (4, 2), (5, 1), (6, 1))
    ev = policy.select_call(legal, hand)
    assert ev is not None
    assert ev.candidate.family == ActionFamily.PON


def test_call_policy_no_call_candidate_returns_pass():
    policy = RuleBasedCallPolicy()
    legal = LegalActionSet(
        decision_player=0,
        candidates=(_candidate(ActionFamily.PASS),),
    )
    hand = _counts((0, 1))
    ev = policy.select_call(legal, hand)
    assert ev is not None
    assert ev.candidate.family == ActionFamily.PASS


# ----------------------------------------------------------------------
# extract_hand_counts / extract_own_meld_count
# ----------------------------------------------------------------------


class _FakeObs:
    """riichienv.Observation を模す薄い stub。"""

    def __init__(self, hand, player_id, melds=None):
        self.hand = hand
        self.player_id = player_id
        self.melds = melds if melds is not None else [[], [], [], []]


def test_extract_hand_counts_converts_tile_ids():
    # tile_ids: 0,1,2 (= 1m×3 in tile_type 0) + 100 (= tile_type 25 = 8s)
    obs = _FakeObs(hand=[0, 1, 2, 100], player_id=0)
    c = extract_hand_counts(obs)
    assert c is not None
    assert c[0] == 3
    assert c[25] == 1
    assert sum(c) == 4


def test_extract_hand_counts_none_obs_returns_none():
    assert extract_hand_counts(None) is None


def test_extract_own_meld_count():
    obs = _FakeObs(
        hand=[],
        player_id=0,
        melds=[[object(), object()], [], [object()], []],
    )
    assert extract_own_meld_count(obs) == 2
    assert extract_own_meld_count(None) == 0
