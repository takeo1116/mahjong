"""Tests for terminal / yaku auxiliary target extraction.

確認内容:
- terminal 5-class の固定 class order と one-hot/index 変換
- 5 つの outcome カテゴリ (win_menzen / win_called / draw_tenpai /
  deal_in / other_non_dealin) を ``RoundOutcome`` で表現できる
- yaku vocab の固定順 / yaku_id <-> index mapping
- yaku multi-hot extraction (winner-only, mask 込み)
- unknown yaku id は fail-fast
- dora-like yaku を diagnostics 上で分離できる
- no-winner / draw / deal_in / other で crash しない
- 実 RiichiEnv ``get_all_yaku()`` snapshot が hardcoded vocab と整合する
"""
from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Terminal class
# ---------------------------------------------------------------------------


def test_terminal_class_order_stable():
    """class order は固定。enum value と TERMINAL_CLASSES tuple の index が一致。"""
    from mahjong_agent.targets import (
        NUM_TERMINAL_CLASSES,
        TERMINAL_CLASSES,
        TerminalClass,
    )

    assert NUM_TERMINAL_CLASSES == 5
    assert TERMINAL_CLASSES == (
        "win_menzen",
        "win_called",
        "draw_tenpai",
        "deal_in",
        "other_non_dealin",
    )
    assert TerminalClass.WIN_MENZEN.value == 0
    assert TerminalClass.WIN_CALLED.value == 1
    assert TerminalClass.DRAW_TENPAI.value == 2
    assert TerminalClass.DEAL_IN.value == 3
    assert TerminalClass.OTHER_NON_DEALIN.value == 4


def test_terminal_one_hot_shape_dtype():
    from mahjong_agent.targets import RoundOutcome, terminal_one_hot

    oh = terminal_one_hot(RoundOutcome())  # other_non_dealin
    assert oh.shape == (5,)
    assert oh.dtype == np.float32
    assert oh.sum() == 1.0


def test_terminal_class_name_round_trip():
    from mahjong_agent.targets import (
        NUM_TERMINAL_CLASSES,
        TERMINAL_CLASSES,
        terminal_class_name,
    )

    for i in range(NUM_TERMINAL_CLASSES):
        assert terminal_class_name(i) == TERMINAL_CLASSES[i]


def test_terminal_class_name_out_of_range_raises():
    from mahjong_agent.targets import terminal_class_name

    with pytest.raises(ValueError):
        terminal_class_name(5)
    with pytest.raises(ValueError):
        terminal_class_name(-1)


# ---------------------------------------------------------------------------
# Terminal: 5 outcome categories
# ---------------------------------------------------------------------------


def test_terminal_win_menzen():
    from mahjong_agent.targets import (
        RoundOutcome,
        TerminalClass,
        terminal_class_index,
    )

    outcome = RoundOutcome(is_winner=True, won_menzen=True)
    assert terminal_class_index(outcome) == TerminalClass.WIN_MENZEN.value


def test_terminal_win_called():
    from mahjong_agent.targets import (
        RoundOutcome,
        TerminalClass,
        terminal_class_index,
    )

    outcome = RoundOutcome(is_winner=True, won_menzen=False)
    assert terminal_class_index(outcome) == TerminalClass.WIN_CALLED.value


def test_terminal_draw_tenpai():
    from mahjong_agent.targets import (
        RoundOutcome,
        TerminalClass,
        terminal_class_index,
    )

    outcome = RoundOutcome(is_draw=True, is_tenpai_at_draw=True)
    assert terminal_class_index(outcome) == TerminalClass.DRAW_TENPAI.value


def test_terminal_deal_in():
    from mahjong_agent.targets import (
        RoundOutcome,
        TerminalClass,
        terminal_class_index,
    )

    outcome = RoundOutcome(is_deal_in_payer=True)
    assert terminal_class_index(outcome) == TerminalClass.DEAL_IN.value


def test_terminal_other_non_dealin_default():
    from mahjong_agent.targets import (
        RoundOutcome,
        TerminalClass,
        terminal_class_index,
    )

    # 完全 default (= 観戦/被ツモ/流局ノーテン/途中流局)
    outcome = RoundOutcome()
    assert terminal_class_index(outcome) == TerminalClass.OTHER_NON_DEALIN.value


def test_terminal_other_non_dealin_draw_noten():
    """流局ノーテン (is_draw=True, is_tenpai_at_draw=False) は other_non_dealin。"""
    from mahjong_agent.targets import (
        RoundOutcome,
        TerminalClass,
        terminal_class_index,
    )

    outcome = RoundOutcome(is_draw=True, is_tenpai_at_draw=False)
    assert terminal_class_index(outcome) == TerminalClass.OTHER_NON_DEALIN.value


def test_terminal_winner_takes_precedence_over_deal_in():
    """winner と deal_in が同時に True の異常入力では winner が優先 (defensive)."""
    from mahjong_agent.targets import (
        RoundOutcome,
        TerminalClass,
        terminal_class_index,
    )

    outcome = RoundOutcome(
        is_winner=True, won_menzen=True, is_deal_in_payer=True
    )
    assert terminal_class_index(outcome) == TerminalClass.WIN_MENZEN.value


# ---------------------------------------------------------------------------
# Yaku vocab
# ---------------------------------------------------------------------------


def test_yaku_vocab_size_and_fixed_order():
    from mahjong_agent.targets import NUM_YAKU, YAKU_VOCAB

    assert NUM_YAKU == 49
    assert len(YAKU_VOCAB) == NUM_YAKU
    # 先頭は Menzen Tsumo (id=1), 末尾は Dai Suusi (id=50)
    assert YAKU_VOCAB[0].id == 1
    assert YAKU_VOCAB[0].name_en == "Menzen Tsumo"
    assert YAKU_VOCAB[-1].id == 50
    assert YAKU_VOCAB[-1].name_en == "Dai Suusi"


def test_yaku_id_to_index_mapping():
    from mahjong_agent.targets import YAKU_ID_TO_INDEX, YAKU_VOCAB

    for i, info in enumerate(YAKU_VOCAB):
        assert YAKU_ID_TO_INDEX[info.id] == i
    # id 46 は欠番なので含まれない
    assert 46 not in YAKU_ID_TO_INDEX


def test_yaku_index_lookup():
    from mahjong_agent.targets import yaku_index

    assert yaku_index(1) == 0  # Menzen Tsumo
    assert yaku_index(2) == 1  # Riichi
    # 46 は欠番
    with pytest.raises(KeyError):
        yaku_index(46)


def test_yaku_multi_hot_basic():
    from mahjong_agent.targets import (
        NUM_YAKU,
        YAKU_ID_TO_INDEX,
        yaku_ids_to_multihot,
    )

    mh = yaku_ids_to_multihot([1, 12])  # Menzen Tsumo + Tanyao
    assert mh.shape == (NUM_YAKU,)
    assert mh.dtype == np.float32
    assert mh.sum() == 2.0
    assert mh[YAKU_ID_TO_INDEX[1]] == 1.0
    assert mh[YAKU_ID_TO_INDEX[12]] == 1.0


def test_yaku_multi_hot_empty_winner():
    from mahjong_agent.targets import NUM_YAKU, yaku_ids_to_multihot

    mh = yaku_ids_to_multihot([])
    assert mh.shape == (NUM_YAKU,)
    assert mh.sum() == 0.0


def test_yaku_multi_hot_unknown_fails_fast():
    from mahjong_agent.targets import yaku_ids_to_multihot

    with pytest.raises(ValueError):
        yaku_ids_to_multihot([46])  # 46 は欠番
    with pytest.raises(ValueError):
        yaku_ids_to_multihot([99])  # 範囲外


def test_yaku_multi_hot_unknown_silenced_when_allowed():
    from mahjong_agent.targets import (
        YAKU_ID_TO_INDEX,
        yaku_ids_to_multihot,
    )

    mh = yaku_ids_to_multihot([46, 2, 99], allow_unknown=True)
    assert mh.sum() == 1.0
    assert mh[YAKU_ID_TO_INDEX[2]] == 1.0


# ---------------------------------------------------------------------------
# extract_yaku_target (winner-only)
# ---------------------------------------------------------------------------


def test_extract_yaku_target_winner():
    from mahjong_agent.targets import (
        NUM_YAKU,
        YAKU_ID_TO_INDEX,
        extract_yaku_target,
    )

    target, mask = extract_yaku_target([1, 12], is_winner=True)
    assert target.shape == (NUM_YAKU,)
    assert mask == 1.0
    assert target[YAKU_ID_TO_INDEX[1]] == 1.0
    assert target[YAKU_ID_TO_INDEX[12]] == 1.0


def test_extract_yaku_target_non_winner_returns_zero_target_and_zero_mask():
    from mahjong_agent.targets import NUM_YAKU, extract_yaku_target

    target, mask = extract_yaku_target([1, 12], is_winner=False)
    assert target.shape == (NUM_YAKU,)
    assert target.sum() == 0.0
    assert mask == 0.0


def test_extract_yaku_target_none_yaku_ids():
    from mahjong_agent.targets import NUM_YAKU, extract_yaku_target

    target, mask = extract_yaku_target(None, is_winner=True)
    assert target.shape == (NUM_YAKU,)
    assert target.sum() == 0.0
    assert mask == 0.0


# ---------------------------------------------------------------------------
# Dora-like yaku grouping (diagnostics)
# ---------------------------------------------------------------------------


def test_dora_like_yaku_ids():
    from mahjong_agent.targets import DORA_LIKE_YAKU_IDS, is_dora_like_yaku

    assert DORA_LIKE_YAKU_IDS == frozenset({31, 32, 33, 34})
    for yid in (31, 32, 33, 34):
        assert is_dora_like_yaku(yid)
    for yid in (1, 2, 12, 50):
        assert not is_dora_like_yaku(yid)


def test_dora_like_indices_consistent_with_vocab():
    from mahjong_agent.diagnostics import (
        DORA_LIKE_YAKU_INDICES,
        NON_DORA_YAKU_INDICES,
    )
    from mahjong_agent.targets import NUM_YAKU, YAKU_VOCAB

    assert len(DORA_LIKE_YAKU_INDICES) == 4
    assert len(NON_DORA_YAKU_INDICES) == NUM_YAKU - 4
    # index で引いた vocab entry が is_dora_like=True
    for idx in DORA_LIKE_YAKU_INDICES:
        assert YAKU_VOCAB[idx].is_dora_like
    for idx in NON_DORA_YAKU_INDICES:
        assert not YAKU_VOCAB[idx].is_dora_like


def test_dora_like_mask_and_non_dora_mask_partition():
    from mahjong_agent.diagnostics import dora_like_mask, non_dora_mask
    from mahjong_agent.targets import NUM_YAKU

    d = dora_like_mask()
    n = non_dora_mask()
    assert d.shape == (NUM_YAKU,)
    assert n.shape == (NUM_YAKU,)
    # 排他和が 1
    assert np.allclose(d + n, np.ones(NUM_YAKU, dtype=np.float32))
    assert d.sum() == 4.0


def test_split_dora_like():
    from mahjong_agent.diagnostics import (
        DORA_LIKE_YAKU_INDICES,
        NON_DORA_YAKU_INDICES,
        split_dora_like,
    )
    from mahjong_agent.targets import NUM_YAKU, yaku_ids_to_multihot

    # Tanyao (12, non-dora) + Dora (31)
    mh = yaku_ids_to_multihot([12, 31])
    non_dora, dora_like = split_dora_like(mh)
    assert non_dora.shape == (len(NON_DORA_YAKU_INDICES),)
    assert dora_like.shape == (len(DORA_LIKE_YAKU_INDICES),)
    assert non_dora.sum() == 1.0  # Tanyao のみ
    assert dora_like.sum() == 1.0  # Dora のみ
    # 元の multihot は変わらず
    assert mh.sum() == 2.0
    assert mh.shape == (NUM_YAKU,)


def test_split_dora_like_wrong_shape_raises():
    from mahjong_agent.diagnostics import split_dora_like

    with pytest.raises(ValueError):
        split_dora_like(np.zeros(10, dtype=np.float32))


# ---------------------------------------------------------------------------
# Real RiichiEnv smoke: snapshot vocab matches runtime get_all_yaku()
# ---------------------------------------------------------------------------


def test_hardcoded_vocab_matches_riichienv_runtime():
    """``riichienv.get_all_yaku()`` の出力が hardcoded snapshot と一致する。

    一致しないときは PyPI 版 ``riichienv`` 側が yaku table を更新した可能性が
    高い。その場合は ``targets/yaku.py`` の ``_RAW_YAKU_TABLE`` を新仕様に
    合わせて再 snapshot する必要がある (= 既存学習との互換性破壊の review が
    必要)。
    """
    import riichienv

    from mahjong_agent.targets import YAKU_VOCAB

    runtime = list(riichienv.get_all_yaku())
    assert len(runtime) == len(YAKU_VOCAB), (
        f"riichienv yaku count changed: runtime={len(runtime)} vs "
        f"snapshot={len(YAKU_VOCAB)}"
    )
    for i, (rt, snap) in enumerate(zip(runtime, YAKU_VOCAB, strict=True)):
        assert rt.id == snap.id, f"index {i}: id {rt.id} vs snapshot {snap.id}"
        assert rt.name == snap.name, (
            f"index {i}: name {rt.name!r} vs snapshot {snap.name!r}")
        assert rt.name_en == snap.name_en, (
            f"index {i}: name_en {rt.name_en!r} vs snapshot {snap.name_en!r}")
