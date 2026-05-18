"""Discard 選択 helper。

合法 tile_type を ``compute_shanten + count_acceptance`` で評価し、
(shanten 最小, ukeire 最大) を取る tile_type の集合を返す。

入力は **public-only**:
- ``hand_counts``: 自分の手牌 (34 種)
- ``legal_mask``: 通常打牌 legal tile_type の 34-dim mask
- ``meld_count``: 自分の副露面子数
- (optional) ``seen_counts``: 全 player discards + 副露 + dora indicators +
  自手から作った 34-dim 既見 counts (= 山残り計算に使う)

hidden info (他家手牌 / 山の中身 / 裏ドラ) は使わない。
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mahjong_agent.baseline.shanten import compute_shanten
from mahjong_agent.baseline.ukeire import count_acceptance

_NUM_TILE_TYPES = 34


@dataclass(frozen=True)
class DiscardSelectResult:
    """``find_best_discard`` の戻り値。

    Attributes
    ----------
    best_tile_type:
        ``best_mask`` で先頭 (= tile_type 昇順最小) の tile_type。``-1`` のとき
        合法 discard が無い。
    best_mask:
        ``(34,) float32``。同率最良 tile_type に ``1.0``、それ以外 ``0.0``。
    best_shanten:
        最良候補で達成された ``shanten`` 値。
    best_acceptance:
        最良候補で達成された ``ukeire`` 値。
    """

    best_tile_type: int
    best_mask: np.ndarray
    best_shanten: int
    best_acceptance: int


def find_best_discard(
    hand_counts: Sequence[int],
    legal_mask: Sequence[float] | np.ndarray,
    *,
    meld_count: int = 0,
    seen_counts: Sequence[int] | None = None,
) -> DiscardSelectResult:
    """合法 tile_type のうち (shanten 最小, ukeire 最大) を取る集合を返す。

    Parameters
    ----------
    hand_counts:
        34 種 tile_type 枚数。手牌 14 (or 13) 枚分。
    legal_mask:
        34-dim float / int。``>= 0.5`` で legal。
    meld_count:
        自分の副露面子数 (0..4)。
    seen_counts:
        山残り計算に使う既見 counts (option)。``None`` のときは
        ``count_acceptance`` 既定の「自分の手牌のみを既見扱い」で動く。

    Returns
    -------
    DiscardSelectResult
    """
    counts = [int(x) for x in hand_counts]
    if len(counts) != _NUM_TILE_TYPES:
        raise ValueError(
            f"hand_counts must be length 34, got {len(counts)}"
        )
    mask = np.asarray(legal_mask, dtype=np.float32).reshape(-1)
    if mask.size != _NUM_TILE_TYPES:
        raise ValueError(
            f"legal_mask must be length 34, got {mask.size}"
        )

    best_mask = np.zeros(_NUM_TILE_TYPES, dtype=np.float32)
    best_tile_type = -1
    best_shanten = 999
    best_acceptance = -1

    # 1st pass: 最良 (shanten, ukeire) を見つける
    for t in range(_NUM_TILE_TYPES):
        if mask[t] < 0.5 or counts[t] <= 0:
            continue
        counts[t] -= 1
        sh = compute_shanten(counts, meld_count)
        acc = count_acceptance(counts, sh, meld_count, seen_counts=seen_counts)
        counts[t] += 1
        if sh < best_shanten or (sh == best_shanten and acc > best_acceptance):
            best_shanten = sh
            best_acceptance = acc

    # 該当無し (= 合法 discard が 0 件 / 手牌に該当 tile が無い)
    if best_shanten == 999:
        return DiscardSelectResult(
            best_tile_type=-1,
            best_mask=best_mask,
            best_shanten=best_shanten,
            best_acceptance=best_acceptance,
        )

    # 2nd pass: 同率最良候補を mask 化
    for t in range(_NUM_TILE_TYPES):
        if mask[t] < 0.5 or counts[t] <= 0:
            continue
        counts[t] -= 1
        sh = compute_shanten(counts, meld_count)
        acc = count_acceptance(counts, sh, meld_count, seen_counts=seen_counts)
        counts[t] += 1
        if sh == best_shanten and acc == best_acceptance:
            best_mask[t] = 1.0
            if best_tile_type == -1:
                best_tile_type = t

    return DiscardSelectResult(
        best_tile_type=best_tile_type,
        best_mask=best_mask,
        best_shanten=best_shanten,
        best_acceptance=best_acceptance,
    )


__all__ = ["DiscardSelectResult", "find_best_discard"]
