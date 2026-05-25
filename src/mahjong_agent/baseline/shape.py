"""手牌 shape hint (Stage02 parity)。

自手 counts から閉じた shape を検出する binary multihot を返す。

- closed_chi (21): 数牌の連続 3 牌 (例 1m2m3m)。3 suit × 中心牌 2-8 の 7 種。
- closed_outside_wait (24): 隣接 2 牌 塔子 (12,23,...,89)。3 suit × 8 種。
- closed_inside_wait (21): 1 つ飛び 嵌張 (例 1m3m)。3 suit × 中心牌 2-8 の 7 種。

合計 66 dim。字牌は対象外。call (副露しても面子/塔子が残るか) 判断や役の方向の
public shape 理解を policy に与えるための feature。hidden info は使わない
(自手 counts のみ)。

C++ fast path (``mahjong_agent._mahjong_fast.compute_shape_hint``) が使える環境では
それを優先し、無い環境では下記 Python 実装に fallback する (同一 semantics)。
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mahjong_agent.baseline import _fast

_NUM_TILE_TYPES = 34
_CHI_DIM = 21
_OUTSIDE_WAIT_DIM = 24
_INSIDE_WAIT_DIM = 21
SHAPE_HINT_DIM = _CHI_DIM + _OUTSIDE_WAIT_DIM + _INSIDE_WAIT_DIM  # 66


def compute_shape_hint(counts: Sequence[int]) -> np.ndarray:
    """``counts`` (34) から shape hint ``(66,) float32`` を返す。

    C++ fast path が利用可能ならそれを使い、無ければ Python 実装に倒す。
    """
    if len(counts) != _NUM_TILE_TYPES:
        raise ValueError(
            f"shape_hint counts must be length 34, got {len(counts)}"
        )
    if _fast.FAST_AVAILABLE:
        return np.asarray(_fast.compute_shape_hint(counts), dtype=np.float32)
    return compute_shape_hint_python(counts)


def compute_shape_hint_python(counts: Sequence[int]) -> np.ndarray:
    """純 Python 実装の shape hint (fast path 無し / 検証用)。"""
    c = [int(x) for x in counts]
    if len(c) != _NUM_TILE_TYPES:
        raise ValueError(
            f"shape_hint counts must be length 34, got {len(c)}"
        )
    chi = np.zeros(_CHI_DIM, dtype=np.float32)
    outside_wait = np.zeros(_OUTSIDE_WAIT_DIM, dtype=np.float32)
    inside_wait = np.zeros(_INSIDE_WAIT_DIM, dtype=np.float32)
    for suit in range(3):
        base = suit * 9
        # 順子 (closed_chi): 中心牌 2-8 (index 1-7)
        for center in range(1, 8):
            idx = base + center
            if c[idx - 1] >= 1 and c[idx] >= 1 and c[idx + 1] >= 1:
                chi[suit * 7 + (center - 1)] = 1.0
        # 塔子 (closed_outside_wait): 隣接 2 牌 (12,23,...,89)
        for pair_start in range(8):
            idx = base + pair_start
            if c[idx] >= 1 and c[idx + 1] >= 1:
                outside_wait[suit * 8 + pair_start] = 1.0
        # 嵌張 (closed_inside_wait): 中心牌 2-8、間が空く
        for center in range(1, 8):
            idx = base + center
            if c[idx - 1] >= 1 and c[idx] < 1 and c[idx + 1] >= 1:
                inside_wait[suit * 7 + (center - 1)] = 1.0
    return np.concatenate([chi, outside_wait, inside_wait])


__all__ = ["SHAPE_HINT_DIM", "compute_shape_hint", "compute_shape_hint_python"]
