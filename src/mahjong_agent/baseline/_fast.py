"""C++ fast path (``mahjong_agent._mahjong_fast``) の薄い wrapper。

extension が import できる環境では ``FAST_AVAILABLE=True`` になり、
``compute_shanten`` / ``analyze_discards`` / ``find_best_discard`` を提供する。
import 失敗時 (compiler / pybind11 が無い / ビルドされていない) は
``FAST_AVAILABLE=False`` になり、呼び出し側は Python fallback を使う。

hidden info 境界: extension に渡すのは ``counts`` / ``legal_mask`` /
``meld_count`` のみ。env / wall / 他家手牌 / state は渡さない。
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:
    from mahjong_agent import _mahjong_fast as _ext

    FAST_AVAILABLE = True
except Exception:  # pragma: no cover - ext 未ビルド環境
    _ext = None
    FAST_AVAILABLE = False


def compute_shanten(counts: Sequence[int], meld_count: int = 0) -> int:
    """C++ fast path で shanten を計算する。ext 未利用時は呼ばないこと。"""
    if _ext is None:
        raise RuntimeError("_mahjong_fast extension unavailable")
    return int(_ext.compute_shanten([int(x) for x in counts], int(meld_count)))


def analyze_discards(
    counts: Sequence[int],
    legal_mask: Sequence[int],
    meld_count: int = 0,
) -> dict[str, Any]:
    """C++ fast path で打牌候補を一括分析する。"""
    if _ext is None:
        raise RuntimeError("_mahjong_fast extension unavailable")
    return _ext.analyze_discards(
        [int(x) for x in counts],
        [int(x) for x in legal_mask],
        int(meld_count),
    )


def find_best_discard(
    counts: Sequence[int],
    legal_mask: Sequence[int],
    meld_count: int = 0,
) -> dict[str, Any]:
    """C++ fast path で最良打牌集合を求める。"""
    if _ext is None:
        raise RuntimeError("_mahjong_fast extension unavailable")
    return _ext.find_best_discard(
        [int(x) for x in counts],
        [int(x) for x in legal_mask],
        int(meld_count),
    )


def compute_shape_hint(counts: Sequence[int]) -> list[float]:
    """C++ fast path で手牌 shape hint (66 dim) を計算する。"""
    if _ext is None:
        raise RuntimeError("_mahjong_fast extension unavailable")
    return list(_ext.compute_shape_hint([int(x) for x in counts]))


__all__ = [
    "FAST_AVAILABLE",
    "analyze_discards",
    "compute_shanten",
    "compute_shape_hint",
    "find_best_discard",
]
