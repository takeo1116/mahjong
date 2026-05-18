"""Yaku-group helpers for diagnostics.

学習 loss は通常 yaku と dora-like yaku をまとめて 1 つの multi-label BCE で
扱うが、diagnostics として「dora-like yaku に偏った予測になっていないか」
等を分離集計したいケースがある。本モジュールは vocab index を group ごとに
slice する helper を提供する。
"""
from __future__ import annotations

import numpy as np

from mahjong_agent.targets.yaku import (
    NUM_YAKU,
    YAKU_VOCAB,
)

DORA_LIKE_YAKU_INDICES: tuple[int, ...] = tuple(
    i for i, y in enumerate(YAKU_VOCAB) if y.is_dora_like
)
NON_DORA_YAKU_INDICES: tuple[int, ...] = tuple(
    i for i, y in enumerate(YAKU_VOCAB) if not y.is_dora_like
)


def dora_like_mask() -> np.ndarray:
    """dora-like yaku の index に 1.0、それ以外 0.0 の ``(NUM_YAKU,)`` mask。"""
    mask = np.zeros(NUM_YAKU, dtype=np.float32)
    for idx in DORA_LIKE_YAKU_INDICES:
        mask[idx] = 1.0
    return mask


def non_dora_mask() -> np.ndarray:
    """non-dora yaku の index に 1.0、それ以外 0.0 の ``(NUM_YAKU,)`` mask。"""
    return 1.0 - dora_like_mask()


def split_dora_like(multihot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(NUM_YAKU,)`` multi-hot を ``(non_dora, dora_like)`` 部分配列に分割する。

    diagnostics で「通常 yaku 部分の hit 率」「dora 部分の hit 率」を別々に
    集計したいときに使う。
    """
    multihot = np.asarray(multihot)
    if multihot.shape[-1] != NUM_YAKU:
        raise ValueError(
            f"multihot last dim must be {NUM_YAKU}, got {multihot.shape}"
        )
    non_dora = multihot[..., list(NON_DORA_YAKU_INDICES)]
    dora_like = multihot[..., list(DORA_LIKE_YAKU_INDICES)]
    return non_dora, dora_like


__all__ = [
    "DORA_LIKE_YAKU_INDICES",
    "NON_DORA_YAKU_INDICES",
    "dora_like_mask",
    "non_dora_mask",
    "split_dora_like",
]
