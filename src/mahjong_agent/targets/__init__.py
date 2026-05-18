"""Auxiliary target extraction (terminal class / yaku multi-hot)."""
from __future__ import annotations

from mahjong_agent.targets.terminal import (
    NUM_TERMINAL_CLASSES,
    TERMINAL_CLASSES,
    RoundOutcome,
    TerminalClass,
    terminal_class_index,
    terminal_class_name,
    terminal_one_hot,
)
from mahjong_agent.targets.yaku import (
    DORA_LIKE_YAKU_IDS,
    NUM_YAKU,
    YAKU_ID_TO_INDEX,
    YAKU_VOCAB,
    YakuInfo,
    extract_yaku_target,
    is_dora_like_yaku,
    yaku_ids_to_multihot,
    yaku_index,
)

__all__ = [
    "DORA_LIKE_YAKU_IDS",
    "NUM_TERMINAL_CLASSES",
    "NUM_YAKU",
    "RoundOutcome",
    "TERMINAL_CLASSES",
    "TerminalClass",
    "YAKU_ID_TO_INDEX",
    "YAKU_VOCAB",
    "YakuInfo",
    "extract_yaku_target",
    "is_dora_like_yaku",
    "terminal_class_index",
    "terminal_class_name",
    "terminal_one_hot",
    "yaku_ids_to_multihot",
    "yaku_index",
]
