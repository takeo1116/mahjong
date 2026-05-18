"""Rule-based baseline helpers (shanten / ukeire / discard selection / call policy).

Stage02 の ``mahjong_rl.baseline`` 相当を public-only / hidden-info-free に
再実装した module 群。``RuleBasedBaselineAgent`` 等の teacher / baseline 経路
から利用される。
"""
from __future__ import annotations

from mahjong_agent.baseline.call_policy import RuleBasedCallPolicy
from mahjong_agent.baseline.discard_select import (
    DiscardSelectResult,
    find_best_discard,
)
from mahjong_agent.baseline.shanten import compute_shanten
from mahjong_agent.baseline.ukeire import count_acceptance

__all__ = [
    "DiscardSelectResult",
    "RuleBasedCallPolicy",
    "compute_shanten",
    "count_acceptance",
    "find_best_discard",
]
