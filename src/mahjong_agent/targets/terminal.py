"""Terminal auxiliary target (5-class).

各 round の終了時に、対象 player の outcome を 5 class のいずれかに分類する。
分類は decision_owner ごとに行う想定 (= round 内の全 decision sample に
同じ terminal class が attach される)。

5-class:
    0 = win_menzen        対象 player が門前で和了
    1 = win_called        対象 player が副露ありで和了
    2 = draw_tenpai       流局時にテンパイ
    3 = deal_in           ron 放銃した
    4 = other_non_dealin  上記以外 (被ツモ / 流局ノーテン / 途中流局 / 他家和了傍観 等)

class order は固定。one-hot 化 / index 化の helper を提供する。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class TerminalClass(IntEnum):
    """Terminal auxiliary 5-class (固定順)."""

    WIN_MENZEN = 0
    WIN_CALLED = 1
    DRAW_TENPAI = 2
    DEAL_IN = 3
    OTHER_NON_DEALIN = 4


# 固定 class order (索引は TerminalClass の値と一致)
TERMINAL_CLASSES: tuple[str, ...] = (
    "win_menzen",
    "win_called",
    "draw_tenpai",
    "deal_in",
    "other_non_dealin",
)
NUM_TERMINAL_CLASSES: int = len(TERMINAL_CLASSES)
assert NUM_TERMINAL_CLASSES == 5


@dataclass(frozen=True)
class RoundOutcome:
    """Round 終了時の facts を 1 player 視点で平坦化したもの。

    self-play loop が RiichiEnv の ``win_results`` / ``score_deltas`` /
    ``Observation.melds`` / draw 判定から組み立てる想定。target extractor は
    この dataclass を読むだけで判定できるよう、複雑な engine state には依存
    しない。

    Attributes
    ----------
    is_winner:
        対象 player がこの round で和了した。
    won_menzen:
        winner=True のとき、和了形が門前 (= 公開副露なし、ankan は許容) かどうか。
        winner=False のときは無視される。
    is_deal_in_payer:
        対象 player が ron を放銃した (= ron 和了の打牌者)。
    is_tenpai_at_draw:
        流局 (ryukyoku) 時にテンパイだった。draw=False のときは無視。
    is_draw:
        round 終了が ryukyoku 流局だった (= 和了者なし)。
    """

    is_winner: bool = False
    won_menzen: bool = False
    is_deal_in_payer: bool = False
    is_tenpai_at_draw: bool = False
    is_draw: bool = False


def terminal_class_index(outcome: RoundOutcome) -> int:
    """``RoundOutcome`` から terminal class index を 1 つ返す。

    優先順位:
    1. winner であれば win_menzen / win_called
    2. ron 放銃であれば deal_in
    3. 流局 + テンパイなら draw_tenpai
    4. それ以外は other_non_dealin
    """
    if outcome.is_winner:
        return (
            TerminalClass.WIN_MENZEN.value
            if outcome.won_menzen
            else TerminalClass.WIN_CALLED.value
        )
    if outcome.is_deal_in_payer:
        return TerminalClass.DEAL_IN.value
    if outcome.is_draw and outcome.is_tenpai_at_draw:
        return TerminalClass.DRAW_TENPAI.value
    return TerminalClass.OTHER_NON_DEALIN.value


def terminal_one_hot(outcome: RoundOutcome) -> np.ndarray:
    """``(5,)`` float32 one-hot を返す。"""
    out = np.zeros(NUM_TERMINAL_CLASSES, dtype=np.float32)
    out[terminal_class_index(outcome)] = 1.0
    return out


def terminal_class_name(index: int) -> str:
    """index -> class 名 (英語 snake_case)。"""
    if not 0 <= int(index) < NUM_TERMINAL_CLASSES:
        raise ValueError(
            f"terminal class index out of range: {index} "
            f"(expected 0..{NUM_TERMINAL_CLASSES - 1})"
        )
    return TERMINAL_CLASSES[int(index)]


__all__ = [
    "TerminalClass",
    "TERMINAL_CLASSES",
    "NUM_TERMINAL_CLASSES",
    "RoundOutcome",
    "terminal_class_index",
    "terminal_one_hot",
    "terminal_class_name",
]
