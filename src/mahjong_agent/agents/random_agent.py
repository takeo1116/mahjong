"""Uniform random agent over model-facing legal actions.

入力は ``LegalActionSet`` (model-facing) のみで、RiichiEnv の内部 state や
hidden info には触れない。``normal_discard`` の tile_type 群と
``candidates`` を 1 つの選択集合と見なし、その union から uniform random に
1 つ選ぶ。
"""
from __future__ import annotations

import random
from typing import Any

from mahjong_agent.actions.types import (
    ActionFamily,
    ActionKey,
    LegalActionSet,
    ModelAction,
)
from mahjong_agent.agents.base import AgentDecision


class RandomAgent:
    """legal な model-facing action から uniform random に 1 つ選ぶ agent。

    Selection 仕様
    --------------
    - ``legal_set.normal_discard`` の各 tile_type と ``legal_set.candidates``
      の各 ``ModelAction`` を **1 つの flat な選択集合** として扱う。
    - 例えば normal_discard 13 種 + candidates 2 件 (例: RiichiDiscard 1 件 +
      Pass 1 件) の場合、計 15 件から uniform 1/15 で選ぶ。
    - 「discard と candidate のどちらに先に振るか」のような hierarchical
      sampling は行わない。理由: hierarchical だと normal_discard の枚数で
      candidate (例えば Tsumo) の確率が薄まらず、family 分布が legal action
      数に従って自然にスケールするほうが分かりやすい baseline になる。
    - ``rng`` を渡すと deterministic に動作する。``rng`` を渡さなければ
      コンストラクタ時の seed から作った内部 ``random.Random`` を使う。
    - legal action が 1 件も無い ``LegalActionSet`` を渡したら ``ValueError``
      で fail-fast する。

    Parameters
    ----------
    seed:
        内部 ``random.Random`` の seed。``None`` で system-entropy。
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select_action(
        self,
        legal_set: LegalActionSet,
        *,
        rng: random.Random | None = None,
        observation: Any | None = None,
    ) -> AgentDecision:
        """``LegalActionSet`` から uniform random に 1 action を選ぶ。

        Parameters
        ----------
        legal_set:
            ``LegalActionSet`` (model-facing legal actions)。
        rng:
            このコールでのみ使う ``random.Random`` (override)。``None`` なら
            agent 内部 rng。
        observation:
            public observation。``RandomAgent`` は使わないが、API 統一のため
            slot だけ持つ。

        Returns
        -------
        AgentDecision
        """
        del observation  # unused (random agent ignores observation)
        choices: list[ModelAction] = []
        # normal_discard
        for tt in sorted(legal_set.normal_discard.keys()):
            choices.append(legal_set.normal_discard[tt])
        # candidates (tuple は既に stable sort 済み)
        choices.extend(legal_set.candidates)
        if not choices:
            raise ValueError(
                f"RandomAgent: legal_set for player {legal_set.decision_player} "
                f"has no normal_discard and no candidates"
            )
        r = rng if rng is not None else self._rng
        chosen = r.choice(choices)
        return AgentDecision(action=chosen, rationale=f"random:{chosen.family.value}")


def _is_normal_discard_key(key: ActionKey) -> bool:
    """テストや diagnostics で使う helper。"""
    return key.family == ActionFamily.NORMAL_DISCARD


__all__ = ["RandomAgent"]
