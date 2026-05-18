"""Minimal rule-based baseline agent.

imitation warm start や evaluation 用に、合法に動き、明らかな win action を
取りこぼさない最小 baseline を提供する。強さは主目的ではなく、後段の
imitation teacher / 比較対象として使える「動く baseline」を作ることが目的。

特徴
----
- hidden info を一切使わない。入力は ``LegalActionSet`` (model-facing legal
  actions) と、optional な public observation のみ。
- 物理 tile id (赤牌か通常牌か) の選択は agent 側で行わず、resolver /
  action abstraction (``ModelAction._raw_actions``) に任せる。
- 同じ tile_type が複数 raw discard を持っていても、agent からは tile_type
  単位でしか選択しない。

優先順位 (上から順に最初に該当した action を選ぶ)
-----------------------------------------------
1. **Tsumo / Ron**: candidate に win action があれば必ず選ぶ。
2. **KyushuKyuhai**: 取れるなら取る。手詰まりを増やさない安全側のデフォルト。
3. **Normal discard**: 通常打牌が legal なら、合法 tile_type から
   deterministic に 1 つ選ぶ。具体的には **最も大きい tile_type** を選ぶ
   (字牌 27..33 > 9m/9p/9s/8m/8p/8s.. などの順)。これは「字牌は早めに
   切られやすい」という弱い prior に従った deterministic な選び方で、
   teacher としてではなく「壊れない default」として動作させる目的。
4. **RiichiDiscard**: 通常打牌が無く RiichiDiscard だけが残っている (env が
   discard 強制している) 例外ケースで、tile_type 昇順の先頭を選ぶ。
   通常時はリーチを **自動で打たない** (rule_base は基本的に dama 寄り)。
5. **Pass**: response (鳴き応答 / 自家 optional) で Pass があれば Pass する。
   teacher としては「鳴かない / 暗槓しない」が安全側の default。
6. **Kita** (3P): 取れるなら取る。4P では発生しない。
7. **Ankan / Kakan**: 他に取れる option が無いときの fallback (基本的には
   起こらない経路)。
8. **Chi / Pon / Daiminkan**: 同上の fallback。

注意
----
- hidden info を見ないため、teacher として強くはない。imitation warm start
  には十分とは限らず、後続 issue でより強い baseline / human data に
  置き換える想定。
- 「常に Pass」「常に highest tile_type を discard」という選び方は単純で
  predictable な baseline だが、deterministic に teacher target を作れる
  という利点もある (同じ legal_set には常に同じ decision が返る)。
"""
from __future__ import annotations

import random
from typing import Any

from mahjong_agent.actions.types import (
    ActionFamily,
    LegalActionSet,
    ModelAction,
)
from mahjong_agent.agents.base import AgentDecision

# 鳴き / 自家 optional で fallback として最後に取りうる family の順序。
# pass を取ったあとに到達する想定なので通常は使われない。
_CALL_FALLBACK_ORDER: tuple[ActionFamily, ...] = (
    ActionFamily.ANKAN,
    ActionFamily.KAKAN,
    ActionFamily.CHI,
    ActionFamily.PON,
    ActionFamily.DAIMINKAN,
)


class RuleBasedBaselineAgent:
    """最小 rule-based baseline。

    Parameters
    ----------
    seed:
        tie-breaking などで rng が必要になった場合に備える slot。現実装では
        deterministic に動くので未使用だが、後続 issue で rule を強化したい
        場合のために用意。
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------

    def select_action(
        self,
        legal_set: LegalActionSet,
        *,
        rng: random.Random | None = None,
        observation: Any | None = None,
    ) -> AgentDecision:
        """priority list に従って 1 つ action を選ぶ。

        Parameters
        ----------
        legal_set:
            model-facing legal actions。
        rng:
            override 用 rng。現実装では使わない。
        observation:
            public observation。現実装では使わない。

        Returns
        -------
        AgentDecision

        Raises
        ------
        ValueError:
            normal_discard も candidates も空のとき。
        """
        del observation, rng  # unused in current minimal heuristic

        cand_by_family = self._group_candidates(legal_set)

        # 1) Tsumo / Ron
        for fam in (ActionFamily.TSUMO, ActionFamily.RON):
            if fam in cand_by_family:
                return AgentDecision(
                    action=cand_by_family[fam][0],
                    rationale=f"win:{fam.value}",
                )

        # 2) Kyushu Kyuhai
        if ActionFamily.KYUSHU_KYUHAI in cand_by_family:
            return AgentDecision(
                action=cand_by_family[ActionFamily.KYUSHU_KYUHAI][0],
                rationale="kyushu_kyuhai",
            )

        # 3) Normal discard
        if legal_set.normal_discard:
            tile_type = self._pick_normal_discard_tile_type(legal_set)
            return AgentDecision(
                action=legal_set.normal_discard[tile_type],
                rationale="normal_discard",
            )

        # 4) RiichiDiscard fallback (通常打牌が無いとき限定)
        if ActionFamily.RIICHI_DISCARD in cand_by_family:
            chosen = self._pick_riichi_discard(
                cand_by_family[ActionFamily.RIICHI_DISCARD]
            )
            return AgentDecision(action=chosen, rationale="riichi_discard_fallback")

        # 5) Pass
        if ActionFamily.PASS in cand_by_family:
            return AgentDecision(
                action=cand_by_family[ActionFamily.PASS][0],
                rationale="pass",
            )

        # 6) Kita
        if ActionFamily.KITA in cand_by_family:
            return AgentDecision(
                action=cand_by_family[ActionFamily.KITA][0],
                rationale="kita",
            )

        # 7-8) Other call fallback
        for fam in _CALL_FALLBACK_ORDER:
            if fam in cand_by_family:
                return AgentDecision(
                    action=cand_by_family[fam][0],
                    rationale=f"fallback:{fam.value}",
                )

        raise ValueError(
            f"RuleBasedBaselineAgent: legal_set for player "
            f"{legal_set.decision_player} has no normal_discard and no "
            f"candidates"
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_candidates(
        legal_set: LegalActionSet,
    ) -> dict[ActionFamily, list[ModelAction]]:
        out: dict[ActionFamily, list[ModelAction]] = {}
        for c in legal_set.candidates:
            out.setdefault(c.family, []).append(c)
        return out

    @staticmethod
    def _pick_normal_discard_tile_type(legal_set: LegalActionSet) -> int:
        """合法 tile_type のうち最大のものを返す (字牌寄りの safe default)。"""
        tts = legal_set.normal_discard_tile_types()
        # normal_discard が空でないことは caller 側で検査済み。
        return tts[-1]

    @staticmethod
    def _pick_riichi_discard(candidates: list[ModelAction]) -> ModelAction:
        """RiichiDiscard candidate から tile_type 昇順の先頭を選ぶ。"""
        sorted_cands = sorted(
            candidates,
            key=lambda c: (
                c.key.tile_type if c.key.tile_type is not None else -1
            ),
        )
        return sorted_cands[0]


__all__ = ["RuleBasedBaselineAgent"]
