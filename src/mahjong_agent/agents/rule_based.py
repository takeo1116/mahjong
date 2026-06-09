"""Rule-based baseline agent (Stage02 parity rev)。

``baseline/`` package の ``compute_shanten`` / ``count_acceptance`` /
``find_best_discard`` / ``RuleBasedCallPolicy`` を使って、shanten 最小化 +
ukeire 最大化の通常打牌と、shape ベース heuristic 系統の副露判断を行う。

特徴
----
- hidden info を一切使わない。入力は ``LegalActionSet`` と (optional な)
  ``riichienv.Observation``。observation が無いときは legacy fallback
  ("最大 tile_type を切る") に倒れる。
- 物理 tile id (赤牌か通常牌か) の選択は resolver / action abstraction
  に任せる。agent は tile_type 単位だけを見る。
- 同率最良候補は 34-dim ``teacher_best_mask`` として
  ``AgentDecision.extras["teacher_best_mask"]`` に格納する。``SelfPlayRunner``
  はこれを ``DecisionSample.teacher_best_mask`` / ``teacher_*`` に書き出す
  (tie-aware imitation loss の soft target で使う)。

優先順位 (上から順に最初に該当した action を選ぶ)
-------------------------------------------------
1. **Tsumo / Ron**: candidate に win action があれば必ず選ぶ。
2. **KyushuKyuhai**: 取れるなら取る。
3. **RiichiDiscard** (``prefer_riichi=True``, default): 立直できる局面なら
   基本立直する (Stage02 parity)。NORMAL_DISCARD より優先。打牌 tile は
   shanten 最小 + ukeire 最大の best set 内から選ぶ (hand_counts 不可なら
   tile_type 昇順先頭)。``prefer_riichi=False`` ではこの step を skip。
4. **Normal discard**: shanten 最小化 + ukeire 最大化で 1 つ選ぶ。
   observation 無し / 計算失敗時は legacy fallback (= 最大 tile_type)。
5. **RiichiDiscard fallback**: 通常打牌が無く RiichiDiscard だけが残る例外
   ケース (``prefer_riichi`` に関わらず) で立直打牌を選ぶ。
6. **副露評価**: ``RuleBasedCallPolicy`` で CHI / PON / DAIMINKAN を採点。
   PASS が最高 score なら PASS。
7. **Pass** (= response phase で call score 0 のとき)。
8. **Kita** (3P): 取れるなら取る。
9. **Ankan / Kakan**: 他に取れる option が無いときの fallback。
10. **Chi / Pon / Daiminkan**: 同上の fallback。
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np

from mahjong_agent.actions.types import (
    ActionFamily,
    LegalActionSet,
    ModelAction,
)
from mahjong_agent.agents.base import AgentDecision
from mahjong_agent.baseline.call_policy import (
    RuleBasedCallPolicy,
    extract_hand_counts,
    extract_own_meld_count,
)
from mahjong_agent.baseline.discard_select import find_best_discard

_CALL_FALLBACK_ORDER: tuple[ActionFamily, ...] = (
    ActionFamily.ANKAN,
    ActionFamily.KAKAN,
    ActionFamily.CHI,
    ActionFamily.PON,
    ActionFamily.DAIMINKAN,
)
_NUM_TILE_TYPES = 34


class RuleBasedBaselineAgent:
    """Stage02 parity rev rule-based baseline。

    Parameters
    ----------
    seed:
        tie-breaking 等で rng が必要になった場合に備える slot。現実装では
        deterministic に動くため未使用だが、後段で同率複数候補から sample
        するモードを足したい場合に使う。
    use_call_policy:
        ``True`` (default) で ``RuleBasedCallPolicy`` を使って CHI/PON/
        DAIMINKAN を評価する。``False`` で legacy "常に PASS" 動作。
    use_shanten_discard:
        ``True`` (default) で shanten 最小化 + ukeire 最大化の通常打牌。
        ``False`` で legacy "最大 tile_type" fallback。
    prefer_riichi:
        ``True`` (default) で「立直できる局面なら基本的に立直する」挙動
        (Stage02 parity)。``RIICHI_DISCARD`` が legal なら ``NORMAL_DISCARD``
        より優先して選ぶ。``False`` で旧挙動 (通常打牌優先、立直は normal
        discard が無い例外時のみ)。eval opponent の強度を Stage02 に揃える
        ための flag。
    """

    def __init__(
        self,
        seed: int | None = None,
        *,
        use_call_policy: bool = True,
        use_shanten_discard: bool = True,
        prefer_riichi: bool = True,
    ) -> None:
        self._rng = random.Random(seed)
        self._use_call_policy = bool(use_call_policy)
        self._use_shanten_discard = bool(use_shanten_discard)
        self._prefer_riichi = bool(prefer_riichi)
        self._call_policy = RuleBasedCallPolicy()

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

        normal_discard を選んだときは ``AgentDecision.extras`` に以下を入れる:

        - ``teacher_best_mask``: ``(34,) float32`` 同率最良 tile_type mask
        - ``teacher_discard_tile_type``: ``int`` 同率最良の先頭 tile_type
        - ``teacher_shanten``: ``int`` 打牌後 shanten
        - ``teacher_ukeire``: ``int`` 打牌後 ukeire (= 受け入れ枚数)
        """
        del rng  # rule_base は deterministic に動くので未使用

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

        # 2.5) RiichiDiscard (prefer_riichi=True): 立直できるなら基本立直する
        # (Stage02 parity)。NORMAL_DISCARD より優先。
        if self._prefer_riichi and ActionFamily.RIICHI_DISCARD in cand_by_family:
            return self._decide_riichi_discard(
                cand_by_family[ActionFamily.RIICHI_DISCARD], observation
            )

        # 3) Normal discard
        if legal_set.normal_discard:
            return self._decide_normal_discard(legal_set, observation)

        # 4) RiichiDiscard fallback (prefer_riichi=False + 通常打牌が無い例外、
        # または prefer_riichi=True でも normal_discard が空のケース)。
        if ActionFamily.RIICHI_DISCARD in cand_by_family:
            return self._decide_riichi_discard(
                cand_by_family[ActionFamily.RIICHI_DISCARD], observation
            )

        # 5) Response (call policy)
        if self._use_call_policy and any(
            f in cand_by_family
            for f in (ActionFamily.CHI, ActionFamily.PON, ActionFamily.DAIMINKAN)
        ):
            hand_counts = extract_hand_counts(observation)
            if hand_counts is not None:
                evaluation = self._call_policy.select_call(
                    legal_set, hand_counts
                )
                if evaluation is not None:
                    return AgentDecision(
                        action=evaluation.candidate,
                        rationale=f"call:{evaluation.rationale}",
                        extras={"call_score": float(evaluation.score)},
                    )

        # 6) Pass
        if ActionFamily.PASS in cand_by_family:
            return AgentDecision(
                action=cand_by_family[ActionFamily.PASS][0],
                rationale="pass",
            )

        # 7) Kita
        if ActionFamily.KITA in cand_by_family:
            return AgentDecision(
                action=cand_by_family[ActionFamily.KITA][0],
                rationale="kita",
            )

        # 8-9) Other call fallback
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
    # normal discard branch
    # ------------------------------------------------------------------

    def _decide_normal_discard(
        self,
        legal_set: LegalActionSet,
        observation: Any | None,
    ) -> AgentDecision:
        """通常打牌を選ぶ。shanten min + ukeire max が default。"""
        # observation が無い or shanten モードが off なら legacy fallback。
        hand_counts = (
            extract_hand_counts(observation)
            if self._use_shanten_discard
            else None
        )
        if hand_counts is None:
            tile_type = self._pick_normal_discard_tile_type(legal_set)
            return AgentDecision(
                action=legal_set.normal_discard[tile_type],
                rationale="normal_discard:fallback_max_tt",
            )

        legal_mask = np.zeros(_NUM_TILE_TYPES, dtype=np.float32)
        for tt in legal_set.normal_discard.keys():
            if 0 <= tt < _NUM_TILE_TYPES:
                legal_mask[tt] = 1.0
        meld_count = extract_own_meld_count(observation)
        result = find_best_discard(
            hand_counts, legal_mask, meld_count=meld_count
        )
        if result.best_tile_type < 0:
            # shanten 計算上 legal discard 不在 (ex: 異常な obs)。fallback。
            tile_type = self._pick_normal_discard_tile_type(legal_set)
            return AgentDecision(
                action=legal_set.normal_discard[tile_type],
                rationale="normal_discard:fallback_no_best",
            )

        chosen_tt = result.best_tile_type
        if chosen_tt not in legal_set.normal_discard:
            # find_best_discard が選んだ tile_type が legal_set 側に無い
            # 病的ケース (= legal_mask と normal_discard.keys() の不一致)。
            # legal_set を真とし、mask から最初の legal を選ぶ。
            tile_type = self._pick_normal_discard_tile_type(legal_set)
            return AgentDecision(
                action=legal_set.normal_discard[tile_type],
                rationale="normal_discard:legal_set_mismatch",
            )
        return AgentDecision(
            action=legal_set.normal_discard[chosen_tt],
            rationale="normal_discard:shanten_min",
            extras={
                "teacher_best_mask": result.best_mask,
                "teacher_discard_tile_type": int(chosen_tt),
                "teacher_shanten": int(result.best_shanten),
                "teacher_ukeire": int(result.best_acceptance),
            },
        )

    # ------------------------------------------------------------------
    # riichi discard branch
    # ------------------------------------------------------------------

    def _decide_riichi_discard(
        self,
        riichi_cands: list[ModelAction],
        observation: Any | None,
    ) -> AgentDecision:
        """``RIICHI_DISCARD`` candidate から打牌 tile を deterministic に選ぶ。

        observation から hand_counts が取れる場合は、立直可能 tile (= riichi
        candidate の tile_type) を legal mask として ``find_best_discard`` を
        呼び、shanten 最小 + ukeire 最大の tile を選ぶ (best set 内)。teacher
        情報も extras に入れる。hand_counts 不可 / 照合不能なら tile_type 昇順
        先頭の deterministic fallback (``_pick_riichi_discard``)。
        """
        # tile_type -> riichi candidate (同 tile_type は先頭を採用、deterministic)
        by_tt: dict[int, ModelAction] = {}
        for c in sorted(
            riichi_cands,
            key=lambda c: (c.key.tile_type if c.key.tile_type is not None else -1),
        ):
            tt = c.key.tile_type
            if tt is not None and 0 <= int(tt) < _NUM_TILE_TYPES and tt not in by_tt:
                by_tt[int(tt)] = c

        hand_counts = (
            extract_hand_counts(observation)
            if self._use_shanten_discard
            else None
        )
        if hand_counts is not None and by_tt:
            legal_mask = np.zeros(_NUM_TILE_TYPES, dtype=np.float32)
            for tt in by_tt:
                legal_mask[tt] = 1.0
            meld_count = extract_own_meld_count(observation)
            result = find_best_discard(
                hand_counts, legal_mask, meld_count=meld_count
            )
            if result.best_tile_type in by_tt:
                chosen_tt = int(result.best_tile_type)
                return AgentDecision(
                    action=by_tt[chosen_tt],
                    rationale="riichi_discard:shanten_min",
                    extras={
                        "teacher_best_mask": result.best_mask,
                        "teacher_discard_tile_type": chosen_tt,
                        "teacher_shanten": int(result.best_shanten),
                        "teacher_ukeire": int(result.best_acceptance),
                    },
                )

        # fallback: tile_type 昇順先頭 (deterministic)
        chosen = self._pick_riichi_discard(riichi_cands)
        return AgentDecision(
            action=chosen, rationale="riichi_discard:fallback"
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
        """fallback: 合法 tile_type のうち最大のもの (legacy)。"""
        tts = legal_set.normal_discard_tile_types()
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
