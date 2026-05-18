"""Call policy (副露 heuristic)。

``RuleBasedBaselineAgent`` が response phase で CHI / PON / DAIMINKAN /
PASS を選ぶための heuristic 副露評価。``LegalActionSet`` / ``ModelAction``
を直接受けて評価する。public-only。

「rule_based agent が response phase で何を選ぶか」の最低限を担うため、
以下の点数表を使う:

- 役牌 (白/發/中) ポン → 100
- 役牌 ダイミンカン → 90
- 対々和方向 (副露後の手牌に刻子・対子が多い) ポン → 60
- 喰い断方向 (中張牌のみ) ポン → 50
- 中張牌の普通ポン → 15
- 喰い断方向 チー → 40
- 一気通貫 / 三色同順方向 チー → 70
- 么九ポン (役牌以外) → 0
- 不明 → 0

score が 0 以下なら Skip / Pass を返す方針。tile_type 単位の評価で、赤牌
等の物理 tile id 区別は resolver 側に閉じる。

注意: ``compute_shanten`` を使うほどの判定はしておらず、shape ベースの
heuristic。本格的に強くするのは後段 issue の責務。
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mahjong_agent.actions.types import ActionFamily, LegalActionSet, ModelAction

_NUM_TILE_TYPES = 34


@dataclass(frozen=True)
class CallEvaluation:
    """1 candidate に対する評価結果。"""

    candidate: ModelAction
    score: int
    rationale: str


class RuleBasedCallPolicy:
    """call (副露) heuristic 評価器。

    `select_call(legal_set, hand_counts)` で legal candidates のうち最高 score
    を返す。Skip (= ``PASS``) を上回る score の call action が無ければ Skip
    を返す。
    """

    def select_call(
        self,
        legal_set: LegalActionSet,
        hand_counts: Sequence[int],
    ) -> CallEvaluation | None:
        """legal candidates を評価し、最大 score の 1 件を返す。

        Returns
        -------
        CallEvaluation | None:
            - 評価対象 candidate (CHI / PON / DAIMINKAN) が無ければ ``None``。
            - PASS が candidate に居て、それより高い score の call が無ければ
              PASS を返す。
            - 全 candidate に PASS が無く、call しか無い病的状況では最大
              score の call を返す (Stage02 と同じ動作)。
        """
        cands = list(legal_set.candidates)
        if not cands:
            return None
        counts = [int(x) for x in hand_counts]
        if len(counts) != _NUM_TILE_TYPES:
            raise ValueError(
                f"hand_counts must be length 34, got {len(counts)}"
            )

        # PASS を score=0 として記録
        pass_cand: ModelAction | None = None
        evaluated: list[CallEvaluation] = []
        for c in cands:
            if c.family == ActionFamily.PASS:
                pass_cand = c
                continue
            score, why = self._score_candidate(c, counts)
            evaluated.append(CallEvaluation(c, score, why))

        # 最高 score を選ぶ。0 以下なら PASS に倒す。
        evaluated.sort(key=lambda e: e.score, reverse=True)
        top = evaluated[0] if evaluated else None
        if top is None:
            # call なし。pass のみあれば pass を返す。
            if pass_cand is not None:
                return CallEvaluation(pass_cand, 0, "pass:no_call_candidate")
            return None
        if top.score > 0:
            return top
        # call 候補は居るが score 0 → PASS 優先
        if pass_cand is not None:
            return CallEvaluation(pass_cand, 0, "pass:call_score_zero")
        # PASS が無い病的ケース → 最大 score の call を返す (Stage02 互換)
        return top

    # ------------------------------------------------------------------
    # candidate scoring
    # ------------------------------------------------------------------

    def _score_candidate(
        self, cand: ModelAction, hand_counts: list[int]
    ) -> tuple[int, str]:
        fam = cand.family
        tt = cand.tile_type
        if fam == ActionFamily.PON:
            return self._score_pon(tt, hand_counts, list(cand.key.consume_tile_types))
        if fam == ActionFamily.DAIMINKAN:
            return self._score_daiminkan(tt)
        if fam == ActionFamily.CHI:
            return self._score_chi(cand, hand_counts)
        return (0, f"unknown_family:{fam.value}")

    def _score_pon(
        self,
        tt: int | None,
        hand_counts: list[int],
        consumed_types: list[int],
    ) -> tuple[int, str]:
        if tt is None:
            return (0, "pon:no_tile_type")
        # 役牌 (白=31, 發=32, 中=33) ポンは常に高 score
        if _is_yakuhai_dragon(tt):
            return (100, "pon:yakuhai_dragon")

        after = list(hand_counts)
        # 消費する 2 枚を手牌から引く
        for ct in consumed_types:
            if 0 <= ct < _NUM_TILE_TYPES and after[ct] > 0:
                after[ct] -= 1

        koutsu_count = sum(1 for c in after if c >= 3)
        pair_count = sum(1 for c in after if c >= 2)
        if koutsu_count + pair_count >= 2:
            return (60, "pon:toitoi_shape")

        # 中張牌 (= 数牌 2-8) のみ かつ手に么九 / 字牌が無ければ喰い断方向
        if not _is_yaochu(tt) and _is_tanyao_compatible(after, extra_tt=None):
            return (50, "pon:tanyao")

        # 么九牌 (役牌以外) のポンは低優先
        if _is_yaochu(tt):
            return (0, "pon:yaochu_non_yakuhai")

        return (15, "pon:neutral_middle")

    def _score_daiminkan(self, tt: int | None) -> tuple[int, str]:
        if tt is None:
            return (0, "kan:no_tile_type")
        if _is_yakuhai_dragon(tt):
            return (90, "kan:yakuhai_dragon")
        if not _is_yaochu(tt):
            return (20, "kan:tanyao_compatible")
        return (0, "kan:yaochu_non_yakuhai")

    def _score_chi(
        self, cand: ModelAction, hand_counts: list[int]
    ) -> tuple[int, str]:
        tt = cand.tile_type
        consumed = list(cand.key.consume_tile_types)
        if tt is None or not consumed:
            return (0, "chi:no_info")
        chi_tts = sorted(list(consumed) + [tt])

        after = list(hand_counts)
        for ct in consumed:
            if 0 <= ct < _NUM_TILE_TYPES and after[ct] > 0:
                after[ct] -= 1

        # 一気通貫 / 三色同順への寄与は高 score
        if _contributes_to_ittsu(chi_tts, after):
            return (70, "chi:ittsu")
        if _contributes_to_sanshoku(chi_tts, after):
            return (70, "chi:sanshoku")

        # 喰い断方向 (=consumed + tt が全て中張牌)
        all_tts = list(consumed) + [tt]
        if all(not _is_yaochu(t) for t in all_tts) and _is_tanyao_compatible(
            after, extra_tt=None
        ):
            return (40, "chi:tanyao")

        return (0, "chi:no_direction")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _is_yakuhai_dragon(tt: int) -> bool:
    """白 (31) / 發 (32) / 中 (33) のいずれか。

    自風 / 場風判定は player_id / round_wind が必要なので本 helper では扱わない
    (= 自/場風牌の役牌判定は本 heuristic では行わない、簡略化)。
    """
    return int(tt) >= 31


def _is_yaochu(tt: int) -> bool:
    """么九牌 (1/9 数牌 + 全字牌) か。"""
    tt = int(tt)
    if tt >= 27:
        return True
    num = (tt % 9) + 1
    return num == 1 or num == 9


def _is_tanyao_compatible(
    counts: list[int], extra_tt: int | None
) -> bool:
    """手牌 + 副露 + 候補 tile が断么九条件 (么九不在) を満たせるか。"""
    for t in range(_NUM_TILE_TYPES):
        if counts[t] > 0 and _is_yaochu(t):
            return False
    if extra_tt is not None and _is_yaochu(extra_tt):
        return False
    return True


def _contributes_to_ittsu(
    chi_tts: list[int], after_counts: list[int]
) -> bool:
    """この chi が一気通貫 (1-2-3 / 4-5-6 / 7-8-9) に寄与するか。"""
    if not chi_tts:
        return False
    first = min(chi_tts)
    suit = first // 9
    if suit >= 3:
        return False
    base = suit * 9
    shuntsu_starts: set[int] = set()
    shuntsu_starts.add(first)
    for s in range(7):
        t = base + s
        if (
            after_counts[t] >= 1
            and after_counts[t + 1] >= 1
            and after_counts[t + 2] >= 1
        ):
            shuntsu_starts.add(t)
    needed = {base, base + 3, base + 6}
    return len(shuntsu_starts & needed) >= 2


def _contributes_to_sanshoku(
    chi_tts: list[int], after_counts: list[int]
) -> bool:
    """この chi が三色同順に寄与するか。

    chi で作った順子のスタート (例 ``num=1`` なら 1-2-3) と同じ num の
    順子が、別スートの ``after_counts`` に居れば +1。chi 自体のスートは
    chi で消費済みなので after_counts 上は 0 になっており、別途 +1 する。
    """
    if not chi_tts:
        return False
    first = min(chi_tts)
    chi_suit = first // 9
    if chi_suit >= 3:
        return False
    num = first % 9
    if num + 2 >= 9:
        return False
    suits_with_shuntsu = 0
    for s in range(3):
        if s == chi_suit:
            continue
        t = s * 9 + num
        if (
            after_counts[t] >= 1
            and after_counts[t + 1] >= 1
            and after_counts[t + 2] >= 1
        ):
            suits_with_shuntsu += 1
    # chi 自体を 1 スート分としてカウント
    suits_with_shuntsu += 1
    return suits_with_shuntsu >= 2


def extract_hand_counts(observation: Any) -> list[int] | None:
    """``riichienv.Observation`` の自手から 34-dim 牌種 counts を作る。

    `observation` が None または `.hand` を持たない場合は ``None``。
    hidden info (他家手牌) には触れない。
    """
    if observation is None:
        return None
    hand = getattr(observation, "hand", None)
    if hand is None:
        return None
    counts = [0] * _NUM_TILE_TYPES
    for tid in hand:
        tt = int(tid) // 4
        if 0 <= tt < _NUM_TILE_TYPES:
            counts[tt] += 1
    return counts


def extract_own_meld_count(observation: Any) -> int:
    """自分の公開副露 + 暗槓を「副露面子」として数える。

    shanten 計算の ``meld_count`` (= 既に確定した面子数) として使う。
    Ankan / Kakan / Pon / Chi / Daiminkan すべて 1 面子分。Kita (3P) は扱わない。
    """
    if observation is None:
        return 0
    pid = getattr(observation, "player_id", None)
    melds = getattr(observation, "melds", None)
    if pid is None or melds is None:
        return 0
    own = melds[int(pid)] if 0 <= int(pid) < len(melds) else []
    return len(own)


__all__ = [
    "CallEvaluation",
    "RuleBasedCallPolicy",
    "extract_hand_counts",
    "extract_own_meld_count",
]
