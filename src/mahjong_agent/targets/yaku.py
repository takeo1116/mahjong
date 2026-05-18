"""Yaku auxiliary target: 固定 vocab と multi-hot extractor.

`mahjong_agent` 内で固定順の yaku vocab を持ち、学習再現性を確保する。
PyPI 版 ``riichienv`` の ``get_all_yaku()`` で得られる ID/名称を snapshot
してハードコードする (vocab が riichienv の version で動かないようにする
ため)。snapshot 一致は ``test_targets.py`` の smoke で検証する。

dora-like yaku (Dora / Aka Dora / Ura Dora / Nuki Dora) は yaku-id 31..34。
通常 yaku と同じ multi-hot に含めるが、diagnostics 側で分離できるよう
``DORA_LIKE_YAKU_IDS`` を export する。
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class YakuInfo:
    """1 yaku の metadata。"""

    id: int
    name: str
    name_en: str
    is_dora_like: bool


# Dora-like yaku の id 集合 (riichienv で id 31..34 が Dora / Aka / Ura / Nuki)。
DORA_LIKE_YAKU_IDS: frozenset[int] = frozenset({31, 32, 33, 34})


# riichienv 0.4.8 の get_all_yaku() snapshot。固定順を保つため hardcode。
# 配列 index = vocab index (0..48)、entry.id = riichienv 内 yaku id。
# 配列の順序は変更しないこと (= 既存 checkpoint の loss head が壊れる)。
_RAW_YAKU_TABLE: tuple[tuple[int, str, str], ...] = (
    (1, "門前清自摸和", "Menzen Tsumo"),
    (2, "立直", "Riichi"),
    (3, "槍槓", "Chankan"),
    (4, "嶺上開花", "Rinshan Kaihou"),
    (5, "海底摸月", "Haitei Raoyue"),
    (6, "河底撈魚", "Houtei Raoyui"),
    (7, "役牌 白", "Yakuhai (haku)"),
    (8, "役牌 發", "Yakuhai (hatsu)"),
    (9, "役牌 中", "Yakuhai (chun)"),
    (10, "自風牌", "Yakuhai (seat wind)"),
    (11, "場風牌", "Yakuhai (round wind)"),
    (12, "断幺九", "Tanyao"),
    (13, "一盃口", "Iipeiko"),
    (14, "平和", "Pinfu"),
    (15, "混全帯幺九", "Chantai"),
    (16, "一気通貫", "Ittsu"),
    (17, "三色同順", "Sanshoku Doujun"),
    (18, "ダブル立直", "Double Riichi"),
    (19, "三色同刻", "Sanshoku Doukou"),
    (20, "三槓子", "San Kantsu"),
    (21, "対々和", "Toitoi"),
    (22, "三暗刻", "San Ankou"),
    (23, "小三元", "Shou Sangen"),
    (24, "混老頭", "Honroutou"),
    (25, "七対子", "Chiitoitsu"),
    (26, "純全帯幺九", "Junchan"),
    (27, "混一色", "Honitsu"),
    (28, "二盃口", "Ryanpeikou"),
    (29, "清一色", "Chinitsu"),
    (30, "一発", "Ippatsu"),
    (31, "ドラ", "Dora"),
    (32, "赤ドラ", "Aka Dora"),
    (33, "裏ドラ", "Ura Dora"),
    (34, "抜きドラ", "Nuki Dora"),
    (35, "天和", "Tenhou"),
    (36, "地和", "Chiihou"),
    (37, "大三元", "Dai Sangen"),
    (38, "四暗刻", "Su Ankou"),
    (39, "字一色", "Tsuu iisou"),
    (40, "緑一色", "Ryuu iisou"),
    (41, "清老頭", "Chinroutou"),
    (42, "国士無双", "Kokushi Musou"),
    (43, "小四喜", "Sho Suusi"),
    (44, "四槓子", "Su Kantsu"),
    (45, "九蓮宝燈", "Chuuren Poutou"),
    (47, "純正九蓮宝燈", "Junsei Chuuren Poutou"),
    (48, "四暗刻単騎", "Su Ankou Tanki"),
    (49, "国士無双十三面待ち", "Kokushi Musou 13-men"),
    (50, "大四喜", "Dai Suusi"),
)

YAKU_VOCAB: tuple[YakuInfo, ...] = tuple(
    YakuInfo(
        id=yid,
        name=jp,
        name_en=en,
        is_dora_like=(yid in DORA_LIKE_YAKU_IDS),
    )
    for (yid, jp, en) in _RAW_YAKU_TABLE
)
NUM_YAKU: int = len(YAKU_VOCAB)

# yaku_id -> vocab index
YAKU_ID_TO_INDEX: dict[int, int] = {y.id: i for i, y in enumerate(YAKU_VOCAB)}


def yaku_index(yaku_id: int) -> int:
    """yaku_id -> vocab index。未知 id は KeyError。"""
    return YAKU_ID_TO_INDEX[int(yaku_id)]


def is_dora_like_yaku(yaku_id: int) -> bool:
    """yaku_id が Dora / Aka Dora / Ura Dora / Nuki Dora かどうか。"""
    return int(yaku_id) in DORA_LIKE_YAKU_IDS


def yaku_ids_to_multihot(
    yaku_ids: Iterable[int],
    *,
    allow_unknown: bool = False,
) -> np.ndarray:
    """yaku_id list を ``(NUM_YAKU,)`` float32 multi-hot に変換する。

    Parameters
    ----------
    yaku_ids:
        list of riichienv yaku id (winner-only)。
    allow_unknown:
        ``False`` (default): 未知 id は ``ValueError`` を投げる (fail-fast)。
        ``True``: 未知 id は silent に無視する (互換性パスに使う)。
    """
    out = np.zeros(NUM_YAKU, dtype=np.float32)
    for yid in yaku_ids:
        yid_int = int(yid)
        idx = YAKU_ID_TO_INDEX.get(yid_int)
        if idx is None:
            if allow_unknown:
                continue
            raise ValueError(
                f"unknown yaku id: {yid_int}; expected one of "
                f"{sorted(YAKU_ID_TO_INDEX.keys())}"
            )
        out[idx] = 1.0
    return out


def extract_yaku_target(
    yaku_ids: Iterable[int] | None,
    *,
    is_winner: bool,
    allow_unknown: bool = False,
) -> tuple[np.ndarray, float]:
    """winner-only multi-label BCE 用の target / loss mask を返す。

    Returns
    -------
    (target, loss_mask):
        target: ``(NUM_YAKU,)`` float32 multi-hot。``is_winner=False`` または
            yaku_ids が None のときは all-zero。
        loss_mask: 0.0 or 1.0 の scalar。winner で yaku_ids が与えられたとき
            のみ 1.0。それ以外は 0.0 (BCE loss を mask out する想定)。
    """
    if not is_winner or yaku_ids is None:
        return np.zeros(NUM_YAKU, dtype=np.float32), 0.0
    target = yaku_ids_to_multihot(yaku_ids, allow_unknown=allow_unknown)
    return target, 1.0


__all__ = [
    "YakuInfo",
    "YAKU_VOCAB",
    "NUM_YAKU",
    "YAKU_ID_TO_INDEX",
    "DORA_LIKE_YAKU_IDS",
    "yaku_index",
    "is_dora_like_yaku",
    "yaku_ids_to_multihot",
    "extract_yaku_target",
]
