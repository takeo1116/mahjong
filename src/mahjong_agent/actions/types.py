"""Model-facing action types and semantic key.

これらの型は policy / model が扱う abstraction を表す。RiichiEnv の raw
``Action`` (physical tile id 等を含む) は、resolver 用 metadata として
``ModelAction._raw_actions`` に閉じ込め、model-facing feature には含めない。

- 通常打牌は 34-way head 用に ``LegalActionSet.normal_discard``
  (tile_type -> ModelAction) で表現する。
- それ以外 (RiichiDiscard / Chi / Pon / Daiminkan / Ankan / Kakan / Ron /
  Tsumo / KyushuKyuhai / Kita / Pass) は candidate head 用に
  ``LegalActionSet.candidates`` (tuple of ModelAction) で表現する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionFamily(str, Enum):
    """model-facing action family. policy/encoder で使う族識別子。"""

    NORMAL_DISCARD = "normal_discard"
    RIICHI_DISCARD = "riichi_discard"
    TSUMO = "tsumo"
    RON = "ron"
    CHI = "chi"
    PON = "pon"
    DAIMINKAN = "daiminkan"
    ANKAN = "ankan"
    KAKAN = "kakan"
    KYUSHU_KYUHAI = "kyushu_kyuhai"
    KITA = "kita"
    PASS = "pass"


# physical tile id 0..135 のうち red 5 牌に割り当てられている id。
# riichienv の parse_tile("0m") = 16, parse_tile("0p") = 52, parse_tile("0s") = 88
# (= 各 5 系列の先頭 id) で確認済み。
_RED_TILE_IDS: frozenset[int] = frozenset({16, 52, 88})


def is_red_tile_id(tile_id: int) -> bool:
    """tile_id (0..135) が red 5 牌かどうかを返す。"""
    return int(tile_id) in _RED_TILE_IDS


def tile_id_to_type(tile_id: int) -> int:
    """tile_id (0..135) を tile_type (0..33) に変換する。"""
    return int(tile_id) // 4


@dataclass(frozen=True, eq=True)
class ActionKey:
    """Semantic key for ModelAction.

    legal action list の順序や具体 tile_id に依存せず、同一 action を一意に
    識別するための key。policy index と直接結びつけない。

    Fields
    ------
    family:
        action family。
    tile_type:
        action の対象となる牌種 (0..33)。Pass / Tsumo / Ron / KyushuKyuhai
        など、tile_type が不要なものは None。
    consume_tile_types:
        副露で消費される牌の tile_type tuple (昇順)。Chi では 2 枚、Pon では
        2 枚、Daiminkan / Ankan / Kakan では 3 枚 (Kakan は 既存 Pon 構成牌)。
        Ron / Tsumo / Discard / Pass / Riichi 系では空 tuple。
    target_rel_seat:
        対象 player への相対席 (0=自家, 1=下家, 2=対面, 3=上家)。Chi/Pon/
        Daiminkan/Ron など対象 player が定まる action でのみ設定。それ以外は
        None。

    Notes
    -----
    Red 5 と通常 5 は同じ tile_type を共有するため key 上は同一になる。
    解決時 (resolver) に通常牌優先で具体 tile_id を選ぶ。
    """

    family: ActionFamily
    tile_type: int | None = None
    consume_tile_types: tuple[int, ...] = ()
    target_rel_seat: int | None = None


@dataclass(frozen=True)
class ModelAction:
    """policy / agent が扱う 1 つの action / candidate。

    ``key`` が semantic identity を表し、policy はこれを基に softmax index
    などに対応付ける。``_raw_actions`` は resolver / env step 用 metadata
    であり、model-facing feature には含めない。

    Attributes
    ----------
    key:
        action の semantic key。
    actor:
        この action を実行する player id (0..3)。
    _raw_actions:
        env step に渡す raw ``riichienv.Action`` の tuple。RiichiDiscard では
        ``(Riichi, Discard)`` の 2 要素になる。それ以外は通常 1 要素。
    """

    key: ActionKey
    actor: int
    _raw_actions: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    @property
    def family(self) -> ActionFamily:
        return self.key.family

    @property
    def tile_type(self) -> int | None:
        return self.key.tile_type

    def raw_actions(self) -> tuple[Any, ...]:
        """env.step に渡す raw Action tuple を返す (resolver 用)。"""
        return self._raw_actions


@dataclass(frozen=True)
class LegalActionSet:
    """ある decision point での model-facing legal action 集合。

    Attributes
    ----------
    decision_player:
        この decision を求められている player id (Phase.WaitAct なら
        current_player、Phase.WaitResponse なら各 claimant ごとに別 set)。
    normal_discard:
        tile_type -> ModelAction の dict。``ActionFamily.NORMAL_DISCARD``
        専用。同一 tile_type に複数 raw discard (例: 赤 / 通常) があれば
        通常牌優先で 1 つに集約済み。空の場合は通常打牌が legal ではない。
    candidates:
        normal discard 以外の全 candidate (RiichiDiscard / Chi / Pon /
        Daiminkan / Ankan / Kakan / Ron / Tsumo / KyushuKyuhai / Kita /
        Pass)。tuple なので順序は安定だが、index に意味は持たせない。
    """

    decision_player: int
    normal_discard: dict[int, ModelAction] = field(default_factory=dict)
    candidates: tuple[ModelAction, ...] = ()

    @property
    def can_normal_discard(self) -> bool:
        return bool(self.normal_discard)

    @property
    def candidate_keys(self) -> list[ActionKey]:
        return [c.key for c in self.candidates]

    def normal_discard_tile_types(self) -> list[int]:
        """通常打牌可能な tile_type list (昇順)。"""
        return sorted(self.normal_discard.keys())

    def find_candidate(self, key: ActionKey) -> ModelAction | None:
        """semantic key 一致する candidate を返す (見つからなければ None)。"""
        for c in self.candidates:
            if c.key == key:
                return c
        return None


__all__ = [
    "ActionFamily",
    "ActionKey",
    "ModelAction",
    "LegalActionSet",
    "is_red_tile_id",
    "tile_id_to_type",
]
