"""RiichiEnv legal action <-> model-facing action conversion."""
from __future__ import annotations

from typing import Any

import riichienv

from mahjong_agent.actions.types import (
    ActionFamily,
    ActionKey,
    LegalActionSet,
    ModelAction,
    is_red_tile_id,
    tile_id_to_type,
)

# RiichiEnv の ActionType.* と model-facing ``ActionFamily`` の対応。
# 通常打牌 (DISCARD) は normal_discard / riichi_discard を tile-type-level で
# 区別する必要があり、後段で別途扱う。
_ACTION_TYPE_TO_FAMILY: dict[Any, ActionFamily] = {
    riichienv.ActionType.TSUMO: ActionFamily.TSUMO,
    riichienv.ActionType.RON: ActionFamily.RON,
    riichienv.ActionType.CHI: ActionFamily.CHI,
    riichienv.ActionType.PON: ActionFamily.PON,
    riichienv.ActionType.DAIMINKAN: ActionFamily.DAIMINKAN,
    riichienv.ActionType.ANKAN: ActionFamily.ANKAN,
    riichienv.ActionType.KAKAN: ActionFamily.KAKAN,
    riichienv.ActionType.KYUSHU_KYUHAI: ActionFamily.KYUSHU_KYUHAI,
    riichienv.ActionType.KITA: ActionFamily.KITA,
    riichienv.ActionType.PASS: ActionFamily.PASS,
}


def _discard_sort_key(raw_action: Any) -> tuple[bool, int]:
    """同一 tile_type 内で通常牌 -> 赤牌 の順に並べる sort key。

    Returns
    -------
    (is_red, tile_id):
        通常牌 (is_red=False) が先頭、その中で tile_id 昇順。
    """
    tid = int(raw_action.tile) if raw_action.tile is not None else -1
    return (is_red_tile_id(tid), tid)


def _consume_tile_types(raw_action: Any) -> tuple[int, ...]:
    """raw action の consume_tiles (tile_id list) を tile_type tuple (昇順) に変換する。"""
    if not raw_action.consume_tiles:
        return ()
    return tuple(sorted(tile_id_to_type(t) for t in raw_action.consume_tiles))


def _relative_seat(actor: int, target_actor: int, num_players: int) -> int:
    """``actor`` から見た ``target_actor`` の相対席を返す (0=self, 1=shimo, ...)。"""
    return (int(target_actor) - int(actor)) % int(num_players)


def _build_normal_discard_set(
    raw_actions: list[Any],
    actor: int,
) -> dict[int, ModelAction]:
    """raw legal actions から normal discard set (tile_type -> ModelAction) を構築する。

    同一 tile_type に複数 raw discard (赤 / 通常) があれば通常牌優先で 1 つに
    集約する。
    """
    by_type: dict[int, list[Any]] = {}
    for a in raw_actions:
        if a.action_type != riichienv.ActionType.DISCARD:
            continue
        if a.tile is None:
            continue
        tt = tile_id_to_type(a.tile)
        by_type.setdefault(tt, []).append(a)
    out: dict[int, ModelAction] = {}
    for tt, opts in by_type.items():
        opts.sort(key=_discard_sort_key)
        chosen = opts[0]
        key = ActionKey(family=ActionFamily.NORMAL_DISCARD, tile_type=tt)
        out[tt] = ModelAction(key=key, actor=actor, _raw_actions=(chosen,))
    return out


def _build_simple_candidate(
    raw_action: Any,
    actor: int,
    num_players: int,
    last_discarder: int | None,
) -> ModelAction | None:
    """非 Riichi の単発 candidate (Chi/Pon/Daiminkan/Ankan/Kakan/Ron/Tsumo/
    KyushuKyuhai/Kita/Pass) を ModelAction に変換する。

    DISCARD は別経路で処理するので None を返す。RIICHI は本関数では扱わない
    (上位で RiichiDiscard candidate に展開する)。
    """
    at = raw_action.action_type
    if at == riichienv.ActionType.DISCARD or at == riichienv.ActionType.RIICHI:
        return None
    family = _ACTION_TYPE_TO_FAMILY.get(at)
    if family is None:
        return None
    tile_type: int | None
    if raw_action.tile is not None:
        tile_type = tile_id_to_type(raw_action.tile)
    else:
        tile_type = None
    consume = _consume_tile_types(raw_action)
    target_rel_seat: int | None = None
    # 対象 player が定まる action は last_discarder への相対席を入れる。
    # Chi / Pon / Daiminkan / Ron は他家の discard に対する反応。
    if family in (
        ActionFamily.CHI,
        ActionFamily.PON,
        ActionFamily.DAIMINKAN,
        ActionFamily.RON,
    ) and last_discarder is not None:
        target_rel_seat = _relative_seat(actor, last_discarder, num_players)
    key = ActionKey(
        family=family,
        tile_type=tile_type,
        consume_tile_types=consume,
        target_rel_seat=target_rel_seat,
    )
    return ModelAction(key=key, actor=actor, _raw_actions=(raw_action,))


def _enumerate_riichi_discards(
    env: Any,
    riichi_action: Any,
    actor: int,
) -> list[ModelAction]:
    """riichi 宣言可能な状態から、宣言後に許される discard を tile_type 別に
    列挙して RiichiDiscard candidate を返す。

    env を clone して Riichi を step し、得られた discard legal actions を
    tile_type で集約する。同一 tile_type 内では通常牌優先で raw_action を選ぶ。
    """
    cloned = env.clone()
    cloned.step({actor: riichi_action})
    obs = cloned.get_observation(actor)
    la = list(obs.legal_actions())
    by_type: dict[int, list[Any]] = {}
    for a in la:
        if a.action_type != riichienv.ActionType.DISCARD:
            continue
        if a.tile is None:
            continue
        tt = tile_id_to_type(a.tile)
        by_type.setdefault(tt, []).append(a)
    out: list[ModelAction] = []
    for tt, opts in by_type.items():
        opts.sort(key=_discard_sort_key)
        discard_action = opts[0]
        key = ActionKey(family=ActionFamily.RIICHI_DISCARD, tile_type=tt)
        # raw_actions は (Riichi, Discard) の 2-step sequence。
        out.append(
            ModelAction(
                key=key,
                actor=actor,
                _raw_actions=(riichi_action, discard_action),
            )
        )
    # 安定性のため tile_type 昇順で返す。
    out.sort(key=lambda m: (m.key.tile_type if m.key.tile_type is not None else -1))
    return out


def legal_actions_to_model_set(
    raw_actions: list[Any],
    actor: int,
    *,
    num_players: int = 4,
    last_discarder: int | None = None,
    env_for_riichi: Any | None = None,
) -> LegalActionSet:
    """1 player 分の raw legal actions から ``LegalActionSet`` を構築する。

    Parameters
    ----------
    raw_actions:
        ``riichienv.Action`` の list。``Observation.legal_actions()`` の結果。
    actor:
        decision player id。
    num_players:
        4 (4 人麻雀) または 3 (3 人麻雀)。
    last_discarder:
        Chi/Pon/Daiminkan/Ron の対象 discarder player id。Response Phase で
        分かる場合のみ渡す。``Phase.WaitAct`` (= 自家行動) のときは None で
        良い。
    env_for_riichi:
        Riichi action が legal の場合に、Riichi 宣言後の discard を列挙する
        ためにに clone して使う ``riichienv.RiichiEnv``。``None`` を渡すと、
        Riichi candidate は生成されない (= 後段で別途扱う前提)。

    Returns
    -------
    LegalActionSet
    """
    actor = int(actor)
    # normal discard set
    normal = _build_normal_discard_set(raw_actions, actor=actor)

    # その他 candidate
    candidates: list[ModelAction] = []
    riichi_actions = [
        a for a in raw_actions if a.action_type == riichienv.ActionType.RIICHI
    ]
    for a in raw_actions:
        mc = _build_simple_candidate(
            a, actor=actor, num_players=num_players, last_discarder=last_discarder
        )
        if mc is not None:
            candidates.append(mc)
    # Riichi candidate: env が渡されていれば RiichiDiscard を tile_type 別に展開
    if riichi_actions and env_for_riichi is not None:
        riichi_action = riichi_actions[0]
        candidates.extend(
            _enumerate_riichi_discards(env_for_riichi, riichi_action, actor)
        )
    # 順序安定化 (family -> tile_type -> consume_tile_types -> target_rel_seat)
    candidates.sort(
        key=lambda m: (
            m.key.family.value,
            m.key.tile_type if m.key.tile_type is not None else -1,
            m.key.consume_tile_types,
            m.key.target_rel_seat if m.key.target_rel_seat is not None else -1,
        )
    )
    return LegalActionSet(
        decision_player=actor,
        normal_discard=normal,
        candidates=tuple(candidates),
    )


__all__ = [
    "legal_actions_to_model_set",
]
