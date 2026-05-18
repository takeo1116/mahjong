"""Resolve model-facing actions back into RiichiEnv raw action sequences."""
from __future__ import annotations

from typing import Any

from mahjong_agent.actions.types import ActionKey, LegalActionSet, ModelAction


def resolve_normal_discard(
    legal_set: LegalActionSet,
    tile_type: int,
) -> tuple[Any, ...]:
    """通常打牌 (34-way) を raw action tuple に解決する。

    Raises
    ------
    KeyError:
        指定した tile_type が ``legal_set.normal_discard`` に存在しないとき。
    """
    action = legal_set.normal_discard.get(int(tile_type))
    if action is None:
        raise KeyError(
            f"normal_discard for tile_type={tile_type} not in legal set "
            f"(player {legal_set.decision_player}); "
            f"available={sorted(legal_set.normal_discard.keys())}"
        )
    return action.raw_actions()


def resolve_candidate(
    legal_set: LegalActionSet,
    key: ActionKey,
) -> tuple[Any, ...]:
    """semantic key で candidate を引き、raw action tuple に解決する。

    Raises
    ------
    KeyError:
        指定 key を持つ candidate が ``legal_set.candidates`` に存在しないとき。
    """
    match = legal_set.find_candidate(key)
    if match is None:
        raise KeyError(
            f"candidate {key!r} not in legal set (player "
            f"{legal_set.decision_player}); "
            f"available_keys={legal_set.candidate_keys}"
        )
    return match.raw_actions()


def resolve(legal_set: LegalActionSet, action: ModelAction) -> tuple[Any, ...]:
    """``ModelAction`` を raw action tuple に解決する。

    通常打牌は ``legal_set.normal_discard`` から、それ以外は
    ``legal_set.candidates`` から引いて raw_actions を返す。
    """
    if action.family.value == "normal_discard":
        if action.tile_type is None:
            raise ValueError("normal_discard ModelAction requires tile_type")
        return resolve_normal_discard(legal_set, action.tile_type)
    return resolve_candidate(legal_set, action.key)


__all__ = [
    "resolve_normal_discard",
    "resolve_candidate",
    "resolve",
]
