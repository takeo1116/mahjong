"""Model-facing action abstraction and RiichiEnv action resolver."""
from __future__ import annotations

from mahjong_agent.actions.convert import legal_actions_to_model_set
from mahjong_agent.actions.resolver import (
    resolve,
    resolve_candidate,
    resolve_normal_discard,
)
from mahjong_agent.actions.types import (
    ActionFamily,
    ActionKey,
    LegalActionSet,
    ModelAction,
    is_red_tile_id,
    tile_id_to_type,
)

__all__ = [
    "ActionFamily",
    "ActionKey",
    "LegalActionSet",
    "ModelAction",
    "is_red_tile_id",
    "tile_id_to_type",
    "legal_actions_to_model_set",
    "resolve",
    "resolve_candidate",
    "resolve_normal_discard",
]
