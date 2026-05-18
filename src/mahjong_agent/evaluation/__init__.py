"""Self-play / evaluation loops and metrics."""
from __future__ import annotations

from mahjong_agent.evaluation.metrics import aggregate_metrics
from mahjong_agent.evaluation.round_tracker import (
    RoundTracker,
    make_initial_sample,
)
from mahjong_agent.evaluation.runner import (
    EpisodeResult,
    SelfPlayConfig,
    SelfPlayRunner,
    make_seat_agents_from_mapping,
)
from mahjong_agent.evaluation.seat_agents import SeatAgents

__all__ = [
    "EpisodeResult",
    "RoundTracker",
    "SeatAgents",
    "SelfPlayConfig",
    "SelfPlayRunner",
    "aggregate_metrics",
    "make_initial_sample",
    "make_seat_agents_from_mapping",
]
