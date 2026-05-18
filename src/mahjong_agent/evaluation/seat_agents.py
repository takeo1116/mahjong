"""SeatAgents: player_id -> (agent, actor_type label) mapping.

self-play / evaluation loop は agent を席に割り当てる必要があるが、agent の
具体型 (RandomAgent / RuleBasedBaselineAgent / 将来の model agent) を限定
したくないので、duck-typed object を受け付ける軽量 container として実装する。

agent には ``select_action(legal_set, *, rng=None, observation=None)`` が
あれば十分。``actor_type`` は ``DecisionSample.actor_type`` に書き込まれる
free-form label (``"random"`` / ``"rule_based"`` / ``"policy"`` 等)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SeatAgents:
    """player_id -> (agent, actor_type) の mapping。

    Attributes
    ----------
    agents:
        ``player_id`` -> agent instance。agent は
        ``select_action(legal_set, *, rng=None, observation=None) -> AgentDecision``
        を持つ duck-typed object。
    actor_types:
        ``player_id`` -> actor_type label。``DecisionSample.actor_type`` に
        記録される free-form 文字列。
    """

    agents: dict[int, Any]
    actor_types: dict[int, str]

    def __post_init__(self) -> None:
        if set(self.agents.keys()) != set(self.actor_types.keys()):
            raise ValueError(
                "SeatAgents: agents and actor_types must have identical keys; "
                f"agents={sorted(self.agents.keys())}, "
                f"actor_types={sorted(self.actor_types.keys())}"
            )

    @classmethod
    def homogeneous(
        cls, agent: Any, actor_type: str, *, num_players: int = 4
    ) -> SeatAgents:
        """全席に同じ agent / actor_type を割り当てる。"""
        return cls(
            agents={i: agent for i in range(num_players)},
            actor_types={i: str(actor_type) for i in range(num_players)},
        )

    @classmethod
    def from_pairs(
        cls, pairs: dict[int, tuple[Any, str]]
    ) -> SeatAgents:
        """``{player_id: (agent, actor_type)}`` から構築する convenience。"""
        agents: dict[int, Any] = {}
        actor_types: dict[int, str] = {}
        for pid, (a, t) in pairs.items():
            agents[int(pid)] = a
            actor_types[int(pid)] = str(t)
        return cls(agents=agents, actor_types=actor_types)

    @property
    def player_ids(self) -> list[int]:
        return sorted(self.agents.keys())


__all__ = ["SeatAgents"]
