"""Agent base types: ``AgentDecision`` and minimal ``Agent`` interface.

Stage03 の self-play / evaluation loop からは agent をできるだけ単純に呼びたい
ので、ここでは抽象基底ではなく軽量な dataclass + duck typing を採用する。

Agent は概念的に以下のシグネチャを持つ:

```python
agent.select_action(
    legal_set: LegalActionSet,
    *,
    rng: random.Random | None = None,
    observation: Any | None = None,
) -> AgentDecision
```

``observation`` は public observation を受け取れる slot として用意する
(rule_base は使わなくてよい)。hidden info は agent に渡さない。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mahjong_agent.actions.types import ActionFamily, ModelAction


@dataclass(frozen=True)
class AgentDecision:
    """Agent の 1 回の選択結果を表す軽量 dataclass。

    Attributes
    ----------
    action:
        agent が選んだ model-facing ``ModelAction``。
    rationale:
        debug / diagnostics 用の短いラベル (``"win"`` / ``"normal_discard"`` /
        ``"pass"`` 等)。downstream の loop で metric を切る用途。
    """

    action: ModelAction
    rationale: str = ""
    # forward-compat 用の free-form metadata (JSON serializable な値のみ)。
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def family(self) -> ActionFamily:
        return self.action.family

    @property
    def tile_type(self) -> int | None:
        return self.action.tile_type

    def raw_actions(self) -> tuple[Any, ...]:
        """``env.step`` に渡せる raw action tuple を返す。

        ``RiichiDiscard`` の場合は ``(Riichi, Discard)`` の 2-step。
        ``LegalActionSet`` を再度引かなくても agent decision から直接
        env step に渡せるよう、resolver と同等の値を返す convenience。
        """
        return self.action.raw_actions()


__all__ = ["AgentDecision"]
