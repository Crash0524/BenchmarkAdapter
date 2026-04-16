from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..base import BenchmarkInstance, Trajectory


@dataclass(slots=True)
class AgentResponse:
    """Standardized agent response for one benchmark instance."""

    prediction: str
    trajectory: Trajectory = field(default_factory=Trajectory)
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentProtocol(Protocol):
    """Protocol to decouple benchmark adapters from agent implementations."""

    def solve(self, instance: BenchmarkInstance, context: dict[str, Any]) -> AgentResponse:
        """Solve one benchmark instance and return a standardized response."""
