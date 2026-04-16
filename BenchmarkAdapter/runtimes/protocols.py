from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..base import BenchmarkInstance, Trajectory


@dataclass(slots=True)
class RuntimeResponse:
    """Standard response emitted by runtimes."""

    prediction: str
    trajectory: Trajectory = field(default_factory=Trajectory)
    metadata: dict[str, Any] = field(default_factory=dict)


RuntimeSolveResult = RuntimeResponse | str


class RuntimeProtocol(Protocol):
    """Protocol for benchmark runtime engines (local agent, API agent, etc.)."""

    def solve(self, instance: BenchmarkInstance, context: dict[str, Any]) -> RuntimeSolveResult:
        """Solve one instance under provided benchmark context."""
