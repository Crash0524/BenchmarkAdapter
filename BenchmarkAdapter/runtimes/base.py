from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..base import BenchmarkInstance
from .protocols import RuntimeSolveResult


class BaseRuntime(ABC):
    """Abstract base class for method/runtime implementations."""

    @abstractmethod
    def solve(self, instance: BenchmarkInstance, context: dict[str, Any]) -> RuntimeSolveResult:
        """Solve one instance under provided benchmark context."""
        pass
