from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..base import BenchmarkInstance, BenchmarkResult, Trajectory


class BenchmarkAdapter(ABC):
    """Abstract interface for benchmark-specific adapters.

    Implementations should only override benchmark-specific behavior:
    loading instances, extracting task text, building environments, and
    judging results. The shared runner can call these methods uniformly.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def load_instances(self) -> Iterable[Mapping[str, Any]]:
        """Load raw benchmark rows or tasks for the configured split."""
        pass

    @abstractmethod
    def normalize_instance(self, instance: Mapping[str, Any]) -> BenchmarkInstance:
        """Convert a raw instance into the canonical representation."""
        pass

    @abstractmethod
    def build_environment(self, instance: BenchmarkInstance) -> Any:
        """Create the environment used to run the agent for one instance."""
        pass

    def build_context(self, instance: BenchmarkInstance, env: Any) -> dict[str, Any]:
        """Optional context builder consumed by the agent."""
        return {"instance": instance, "env": env}

    @abstractmethod
    def judge(self, instance: BenchmarkInstance, trajectory: Trajectory) -> BenchmarkResult:
        """Evaluate one trajectory and return a canonical result."""
        pass

    def before_run(self) -> None:
        """Optional hook invoked once before iterating over instances."""
        return None

    def after_run(self) -> None:
        """Optional hook invoked once after iterating over instances."""
        return None

    def teardown_environment(self, env: Any) -> None:
        """Optional cleanup hook for one instance environment."""
        return None

    def iter_instances(self) -> Iterable[BenchmarkInstance]:
        """Default helper that normalizes all loaded raw instances."""
        for instance in self.load_instances():
            yield self.normalize_instance(instance)


class BenchmarkTaskSelector(ABC):
    def __init__(self, task_cfg: Mapping[str, Any], method_cfg: Mapping[str, Any] | None = None) -> None:
        self.task_cfg = task_cfg

    @abstractmethod
    def get_run_objects(self) -> list["RunObject"]:
        """Return benchmark-specific run objects used by the pipeline."""


@dataclass(frozen=True)
class RunObject:
    name: str  # task name expmple: "webarena.1"
    cli_args: dict[str, Any]
    instances: dict[str, Any]
    task_description: str | None = None
    output_stem: str | None = None
    tags: dict[str, str] = field(default_factory=dict)



