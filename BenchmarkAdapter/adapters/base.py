from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..base import BenchmarkInstance, BenchmarkResult, Trajectory


class BenchmarkAdapter(ABC):
    """Abstract interface for benchmark-specific integrations.

    The repository currently supports two execution styles:
    1. benchmark-native execution via ``run(...)`` on one ``RunObject``;
    2. generic execution via ``iter_instances()`` for the shared runner.

    WebArena uses the benchmark-native path today. Future benchmarks such as
    Mind2Web should implement the smallest surface they actually need instead
    of inheriting assumptions from a different execution model.
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

    @abstractmethod
    def build_environment(self, instance: BenchmarkInstance) -> Any:
        """Create the environment used to run the agent for one instance."""

    def build_context(self, instance: BenchmarkInstance, env: Any) -> dict[str, Any]:
        """Optional context builder consumed by the agent."""
        return {"instance": instance, "env": env}

    @abstractmethod
    def judge(self, instance: BenchmarkInstance, trajectory: Trajectory) -> BenchmarkResult:
        """Evaluate one trajectory and return a canonical result."""

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

    def run(
        self,
        run_object: "RunObject",
        chat_model_args: Any | None = None,
        memory_path: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> Path:
        """Execute one benchmark-specific run object.

        Adapters should override this when the benchmark has a native execution
        stack that is more appropriate than the shared generic runner.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement benchmark-native run(...)"
        )


class BenchmarkTaskSelector(ABC):
    """Select benchmark-specific run objects from task configuration."""

    def __init__(self, task_cfg: Mapping[str, Any]) -> None:
        self.task_cfg = task_cfg

    @abstractmethod
    def get_run_objects(self) -> list["RunObject"]:
        """Return benchmark-specific run objects used by the pipeline."""


@dataclass(frozen=True)
class RunObject:
    """Canonical task unit emitted by a benchmark task selector."""

    name: str
    cli_args: dict[str, Any]
    instances: dict[str, Any]
    task_description: str | None = None
    output_stem: str | None = None
    tags: dict[str, str] = field(default_factory=dict)

