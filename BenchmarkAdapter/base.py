"""Abstract benchmark adapter layer.

This module defines the contract between:
1) benchmark-specific logic (dataset schema, env, judging), and
2) benchmark-agnostic execution logic (runner, agent orchestration).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class BenchmarkInstance:
    """Canonical representation of a benchmark instance.

    Benchmark adapters can normalize their raw dataset rows into this schema,
    so the runner can remain benchmark-agnostic.
    """

    instance_id: str
    task: str
    raw: Mapping[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Trajectory:
    """Canonical trajectory artifact returned by an agent run."""

    steps: list[dict[str, Any]] = field(default_factory=list)
    final_output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkResult:
    """Canonical benchmark result for one instance."""

    instance_id: str
    success: bool
    score: float | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
