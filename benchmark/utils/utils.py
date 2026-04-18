from __future__ import annotations

from typing import Any, Mapping

from BenchmarkAdapter.adapters.base import BenchmarkAdapter, BenchmarkTaskSelector
from BenchmarkAdapter.registry import build_adapter as _build_adapter
from BenchmarkAdapter.registry import build_task_selector as _build_task_selector


def build_task_selector(
    benchmark: str,
    task_cfg: Mapping[str, Any],
) -> BenchmarkTaskSelector:
    return _build_task_selector(benchmark, dict(task_cfg))


def adapter_selector(
    benchmark: str,
    task_cfg: Mapping[str, Any],
) -> BenchmarkAdapter:
    """Build the configured benchmark adapter."""
    return _build_adapter(benchmark, dict(task_cfg))
