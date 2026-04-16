from __future__ import annotations

from typing import Any, Mapping


from BenchmarkAdapter.adapters.base import RunObject
from BenchmarkAdapter.adapters.base import BenchmarkAdapter, BenchmarkTaskSelector
from benchmark.WebArena.adapter import WebArenaAdapter, WebArenaTaskSelector


def build_task_selector(
    benchmark: str,
    task_cfg: Mapping[str, Any],
) -> BenchmarkTaskSelector:
    if benchmark == "webarena":
        return WebArenaTaskSelector(task_cfg)
    raise ValueError(f"Unsupported benchmark for task selection: {benchmark}")


def adapter_selector(
    benchmark: str,
    task_cfg: Mapping[str, Any],
) -> BenchmarkAdapter:
    """
    select adapter and env for task
    """
    if benchmark == "webarena":
        config: dict[str, Any] = dict(task_cfg)
        return WebArenaAdapter(config)
    raise ValueError(f"Unsupported benchmark for adapter selection: {benchmark}")
