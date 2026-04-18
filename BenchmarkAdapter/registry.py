from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Type

from .adapters.base import BenchmarkAdapter, BenchmarkTaskSelector


@dataclass(frozen=True, slots=True)
class BenchmarkBinding:
    adapter_path: str
    task_selector_path: str


BENCHMARK_REGISTRY: dict[str, BenchmarkBinding] = {
    "webarena": BenchmarkBinding(
        adapter_path="benchmark.WebArena.adapter:WebArenaAdapter",
        task_selector_path="benchmark.WebArena.adapter:WebArenaTaskSelector",
    ),
}


def _load_attr(import_path: str) -> Any:
    module_name, attr_name = import_path.split(":", maxsplit=1)
    module = import_module(module_name)
    return getattr(module, attr_name)


def register_benchmark(name: str, binding: BenchmarkBinding) -> None:
    BENCHMARK_REGISTRY[name.strip().lower()] = binding


def register_adapter(
    name: str,
    adapter_cls: Type[BenchmarkAdapter],
    task_selector_cls: Type[BenchmarkTaskSelector],
) -> None:
    register_benchmark(
        name,
        BenchmarkBinding(
            adapter_path=f"{adapter_cls.__module__}:{adapter_cls.__name__}",
            task_selector_path=f"{task_selector_cls.__module__}:{task_selector_cls.__name__}",
        ),
    )


def list_adapter_names() -> list[str]:
    return sorted(BENCHMARK_REGISTRY.keys())


def get_adapter_cls(name: str) -> Type[BenchmarkAdapter]:
    key = name.strip().lower()
    if key not in BENCHMARK_REGISTRY:
        valid = ", ".join(list_adapter_names())
        raise KeyError(f"Unknown benchmark adapter: {name}. Valid: {valid}")
    return _load_attr(BENCHMARK_REGISTRY[key].adapter_path)


def get_task_selector_cls(name: str) -> Type[BenchmarkTaskSelector]:
    key = name.strip().lower()
    if key not in BENCHMARK_REGISTRY:
        valid = ", ".join(list_adapter_names())
        raise KeyError(f"Unknown benchmark task selector: {name}. Valid: {valid}")
    return _load_attr(BENCHMARK_REGISTRY[key].task_selector_path)


def build_adapter(name: str, config: dict[str, Any]) -> BenchmarkAdapter:
    return get_adapter_cls(name)(config)


def build_task_selector(name: str, task_cfg: dict[str, Any]) -> BenchmarkTaskSelector:
    return get_task_selector_cls(name)(task_cfg)
