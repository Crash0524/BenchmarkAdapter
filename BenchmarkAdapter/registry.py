from __future__ import annotations

from typing import Type

from .adapters.base import BenchmarkAdapter
from benchmark.WebArena import WebArenaAdapter


ADAPTER_REGISTRY: dict[str, Type[BenchmarkAdapter]] = {
    "webarena": WebArenaAdapter,
}


def register_adapter(name: str, adapter_cls: Type[BenchmarkAdapter]) -> None:
    ADAPTER_REGISTRY[name.strip().lower()] = adapter_cls


def list_adapter_names() -> list[str]:
    return sorted(ADAPTER_REGISTRY.keys())


def get_adapter_cls(name: str) -> Type[BenchmarkAdapter]:
    key = name.strip().lower()
    if key not in ADAPTER_REGISTRY:
        valid = ", ".join(list_adapter_names())
        raise KeyError(f"Unknown benchmark adapter: {name}. Valid: {valid}")
    return ADAPTER_REGISTRY[key]
