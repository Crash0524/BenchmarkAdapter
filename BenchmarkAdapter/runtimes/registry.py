from __future__ import annotations

from typing import Type

from .base import BaseRuntime
from methods.echo import EchoRuntime


RUNTIME_REGISTRY: dict[str, Type[BaseRuntime]] = {
    "echo": EchoRuntime,
}


def register_runtime(name: str, runtime_cls: Type[BaseRuntime]) -> None:
    RUNTIME_REGISTRY[name.strip().lower()] = runtime_cls


def list_runtime_names() -> list[str]:
    return sorted(RUNTIME_REGISTRY.keys())


def get_runtime_cls(name: str) -> Type[BaseRuntime]:
    key = name.strip().lower()
    if key not in RUNTIME_REGISTRY:
        valid = ", ".join(list_runtime_names())
        raise KeyError(f"Unknown runtime: {name}. Valid: {valid}")
    return RUNTIME_REGISTRY[key]
