from __future__ import annotations

from typing import Type

from benchmark.browsergym import BrowserGymDriver
from benchmark.local import AdapterNativeDriver


DRIVER_REGISTRY: dict[str, Type] = {
    "adapter-native": AdapterNativeDriver,
    "browsergym": BrowserGymDriver,
}


def register_driver(name: str, driver_cls: Type) -> None:
    DRIVER_REGISTRY[name.strip().lower()] = driver_cls


def list_driver_names() -> list[str]:
    return sorted(DRIVER_REGISTRY.keys())


def get_driver_cls(name: str) -> Type:
    key = name.strip().lower()
    if key not in DRIVER_REGISTRY:
        valid = ", ".join(list_driver_names())
        raise KeyError(f"Unknown driver: {name}. Valid: {valid}")
    return DRIVER_REGISTRY[key]
