from __future__ import annotations

from typing import Any

from BenchmarkAdapter.adapters.base import BenchmarkAdapter
from BenchmarkAdapter.base import BenchmarkInstance


class AdapterNativeDriver:
    """Default driver: delegate environment lifecycle to adapter methods."""

    def setup(self, adapter: BenchmarkAdapter, instance: BenchmarkInstance) -> Any:
        return adapter.build_environment(instance)

    def teardown(self, adapter: BenchmarkAdapter, env: Any) -> None:
        adapter.teardown_environment(env)
