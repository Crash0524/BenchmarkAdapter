from __future__ import annotations

from typing import Any, Protocol

from ..adapters.base import BenchmarkAdapter
from ..base import BenchmarkInstance


class EnvironmentDriver(Protocol):
    """Driver protocol for provisioning and tearing down benchmark environments."""

    def setup(self, adapter: BenchmarkAdapter, instance: BenchmarkInstance) -> Any:
        """Create/attach environment for one instance."""

    def teardown(self, adapter: BenchmarkAdapter, env: Any) -> None:
        """Cleanup the environment for one instance."""
