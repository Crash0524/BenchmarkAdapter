from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

from .adapters.base import BenchmarkAdapter
from .base import BenchmarkResult, Trajectory
from .drivers.protocols import EnvironmentDriver
from .runtimes.base import BaseRuntime
from .runtimes.protocols import RuntimeResponse, RuntimeSolveResult


class BenchmarkRunner:
    """Benchmark-agnostic orchestration runner."""

    def __init__(
        self,
        adapter: BenchmarkAdapter,
        runtime: BaseRuntime,
        driver: EnvironmentDriver,
    ) -> None:
        self.adapter = adapter
        self.runtime = runtime
        self.driver = driver

    def run(self, limit: int | None = None) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        self.adapter.before_run()
        for i, instance in enumerate(self.adapter.iter_instances()):
            if limit is not None and i >= limit:
                break
            env = self.driver.setup(self.adapter, instance)
            try:
                context = self.adapter.build_context(instance, env)
                raw_runtime_result = self.runtime.solve(instance, context)
                runtime_response = self._normalize_runtime_result(raw_runtime_result)
                result = self.adapter.judge(instance, runtime_response.trajectory)
                if runtime_response.prediction:
                    result.metadata.setdefault("prediction", runtime_response.prediction)
                if runtime_response.metadata:
                    result.metadata.setdefault("runtime", runtime_response.metadata)
            finally:
                self.driver.teardown(self.adapter, env)
            results.append(result)
        self.adapter.after_run()
        return results

    @staticmethod
    def _normalize_runtime_result(runtime_result: RuntimeSolveResult) -> RuntimeResponse:
        if isinstance(runtime_result, RuntimeResponse):
            if not runtime_result.trajectory.final_output and runtime_result.prediction:
                runtime_result.trajectory.final_output = runtime_result.prediction
            return runtime_result

        prediction = str(runtime_result)
        return RuntimeResponse(prediction=prediction, trajectory=Trajectory(final_output=prediction))

    @staticmethod
    def save_results(results: list[BenchmarkResult], output_file: Path) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(r) for r in results]
        output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
