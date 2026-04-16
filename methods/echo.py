from __future__ import annotations

from BenchmarkAdapter.base import BenchmarkInstance, Trajectory
from BenchmarkAdapter.runtimes.base import BaseRuntime
from BenchmarkAdapter.runtimes.protocols import RuntimeResponse


class EchoRuntime(BaseRuntime):
    """Minimal runtime for smoke tests and scaffolding."""

    def solve(self, instance: BenchmarkInstance, context: dict) -> RuntimeResponse:
        output = f"Echo runtime solve for: {instance.task[:120]}"
        traj = Trajectory(
            steps=[
                {"type": "thought", "content": "Runtime scaffold execution."},
                {"type": "action", "content": output},
            ],
            final_output=output,
            metadata={"runtime": "EchoRuntime"},
        )
        return RuntimeResponse(prediction=output, trajectory=traj)
