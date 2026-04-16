from __future__ import annotations

from BenchmarkAdapter.agents.protocols import AgentResponse
from BenchmarkAdapter.base import BenchmarkInstance, Trajectory


class EchoAgent:
    """Minimal agent used for smoke tests and integration scaffolding."""

    def solve(self, instance: BenchmarkInstance, context: dict) -> AgentResponse:
        output = f"Echo solve for: {instance.task[:120]}"
        traj = Trajectory(
            steps=[
                {
                    "type": "thought",
                    "content": "This is a scaffold run. Replace with real agent API calls.",
                },
                {"type": "action", "content": output},
            ],
            final_output=output,
            metadata={"agent": "EchoAgent"},
        )
        return AgentResponse(prediction=output, trajectory=traj)
