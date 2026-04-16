from __future__ import annotations

from typing import Any

from BenchmarkAdapter.adapters.base import BenchmarkAdapter
from BenchmarkAdapter.base import BenchmarkInstance


class BrowserGymDriver:
    """Environment driver for BrowserGym tasks via gymnasium.

    Expected configuration via ``adapter.config.extra``:
    - ``browsergym_task``: str, required if not present in instance
    - ``browsergym_env_kwargs``: dict, optional kwargs passed to ``gym.make``
    - ``seed``: int, optional reset seed
    """

    def setup(self, adapter: BenchmarkAdapter, instance: BenchmarkInstance) -> Any:
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise ImportError(
                "BrowserGymDriver requires gymnasium. Install it before using this driver."
            ) from exc

        extra = adapter.config.extra or {}
        task_name = (
            instance.metadata.get("browsergym_task")
            or instance.raw.get("browsergym_task")
            or extra.get("browsergym_task")
            or extra.get("task_name")
        )
        if not task_name:
            raise ValueError(
                "BrowserGymDriver requires task name in one of: "
                "instance.metadata['browsergym_task'], instance.raw['browsergym_task'], "
                "config.extra['browsergym_task'], or config.extra['task_name']."
            )

        if str(task_name).startswith("webarena"):
            try:
                import browsergym  # noqa: F401
            except ImportError:
                # Some deployments auto-register BrowserGym envs through sitecustomize or startup code.
                pass

        env_kwargs = dict(extra.get("browsergym_env_kwargs", {}))
        instance_env_kwargs = instance.raw.get("browsergym_env_kwargs")
        if isinstance(instance_env_kwargs, dict):
            env_kwargs.update(instance_env_kwargs)
        env = gym.make(task_name, **env_kwargs)

        seed = extra.get("seed", adapter.config.seed)
        try:
            env.reset(seed=seed)
        except TypeError:
            # Some environments may not accept a seed argument.
            env.reset()

        return env

    def teardown(self, adapter: BenchmarkAdapter, env: Any) -> None:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
