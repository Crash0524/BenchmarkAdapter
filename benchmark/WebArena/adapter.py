from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from browsergym.experiments import ExpArgs, EnvArgs

from BenchmarkAdapter.adapters.base import BenchmarkAdapter, BenchmarkTaskSelector, RunObject
from BenchmarkAdapter.base import  BenchmarkInstance, BenchmarkResult, Trajectory
from .observation import Flags
from .agent import WebArenaAgentArgs


class WebArenaAdapter(BenchmarkAdapter):
    """Example adapter for WebArena-style tasks."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)

        env_kwargs = config.get("env_kwargs_json")
        assert env_kwargs is None or isinstance(env_kwargs, Mapping)
        self.env_kwargs_json = dict(env_kwargs)

    def load_instances(self) -> Iterable[Mapping[str, Any]]:
        instances = self.config.get("instances", [])
        if isinstance(instances, list):
            return instances
        if isinstance(instances, Mapping):
            return [instances]
        return []

    def normalize_instance(self, instance: Mapping[str, Any]) -> BenchmarkInstance:
        task_id = instance.get("task_id", instance.get("id", instance.get("instance_id", "unknown")))
        instance_id = str(instance.get("instance_id", task_id))
        task = str(
            instance.get(
                "intent",
                instance.get("goal", instance.get("task", instance.get("query", ""))),
            )
        )
        browsergym_task = (
            instance.get("browsergym_task")
            or instance.get("task_name")
            or (f"webarena.{task_id}" if str(task_id).isdigit() else None)
        )
        metadata = {
            "browsergym_task": browsergym_task,
            "task_id": task_id,
            "start_url": instance.get("start_url"),
            "eval": instance.get("eval", {}),
        }
        return BenchmarkInstance(instance_id=instance_id, task=task, raw=instance, metadata=metadata)

    def build_environment(self, instance: BenchmarkInstance) -> dict[str, Any]:
        return {
            "task_name": instance.metadata.get("browsergym_task"),
            "instance_id": instance.instance_id,
            "start_url": instance.metadata.get("start_url"),
        }

    def judge(self, instance: BenchmarkInstance, trajectory: Trajectory) -> BenchmarkResult:
        text = (trajectory.final_output or "").lower()
        ok = "answer(" in text or "click(" in text or "type(" in text or "goto(" in text
        return BenchmarkResult(
            instance_id=instance.instance_id,
            success=ok,
            score=1.0 if ok else 0.0,
            message="judged by action-string heuristic",
            metadata={"judge_meta": trajectory.metadata or {}},
        )

    def run(
        self,
        run_object: RunObject,
        chat_model_args: Any | None = None,
        memory_path: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> Path:
        """Create EnvArgs/ExpArgs and execute one BrowserGym experiment run."""

        if not isinstance(run_object.instances, Mapping):
            raise TypeError("RunObject.instances must be a mapping task payload")
        first_instance = run_object.instances

        task_id = run_object.name.split(".")[-1]
        task_name = run_object.name
        if not task_name:
            raise ValueError(f"Cannot resolve BrowserGym task name for run object: {run_object.name}")

        viewport = self.env_kwargs_json.get("viewport")
        if viewport is not None and not isinstance(viewport, Mapping):
            raise TypeError("env_kwargs_json.viewport must be a mapping when provided")

        env_args = EnvArgs(
            task_name=str(task_name),
            task_seed=None,
            max_steps=int(self.env_kwargs_json.get("max_steps", 30)),
            headless=bool(self.env_kwargs_json.get("headless", True)),
            viewport=dict(viewport),
            slow_mo=int(self.env_kwargs_json.get("slow_mo", 0)),
        )

        flags_cfg = self.config.get("flags", {})

        exp_args = ExpArgs(
            env_args=env_args,
            agent_args=WebArenaAgentArgs(
                chat_model_args=chat_model_args,
                flags=Flags(
                    use_html=flags_cfg.get("use_html", True),
                    use_ax_tree=flags_cfg.get("use_ax_tree"),
                    use_thinking=flags_cfg.get("use_thinking"),  # "Enable the agent with a memory (scratchpad)."
                    use_error_logs=flags_cfg.get("use_error_logs"),  # "Prompt the agent with the error logs."
                    use_memory=flags_cfg.get("use_memory"),  # "Enables the agent with a memory (scratchpad)."
                    use_history=flags_cfg.get("use_history"),
                    use_diff=flags_cfg.get("use_diff", False),  # "Prompt the agent with the difference between the current and past observation."
                    use_past_error_logs=flags_cfg.get("use_past_error_logs", True),  # "Prompt the agent with the past error logs."
                    use_action_history=flags_cfg.get("use_action_history", True),  # "Prompt the agent with the action history."
                    multi_actions=flags_cfg.get("multi_actions", False),
                    use_abstract_example=flags_cfg.get("use_abstract_example", True),  # "Prompt the agent with an abstract example."
                    use_concrete_example=flags_cfg.get("use_concrete_example", True),  # "Prompt the agent with a concrete example."
                    use_screenshot=flags_cfg.get("use_screenshot", True),
                    enable_chat=True,
                    demo_mode="default" if flags_cfg.get("demo_mode", False) else "off",
                    memory_path=memory_path,
                    ),
            ),
        )
        out_dir = Path(output_dir) if output_dir is not None else self.default_output_dir

        exp_args.prepare(out_dir)
        exp_args.run()
        exp_dir = Path(exp_args.exp_dir)
        self._flatten_experiment_output(task_dir=out_dir, exp_dir=exp_dir)
        return out_dir

    @staticmethod
    def _flatten_experiment_output(task_dir: Path, exp_dir: Path) -> None:
        """Merge logs and move artifacts from timestamped exp_dir into task_dir."""

        if exp_dir.resolve() == task_dir.resolve():
            return

        run_log = task_dir / "run.log"
        exp_log = exp_dir / "experiment.log"
        if exp_log.exists():
            with run_log.open("a", encoding="utf-8") as dst:
                dst.write("\n\n===== BrowserGym experiment.log =====\n")
                dst.write(exp_log.read_text(encoding="utf-8"))

        for child in exp_dir.iterdir():
            if child.name == "experiment.log":
                continue
            dst = task_dir / child.name
            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            shutil.move(str(child), str(dst))

        if exp_dir.exists():
            shutil.rmtree(exp_dir)




class WebArenaTaskSelector(BenchmarkTaskSelector):
    @staticmethod
    def _should_keep(config: dict, website: str) -> bool:
        sites = config.get("sites", [])
        if not isinstance(sites, list) or not sites:
            return False
        if website == "multi":
            return len(sites) != 1 and "map" not in sites
        return sites[0] == website

    def __init__(self, task_cfg: Mapping[str, Any]) -> None:
        super().__init__(task_cfg)

        self.website = str(task_cfg.get("website", "shopping"))
        self.start_index = int(task_cfg.get("start_index", 0))
        end_index = task_cfg.get("end_index")
        self.end_index = int(end_index) if end_index is not None else None
        self.prev_id = int(task_cfg.get("prev_id", -1))
        self.config_dir = Path(str(task_cfg.get("config_dir", "config_files")))

    def get_run_objects(self) -> list[RunObject]:
        config_files = sorted(
            [p for p in self.config_dir.glob("*.json") if p.stem.isdigit()],
            key=lambda p: int(p.stem),
        )

        selected_items: list[tuple[int, dict[str, Any]]] = []
        for path in config_files:
            cfg = json.loads(path.read_text(encoding="utf-8"))
            task_id = int(path.stem)
            if task_id <= self.prev_id:
                continue
            if self._should_keep(cfg, self.website):
                selected_items.append((task_id, cfg))

        slice_end = self.end_index if self.end_index is not None else len(selected_items)
        selected = selected_items[self.start_index:slice_end]
        return [
            RunObject(
                name=f"webarena.{tid}",
                cli_args={"instances_json": str(self.config_dir / f"{tid}.json")},
                instances=cfg,
                task_description=str(cfg.get("intent")),
                output_stem=f"webarena.{tid}",
                tags={"website": self.website},
            )
            for tid, cfg in selected
        ]
