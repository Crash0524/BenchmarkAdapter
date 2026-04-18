from __future__ import annotations

import json
import logging
import importlib
import time
from logging import FileHandler
from pathlib import Path
from typing import Any, Mapping

from BenchmarkAdapter.drivers.registry import get_driver_cls
from BenchmarkAdapter.registry import get_adapter_cls
from BenchmarkAdapter.adapters.base import RunObject
from benchmark.utils.utils import build_task_selector, adapter_selector
from .induce_memory import induce_memory
from .memory_management import select_memory
from .utils.evaluate_trajectory import evaluate_trajectory
from agent.agent import ChatModelArgs


def _redact_config(config: Mapping[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in config.items():
        lower = key.lower()
        if any(token in lower for token in ("key", "token", "secret", "password")):
            redacted[key] = "***"
        else:
            redacted[key] = value
    return redacted


def build_method_config(method_cfg: Mapping[str, Any], api_name: str, api_cfg: Mapping[str, Any]) -> dict[str, Any]:
    runtime_cfg = method_cfg.get("runtime_config", {})
    if not isinstance(runtime_cfg, dict):
        raise ValueError("method_config.runtime_config must be a JSON object")

    config_cfg = runtime_cfg.get("config", {})
    if not isinstance(config_cfg, dict):
        config_cfg = {}

    merged = dict(config_cfg)
    merged.update(dict(api_cfg))
    merged["model_name"] = api_name
    return merged


def setup_task_logging(log_file: Path) -> tuple[logging.Logger, logging.Handler]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = FileHandler(log_file, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))

    task_logger = logging.getLogger(f"methods.reasoning_bank.{log_file.parent.name}")
    task_logger.setLevel(logging.DEBUG)
    task_logger.addHandler(handler)
    task_logger.propagate = True
    return task_logger, handler



def _prepare_memory_for_run(
    run_object: RunObject,
    method_cfg: Mapping[str, Any],
    output_dir: Path,
    logger: logging.Logger,
) -> Path:
    config_cfg = method_cfg.get("config")

    memory_path_raw = config_cfg.get("memory_path", "reasoning_bank")
    memory_dir = Path(output_dir) / str(memory_path_raw)
    memory_path = memory_dir / "memory.txt"
    memory_prefix = memory_dir.name or "default"

    memory_dir.mkdir(parents=True, exist_ok=True)
    logger.info("MEMORY_PREP start task=%s memory_path=%s memory_dir=%s", run_object.name, memory_path, memory_dir)

    reasoning_bank_path = memory_dir / f"{memory_prefix}.jsonl"
    if not reasoning_bank_path.exists():
        reasoning_bank_path.write_text("", encoding="utf-8")

    reasoning_bank: list[dict[str, Any]] = []
    for line in reasoning_bank_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                reasoning_bank.append(data)
        except json.JSONDecodeError:
            continue
    logger.info("MEMORY_PREP bank_loaded task=%s records=%d source=%s", run_object.name, len(reasoning_bank), reasoning_bank_path)

    task_id = run_object.name.split(".")[-1]
    if not isinstance(run_object.instances, Mapping):
        raise TypeError("RunObject.instances must be a mapping task payload")
    cur_query = str(run_object.task_description or run_object.name)

    logger.debug("MEMORY_PREP cur_query task=%s query=%s", run_object.name, cur_query)

    cache_path = str(memory_dir / f"{memory_prefix}_embeddings.jsonl")
    prefer_model = str(config_cfg.get("embedding_model", method_cfg.get("embedding_model", "")))

    res = select_memory(
        n=1,
        reasoning_bank=reasoning_bank,
        cur_query=cur_query,
        task_id=task_id,
        cache_path=cache_path,
        prefer_model=prefer_model,
    )
    logger.info("MEMORY_PREP retrieved task=%s selected=%d embedding_model=%s", run_object.name, len(res), prefer_model or "default")

    if not res:
        memory_path.write_text("", encoding="utf-8")
        logger.info("MEMORY_PREP empty task=%s output=%s", run_object.name, memory_path)
        return memory_path

    mem_items: list[str] = []
    for item in res:
        if isinstance(item, Mapping) and "memory_items" in item:
            mem_items.append(str(item["memory_items"]))

    memory_path.write_text("\n\n".join(mem_items) + ("\n" if mem_items else ""), encoding="utf-8")
    logger.info("MEMORY_PREP done task=%s memory_items=%d output=%s", run_object.name, len(mem_items), memory_path)
    return memory_path


def run(
    task_cfg: Mapping[str, Any],
    method_cfg: Mapping[str, Any],
    api_name: str,
    api_cfg: Mapping[str, Any],
    output_dir: Path,
    logger: logging.Logger | None = None,
) -> None:
    logger = logger or logging.getLogger("methods.reasoning_bank.method_main")

    # get task data
    benchmark = str(task_cfg.get("benchmark", "webarena"))
    selector = build_task_selector(benchmark, task_cfg)
    run_objects = selector.get_run_objects()

    logger.info(
        "Method start benchmark=%s api=%s selected_instances=%d",
        benchmark,
        api_name,
        len(run_objects),
    )

    method_config = build_method_config(method_cfg, api_name, api_cfg)
    logger.info("Method config merged keys=%s", sorted(method_config.keys()))
    logger.debug("Method config merged values=%s", _redact_config(method_config))
    failures: list[str] = []
    total_start = time.perf_counter()


    for run_object in run_objects:
        name = run_object.name
        stem = run_object.output_stem or name
        task_out_dir = output_dir / stem.replace("/", "_").replace("\\", "_").replace(" ", "_")
        task_log_file = task_out_dir / "run.log"
        task_logger, task_handler = setup_task_logging(task_log_file)
        task_start = time.perf_counter()
        task_logger.info("RUN_START instance=%s output_dir=%s", name, task_out_dir)

        try:
            # memory
            memory_path = _prepare_memory_for_run(run_object, method_cfg, output_dir, task_logger)
            
            # prepare agent config
            chat_model_args = ChatModelArgs(
                model_name=str(method_config.get("model_name", api_name)),
                temperature=method_config.get("temperature", 0.7),
                max_total_tokens=method_config.get("max_total_tokens", 128000),
                max_input_tokens=method_config.get("max_input_tokens", 126000),
                max_new_tokens=method_config.get("max_new_tokens", 65536),
            )

            # prepare benchmark env
            adapter = adapter_selector(
                benchmark=benchmark,
                task_cfg=task_cfg,
                )
            exp_dir = adapter.run(
                run_object=run_object,
                chat_model_args=chat_model_args,
                memory_path=memory_path,
                output_dir=task_out_dir,
            )
            elapsed = time.perf_counter() - task_start
            task_logger.info("RUN_OK instance=%s exp_dir=%s elapsed_sec=%.2f", name, exp_dir, elapsed)


            eval_output = evaluate_trajectory(
                method_cfg=method_cfg,
                task_cfg=task_cfg,
                result_dir=task_out_dir,
                default_model=str(method_config.get("model_name", api_name)),
            )
            task_logger.info("EVAL_OK instance=%s output=%s", name, eval_output)

            induce_cfg = method_cfg.get("induce_memory")
            if isinstance(induce_cfg, Mapping):
                memory_output = induce_memory(
                    method_cfg=method_cfg,
                    task_cfg=task_cfg,
                    result_dir=task_out_dir,
                    task_name=name,
                    default_model=str(method_config.get("model_name", api_name)),
                    output_dir=output_dir,
                )
                task_logger.info("INDUCE_MEMORY_OK instance=%s output=%s", name, memory_output)



        except Exception:
            failures.append(name)
            task_logger.exception("RUN_FAIL instance=%s", name)
        finally:
            task_logger.removeHandler(task_handler)
            task_handler.close()

    time.sleep(10)

    total_elapsed = time.perf_counter() - total_start
    success_count = len(run_objects) - len(failures)
    logger.info(
        "Method summary total=%d success=%d failed=%d elapsed_sec=%.2f",
        len(run_objects),
        success_count,
        len(failures),
        total_elapsed,
    )

    if failures:
        logger.error("Method finished with failures:")
        for item in failures:
            logger.error("- %s", item)
        raise SystemExit(1)

    logger.info("Method finished successfully")
