#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from importlib import import_module

import json


def setup_logging(task_cfg: dict) -> logging.Logger:
    cfg = task_cfg.get("logging", {})
    if not isinstance(cfg, dict):
        cfg = {}

    level_name = str(cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_file = cfg.get("file")
    if log_file:
        log_path = Path(str(log_file))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("pipeline")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WebArena with task/method/api config files")
    parser.add_argument("--task_name", default="webarena", help="The Benchmark to run")
    parser.add_argument("--method_name", default="reasoning_bank", help="Runtime/method name")
    parser.add_argument("--api_name", default="qwen3.5-flash", help="API profile/model name")
    parser.add_argument("--output_dir", default="outputs/", help="All output directory for results")
    return parser.parse_args()


def load_json(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a JSON object: {path}")
    return data


def main() -> None:
    args = parse_args()

    # load configs
    task_cfg = load_json(f"run_configs/tasks/{args.task_name}.json")
    method_cfg = load_json(f"run_configs/methods/{args.method_name}.json")
    api_cfg = load_json(f"run_configs/apis/{args.api_name}.json")
    logger = setup_logging(task_cfg)
    logger.info(
        "Pipeline start task=%s method=%s api=%s",
        args.task_name,
        args.method_name,
        args.api_name,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory ready path=%s", output_dir.resolve())

    method_module = import_module(f"methods.{args.method_name}.method_main")
    if not hasattr(method_module, "run"):
        raise AttributeError(f"methods.{args.method_name}.method_main must define run(...) ")

    logger.info("Dispatch method=%s api=%s output_dir=%s", args.method_name, args.api_name, output_dir)
    method_module.run(task_cfg, method_cfg, args.api_name, api_cfg, output_dir=output_dir, logger=logger)




if __name__ == "__main__":
    main()
