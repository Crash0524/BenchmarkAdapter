# Copyright 2026 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import traceback
from typing import Any, Mapping
from pathlib import Path
from .evaluator import Evaluator
from .clients import CLIENT_DICT


def load_blocks(path: str) -> list[list[str]]:
    """Load blank-line separated blocks from the log file."""
    blocks, block = [], []
    for line in open(path, 'r'):
        if line.strip() == "":
            blocks.append(block)
            block = []
        else:
            if line.strip():
                block.append(line.strip())
    print(blocks)
    assert len(blocks) % 2 == 0
    return blocks

def remove_invalid_steps(actions: list[str]) -> list[str]:
    """Remove invalid steps from the action sequence."""
    valid_actions = []
    for a in actions:
        if "click(" in a:
            arg = a[a.index("(")+1: a.index(")")]
            if type(eval(arg)) == str:
                valid_actions.append(a)
        elif "fill(" in a:
            arg = a[a.index("(")+1: a.index(",")].strip()
            if type(eval(arg)) == str:
                valid_actions.append(a)
        else:
            valid_actions.append(a)
    return valid_actions


def extract_think_and_action(path: str) -> tuple[list[str], list[str]]:
    """Extract the task trajectory from the log file."""
    log_text = open(path, 'r').read()
    lines = log_text.splitlines()
    think_list = []
    action_list = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("action:"):
            # Parse the full action block (can span multiple lines)
            action_lines = []
            if line.strip() != "action:":
                action_lines.append(line[len("action:"):].strip())
            i += 1
            while i < len(lines) and lines[i].strip() != "":
                action_lines.append(lines[i].strip())
                i += 1
            action_text = "".join(action_lines).strip()

            # Now look backward for the most recent loop-INFO thinking block
            thinking_lines = []
            for j in range(i - 1, -1, -1):
                if "browsergym.experiments.loop - INFO -" in lines[j]:
                    thinking = lines[j].split("browsergym.experiments.loop - INFO -", 1)[-1].strip()
                    thinking_lines.insert(0, thinking)
                    break
            thinking_text = "\n".join(thinking_lines).strip()
            think_list.append(thinking_text)
            action_list.append(action_text)
        else:
            i += 1

    assert len(think_list) == len(action_list)
    return think_list, action_list

def extract_response(action: str) -> str:
    s, e = action.index("(")+1, action.index(")")
    return action[s: e]


def process_sample(
    idx: str, traj_info: dict, log_save_path,
    model: str, eval_version: str,
) -> list[dict]:
    clients = {model: CLIENT_DICT[model](model_name=model)}
    evaluator = Evaluator(clients, log_save_path=log_save_path + "/trajs")
    try:
        out, _ = evaluator(traj_info, model, eval_version)
        eval_result = None
        if out["status"].lower() == "success": eval_result = True
        else: eval_result = False
        return [{
                "idx": idx,
                "gt": traj_info["eval"],
                "rm": eval_result,
                "thoughts": out["thoughts"], 
                "uid": traj_info["traj_name"],
        }]
    except Exception as e:
        print(f"Error on {idx}, {e}")
        print(traceback.format_exc())
        return [{
            "idx": idx,
            "gt": traj_info["eval"],
            "rm": None,
            "thoughts": None, 
            "uid": traj_info["traj_name"],
        }]


def evaluate_trajectory(
    method_cfg: Mapping[str, Any],
    result_dir: str | Path,
    default_model: str,
) -> Path:
    assert "evaluate" in method_cfg
    evaluate_cfg = method_cfg.get("evaluate")
    if not isinstance(evaluate_cfg, Mapping):
        evaluate_cfg = {}

    if "model" not in evaluate_cfg:
        evaluate_cfg["model"] = default_model

    result_dir_path = Path(result_dir)
    task_name = str(result_dir_path).split("/")[-1]
    task_id = task_name.split(".")[-1]

    model = str(evaluate_cfg.get("model"))
    prompt = str(evaluate_cfg.get("prompt", "text"))
    if prompt not in {"text", "vision"}:
        raise ValueError("evaluate.prompt must be 'text' or 'vision'")
    if model == "gpt-4o" and prompt != "vision":
        print(f"Warning: use vision prompt by default for {model}.")
        prompt = "vision"

    log_dir = Path(str(evaluate_cfg.get("log_dir", "outputs/evaluate_log")))

    config_path = Path("config_files") / f"{task_id}.json"
    config = json.load(open(config_path))

    # load trajectory log (prefer merged run.log, fallback to experiment.log)
    log_path = result_dir_path / "run.log"
    if not log_path.exists():
        log_path = result_dir_path / "experiment.log"
    think_list, action_list = extract_think_and_action(str(log_path))
    # actions = [act for acts in action_list for act in acts]
    actions = [act for act in action_list]
    if action_list and "send_msg_to_user" in action_list[-1]:
        response = extract_response(action_list[-1])
    else:
        response = ""
    
    # load summary info
    summary_path = result_dir_path / "summary_info.json"
    summary = json.load(open(summary_path, 'r'))

    # collect traj info
    image_paths = [
        os.path.join(result_dir_path, f) for f in os.listdir(result_dir_path)
        if f.startswith("screenshot_step_") and f.endswith(".jpg")
    ]
    image_paths = sorted(image_paths, key=lambda x: int(x.split('/')[-1].split("_")[-1].split(".")[0]))
    traj_info = {
        "intent": config["intent"],
        "response": response,
        "captions": think_list,
        "actions": actions,
        "traj_name": config["task_id"],
        "image_paths": image_paths,
        "images": image_paths,
        "eval": summary["cum_reward"]
    }

    # evaluate trajectory
    log_save_path = str(log_dir / task_name)
    print("Log Save Path:", log_save_path)
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)
        os.makedirs(log_save_path + "/trajs")
    eval_info = process_sample(
        idx=config["task_id"], traj_info=traj_info,
        log_save_path=log_save_path, 
        model=model, eval_version=prompt,
    )
    output_eval_path = result_dir_path / f"{model}_autoeval.json"
    json.dump(eval_info, open(output_eval_path, 'w'))
    return output_eval_path
