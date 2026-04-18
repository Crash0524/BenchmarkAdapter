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
import random
from pathlib import Path
from typing import Any, Mapping

from .prompts.memory_instruction import (
    SUCCESSFUL_SI,
    FAILED_SI,
    AWM_INSTRUCTION,
    AWM_EXAMPLE,
)
from methods.reasoning_bank.utils.clients import CLIENT_DICT


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
    assert len(blocks) % 2 == 0
    return blocks


def remove_invalid_steps(actions: list[str]) -> list[str]:
    """Remove invalid steps from the action sequence."""
    valid_actions = []
    for a in actions:
        if "click(" in a:
            arg = a[a.index("(")+1: a.index(")")]
            try:
                if type(eval(arg)) == str and type(eval(arg[1:-1])) == int:
                    valid_actions.append(a)
            except:
                continue
        elif "fill(" in a:
            arg = a[a.index("(")+1: a.index(",")].strip()
            if type(eval(arg)) == str:
                valid_actions.append(a)
        elif "scroll(" in a or "noop(" in a:
            continue
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

def format_trajectory(think_list: list[str], action_list: list[str]) -> str:
    trajectory = []
    for t, a in zip(think_list, action_list):
        # acts = '\n'.join(a)
        acts = a
        trajectory.append(f"<think>\n{t}\n</think>\n<action>\n{acts}\n</action>")
    return '\n\n'.join(trajectory)

def random_group_sample(d: dict, n) -> list:
    """Randomly sample n groups from the dictionary."""
    return [ex for v in d.values() for ex in random.sample(v, min(n, len(v)))]


def format_examples(examples: list[dict], flag=False) -> str:
    """Format examples to the prompt."""
    formatted_examples = []
    for ex in examples:
        trajectory = format_trajectory(ex["think_list"], ex["action_list"])
        formatted_examples.append(f"Query: {ex['query']}\nTrajectory:\n{trajectory}")
    # return '\n\n'.join(["## Concrete Examples"] + formatted_examples + ["## Summary Workflow"])
    if flag:
        return '\n\n'.join(["## Query and Trajectory Generated Using Previous Memory"] + formatted_examples + ["## Correctness Signal"]+ ["The result is CORRECT."] + ["## Updated Memory"])
    else:
        return '\n\n'.join(["## Query and Trajectory Generated Using Previous Memory"] + formatted_examples + ["## Correctness Signal"]+ ["The result is INCORRECT."] + ["## Updated Memory"])


def get_info(result_dir: str | Path, status: str, data_dir: str | Path) -> dict[str, Any]:
    result_dir = str(result_dir)
        
    # get query -> task objective
    task_id = result_dir.split('/')[-1].split("_")[0].split(".")[1]
    config_path = os.path.join(str(data_dir), f"{task_id}.json")
    config = json.load(open(config_path))
    query = config["intent"]

    template_id = config["intent_template_id"]  # for deduplication

    # parse trajectory
    log_path = os.path.join(result_dir, "run.log")
    if not os.path.exists(log_path):
        log_path = os.path.join(result_dir, "experiment.log")
    think_list, action_list = extract_think_and_action(log_path)

    # add to template dict
    if status == 'success':
        wdict = {"query": query, "template_id": template_id, "think_list": think_list, "action_list": action_list, "status": "success"}
    elif status == 'fail':
        wdict = {"query": query, "template_id": template_id, "think_list": think_list, "action_list": action_list, "status": "fail"}

    return wdict


def induce_memory(
    method_cfg: Mapping[str, Any],
    task_cfg: Mapping[str, Any],
    result_dir: str | Path,
    task_name: str,
    default_model: str,
    output_dir: str | Path | None = None,
) -> Path:
    induce_cfg = method_cfg.get("induce_memory", {})
    if not isinstance(induce_cfg, Mapping):
        induce_cfg = {}

    criteria = str(induce_cfg.get("criteria", "autoeval"))
    model = str(induce_cfg.get("model") or default_model)
    memory_mode = str(induce_cfg.get("memory_mode", "reasoningbank"))

    result_dir_path = Path(result_dir)
    cur_task = result_dir_path

    # correctness signals for trajectories
    if criteria == "gt":
        reward = json.load(open(os.path.join(cur_task, "summary_info.json")))["cum_reward"]
    elif criteria == "autoeval":
        reward = json.load(open(os.path.join(cur_task, f"{model}_autoeval.json")))[0]["rm"]
    else:
        raise ValueError(f"Invalid criteria: {criteria}.")

    status = "success" if reward > 0 else "fail"

    ex = get_info(result_dir=result_dir_path, status=status, data_dir=task_cfg.get("config_dir", "config_files"))

    # Define the LLM client based on the model choice
    llm_client = CLIENT_DICT[model](model_name=model)

    # memory extraction based on the trajectory and user queries
    trajectory = format_trajectory(ex["think_list"], ex["action_list"])
    trajectory = f"**Query:** {ex['query']}\n\n**Trajectory:**\n{trajectory}"

    generated_memory_item = ""
    if memory_mode == "reasoningbank":
        if ex['status'] == 'success':
            generated_memory_item, _ = llm_client.one_step_chat(trajectory, system_msg=SUCCESSFUL_SI, temperature=0.7)
        else:
            generated_memory_item, _ = llm_client.one_step_chat(trajectory, system_msg=FAILED_SI, temperature=0.7)
    
    elif memory_mode == "awm":
        if ex['status'] == 'success':
            generated_memory_item, _ = llm_client.one_step_chat(trajectory, system_msg=AWM_INSTRUCTION + AWM_EXAMPLE, temperature=0.7)

    elif memory_mode == "synapse":
        if ex['status'] == 'success':
            generated_memory_item = trajectory

    method_config = method_cfg.get("config", {})
    if not isinstance(method_config, Mapping):
        method_config = {}

    output_path_raw = induce_cfg.get("output_path")
    if output_path_raw:
        output_path = Path(str(output_path_raw))
    else:
        base_output = Path(output_dir) if output_dir is not None else result_dir_path.parent
        memory_prefix = str(method_config.get("memory_path", "reasoning_bank"))
        memory_dir = base_output / memory_prefix
        output_path = memory_dir / f"{memory_prefix}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # write memory to jsonl file 
    with open(output_path, 'a') as f:
        f.write(json.dumps({
            "task_id": task_name.split(".")[-1],
            "query": ex["query"],
            "think_list": ex["think_list"],
            "action_list": ex["action_list"],
            "status": ex["status"],
            "memory_items": generated_memory_item,
            "template_id": ex["template_id"]
        }) + '\n')
    return output_path