# BenchmarkAdapter

BenchmarkAdapter is a lightweight adapter layer for running web agents across benchmarks, methods, and API backends with a single execution entrypoint.

It is designed around three ideas:

- benchmark-specific logic lives in `benchmark/`
- method-specific logic lives in `methods/`
- shared execution and dispatch logic lives in `BenchmarkAdapter/` and `run_pipeline.py`

The current repository focus is a WebArena benchmark integration with the `reasoning_bank` method.

## Overview

The execution flow is:

1. `run_pipeline.py` loads the task, method, and API configuration files.
2. The selected method module is imported dynamically.
3. The method prepares per-task memory, model settings, and output directories.
4. The benchmark adapter runs the task in the environment.
5. The trajectory is evaluated.
6. New memory items are induced from the completed trajectory and appended to the memory store.

This repository is structured so that you can add new benchmarks or methods without rewriting the whole pipeline.

## Repository Layout

- `BenchmarkAdapter/`
  - core adapter contracts
  - registry and dispatch helpers
  - runtime/driver abstractions
- `benchmark/`
  - benchmark-specific adapters
  - WebArena prompt and agent integration
- `methods/`
  - method-specific orchestration logic
  - memory preparation, evaluation, and memory induction
- `run_configs/`
  - task, method, and API JSON configuration files
- `run_pipeline.py`
  - main entrypoint for running one benchmark/method/API combination
- `run_pipeline.sh`
  - convenience wrapper for the WebArena + reasoning_bank default setup

## Current Support

### Benchmarks
- WebArena
- More benchmarks and datasets are still being adapted.

### Methods
- reasoning_bank

### API Backends
- OpenAI-compatible Qwen endpoints
- Gemini
- Claude
- local vLLM-style endpoints

## Installation

### 1. Create and activate an environment

Use your preferred Python environment manager. Python 3.10+ is recommended.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If you are using the included Qwen setup, make sure the required API key is available in the environment.

### 3. Set environment variables

At minimum, the Qwen/OpenAI-compatible path expects:

```bash
dashexport DASHSCOPE_API_KEY="your-api-key"
```

Depending on the backend you use, you may also need:

```bash
export OPENAI_API_KEY="your-openai-key"
export OPENAI_API_BASE="https://your-compatible-endpoint/v1"
export GOOGLE_CLOUD_PROJECT="your-project"
export GOOGLE_CLOUD_LOCATION="your-region"
```

For WebArena itself, the browser environment URLs are typically exported by `run_pipeline.sh`.

## WebArena Setup

This repository assumes that WebArena is configured according to the official BrowserGym and WebArena documentation.

### 1. Install BrowserGym

Please follow the official BrowserGym installation guide first. The WebArena integration in this repository expects BrowserGym to be installed and working before you continue.

### 2. Configure the WebArena Docker environment

After BrowserGym is available, download and configure the WebArena Docker environment.

Follow the official WebArena tutorial and execute the setup scripts in numerical order. Before running any script, make sure the website URLs inside the corresponding script files are updated to [...]

The default WebArena environment URLs are usually exposed through `run_pipeline.sh`, but your local Docker services must also be reachable.

### 3. WebArena directory structure

The WebArena-related files are organized as follows:

- `WebArena/agents/`: web agent implementations that integrate with BrowserGym
- `WebArena/autoeval/`: LLM-as-a-judge evaluation for trajectory correctness
- `WebArena/config_files/`: data processing and generated task configs
- `WebArena/prompt/`: instructions used across the WebArena implementation

### 4. Data preprocessing

Download the raw test files from the WebArena release or tutorial source and place them into `config_files/`.

Then run `generate_config_files.py` to convert the raw test data into the config files used as benchmark inputs.

The generated configs are the files consumed by the task selector in this repository.

### 5. Practical notes

- Make sure the website URLs in the Docker setup scripts are correct before running them.
- Run the scripts in the documented numerical order.
- If you change the raw WebArena data or the generated configs, regenerate the config files before running the pipeline again.

## Quick Start

The repository ships with a default WebArena + reasoning_bank setup.

```bash
bash run_pipeline.sh
```

Available arguments:

- `--task_name`: task config name under `run_configs/tasks/`
- `--method_name`: method config name under `run_configs/methods/`
- `--api_name`: API config name under `run_configs/apis/`
- `--output_dir`: root directory for all generated artifacts

## Configuration Files

### Task configuration

Task configs live under `run_configs/tasks/`.

Example: `run_configs/tasks/webarena.json`

Important fields:

- `benchmark`: benchmark selector used by the task dispatcher
- `driver`: environment driver name
- `website`: WebArena site group to run
- `start_index`: first selected task index
- `end_index`: optional end index
- `prev_id`: skip task IDs up to and including this value
- `env_kwargs_json`: environment arguments passed to the benchmark adapter
- `flags`: prompt and observation flags used by the WebArena adapter

Note:
- The code currently reads the key `flags` in the WebArena adapter.
- If your task JSON still uses `flag`, rename it to `flags` to match the implementation.

### Method configuration

Method configs live under `run_configs/methods/`.

Example: `run_configs/methods/reasoning_bank.json`

Structure:

- `runtime`: method runtime name
- `config`: method runtime configuration
- `evaluate`: evaluation settings
- `induce_memory`: memory induction settings

For the current reasoning_bank setup:

- `config.memory_path` controls where the selected memory text is written
- `config.embedding_model` controls the embedding backend used during memory selection
- `evaluate.prompt` controls the evaluation prompt type (`text` or `vision`)
- `evaluate.log_dir` controls where evaluation reports are stored
- `induce_memory.criteria` chooses the reward source (`gt` or `autoeval`)
- `induce_memory.memory_mode` chooses the memory extraction style

### API configuration

API configs live under `run_configs/apis/`.

Example: `run_configs/apis/qwen3.5-flash.json`

This file defines model-related defaults such as:

- `api_base_url`
- `temperature`
- `max_total_tokens`
- `max_input_tokens`
- `max_new_tokens`
- `enable_thinking`



## How To Add a New Method

1. Create a new folder under `methods/`.
2. Add a `method_main.py` with a `run(...)` function.
3. Add any helper modules you need.
4. Create a matching config file in `run_configs/methods/`.
5. Update the dispatcher if your method needs custom selection logic.

## How To Add a New Benchmark

1. Create a new adapter package under `benchmark/`.
2. Implement the benchmark adapter and task selector.
3. Register the benchmark in `benchmark/utils/utils.py`.
4. Add a task config in `run_configs/tasks/`.
5. Add any required environment or prompt helpers.


## Security Notes

If you plan to open-source this repository:

- do not commit real API keys
- prefer environment variables for secrets
- keep local-only config files out of Git
- review logs before publishing them

## License

Licensed under the Apache License, Version 2.0. See LICENSE.

## Acknowledgements

This project builds on WebArena, BrowserGym, and OpenAI-compatible model backends.
