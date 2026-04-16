# BenchmarkAdapter

Unified benchmark execution scaffold with a stricter split between:

- `BenchmarkAdapter/`: abstract interfaces, shared schemas, runner, CLI
- `benchmark/`: concrete benchmark adapters and benchmark-side drivers
- `methods/`: concrete method/runtime implementations
- `agent/`: concrete agent implementations

## Current Layout

```text
BenchmarkAdapter/
  base.py
  runner.py
  cli.py
  registry.py
  adapters/base.py
  drivers/protocols.py
  runtimes/base.py
  runtimes/protocols.py
  agents/protocols.py

benchmark/
  browsergym.py
  local.py
  WebArena/
    adapter.py
    configs/
  SWEBench/
    adapter.py

methods/
  echo.py
  agent_api.py
  reasoning_bank/
    runtime.py
    config.py
    observation.py
    prompting.py
    action_parser.py
    memory.py

agent/
  echo.py
```

## Design Rules

- Benchmark-specific task parsing and judging go in `benchmark/`
- Environment provisioning and teardown go in benchmark-side drivers
- Method logic goes in `methods/`
- Shared contracts only stay in `BenchmarkAdapter/`

For WebArena specifically:

- observation and action execution come from the environment/driver
- task text and benchmark metadata come from the adapter
- prompt construction, model calls, action parsing, retry logic, and memory updates belong in the method package

## Quick Start

Single WebArena run with the built-in `reasoning_bank` method:

```bash
python -m BenchmarkAdapter.cli \
  --benchmark webarena \
  --driver browsergym \
  --runtime reasoning_bank \
  --instances_json benchmark/WebArena/configs/instances.json \
  --env_kwargs_json benchmark/WebArena/configs/env_kwargs.json \
  --runtime_config_json run_configs/methods/reasoning_bank.json \
  --output outputs/webarena_results.json
```

`--runtime_config_json` accepts either:

- a direct runtime kwargs object
- a wrapped config object with `runtime` + `runtime_config`, such as `run_configs/methods/reasoning_bank.json`

Quick one-task run:

```bash
python -m BenchmarkAdapter.cli \
  --benchmark webarena \
  --driver browsergym \
  --runtime reasoning_bank \
  --task_name webarena.21 \
  --runtime_config_json run_configs/methods/reasoning_bank.json \
  --output outputs/webarena_quick.json
```

## Adding A Benchmark

1. Create a benchmark package, e.g. `benchmark/MyBenchmark/adapter.py`
2. Implement `BenchmarkAdapter`
3. Register it in [registry.py](/D:/Metic/Desktop/BenchmarkAdapter/BenchmarkAdapter/registry.py)
4. If it needs a concrete driver, place that driver under `benchmark/` and register it in [registry.py](/D:/Metic/Desktop/BenchmarkAdapter/BenchmarkAdapter/drivers/registry.py)

## Adding A Method

Small method:

```text
methods/my_method.py
```

Real method:

```text
methods/my_method/
  __init__.py
  runtime.py
  config.py
  prompting.py
  parser.py
  memory.py
```

Method contract:

- implement `solve(instance, context) -> RuntimeResponse | str`
- return a plain string if you only need the final answer
- return `RuntimeResponse` if you also need trajectory steps and metadata

Register the method in [registry.py](/D:/Metic/Desktop/BenchmarkAdapter/BenchmarkAdapter/runtimes/registry.py).

## Testing A Method

Test through the abstraction in three layers:

1. Runtime unit test  
   Build a fake `BenchmarkInstance` and fake `context`, then call `solve(...)` directly.

2. Adapter contract test  
   Use a real adapter instance and verify the method consumes:
   - `instance.task`
   - `instance.raw`
   - `context.api_context`
   - `context.env` when needed

3. End-to-end single-task test  
   Use `python -m BenchmarkAdapter.cli ...` with one task JSON.

## Notes

- `run_pipeline.py` uses `run_configs/tasks`, `run_configs/methods`, and `run_configs/apis` to batch-launch runs.
- `config_files/*.json` are single-task WebArena task files and can be passed directly to the CLI.
