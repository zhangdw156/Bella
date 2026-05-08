# Evaluation

This document describes how to configure and run BELLA evaluations.

## Configuration

BELLA uses a single YAML configuration file with [Hydra](https://hydra.cc/) for parameter management. All fields can be overridden via CLI arguments.

### bella.yaml

```yaml
# Model under test
model:
  name: "Qwen3-8B"               # Display name (used in output directories and reports)
  provider: "openai"              # openai | anthropic
  model_id: "Qwen3-8B"           # Model identifier passed to the SDK
  base_url: "http://localhost:8000/v1"
  api_key: "xxx"
  adapter: null                   # Path to custom ModelAdapter file (null = use default)

# User agent (globally fixed by benchmark for fairness)
user_agent:
  provider: "openai"
  model_id: "gpt-4.1"
  base_url: "https://api.openai.com/v1"
  api_key: "${oc.env:OPENAI_API_KEY}"

# Evaluation parameters
subset: "all"                     # Case subset to evaluate (see Case Selection)
count: null                       # Cases per category (null = all)
n: 1                              # Number of runs per case
workers: 4                        # Parallel workers
max_turns: 30                     # Max conversation turns for dynamic mode

# Output
output_dir: "results"
```

## CLI

The entry point is `bella run`, decorated with `@hydra.main()`.

```bash
# Full benchmark with defaults
bella run

# Override model config
bella run model.name=GPT-4.1 model.provider=openai model.model_id=gpt-4.1 \
         model.base_url=https://api.openai.com/v1 model.api_key=sk-xxx

# Custom adapter for non-standard models
bella run model.adapter=adapters/my_model.py

# Subset selection (see below)
bella run subset=bfcl
bella run subset=bfcl/base
bella run subset="bfcl/base,tau2/airline"

# Count per category
bella run subset=bfcl count=5

# pass@k evaluation
bella run n=4

# Concurrency
bella run workers=8

# Combined
bella run subset="bfcl,tau2" count=10 n=4 workers=8
```

## Case Selection

Cases are organized in a two-level category hierarchy: `level1/level2`.

### Category Map

| Level 1 | Level 2 | Description |
|---------|---------|-------------|
| `bfcl` | `base` | Multi-turn with complete information |
| `bfcl` | `miss_func` | Functions held out to test error recognition |
| `bfcl` | `miss_param` | Parameters must be inferred from context |
| `bfcl` | `long_context` | Large datasets to test information filtering |
| `mcpmark` | `filesystem` | File system operations |
| `mcpmark` | `postgres` | Database operations |
| `tau2` | `airline` | Airline booking, cancellation, modification |
| `tau2` | `retail` | Retail order management, returns, exchanges |
| `astra` | `{env_name}` | Astra-constructed environments (e.g., `travel_booking`) |

### Selection Rules

- `subset=all` — run all cases (default).
- `subset=bfcl` — run all cases under `bfcl/*`.
- `subset=bfcl/base` — run only `bfcl/base` cases.
- `subset="bfcl/base,tau2/airline"` — run cases from both categories.
- `count=5` — limit to 5 cases **per selected level-2 category**. Cases are sampled deterministically (sorted by `case_id`, take first N).

When `subset=bfcl` and `count=5`: runs 5 cases from each of `bfcl/base`, `bfcl/miss_func`, `bfcl/miss_param`, `bfcl/long_context` (20 total).

## Model Adapter

Most models work with the built-in default adapters (OpenAI and Anthropic). For models with non-standard tool call output, provide a custom adapter file:

```python
# adapters/my_model.py

class Adapter:
    def is_tool_call(self, response) -> bool:
        """Check if the model response contains tool calls."""
        ...

    def parse_tool_call(self, response) -> list[dict]:
        """Parse tool calls from the response.
        Returns: [{"name": "...", "arguments": {...}}, ...]
        """
        ...
```

Specify via config: `model.adapter: adapters/my_model.py`

The adapter file is dynamically loaded at startup. The class must be named `Adapter`.

## Metrics

### Single run (n=1)

- **pass@1**: fraction of cases that pass.

### Multiple runs (n>1)

- **pass@1**: unbiased estimate of single-attempt pass probability (Chen et al., 2021).
- **pass@k**: probability that at least 1 of k attempts passes.
- **pass^k**: probability that all k attempts pass.

All metrics are reported both overall and per-category.

### Formula (Chen et al., 2021)

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

Where `n` = total runs, `c` = number of passing runs, `k` = attempts.

## Output Structure

```
results/{model_name}/
├── run_1/
│   ├── bfcl/
│   │   ├── base/
│   │   │   ├── bfcl_base_001.json
│   │   │   └── ...
│   │   └── miss_func/
│   │       └── ...
│   ├── tau2/
│   │   └── airline/
│   │       └── ...
│   └── ...
├── run_2/                          # Only when n > 1
│   └── ...
└── summary.json
```

### Per-case output (`bfcl_base_001.json`)

```json
{
  "case_id": "bfcl_base_001",
  "category": "bfcl/base",
  "env_name": "gorilla_fs_twitter",
  "model": "Qwen3-8B",
  "interaction_mode": "fixed",
  "messages": [...],
  "tool_calls": [
    {"name": "mkdir", "arguments": {"dir_name": "temp"}, "result": {...}},
    ...
  ],
  "replay": {
    "total": 5,
    "matched": 5,
    "mismatched": 0,
    "token_substitutions": 1
  },
  "verify": [
    {"sql": "SELECT ...", "expected": [...], "actual": [...], "pass": true},
    ...
  ],
  "pass": true
}
```

### Summary (`summary.json`)

```json
{
  "model": "Qwen3-8B",
  "n": 4,
  "total_cases": 200,
  "metrics": {
    "overall": {
      "pass@1": 0.72,
      "pass@4": 0.89,
      "pass^4": 0.51
    },
    "by_category": {
      "bfcl/base": {"pass@1": 0.80, "pass@4": 0.95, "pass^4": 0.60},
      "bfcl/miss_func": {"pass@1": 0.65, "pass@4": 0.82, "pass^4": 0.42},
      "tau2/airline": {"pass@1": 0.70, "pass@4": 0.88, "pass^4": 0.48},
      "mcpmark/filesystem": {"pass@1": 0.75, "pass@4": 0.90, "pass^4": 0.55}
    },
    "by_mode": {
      "fixed": {"pass@1": 0.74},
      "dynamic": {"pass@1": 0.69}
    }
  }
}
```

## Execution Flow

```
bella run
  │
  ├── Load bella.yaml + CLI overrides (Hydra)
  ├── Load model adapter (default or custom)
  ├── Select cases by subset + count
  │
  ├── For each case × n runs (parallelized by workers):
  │   │
  │   ├── 1. Copy world.db → session.db
  │   ├── 2. Execute world_setup SQL on session.db
  │   ├── 3. Load backend.py with session.db
  │   ├── 4. Load tools from tools.jsonl
  │   ├── 5. Run agent:
  │   │      Fixed:   send user_demands sequentially
  │   │      Dynamic: user agent ↔ react agent loop
  │   ├── 6. Collect tool call chain
  │   ├── 7. Replay tool call chain on fresh DB (token substitution)
  │   ├── 8. Execute verify SQL on replayed DB
  │   ├── 9. Score: pass / fail
  │   └── 10. Write per-case result JSON
  │
  ├── Compute metrics (pass@1, pass@k, pass^k)
  └── Write summary.json
```
