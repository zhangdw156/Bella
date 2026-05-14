# Evaluation

This document describes how to configure and run BELLA evaluations.

## Configuration

BELLA uses a single YAML configuration file with [Hydra](https://hydra.cc/) for parameter management. All fields can be overridden via CLI arguments.

### bella.yaml

```yaml
# Model under test
model:
  name: "Qwen3-8B"               # Display name (used in output directories and reports)
  protocol: "openai_chat_completions"  # anthropic | openai_chat_completions | openai_responses
  model_id: "Qwen3-8B"           # Model identifier passed to the SDK
  base_url: "http://localhost:8000/v1"
  api_key: "xxx"
  adapter: null                   # Path to custom ModelAdapter file (null = use default)
  temperature: 1.0                # Default 1.0, user-configurable
  max_context_tokens: 128000      # Context window size for auto-compaction

# User agent (globally fixed by benchmark for fairness)
user_agent:
  protocol: "openai_chat_completions"
  model_id: "gpt-4.1"
  base_url: "https://api.openai.com/v1"
  api_key: "${oc.env:OPENAI_API_KEY}"
  max_context_tokens: 128000

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
bella run model.name=GPT-4.1 model.protocol=openai_chat_completions model.model_id=gpt-4.1 \
         model.base_url=https://api.openai.com/v1 model.api_key=sk-xxx

# Custom adapter for non-standard models
bella run model.adapter=adapters/my_model.py

# Subset selection (see below)
bella run subset=mcpmark
bella run subset=mcpmark_postgres
bella run subset="mcpmark_postgres,tau3_airline"

# Count per category
bella run subset=mcpmark count=5

# pass@k evaluation
bella run n=4

# Concurrency
bella run workers=8

# Combined
bella run subset="mcpmark,tau3" count=10 n=4 workers=8
```

## Case Selection

Cases are categorized using underscore-separated hierarchical names. The `subset` parameter performs **prefix matching** — specifying a prefix selects all categories that start with it.

### Category Map

| Category | Cases | Description |
|----------|-------|-------------|
| `mcpmark_filesystem` | ~20 | File system operations |
| `mcpmark_postgres` | ~20 | Database operations |
| `tau3_airline` | ~25 | Airline booking, cancellation, modification |
| `tau3_retail` | ~25 | Retail order management, returns, exchanges |
| `astra_{env_name}` | TBD | Astra-constructed environments |

### Prefix Matching

| `subset=` | Matches | Total |
|-----------|---------|-------|
| `all` | Everything | ~100 |
| `mcpmark` | filesystem + postgres | ~40 |
| `tau3` | airline + retail | ~50 |
| `"mcpmark_postgres,tau3_airline"` | Two specific subsets | ~45 |

### Selection Rules

- `subset=all` — run all cases (default).
- `subset=mcpmark` — prefix match: all categories starting with `mcpmark`.
- `subset=mcpmark_postgres` — exact match: only this category.
- `subset="mcpmark_postgres,tau3_airline"` — comma-separated: multiple prefixes.
- `count=5` — limit to 5 cases **per matched leaf category**. Cases are sampled deterministically (sorted by `case_id`, take first N).

When `subset=mcpmark` and `count=5`: runs 5 cases from each of the 2 MCPMark subsets (10 total).

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

## Temperature

BELLA defaults to `temperature=1.0`, but users may override it via config (`model.temperature`).

Default rationale:
- Thinking/reasoning models require `temperature=1`.
- The `pass@k` and `pass^k` metrics are designed to measure model reliability under stochastic sampling.

Users may set a different temperature if their model performs better at another value. The chosen temperature is recorded in `summary.json` for reproducibility.

## System Prompt

The ReactAgent's system prompt is assembled from two parts:

1. **Common block**: hardcoded in ReactAgent source code — universal behavioral rules.
2. **Category block** (optional): looked up from `category_prompts.json` by the case's `category` field. Contains domain-specific business rules and policies.

`category_prompts.json` is a top-level file alongside `environments/` and `cases/`:

```
bella/
├── environments/
├── cases/
└── category_prompts.json
```

Format:

```json
{
  "tau3_airline": "The current time is 2024-05-15 ...\n\nAs an airline agent ...",
  "tau3_retail": "...",
  "mcpmark_postgres": null
}
```

Categories with `null` or absent entries use only the common block. Users cannot modify or override system prompts — this ensures fair comparison across models.

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
│   ├── mcpmark_postgres/
│   │   ├── mcpmark_postgres_000.json
│   │   └── ...
│   ├── tau3_airline/
│   │   └── ...
│   └── ...
├── run_2/                          # Only when n > 1
│   └── ...
└── summary.json
```

### Per-case output (`mcpmark_postgres_000.json`)

```json
{
  "case_id": "mcpmark_postgres_000",
  "category": "mcpmark_postgres",
  "env_name": "mcpmark_postgres_employees",
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
  "total_cases": 100,
  "metrics": {
    "overall": {
      "pass@1": 0.72,
      "pass@4": 0.89,
      "pass^4": 0.51
    },
    "by_category": {
      "mcpmark_postgres": {"pass@1": 0.75, "pass@4": 0.90, "pass^4": 0.55},
      "tau3_airline": {"pass@1": 0.70, "pass@4": 0.88, "pass^4": 0.48},
      "mcpmark_filesystem": {"pass@1": 0.75, "pass@4": 0.90, "pass^4": 0.55}
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
  │   │      Fixed:   send demand, agent completes autonomously
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
