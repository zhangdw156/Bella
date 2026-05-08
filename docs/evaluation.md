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
  temperature: 1.0                # Default 1.0, user-configurable

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
bella run subset=bfclv4_multi
bella run subset=bfclv4_multi_base
bella run subset="bfclv4_multi_base,tau3_airline"

# Count per category
bella run subset=bfclv4_multi count=5

# pass@k evaluation
bella run n=4

# Concurrency
bella run workers=8

# Combined
bella run subset="bfclv4_multi,tau3" count=10 n=4 workers=8
```

## Case Selection

Cases are categorized using underscore-separated hierarchical names. The `subset` parameter performs **prefix matching** — specifying a prefix selects all categories that start with it.

### Category Map

| Category | Cases | Description |
|----------|-------|-------------|
| `bfclv4_multi_base` | 25 | Multi-turn with complete information |
| `bfclv4_multi_miss_func` | 25 | Functions held out to test error recognition |
| `bfclv4_multi_miss_param` | 25 | Parameters must be inferred from context |
| `bfclv4_multi_long_context` | 25 | Large datasets to test information filtering |
| `mcpmark_filesystem` | ~20 | File system operations |
| `mcpmark_postgres` | ~20 | Database operations |
| `tau3_airline` | ~25 | Airline booking, cancellation, modification |
| `tau3_retail` | ~25 | Retail order management, returns, exchanges |
| `astra_{env_name}` | TBD | Astra-constructed environments |

### Prefix Matching

| `subset=` | Matches | Total |
|-----------|---------|-------|
| `all` | Everything | ~200 |
| `bfclv4_multi` | All 4 BFCL multi-turn subsets | 100 |
| `bfclv4_multi_base` | Only base subset | 25 |
| `mcpmark` | filesystem + postgres | ~40 |
| `tau3` | airline + retail | ~50 |
| `"bfclv4_multi_base,tau3_airline"` | Two specific subsets | ~50 |

### Selection Rules

- `subset=all` — run all cases (default).
- `subset=bfclv4_multi` — prefix match: all categories starting with `bfclv4_multi`.
- `subset=bfclv4_multi_base` — exact match: only this category.
- `subset="bfclv4_multi_base,tau3_airline"` — comma-separated: multiple prefixes.
- `count=5` — limit to 5 cases **per matched leaf category**. Cases are sampled deterministically (sorted by `case_id`, take first N).

When `subset=bfclv4_multi` and `count=5`: runs 5 cases from each of the 4 BFCL subsets (20 total).

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

The ReactAgent's system prompt is assembled from two blocks:

1. **Common block**: a universal prompt defining the agent's basic behavioral rules (e.g., "You are a helpful assistant that uses tools to complete tasks. Do not fabricate information."). Shared across all cases.
2. **Category block**: a category-specific prompt containing domain knowledge, business rules, and policies relevant to that category. For example, `tau3_airline` includes the airline's cancellation and refund policies.

The final system prompt is: `common_block + category_block`.

Both blocks are maintained by the benchmark — **users cannot modify or override them**. This ensures fair comparison across models.

Prompt files are stored in the repository:

```
prompts/
├── common.md                         # Universal agent instructions
├── bfclv4_multi_base.md
├── bfclv4_multi_miss_func.md
├── bfclv4_multi_miss_param.md
├── bfclv4_multi_long_context.md
├── mcpmark_filesystem.md
├── mcpmark_postgres.md
├── tau3_airline.md
├── tau3_retail.md
└── astra_{env_name}.md
```

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
│   ├── bfclv4_multi_base/
│   │   ├── bfclv4_multi_base_001.json
│   │   └── ...
│   ├── bfclv4_multi_miss_func/
│   │   └── ...
│   ├── tau3_airline/
│   │   └── ...
│   └── ...
├── run_2/                          # Only when n > 1
│   └── ...
└── summary.json
```

### Per-case output (`bfcl_base_001.json`)

```json
{
  "case_id": "bfcl_base_001",
  "category": "bfclv4_multi_base",
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
      "bfclv4_multi_base": {"pass@1": 0.80, "pass@4": 0.95, "pass^4": 0.60},
      "bfclv4_multi_miss_func": {"pass@1": 0.65, "pass@4": 0.82, "pass^4": 0.42},
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
