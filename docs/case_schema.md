# Case Schema

Each case is a JSON file describing a single evaluation task. Cases reference an environment by name and define what the agent must accomplish, how the interaction proceeds, and how the final result is verified.

## Directory Structure

Cases and environments are stored at the same level:

```
bella/
├── environments/
│   ├── airline/
│   ├── travel_booking/
│   └── ...
└── cases/
    ├── case_001.json
    ├── case_002.json
    └── ...
```

## Schema Definition

```json
{
  "case_id": "string",
  "env_name": "string",
  "category": "string",
  "source": "bfcl | mcpmark | tau2 | astra",
  "tags": ["string"],

  "interaction_mode": "fixed | dynamic",

  "demand": "string (dynamic only)",
  "user_demands": ["string"] ,
  "world_setup": ["string"],
  "user_agent_config": {
    "role": "string",
    "personality": "string",
    "knowledge_boundary": "string"
  },

  "verify": [
    {
      "sql": "string",
      "expected": [[]],
      "order_matters": false
    }
  ]
}
```

## Field Reference

### Metadata

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `case_id` | string | yes | Unique identifier across the entire benchmark. |
| `env_name` | string | yes | Name of the environment directory this case runs in. |
| `category` | string | yes | Two-level classification in `"level1/level2"` format (e.g., `"bfcl/base"`, `"tau2/airline"`). Used for subset selection during evaluation. |
| `source` | string | yes | Origin of this case: `"bfcl"`, `"mcpmark"`, `"tau2"`, or `"astra"`. |
| `tags` | string[] | no | Categorization tags (e.g., `["booking", "multi-step"]`). |

### Interaction Mode

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `interaction_mode` | string | yes | `"fixed"` — pre-scripted user messages, no user agent. `"dynamic"` — LLM-simulated user agent. |

### Task Content

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `demand` | string | dynamic only | The user's goal. Drives the user agent's behavior. Not present in fixed mode. |
| `user_demands` | string[] | fixed only | Pre-scripted user messages sent to the assistant agent sequentially. Not present in dynamic mode. |
| `world_setup` | string[] | no | SQL statements executed on the world.db copy before the case runs. Default `[]`. |
| `user_agent_config` | object | dynamic only | Configuration for the LLM-based user agent. Not present in fixed mode. |

### user_agent_config

| Field | Type | Description |
|-------|------|-------------|
| `role` | string | Who the user is (e.g., "Budget-conscious traveler planning a weekend trip"). |
| `personality` | string | How the user behaves (e.g., "Price-sensitive, asks about options before committing"). |
| `knowledge_boundary` | string | What the user knows and does not know (e.g., "Knows city names. Does not know zipcodes or flight IDs."). |

The user agent LLM is globally configured by the benchmark (not per-case), ensuring fair comparison across models.

### Verification

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `verify` | object[] | yes | SQL queries + expected results for final DB state verification. |

Each entry in `verify`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sql` | string | — | SELECT query to execute against the final (replayed) DB. |
| `expected` | any[][] | — | Expected result rows. Each row is a list of values. |
| `order_matters` | bool | `false` | Whether row order must match. |

## Interaction Modes

### Fixed Mode (Track A)

Used for cases from BFCL and MCPMark. User messages are pre-scripted — no LLM user agent is involved.

The React agent receives `user_demands[0]`, processes it (making tool calls as needed), responds, then receives `user_demands[1]`, and so on until all messages are exhausted.

For single-shot autonomous tasks (MCPMark style), `user_demands` has exactly one entry.

### Dynamic Mode (Track B)

Used for cases from Tau2-bench and Astra. A user agent LLM generates messages based on `demand` and `user_agent_config`.

The user agent sends the first message (derived from `demand` and persona), the React agent responds, and they alternate until the user agent signals completion or a turn limit is reached.

## Evaluation Flow

```
1. Copy world.db → session.db
2. Execute world_setup SQL statements on session.db (if any)
3. Load backend.py with session.db
4. Load tools from tools.jsonl
5. Run agent:
   - Fixed mode: send user_demands[0], agent processes, send user_demands[1], ...
   - Dynamic mode: user agent generates first message, alternates with react agent
6. Collect the complete tool call chain from the agent's execution
7. Replay:
   a. Copy world.db → replay.db
   b. Execute world_setup SQL on replay.db
   c. Load backend.py with replay.db
   d. Replay the tool call chain against replay.db (with token substitution)
8. Execute each verify[].sql on replay.db
9. Compare results with verify[].expected
10. Score: 1 if ALL verify queries match, 0 if ANY mismatch
```

### Why Replay?

The agent's tool calls are replayed on a fresh database copy rather than verifying the agent's live database directly. This ensures:

- **Reproducibility**: the tool call chain alone is sufficient to produce the final state.
- **Token safety**: during replay, non-deterministic tokens (e.g., auth tokens with timestamps) are automatically substituted using the same mechanism as [Astra's EvalRunner](https://github.com/zhangdw156/Astra). When a replayed auth call returns a different token, all subsequent tool calls are patched with the new token.
- **Integrity**: no hidden side effects or in-memory state influenced the result.

### Verify SQL Guidelines

Learned from Astra's token substitution issue:

1. **Never verify non-deterministic columns**: `access_token`, `session_id`, `token_expires_at`, `created_at`, `updated_at`, or any auto-generated timestamp.
2. **Always verify semantically meaningful state**: status fields, counts, foreign key references, amounts, names, deterministic identifiers.
3. **Use deterministic WHERE clauses**: filter by known IDs (`reservation_id`, `user_id`) not by generated tokens.
4. **Prefer aggregates for counts**: `SELECT COUNT(*) FROM bookings` rather than checking auto-generated `booking_id` values.
5. **Handle floating-point carefully**: use `ROUND()` or integer comparisons where possible.

## Examples

### BFCL Case (Fixed, Multi-Turn)

```json
{
  "case_id": "bfcl_base_001",
  "env_name": "gorilla_fs_twitter",
  "category": "bfcl/base",
  "source": "bfcl",
  "tags": ["file-ops", "social-media"],
  "interaction_mode": "fixed",
  "user_demands": [
    "Move 'final_report.pdf' within document directory to 'temp' directory in document. Make sure to create the directory.",
    "Search for 'budget analysis' sections in the file.",
    "Sort the 'final_report.pdf' by line for improved clarity.",
    "Move 'previous_report.pdf' to temp as well and compare it with 'final_report.pdf'."
  ],
  "world_setup": [],
  "verify": [
    {
      "sql": "SELECT path FROM files WHERE name = 'final_report.pdf'",
      "expected": [["/workspace/document/temp/final_report.pdf"]],
      "order_matters": false
    },
    {
      "sql": "SELECT path FROM files WHERE name = 'previous_report.pdf'",
      "expected": [["/workspace/document/temp/previous_report.pdf"]],
      "order_matters": false
    }
  ]
}
```

### MCPMark Case (Fixed, Single-Turn)

```json
{
  "case_id": "mcpmark_fs_merge_001",
  "env_name": "filesystem_ops",
  "category": "mcpmark/filesystem",
  "source": "mcpmark",
  "tags": ["file-merging"],
  "interaction_mode": "fixed",
  "user_demands": [
    "Find the 10 smallest .txt files in the test directory, merge their content in alphabetical order, and create merged_content.txt with filename headers before each file's content."
  ],
  "world_setup": [],
  "verify": [
    {
      "sql": "SELECT COUNT(*) FROM files WHERE name = 'merged_content.txt'",
      "expected": [[1]],
      "order_matters": false
    }
  ]
}
```

### Tau2-bench Case (Dynamic, Multi-Turn)

```json
{
  "case_id": "tau2_airline_001",
  "env_name": "airline",
  "category": "tau2/airline",
  "source": "tau2",
  "tags": ["cancellation", "policy-compliance"],
  "interaction_mode": "dynamic",
  "demand": "Cancel the reservation from Philadelphia to LaGuardia.",
  "world_setup": [
    "UPDATE reservations SET created_at = datetime('now', '-36 hours') WHERE reservation_id = 'Q69X3R'"
  ],
  "user_agent_config": {
    "role": "A traveler who wants to cancel a trip",
    "personality": "Insistent, claims a phone agent approved the cancellation",
    "knowledge_boundary": "Knows own name (Raj Sanchez) and user ID (raj_sanchez_7340). Does not know reservation ID or cancellation policy."
  },
  "verify": [
    {
      "sql": "SELECT status FROM reservations WHERE reservation_id = 'Q69X3R'",
      "expected": [["cancelled"]],
      "order_matters": false
    }
  ]
}
```

### Astra Case (Dynamic, Multi-Turn)

```json
{
  "case_id": "astra_travel_001",
  "env_name": "travel_booking",
  "category": "astra/travel_booking",
  "source": "astra",
  "tags": ["booking", "implicit-chain"],
  "interaction_mode": "dynamic",
  "demand": "Purchase the cheapest economy flight from Rivermist to Stonebrook for next Friday.",
  "world_setup": [
    "INSERT INTO cities (name, zipcode) VALUES ('Rivermist', '83214')",
    "INSERT INTO cities (name, zipcode) VALUES ('Stonebrook', '74532')",
    "INSERT INTO flights (id, origin_zip, dest_zip, date, price, cabin) VALUES ('FL001', '83214', '74532', '2026-05-09', 299.00, 'economy')",
    "INSERT INTO flights (id, origin_zip, dest_zip, date, price, cabin) VALUES ('FL002', '83214', '74532', '2026-05-09', 459.00, 'economy')"
  ],
  "user_agent_config": {
    "role": "Budget-conscious traveler planning a weekend trip",
    "personality": "Price-sensitive, asks about options before committing",
    "knowledge_boundary": "Knows city names and travel date. Does not know zipcodes, flight IDs, or booking procedure."
  },
  "verify": [
    {
      "sql": "SELECT travel_from, travel_to, travel_class, travel_cost FROM bookings ORDER BY booking_id DESC LIMIT 1",
      "expected": [["83214", "74532", "economy", 299.0]],
      "order_matters": false
    }
  ]
}
```
