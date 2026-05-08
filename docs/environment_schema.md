# Environment Schema

BELLA environments use a packaged directory format derived from the [Astra](https://github.com/zhangdw156/Astra) project. Each environment is a self-contained, executable sandbox with tools backed by a SQLite database.

## Directory Structure

```
environments/{env_name}/
├── contract/
│   ├── tools.jsonl        # Tool definitions, one JSON object per line
│   └── ENVIRONMENT.md     # Human-readable environment description
├── runtime/
│   └── backend.py         # Executable backend
└── world/
    ├── world.db           # SQLite database with seed data (initial world state)
    └── schema.sql         # DDL matching world.db (for reference and recreation)
```

## File Specifications

### contract/tools.jsonl

One tool definition per line. Each line is a JSON object:

```json
{
  "name": "book_ticket",
  "description": "Book a flight ticket for the given flight ID. Returns the booking confirmation. Requires a valid access token.",
  "category": "mutation",
  "inputSchema": {
    "type": "object",
    "properties": {
      "access_token": {
        "type": "string",
        "description": "Session access token obtained from authenticate."
      },
      "flight_id": {
        "type": "string",
        "description": "Flight identifier from search_flights results."
      }
    },
    "required": ["access_token", "flight_id"]
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Tool function name, unique within the environment. |
| `description` | string | What the tool does. Shown to the LLM as part of the tool schema. |
| `category` | string | `"query"` (read-only, does not modify DB) or `"mutation"` (modifies DB state). |
| `inputSchema` | object | [JSON Schema](https://json-schema.org/) describing the tool's input parameters. |

The `inputSchema` follows the standard JSON Schema format used by OpenAI and Anthropic SDKs for tool definitions. The `category` field is for documentation and analysis only — it does not affect runtime behavior.

### contract/ENVIRONMENT.md

A human-readable Markdown document describing the environment. It should include:

- **Overview**: what domain this environment models.
- **Tools**: a table listing all tools with brief descriptions.
- **Database Schema**: key tables, their columns, and relationships.
- **Sample Data**: representative rows from important tables.
- **Key Workflows**: common multi-step operations (e.g., "authenticate → search → book").

This file is not consumed by the evaluation pipeline. It exists for human understanding and case authoring.

### runtime/backend.py

Must export an `EnvironmentBackend` class with the following interface:

```python
from pathlib import Path

class EnvironmentBackend:
    """Executable backend for a BELLA environment."""

    def __init__(self, *, db_path: Path) -> None:
        """
        Initialize the backend with a session-local copy of world.db.

        Args:
            db_path: Path to the session-local SQLite database.
        """
        ...

    def call(self, tool_name: str, arguments: dict) -> dict:
        """
        Execute a tool call. Must never raise — errors are returned as dicts.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments as a dict.

        Returns:
            dict: Tool result on success, {"error": "..."} on failure.
        """
        ...
```

Implementation requirements:

- **Never raise exceptions** from `call()`. All errors must be returned as `{"success": False, "error": {...}}` dicts.
- **All state lives in SQLite**. The backend reads from and writes to the database at `db_path`. No in-memory-only state that would be lost on replay.
- **Deterministic behavior**. Given the same database state and arguments, `call()` must return the same result. Exception: fields like `access_token` and `created_at` that contain timestamps are expected to vary — verify SQL must not check these fields (see [Case Schema](case_schema.md)).
- **Session isolation**. The evaluation pipeline passes a fresh copy of `world.db` for each case run. The backend must not reference any external state.

### world/world.db

A SQLite database containing the environment's default initial state (seed data). This is the baseline — individual cases may apply additional SQL via `world_setup` before execution.

### world/schema.sql

DDL statements that recreate the `world.db` schema. This file is for human reference and tooling support. It must match the actual schema of `world.db`.

## Environment Conversion Guide

All environments in BELLA are converted to this format, regardless of their original source:

| Source | Original Format | Conversion |
|--------|----------------|------------|
| BFCL (Gorilla) | Python classes with in-memory dict/list state | State → SQLite tables; class methods → `backend.py` dispatch |
| MCPMark | Real filesystem / PostgreSQL via MCP servers | Operations modeled as SQLite tables; MCP tool schemas → `tools.jsonl` |
| Tau2-bench | JSON data model (`db.json`) + Python ToolKit | JSON → SQLite import; `@is_tool` methods → `backend.py` dispatch |
| Astra | Already in this format | Direct reuse (minus `tool_graph.json`) |
