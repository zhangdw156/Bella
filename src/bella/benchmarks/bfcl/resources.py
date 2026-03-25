"""
Load BFCL config, prompt templates, datasets, and tool schemas.

All BFCL data lives under ``datasets/bfcl/`` in the project root::

    datasets/bfcl/
    ├── categories.yaml
    ├── prompts/{category}/system.txt
    ├── prompts/{category}/user_template.txt
    ├── raw/{category}.jsonl
    └── tool_schemas/*.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MULTI_TURN_FUNC_DOC_FILE_MAPPING = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
    "WebSearchAPI": "web_search.json",
    "MemoryAPI_kv": "memory_kv.json",
    "MemoryAPI_vector": "memory_vector.json",
    "MemoryAPI_rec_sum": "memory_rec_sum.json",
}


def _bella_root() -> Path:
    """Project root (directory containing pyproject.toml)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Bella project root not found (no pyproject.toml in parents).")


def _bfcl_data_root() -> Path:
    """Root of BFCL benchmark data: ``<project>/datasets/bfcl/``."""
    return _bella_root() / "datasets" / "bfcl"


def load_bfcl_categories() -> dict[str, Any]:
    """Load ``datasets/bfcl/categories.yaml``; keys are category names."""
    import yaml
    path = _bfcl_data_root() / "categories.yaml"
    if not path.exists():
        raise FileNotFoundError(f"BFCL categories config not found: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_prompt_system(category: str) -> str:
    """Load ``datasets/bfcl/prompts/{category}/system.txt``."""
    path = _bfcl_data_root() / "prompts" / category / "system.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt system not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_prompt_user_template(category: str) -> str:
    """Load ``datasets/bfcl/prompts/{category}/user_template.txt``."""
    path = _bfcl_data_root() / "prompts" / category / "user_template.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt user template not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def render_user_prompt(category: str, **kwargs: Any) -> str:
    """Render user_template.txt for *category* with given variables."""
    template = load_prompt_user_template(category)
    return template.format(**kwargs)


def _raw_dataset_path(category: str) -> Path:
    """Path to ``datasets/bfcl/raw/{category}.jsonl``."""
    return _bfcl_data_root() / "raw" / f"{category}.jsonl"


def _tool_schema_path(filename: str) -> Path:
    """Path to ``datasets/bfcl/tool_schemas/{filename}``."""
    return _bfcl_data_root() / "tool_schemas" / filename


def load_multi_turn_functions(involved_classes: list[str]) -> list[dict]:
    """Load and merge function doc lists for the given *involved_classes*."""
    result: list[dict] = []
    for cls in involved_classes:
        filename = MULTI_TURN_FUNC_DOC_FILE_MAPPING.get(cls)
        if not filename:
            continue
        path = _tool_schema_path(filename)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                result.append(json.loads(line))
    return result


def load_bella_dataset(category: str) -> list[dict]:
    """Load dataset entries for the given *category* from ``datasets/bfcl/raw/``."""
    path = _raw_dataset_path(category)
    if not path.exists():
        raise FileNotFoundError(
            f"BFCL dataset not found: {path}. "
            "Run scripts/mirror_bfcl_resources.py first."
        )
    entries: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    if category.startswith("multi_turn_"):
        for entry in entries:
            involved = entry.get("involved_classes", [])
            entry["function"] = load_multi_turn_functions(involved)

    return entries
