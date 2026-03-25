"""
Load BFCL v4 config and prompt templates from Bella project directories.
Used by adapters for result_group, result_filename, and prompt rendering.
Also provides dataset and multi_turn tool schema loaders so inference does not
depend on BFCL runtime data loader.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# involved_classes -> func doc filename (mirrors BFCL MULTI_TURN_FUNC_DOC_FILE_MAPPING)
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

# Bella project root: directory containing pyproject.toml (same as config.load_settings env_path parent)
def _bella_root() -> Path:
    # __file__ is src/bella/bfcl_resources.py -> parents[1]=src, parents[2]=repo root
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").exists():
        raise FileNotFoundError(
            f"Bella project root not found (expected pyproject.toml in {root}). "
            "Run from Bella repo root or set BELLA_PROJECT_ROOT."
        )
    return root


def load_bfcl_categories() -> dict[str, Any]:
    """Load config/bfcl_v4/categories.yaml; keys are category names."""
    import yaml
    path = _bella_root() / "config" / "bfcl_v4" / "categories.yaml"
    if not path.exists():
        raise FileNotFoundError(f"BFCL categories config not found: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_prompt_system(category: str) -> str:
    """Load prompts/bfcl_v4/{category}/system.txt."""
    path = _bella_root() / "prompts" / "bfcl_v4" / category / "system.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt system not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_prompt_user_template(category: str) -> str:
    """Load prompts/bfcl_v4/{category}/user_template.txt."""
    path = _bella_root() / "prompts" / "bfcl_v4" / category / "user_template.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt user template not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def render_user_prompt(category: str, **kwargs: Any) -> str:
    """Render user_template.txt for category with given variables."""
    template = load_prompt_user_template(category)
    return template.format(**kwargs)


def _raw_dataset_path(category: str) -> Path:
    """Path to datasets/bfcl_v4/raw/<category>.jsonl."""
    return _bella_root() / "datasets" / "bfcl_v4" / "raw" / f"{category}.jsonl"


def _multi_turn_schema_path(filename: str) -> Path:
    """Path to tool_schemas/bfcl_v4/multi_turn/<filename>."""
    return _bella_root() / "tool_schemas" / "bfcl_v4" / "multi_turn" / filename


def load_multi_turn_functions(involved_classes: list[str]) -> list[dict]:
    """
    Load and merge function doc lists for the given involved_classes.
    Returns a list of function doc dicts (name, description, parameters, ...).
    """
    result: list[dict] = []
    for cls in involved_classes:
        filename = MULTI_TURN_FUNC_DOC_FILE_MAPPING.get(cls)
        if not filename:
            continue
        path = _multi_turn_schema_path(filename)
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
    """
    Load dataset entries for the given category from Bella's mirrored data.
    Entries have the same shape as BFCL (id, question, function for non_live;
    for multi_turn_base also involved_classes, initial_config, etc.).
    For multi_turn_base, entry['function'] is filled from tool_schemas/bfcl_v4/multi_turn/.
    """
    path = _raw_dataset_path(category)
    if not path.exists():
        raise FileNotFoundError(
            f"Bella dataset not found: {path}. Run scripts/mirror_bfcl_resources.py first."
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
