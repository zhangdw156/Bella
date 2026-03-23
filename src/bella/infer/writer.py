from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json

from bella.infer.types import BellaResult


def result_file_path(
    registry_name: str,
    result_root: Path,
    group: str,
    filename: str,
) -> Path:
    registry_dir = registry_name.replace("/", "_")
    out_dir = result_root / registry_dir / group
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def serialize_result(result: BellaResult) -> Dict[str, object]:
    base: Dict[str, object] = {
        "id": result.id,
        "result": result.result,
        "input_token_count": result.input_token_count,
        "output_token_count": result.output_token_count,
        "latency": result.latency,
    }
    if result.extra:
        base.update(result.extra)
    return base


def load_existing_result_ids(file_path: Path) -> set[str]:
    if not file_path.exists():
        return set()

    existing_ids: set[str] = set()
    with file_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry_id = entry.get("id")
            if isinstance(entry_id, str) and entry_id:
                existing_ids.add(entry_id)
    return existing_ids


def append_result_jsonl(result: BellaResult, file_path: Path) -> None:
    entry = serialize_result(result)
    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def upsert_result_jsonl(result: BellaResult, file_path: Path) -> None:
    entry = serialize_result(result)
    existing_entries: dict[str, Dict[str, object]] = {}

    if file_path.exists():
        with file_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError:
                    continue
                existing_id = existing.get("id")
                if isinstance(existing_id, str) and existing_id:
                    existing_entries[existing_id] = existing

    existing_entries[str(entry["id"])] = entry

    with file_path.open("w", encoding="utf-8") as f:
        for existing_id in sorted(existing_entries):
            f.write(json.dumps(existing_entries[existing_id], ensure_ascii=False) + "\n")


def write_results_jsonl(
    results: List[BellaResult],
    registry_name: str,
    result_root: Path,
    group: str,
    filename: str,
) -> Path:
    """
    Generic BFCL-compatible JSONL writer for BellaResult entries.
    """
    file_path = result_file_path(
        registry_name=registry_name,
        result_root=result_root,
        group=group,
        filename=filename,
    )

    entries = []
    for r in sorted(results, key=lambda x: x.id):
        entries.append(serialize_result(r))

    with file_path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    return file_path
