from __future__ import annotations

from pathlib import Path
from typing import List
import json

from bella.infer.types import BellaResult


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
    registry_dir = registry_name.replace("/", "_")
    out_dir = result_root / registry_dir / group
    out_dir.mkdir(parents=True, exist_ok=True)

    file_path = out_dir / filename

    entries = []
    for r in sorted(results, key=lambda x: x.id):
        base = {
            "id": r.id,
            "result": r.result,
            "input_token_count": r.input_token_count,
            "output_token_count": r.output_token_count,
            "latency": r.latency,
        }
        if r.extra:
            # extra 字段允许未来扩展 debug / cache / memory 等信息。
            base.update(r.extra)
        entries.append(base)

    with file_path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    return file_path

