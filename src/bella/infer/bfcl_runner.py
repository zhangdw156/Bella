from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict, Any
import json

from bella.config import load_settings
from bella.infer.openai_client import OpenAIClient
from bella.infer.bfcl_formatter import build_simple_python_request
from bella.infer.bfcl_parser import parse_simple_python_tool_calls


def iter_limited(entries: list[dict], limit: int | None) -> Iterable[dict]:
    if limit is None or limit <= 0:
        yield from entries
    else:
        yield from entries[:limit]


def _write_simple_python_results_jsonl(
    entries: List[Dict[str, Any]],
    registry_name: str,
    result_root: Path,
) -> Path:
    """
    Write BFCL-compatible JSONL result file for simple_python.

    File layout follows BFCL's conventions:
      result/<registry_dir_name>/non_live/BFCL_v4_simple_python_result.json
    """
    registry_dir = registry_name.replace("/", "_")
    out_dir = result_root / registry_dir / "non_live"
    out_dir.mkdir(parents=True, exist_ok=True)

    file_path = out_dir / "BFCL_v4_simple_python_result.json"

    # 保持按 id 排序，方便 evaluator / diff。
    entries_sorted = sorted(entries, key=lambda e: e.get("id", ""))

    with file_path.open("w", encoding="utf-8") as f:
        for e in entries_sorted:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    return file_path


def run_bfcl_infer(category: str = "simple_python", limit: int = 3) -> None:
    """
    Run minimal BFCL inference loop for a single category (MVP: simple_python).

    Bella fully owns:
    - request construction (messages + tools)
    - API calling (OpenAI-compatible client)
    - tool call parsing
    - result entry construction and JSONL writing
    """
    if category != "simple_python":
        raise ValueError(
            "Current Bella MVP only supports category='simple_python' for inference."
        )

    settings = load_settings()

    # 延迟导入 BFCL 数据加载和结果路径工具，确保 BFCL_PROJECT_ROOT 已由 load_settings 写入环境变量。
    from bfcl_eval.constants.eval_config import RESULT_PATH
    from bfcl_eval.utils import find_file_by_category, load_dataset_entry

    # Load BFCL official dataset entries for the given category.
    entries = load_dataset_entry(
        category,
        include_prereq=False,
        include_language_specific_hint=True,
    )

    if not entries:
        raise RuntimeError(f"No BFCL entries found for category '{category}'.")

    client = OpenAIClient()

    result_entries: List[Dict[str, Any]] = []
    for entry in iter_limited(entries, limit):
        assert isinstance(entry.get("function"), list)

        # 1) BFCL entry -> OpenAI messages + tools
        messages, tools = build_simple_python_request(entry)

        # 2) Call model
        resp = client.chat_with_tools(messages=messages, tools=tools, temperature=0.0)

        # 3) Parse tool calls into BFCL result format
        parsed_result = parse_simple_python_tool_calls(resp)

        # 4) Construct BFCL-compatible result entry
        result_entry: Dict[str, Any] = {
            "id": entry["id"],
            "result": parsed_result,
        }

        # 可选：保留一些基础 metadata，便于 BFCL 统计 token / cost（非必须）
        usage = getattr(resp, "usage", None)
        if usage:
            result_entry["input_token_count"] = getattr(usage, "prompt_tokens", 0)
            result_entry["output_token_count"] = getattr(usage, "completion_tokens", 0)

        # latency 在 client 内部没显式测量，这里暂不填或置 0。
        result_entry["latency"] = 0.0

        result_entries.append(result_entry)

    if not result_entries:
        raise RuntimeError("No inference results produced.")

    # 5) Write JSONL in BFCL-compatible layout (Bella owns writer logic).
    result_file = _write_simple_python_results_jsonl(
        entries=result_entries,
        registry_name=settings.bfcl_registry_name,
        result_root=Path(RESULT_PATH),
    )

    # Try to locate the result file for this category via BFCL helper for sanity.
    try:
        located = find_file_by_category(
            category,
            Path(RESULT_PATH) / settings.bfcl_registry_name.replace("/", "_"),
            is_result_file=True,
        )
        print(f"[Bella] BFCL inference done. Category='{category}', limit={limit}")
        print(f"[Bella] Result file (by writer): {result_file}")
        print(f"[Bella] Result file (by BFCL locator): {located}")
    except FileNotFoundError:
        print(
            f"[Bella] BFCL inference done for category='{category}', "
            f"but BFCL locator could not find result file under RESULT_PATH={RESULT_PATH}."
        )

