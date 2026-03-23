from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, List

from bella.config import load_settings
from bella.infer.openai_client import OpenAIClient
from bella.infer.types import BellaResult
from bella.infer.adapters.base import get_adapter
from bella.infer.writer import write_results_jsonl


def iter_limited(entries: list[dict], limit: int | None) -> Iterable[dict]:
    if limit is None or limit <= 0:
        yield from entries
    else:
        yield from entries[:limit]


def run_bfcl_infer(category: str = "simple_python", limit: int = 3) -> None:
    """
    Run BFCL inference loop for a single category using Bella's unified
    adapter + client + writer pipeline.
    """
    settings = load_settings()

    # 延迟导入 BFCL 结果路径与定位工具（evaluator 仍用 BFCL）；数据改用 Bella 镜像。
    from bfcl_eval.constants.eval_config import RESULT_PATH
    from bfcl_eval.utils import find_file_by_category

    from bella.bfcl_resources import load_bella_dataset

    adapter = get_adapter(category)
    client = OpenAIClient()

    # Load dataset from Bella mirrored data (no BFCL runtime data loader).
    entries = load_bella_dataset(category)

    if not entries:
        raise RuntimeError(f"No BFCL entries found for category '{category}'.")

    selected_entries = list(iter_limited(entries, limit))
    total_entries = len(selected_entries)

    results: List[BellaResult] = []
    for idx, entry in enumerate(selected_entries, start=1):
        entry_id = entry.get("id", "<unknown>")
        print(
            f"[Bella] Inference progress: category='{category}' entry {idx}/{total_entries} id='{entry_id}'"
        )
        # Per-entry state to support both single-turn and multi-turn adapters.
        state_for_entry: dict[str, Any] = adapter.init_state(entry)
        last_bella_result: BellaResult | None = None

        while True:
            request = adapter.build_request(entry, state_for_entry)
            resp = client.chat_with_tools(
                messages=request.messages,
                tools=request.tools,
                temperature=request.temperature,
                tool_choice=request.tool_choice,
            )
            last_bella_result = adapter.parse_response(entry, resp, state_for_entry)

            if not adapter.has_next_turn(entry, state_for_entry):
                break

        assert last_bella_result is not None
        final_result = adapter.finalize_result(entry, state_for_entry, last_bella_result)
        results.append(final_result)

    if not results:
        raise RuntimeError("No inference results produced.")

    result_file = write_results_jsonl(
        results=results,
        registry_name=settings.bfcl_registry_name,
        result_root=Path(RESULT_PATH),
        group=adapter.result_group(category),
        filename=adapter.result_filename(category),
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BFCL inference via Bella using BFCL official handlers."
    )
    parser.add_argument(
        "--category",
        type=str,
        default="simple_python",
        help="BFCL test category to run (default: simple_python).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum number of test entries to run (default: 3).",
    )
    args = parser.parse_args()
    run_bfcl_infer(category=args.category, limit=args.limit)
