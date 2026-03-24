from __future__ import annotations

import argparse
import os
from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from threading import Lock
from typing import Any, Iterable, List

from bella.config import load_settings
from bella.infer.openai_client import OpenAIClient
from bella.infer.types import BellaResult
from bella.infer.adapters.base import get_adapter
from bella.infer.writer import (
    load_existing_result_ids,
    result_file_path,
    upsert_result_jsonl,
)


def iter_limited(entries: list[dict], limit: int | None) -> Iterable[dict]:
    if limit is None or limit <= 0:
        yield from entries
    else:
        yield from entries[:limit]


def _run_single_entry(category: str, entry: dict[str, Any]) -> BellaResult:
    adapter = get_adapter(category)
    client = OpenAIClient()

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
    return adapter.finalize_result(entry, state_for_entry, last_bella_result)


def _build_failed_result(entry: dict[str, Any], exc: Exception) -> BellaResult:
    entry_id = str(entry.get("id", ""))
    return BellaResult(
        id=entry_id,
        result=[],
        input_token_count=0,
        output_token_count=0,
        latency=0.0,
        extra={
            "failed": True,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        },
    )


def _run_single_entry_safe(category: str, entry: dict[str, Any]) -> BellaResult:
    try:
        return _run_single_entry(category, entry)
    except Exception as exc:
        return _build_failed_result(entry, exc)


def run_bfcl_infer(
    category: str = "simple_python",
    limit: int = 3,
    max_workers: int | None = None,
) -> None:
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

    # Load dataset from Bella mirrored data (no BFCL runtime data loader).
    entries = load_bella_dataset(category)

    if not entries:
        raise RuntimeError(f"No BFCL entries found for category '{category}'.")

    result_file = result_file_path(
        registry_name=settings.bfcl_registry_name,
        result_root=Path(RESULT_PATH),
        group=adapter.result_group(category),
        filename=adapter.result_filename(category),
    )
    existing_ids = load_existing_result_ids(result_file)

    selected_entries = list(iter_limited(entries, limit))
    pending_entries = [
        entry for entry in selected_entries if str(entry.get("id", "")) not in existing_ids
    ]
    total_entries = len(selected_entries)
    skipped_entries = total_entries - len(pending_entries)

    if max_workers is None:
        max_workers = int(os.getenv("BELLA_BFCL_MAX_WORKERS", "1"))
    max_workers = max(1, int(max_workers))

    if skipped_entries:
        print(
            f"[Bella] Resume detected: category='{category}' skipping {skipped_entries} completed entries."
        )

    if not pending_entries:
        print(
            f"[Bella] BFCL inference already complete for category='{category}'. "
            f"Using existing result file: {result_file}"
        )
    else:
        write_lock = Lock()
        completed_count = skipped_entries
        entry_id_to_index = {
            str(entry.get("id", "")): idx for idx, entry in enumerate(selected_entries, start=1)
        }

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            in_flight: dict[Future[BellaResult], str] = {}
            pending_iter = iter(pending_entries)

            while len(in_flight) < max_workers:
                try:
                    entry = next(pending_iter)
                except StopIteration:
                    break
                future = pool.submit(_run_single_entry_safe, category, entry)
                in_flight[future] = str(entry.get("id", ""))

            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    entry_id = in_flight.pop(future)
                    result = future.result()
                    with write_lock:
                        upsert_result_jsonl(result, result_file)
                        completed_count += 1
                    entry_index = entry_id_to_index.get(entry_id, completed_count)
                    if result.extra and result.extra.get("failed"):
                        print(
                            f"[Bella] Inference failed: category='{category}' "
                            f"entry {entry_index}/{total_entries} id='{entry_id}' "
                            f"error_type='{result.extra.get('error_type', '')}' "
                            f"message={result.extra.get('error_message', '')!r}"
                        )
                    else:
                        print(
                            f"[Bella] Inference progress: category='{category}' "
                            f"entry {entry_index}/{total_entries} id='{entry_id}'"
                        )

                while len(in_flight) < max_workers:
                    try:
                        entry = next(pending_iter)
                    except StopIteration:
                        break
                    future = pool.submit(_run_single_entry_safe, category, entry)
                    in_flight[future] = str(entry.get("id", ""))

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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of concurrent entry workers. Defaults to BELLA_BFCL_MAX_WORKERS or 1.",
    )
    args = parser.parse_args()
    run_bfcl_infer(category=args.category, limit=args.limit, max_workers=args.max_workers)
