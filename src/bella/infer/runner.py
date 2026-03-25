"""
Benchmark-agnostic inference runner.

Replaces the old BFCL-specific ``bfcl_runner``.  Works with any
``Benchmark`` registered via ``@register_benchmark``.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from threading import Lock
from typing import Any, Dict, Iterable

from bella.benchmarks import get_benchmark
from bella.benchmarks.base import InferAdapter
from bella.config import load_settings
from bella.infer.openai_client import OpenAIClient
from bella.infer.types import BellaResult
from bella.infer.writer import load_existing_result_ids, upsert_result_jsonl


def _iter_limited(entries: list[dict], limit: int | None) -> Iterable[dict]:
    if limit is None or limit <= 0:
        yield from entries
    else:
        yield from entries[:limit]


def _run_single_entry(adapter: InferAdapter, entry: Dict[str, Any]) -> BellaResult:
    client = OpenAIClient()
    state = adapter.init_state(entry)
    last_result: BellaResult | None = None

    while True:
        request = adapter.build_request(entry, state)
        resp = client.chat_with_tools(
            messages=request.messages,
            tools=request.tools,
            temperature=request.temperature,
            tool_choice=request.tool_choice,
        )
        last_result = adapter.parse_response(entry, resp, state)

        if not adapter.has_next_turn(entry, state):
            break

    assert last_result is not None
    return adapter.finalize_result(entry, state, last_result)


def _build_failed_result(entry: Dict[str, Any], exc: Exception) -> BellaResult:
    return BellaResult(
        id=str(entry.get("id", "")),
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


def _run_single_entry_safe(adapter: InferAdapter, entry: Dict[str, Any]) -> BellaResult:
    try:
        return _run_single_entry(adapter, entry)
    except Exception as exc:
        return _build_failed_result(entry, exc)


def run_infer(
    benchmark_name: str,
    category: str,
    limit: int = 0,
    max_workers: int | None = None,
) -> None:
    """Run inference for *category* under *benchmark_name*."""
    load_settings()
    benchmark = get_benchmark(benchmark_name)

    dataset = benchmark.load_dataset(category, limit)
    if not dataset:
        raise RuntimeError(
            f"No entries found for benchmark={benchmark_name!r} category={category!r}."
        )

    adapter = benchmark.create_adapter(category)
    result_file = benchmark.result_file(category)
    existing_ids = load_existing_result_ids(result_file)

    selected = list(_iter_limited(dataset, limit))
    pending = [e for e in selected if str(e.get("id", "")) not in existing_ids]
    total = len(selected)
    skipped = total - len(pending)

    if max_workers is None:
        max_workers = int(os.getenv("BELLA_MAX_WORKERS", "1"))
    max_workers = max(1, int(max_workers))

    if skipped:
        print(
            f"[Bella] Resume: benchmark={benchmark_name!r} category={category!r} "
            f"skipping {skipped} completed entries."
        )

    if not pending:
        print(
            f"[Bella] Inference already complete for {benchmark_name!r}/{category!r}. "
            f"Result file: {result_file}"
        )
        return

    write_lock = Lock()
    completed = skipped
    id_to_idx = {str(e.get("id", "")): i for i, e in enumerate(selected, 1)}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        in_flight: dict[Future[BellaResult], str] = {}
        pending_iter = iter(pending)

        while len(in_flight) < max_workers:
            try:
                entry = next(pending_iter)
            except StopIteration:
                break
            future = pool.submit(_run_single_entry_safe, adapter, entry)
            in_flight[future] = str(entry.get("id", ""))

        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                entry_id = in_flight.pop(future)
                result = future.result()
                with write_lock:
                    upsert_result_jsonl(result, result_file)
                    completed += 1
                idx = id_to_idx.get(entry_id, completed)
                if result.extra and result.extra.get("failed"):
                    print(
                        f"[Bella] FAILED: {benchmark_name}/{category} "
                        f"entry {idx}/{total} id={entry_id!r} "
                        f"error={result.extra.get('error_message', '')!r}"
                    )
                else:
                    print(
                        f"[Bella] Progress: {benchmark_name}/{category} "
                        f"entry {idx}/{total} id={entry_id!r}"
                    )

            while len(in_flight) < max_workers:
                try:
                    entry = next(pending_iter)
                except StopIteration:
                    break
                future = pool.submit(_run_single_entry_safe, adapter, entry)
                in_flight[future] = str(entry.get("id", ""))

    print(f"[Bella] Inference done: {benchmark_name}/{category} limit={limit}")
    print(f"[Bella] Result file: {result_file}")


# ── backward-compatible wrapper ──────────────────────────────────────

def run_bfcl_infer(
    category: str = "simple_python",
    limit: int = 3,
    max_workers: int | None = None,
) -> None:
    """Legacy wrapper — delegates to ``run_infer("bfcl", ...)``."""
    run_infer("bfcl", category, limit=limit, max_workers=max_workers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark inference via Bella.")
    parser.add_argument("--benchmark", type=str, default="bfcl", help="Benchmark name (default: bfcl).")
    parser.add_argument("--category", type=str, default="simple_python", help="Category/subset to run.")
    parser.add_argument("--limit", type=int, default=3, help="Max entries (0 = all).")
    parser.add_argument("--max-workers", type=int, default=None, help="Concurrent workers.")
    args = parser.parse_args()
    run_infer(args.benchmark, args.category, limit=args.limit, max_workers=args.max_workers)
