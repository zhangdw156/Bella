from __future__ import annotations

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

    # 延迟导入 BFCL 数据加载和结果路径工具，确保 BFCL_PROJECT_ROOT 已由 load_settings 写入环境变量。
    from bfcl_eval.constants.eval_config import RESULT_PATH
    from bfcl_eval.utils import find_file_by_category, load_dataset_entry

    adapter = get_adapter(category)
    client = OpenAIClient()

    # Load BFCL official dataset entries for the given category.
    entries = load_dataset_entry(
        category,
        include_prereq=False,
        include_language_specific_hint=True,
    )

    if not entries:
        raise RuntimeError(f"No BFCL entries found for category '{category}'.")

    results: List[BellaResult] = []
    for entry in iter_limited(entries, limit):
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

