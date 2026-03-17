from __future__ import annotations

from copy import deepcopy
from typing import Iterable

from bella.config import load_settings


def iter_limited(entries: list[dict], limit: int | None) -> Iterable[dict]:
    if limit is None or limit <= 0:
        yield from entries
    else:
        yield from entries[:limit]


def run_bfcl_infer(category: str = "simple_python", limit: int = 3) -> None:
    """
    Run minimal BFCL inference loop for a single category.

    This mimics BFCL's own _llm_response_generation.multi_threaded_inference
    for a small subset of test cases, and always delegates result writing
    to BaseHandler.write() to keep full compatibility with the official
    evaluator and file format.
    """
    settings = load_settings()

    # 延迟导入 BFCL 相关模块，确保 BFCL_PROJECT_ROOT 已由 load_settings 写入环境变量。
    from bfcl_eval.constants.eval_config import RESULT_PATH
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )
    from bfcl_eval.utils import find_file_by_category, load_dataset_entry

    # Load BFCL official dataset entries for the given category.
    # For MVP we disable prereq and keep language hint enabled.
    entries = load_dataset_entry(
        category,
        include_prereq=False,
        include_language_specific_hint=True,
    )

    if not entries:
        raise RuntimeError(f"No BFCL entries found for category '{category}'.")

    handler = OpenAICompletionsHandler(
        model_name=settings.openai_model,
        temperature=0.0,
        registry_name=settings.bfcl_registry_name,
        is_fc_model=True,
    )

    # Run inference on a small subset and write results via official handler.
    for entry in iter_limited(entries, limit):
        # Closely follow bfcl_eval._llm_response_generation.multi_threaded_inference
        assert isinstance(entry.get("function"), list)

        result, metadata = handler.inference(
            deepcopy(entry),
            include_input_log=False,
            exclude_state_log=False,
        )

        result_to_write = {
            "id": entry["id"],
            "result": result,
            **(metadata or {}),
        }

        handler.write(result_to_write, RESULT_PATH, update_mode=True)

    # Try to locate the result file for this category for user reference.
    try:
        result_file = find_file_by_category(
            category,
            RESULT_PATH / handler.registry_dir_name,
            is_result_file=True,
        )
        print(f"[Bella] BFCL inference done. Category='{category}', limit={limit}")
        print(f"[Bella] Result file: {result_file}")
    except FileNotFoundError:
        print(
            f"[Bella] BFCL inference done for category='{category}', "
            f"but result file could not be located under RESULT_PATH={RESULT_PATH}."
        )

