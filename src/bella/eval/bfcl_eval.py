from __future__ import annotations

from bella.config import load_settings
from bella.utils.bfcl_compat import ensure_bfcl_model_alias


def run_bfcl_eval(category: str = "simple_python", partial_eval: bool = True) -> None:
    """
    Run BFCL official evaluator for the given category.

    This assumes compatible *_result.json files already exist under RESULT_PATH
    for the configured registry/model name.
    """
    settings = load_settings()

    # 为自定义 registry_name（如 bella-mvp）创建 BFCL 兼容的 model alias，
    # 以便官方 evaluator 能识别该模型名并加载对应 handler 配置。
    ensure_bfcl_model_alias(settings.bfcl_registry_name)

    # 延迟导入 BFCL 相关模块，确保 BFCL_PROJECT_ROOT 已由 load_settings 写入环境变量。
    from bfcl_eval.constants.eval_config import RESULT_PATH, SCORE_PATH
    from bfcl_eval.eval_checker import eval_runner
    from bfcl_eval.utils import find_file_by_category

    # Run official evaluator. We use partial_eval=True because Bella only
    # generates a subset of test entries in the MVP.
    eval_runner.main(
        model=[settings.bfcl_registry_name],
        test_categories=[category],
        result_dir=None,
        score_dir=None,
        partial_eval=partial_eval,
    )

    # Try to locate the score file for this category for user reference.
    try:
        score_file = find_file_by_category(
            category,
            SCORE_PATH / settings.bfcl_registry_name.replace("/", "_"),
            is_score_file=True,
        )
        print(f"[Bella] BFCL evaluation done. Category='{category}'.")
        print(f"[Bella] Score file: {score_file}")
    except FileNotFoundError:
        print(
            f"[Bella] BFCL evaluation finished for category='{category}', "
            f"but score file could not be located under SCORE_PATH={SCORE_PATH}."
        )

