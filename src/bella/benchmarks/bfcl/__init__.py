"""
BFCL (Berkeley Function Call Leaderboard) benchmark implementation.

Registers itself as ``@register_benchmark("bfcl")``.  Importing this
package also triggers per-category adapter registration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from bella.benchmarks import register_benchmark
from bella.benchmarks.base import Benchmark, InferAdapter

# Import adapters to trigger @register_adapter decorators
import bella.benchmarks.bfcl.adapters  # noqa: F401


@register_benchmark("bfcl")
class BFCLBenchmark(Benchmark):
    """BFCL benchmark: function-call evaluation with single/multi-turn categories."""

    @property
    def name(self) -> str:
        return "bfcl"

    def list_categories(self) -> List[str]:
        from bella.benchmarks.bfcl.resources import load_bfcl_categories
        return list(load_bfcl_categories().keys())

    def load_dataset(self, category: str, limit: int = 0) -> List[Dict[str, Any]]:
        from bella.benchmarks.bfcl.resources import load_bella_dataset
        entries = load_bella_dataset(category)
        if limit > 0:
            entries = entries[:limit]
        return entries

    def create_adapter(self, category: str) -> InferAdapter:
        from bella.benchmarks.bfcl.adapters.base import get_adapter
        return get_adapter(category)

    def result_file(self, category: str) -> Path:
        from bella.config import load_settings
        from bella.infer.writer import result_file_path
        from bfcl_eval.constants.eval_config import RESULT_PATH

        settings = load_settings()
        from bella.benchmarks.bfcl.adapters.base import get_adapter
        adapter = get_adapter(category)

        return result_file_path(
            registry_name=settings.bfcl_registry_name,
            result_root=Path(RESULT_PATH),
            group=adapter.result_group(category),
            filename=adapter.result_filename(category),
        )

    def evaluate(self, category: str, **kwargs: Any) -> None:
        from bella.config import load_settings
        from bella.benchmarks.bfcl.compat import ensure_bfcl_model_alias

        partial_eval: bool = kwargs.get("partial_eval", True)
        settings = load_settings()
        ensure_bfcl_model_alias(settings.bfcl_registry_name)

        from bfcl_eval.constants.eval_config import SCORE_PATH
        from bfcl_eval.eval_checker import eval_runner
        from bfcl_eval.utils import find_file_by_category

        isolated_score_dir = (
            Path(SCORE_PATH)
            / "__bella_isolated_eval__"
            / settings.bfcl_registry_name.replace("/", "_")
        )
        isolated_score_dir.mkdir(parents=True, exist_ok=True)

        eval_runner.main(
            model=[settings.bfcl_registry_name],
            test_categories=[category],
            result_dir=None,
            score_dir=isolated_score_dir,
            partial_eval=partial_eval,
        )

        try:
            score_file = find_file_by_category(
                category,
                isolated_score_dir / settings.bfcl_registry_name.replace("/", "_"),
                is_score_file=True,
            )
            print(f"[Bella] BFCL evaluation done. Category='{category}'.")
            print(f"[Bella] Score file: {score_file}")
        except FileNotFoundError:
            print(
                f"[Bella] BFCL evaluation finished for category='{category}', "
                f"but score file could not be located."
            )
