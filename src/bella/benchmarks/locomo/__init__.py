"""
LoCoMo (Long-term Conversational Memory) benchmark implementation.

Evaluates LLM memory over very long multi-session dialogues with the
Question Answering task from the LoCoMo benchmark (ACL 2024).

QA sub-categories:
  qa              – all questions combined
  qa_single_hop   – category 4: direct recall
  qa_multi_hop    – category 1: reasoning across sessions
  qa_temporal     – category 2: time-related questions
  qa_open_domain  – category 3: commonsense inference
  qa_adversarial  – category 5: questions with no answer in context

Dataset: Place ``locomo10.json`` at ``datasets/locomo/locomo10.json`` or set
``BELLA_LOCOMO_DATA_FILE`` to the path.  Download from:
  https://github.com/snap-research/locomo/blob/main/data/locomo10.json

Reference: https://snap-research.github.io/locomo/
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from bella.benchmarks import register_benchmark
from bella.benchmarks.base import Benchmark, InferAdapter
from bella.benchmarks.locomo.evaluation import (
    CATEGORY_NAMES,
    aggregate_scores,
    score_qa,
)

_CATEGORY_NAME_TO_INT: Dict[str, int] = {v: k for k, v in CATEGORY_NAMES.items()}

_VALID_CATEGORIES = {"qa"} | {f"qa_{name}" for name in CATEGORY_NAMES.values()}


def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _default_data_file() -> str:
    return os.getenv("BELLA_LOCOMO_DATA_FILE") or str(
        _find_project_root() / "datasets" / "locomo" / "locomo10.json"
    )


def _load_raw_data(data_file: str | None = None) -> List[Dict[str, Any]]:
    path = data_file or _default_data_file()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"LoCoMo data file not found: {path}\n"
            "Download locomo10.json from "
            "https://github.com/snap-research/locomo/blob/main/data/locomo10.json "
            "and place it at datasets/locomo/locomo10.json"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _flatten_qa_entries(
    raw_data: List[Dict[str, Any]],
    category_filter: int | None = None,
) -> List[Dict[str, Any]]:
    """Flatten LoCoMo samples × QA pairs into per-question entries."""
    entries: List[Dict[str, Any]] = []
    for sample in raw_data:
        sample_id = sample["sample_id"]
        conversation = sample["conversation"]
        for idx, qa in enumerate(sample.get("qa", [])):
            cat = qa.get("category", 0)
            if category_filter is not None and cat != category_filter:
                continue
            entries.append({
                "id": f"{sample_id}_qa_{idx}",
                "sample_id": sample_id,
                "conversation": conversation,
                "question": qa["question"],
                "answer": str(qa.get("answer", qa.get("adversarial_answer", ""))),
                "category": cat,
                "evidence": qa.get("evidence", []),
            })
    return entries


@register_benchmark("locomo")
class LoCoMoBenchmark(Benchmark):
    """LoCoMo benchmark: long-term conversational memory QA evaluation."""

    @property
    def name(self) -> str:
        return "locomo"

    def list_categories(self) -> List[str]:
        return sorted(_VALID_CATEGORIES)

    def load_dataset(self, category: str, limit: int = 0) -> List[Dict[str, Any]]:
        if category not in _VALID_CATEGORIES:
            raise ValueError(
                f"Unknown LoCoMo category: {category!r}. "
                f"Available: {sorted(_VALID_CATEGORIES)}"
            )

        cat_filter: int | None = None
        if category != "qa":
            suffix = category.removeprefix("qa_")
            cat_filter = _CATEGORY_NAME_TO_INT.get(suffix)
            if cat_filter is None:
                raise ValueError(f"Cannot map category {category!r} to LoCoMo int")

        raw = _load_raw_data()
        entries = _flatten_qa_entries(raw, category_filter=cat_filter)

        if limit > 0:
            entries = entries[:limit]
        return entries

    def create_adapter(self, category: str) -> InferAdapter:
        from bella.benchmarks.locomo.adapter import LoCoMoQAAdapter
        return LoCoMoQAAdapter()

    def result_file(self, category: str) -> Path:
        root = _find_project_root()
        out_dir = root / "outputs" / "locomo" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"locomo_{category}_result.jsonl"

    def evaluate(self, category: str, **kwargs: Any) -> None:
        result_path = self.result_file(category)
        if not result_path.exists():
            print(f"[Bella] No result file found: {result_path}")
            return

        scored: List[Dict[str, Any]] = []
        with open(result_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                result_list = entry.get("result", [])
                prediction = (
                    result_list[0].get("prediction", "") if result_list else ""
                )
                gold = entry.get("gold_answer", "")
                cat = entry.get("category", 0)

                sc = score_qa(prediction, gold, cat)
                scored.append({"id": entry["id"], "category": cat, "score": sc})

        if not scored:
            print("[Bella] No entries to evaluate.")
            return

        cat_scores, overall = aggregate_scores(scored)

        print(f"\n[Bella] LoCoMo evaluation: {category}")
        print(f"{'Category':<20} {'Score':>8} {'Count':>8}")
        print("-" * 38)

        cat_counts: Dict[int, int] = {}
        for s in scored:
            cat_counts[s["category"]] = cat_counts.get(s["category"], 0) + 1

        for cat_name, score_val in cat_scores.items():
            cat_int = _CATEGORY_NAME_TO_INT.get(cat_name, -1)
            count = cat_counts.get(cat_int, 0)
            print(f"{cat_name:<20} {score_val:>8.3f} {count:>8}")

        print("-" * 38)
        print(f"{'Overall':<20} {overall:>8.3f} {len(scored):>8}")

        score_dir = result_path.parent.parent / "scores"
        score_dir.mkdir(parents=True, exist_ok=True)
        score_file = score_dir / f"locomo_{category}_score.json"
        with open(score_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "category": category,
                    "overall": round(overall, 4),
                    "per_category": {k: round(v, 4) for k, v in cat_scores.items()},
                    "total_entries": len(scored),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[Bella] Score file: {score_file}\n")
