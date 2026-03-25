"""
LoCoMo QA evaluation metrics.

Ported from the official LoCoMo evaluation code
(https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py)
to work without external dependencies beyond the standard library.

QA categories:
  1 = multi_hop    – F1 with sub-answer splitting
  2 = temporal     – F1
  3 = open_domain  – F1 (commonsense)
  4 = single_hop   – F1
  5 = adversarial  – binary (checks for "no information" / "not mentioned")
"""
from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple

CATEGORY_NAMES: Dict[int, str] = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}


def normalize_answer(s: str) -> str:
    """Lower text, remove punctuation/articles, fix whitespace."""
    s = s.replace(",", "")
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())


def _token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def _multi_answer_f1(prediction: str, ground_truth: str) -> float:
    """Split by comma for multi-answer, compute partial F1 per sub-answer."""
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    if not ground_truths:
        return 0.0
    scores = []
    for gt in ground_truths:
        best = max((_token_f1(p, gt) for p in predictions), default=0.0)
        scores.append(best)
    return sum(scores) / len(scores)


def score_qa(prediction: str, answer: str, category: int) -> float:
    """Score a single QA pair according to its category.

    Returns a float in [0, 1].
    """
    if category == 5:
        pred_lower = prediction.strip().lower()
        if "no information available" in pred_lower or "not mentioned" in pred_lower:
            return 1.0
        return 0.0

    if category == 3:
        answer = answer.split(";")[0].strip()

    if category == 1:
        return _multi_answer_f1(prediction, answer)

    return _token_f1(prediction, answer)


def aggregate_scores(
    results: List[Dict],
) -> Tuple[Dict[str, float], float]:
    """Aggregate per-entry scores into per-category and overall accuracy.

    Each *result* dict must have ``"category"`` (int) and ``"score"`` (float).

    Returns ``(category_scores, overall_score)`` where ``category_scores``
    maps category name → mean score.
    """
    by_cat: Dict[int, List[float]] = {}
    for r in results:
        cat = r["category"]
        by_cat.setdefault(cat, []).append(r["score"])

    cat_scores: Dict[str, float] = {}
    all_scores: List[float] = []
    for cat in sorted(by_cat):
        scores = by_cat[cat]
        name = CATEGORY_NAMES.get(cat, f"cat_{cat}")
        cat_scores[name] = sum(scores) / len(scores) if scores else 0.0
        all_scores.extend(scores)

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return cat_scores, overall
