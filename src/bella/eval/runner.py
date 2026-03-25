"""
Benchmark-agnostic evaluation runner.

Replaces the old BFCL-specific ``bfcl_eval``.  Works with any
``Benchmark`` registered via ``@register_benchmark``.
"""
from __future__ import annotations

import argparse

from bella.benchmarks import get_benchmark
from bella.config import load_settings


def run_eval(
    benchmark_name: str,
    category: str,
    **kwargs,
) -> None:
    """Run evaluation for *category* under *benchmark_name*."""
    load_settings()
    benchmark = get_benchmark(benchmark_name)
    benchmark.evaluate(category, **kwargs)


# ── backward-compatible wrapper ──────────────────────────────────────

def run_bfcl_eval(category: str = "simple_python", partial_eval: bool = True) -> None:
    """Legacy wrapper — delegates to ``run_eval("bfcl", ...)``."""
    run_eval("bfcl", category, partial_eval=partial_eval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark evaluation via Bella.")
    parser.add_argument("--benchmark", type=str, default="bfcl", help="Benchmark name (default: bfcl).")
    parser.add_argument("--category", type=str, default="simple_python", help="Category/subset to evaluate.")
    parser.add_argument("--no-partial-eval", action="store_true", help="Disable partial evaluation.")
    args = parser.parse_args()
    run_eval(args.benchmark, args.category, partial_eval=not args.no_partial_eval)
