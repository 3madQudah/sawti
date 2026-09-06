"""Evaluation run orchestration: load ground truth, run the agent, score per category, write results.

Phase 1: eval runner.
"""

from __future__ import annotations

from pathlib import Path

from sawti.schemas import Language


def run_eval(ground_truth_path: Path, *, output_path: Path) -> dict[str, dict[Language, float]]:
    """Run the full evaluation pipeline and write results to `output_path`.

    Args:
        ground_truth_path: Path to reviewed ground-truth records.
        output_path: Where to write the results table (e.g. `eval_results.md`).

    Returns:
        A mapping from metric name to its per-`Language` category scores.

    Raises:
        NotImplementedError: Until the runner is implemented.
    """
    # TODO(phase-1): load ground truth, run agent graph per call, compute sawti.eval.metrics,
    # and append a dated round to output_path. Never collapse categories into one number.
    raise NotImplementedError
