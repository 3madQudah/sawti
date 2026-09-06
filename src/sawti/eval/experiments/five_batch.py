"""Phase 4 experiment: memory vs. no-memory control across five successive batches.

Phase 4: memory evaluation.
"""

from __future__ import annotations

from pathlib import Path


def run_five_batch(ground_truth_path: Path, *, output_path: Path) -> None:
    """Run the phase 4 five-batch memory vs. no-memory control experiment.

    Args:
        ground_truth_path: Path to reviewed ground-truth records, split into five batches.
        output_path: Where to write the results table.

    Raises:
        NotImplementedError: Until the experiment is implemented.
    """
    # TODO(phase-4): run five successive batches with memory on vs. off, compare
    # per-category deltas via sawti.eval.metrics. Never blend languages together.
    raise NotImplementedError
