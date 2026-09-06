"""Phase 1 experiment: baseline agent performance with no memory.

Phase 1: baseline experiment.
"""

from __future__ import annotations

from pathlib import Path


def run_baseline(ground_truth_path: Path, *, output_path: Path) -> None:
    """Run the phase 1 baseline evaluation (no memory system active).

    Args:
        ground_truth_path: Path to reviewed ground-truth records.
        output_path: Where to write the results table.

    Raises:
        NotImplementedError: Until the baseline experiment is implemented.
    """
    # TODO(phase-1): invoke sawti.eval.runner.run_eval with memory retrieval disabled.
    raise NotImplementedError
