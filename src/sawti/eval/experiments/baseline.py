"""Phase 1 experiment: baseline agent performance with no memory.

Phase 1: baseline experiment.
"""

from __future__ import annotations

from pathlib import Path

from sawti.eval.metrics import DEFAULT_SYNTHETIC_DIR
from sawti.eval.runner import run_eval
from sawti.schemas import Language

DEFAULT_GROUND_TRUTH_DIR = Path("data/ground_truth")
DEFAULT_OUTPUT_PATH = Path("eval_results.md")


def run_baseline(
    ground_truth_path: Path = DEFAULT_GROUND_TRUTH_DIR,
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    synthetic_dir: Path = DEFAULT_SYNTHETIC_DIR,
) -> dict[str, dict[Language, float]]:
    """Run the phase 1 baseline evaluation (no memory system active).

    The memory system is phase 4 and not yet implemented, so "no memory" is
    currently the only configuration there is: `run_plain_extraction` calls
    the provider directly and consults no rule store. When memory lands, this
    is the experiment that must keep running without it, as the control.

    Args:
        ground_truth_path: Directory of reference-label records.
        output_path: Where to append the results table.
        synthetic_dir: Directory of `<call_id>.txt` transcripts.

    Returns:
        A mapping from metric name to its per-`Language` category scores.
    """
    return run_eval(
        ground_truth_path,
        output_path=output_path,
        synthetic_dir=synthetic_dir,
    )
