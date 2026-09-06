"""Tests for `sawti.eval.experiments.five_batch`.

Phase 4: mirrors `src/sawti/eval/experiments/five_batch.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 4")
def test_run_five_batch_compares_memory_on_vs_off_per_batch() -> None:
    """run_five_batch() runs each of the five batches with memory both enabled and disabled."""


@pytest.mark.skip(reason="phase 4")
def test_run_five_batch_reports_deltas_per_language_category() -> None:
    """run_five_batch() never collapses the memory-vs-no-memory delta across languages."""
