"""Tests for `sawti.eval.runner`.

Phase 1: mirrors `src/sawti/eval/runner.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 1")
def test_run_eval_writes_a_dated_round_to_output_path() -> None:
    """run_eval() appends a new dated round table to output_path."""


@pytest.mark.skip(reason="phase 1")
def test_run_eval_never_writes_a_blended_score_across_languages() -> None:
    """run_eval()'s written results table has no single blended-accuracy column."""


@pytest.mark.skip(reason="phase 1")
def test_run_eval_returns_metric_name_to_per_category_mapping() -> None:
    """run_eval() returns dict[str, dict[Language, float]]."""
