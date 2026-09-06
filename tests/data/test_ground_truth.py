"""Tests for `sawti.data.ground_truth`.

Phase 1: mirrors `src/sawti/data/ground_truth.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 1")
def test_load_ground_truth_returns_validated_call_analyses() -> None:
    """load_ground_truth() returns a list of valid CallAnalysis instances."""


@pytest.mark.skip(reason="phase 1")
def test_load_ground_truth_raises_on_malformed_record() -> None:
    """load_ground_truth() raises when a record fails CallAnalysis validation."""


@pytest.mark.skip(reason="phase 1")
def test_validate_ground_truth_rejects_duplicate_call_ids() -> None:
    """validate_ground_truth() raises when two records share the same call_id."""
