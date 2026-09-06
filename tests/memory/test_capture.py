"""Tests for `sawti.memory.capture`.

Phase 4: mirrors `src/sawti/memory/capture.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 4")
def test_capture_correction_stamps_timestamp_and_reviewer_id() -> None:
    """capture_correction() produces a Correction with timestamp and reviewer_id set."""


@pytest.mark.skip(reason="phase 4")
def test_capture_correction_preserves_original_and_corrected_analyses() -> None:
    """capture_correction() retains both the original and corrected CallAnalysis unmodified."""
