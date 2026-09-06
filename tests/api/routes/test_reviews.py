"""Tests for `sawti.api.routes.reviews`.

Phase 3: mirrors `src/sawti/api/routes/reviews.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 3")
def test_submit_correction_requires_reviewer_id() -> None:
    """POST /reviews/{call_id}/corrections rejects a submission missing reviewer_id."""


@pytest.mark.skip(reason="phase 3")
def test_submit_correction_resumes_interrupted_graph_run() -> None:
    """POST /reviews/{call_id}/corrections resumes a graph run that was escalated for review."""
