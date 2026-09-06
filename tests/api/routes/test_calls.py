"""Tests for `sawti.api.routes.calls`.

Phase 3: mirrors `src/sawti/api/routes/calls.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 3")
def test_submit_call_enqueues_analysis_task() -> None:
    """POST /calls/ enqueues an analyze_call_task and returns the new call id."""


@pytest.mark.skip(reason="phase 3")
def test_get_call_analysis_returns_404_for_unknown_call() -> None:
    """GET /calls/{call_id} returns 404 when no analysis exists for call_id."""


@pytest.mark.skip(reason="phase 3")
def test_get_call_analysis_returns_validated_call_analysis() -> None:
    """GET /calls/{call_id} returns a payload that validates as CallAnalysis."""
