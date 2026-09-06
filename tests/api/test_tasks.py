"""Tests for `sawti.api.tasks`.

Phase 3: mirrors `src/sawti/api/tasks.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 3")
def test_analyze_call_task_redacts_before_running_the_agent_graph() -> None:
    """analyze_call_task() redacts PII before any text reaches the analysis graph."""


@pytest.mark.skip(reason="phase 3")
def test_analyze_call_task_persists_call_analysis_record() -> None:
    """analyze_call_task() writes a CallAnalysisRecord for the analyzed call."""
