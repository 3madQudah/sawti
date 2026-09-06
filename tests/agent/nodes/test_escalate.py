"""Tests for `sawti.agent.nodes.escalate`.

Phase 2: mirrors `src/sawti/agent/nodes/escalate.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 2")
def test_escalate_interrupts_graph_execution() -> None:
    """escalate() suspends the graph via LangGraph's interrupt() rather than returning a verdict."""


@pytest.mark.skip(reason="phase 2")
def test_escalate_resumes_with_reviewer_correction() -> None:
    """escalate() resumes execution using the reviewer's submitted Correction."""
