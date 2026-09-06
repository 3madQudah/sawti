"""Tests for `sawti.agent.graph`.

Phase 2: mirrors `src/sawti/agent/graph.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 2")
def test_build_graph_wires_extract_through_confidence_in_order() -> None:
    """build_graph() connects extract -> ground -> score -> compliance -> confidence in sequence."""


@pytest.mark.skip(reason="phase 2")
def test_route_after_confidence_returns_escalate_when_review_required() -> None:
    """route_after_confidence() routes to 'escalate' when requires_human_review is True."""


@pytest.mark.skip(reason="phase 2")
def test_route_after_confidence_ends_graph_when_review_not_required() -> None:
    """route_after_confidence() ends the graph (no escalation) when requires_human_review is False."""


@pytest.mark.skip(reason="phase 2")
def test_low_confidence_run_never_reaches_an_automated_verdict() -> None:
    """A run whose confidence node sets requires_human_review=True never exits via the automated path."""
