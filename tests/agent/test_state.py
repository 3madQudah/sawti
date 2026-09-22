"""Tests for `sawti.agent.state`.

Phase 2, stage 1: mirrors `src/sawti/agent/state.py`.

These tests pin down the two properties the rest of the graph relies on:
partial construction (nodes write only their own keys) and the fact that
claim-bearing keys hold validated Pydantic models, not raw dicts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import get_type_hints

from sawti.agent.state import AgentState
from sawti.schemas import CallAnalysis, Commitment, Language, Quote

TRANSCRIPT = "Agent: I will send the refund by Sunday.\nCustomer: Thank you."


def _quote(text: str, speaker: str = "Agent") -> Quote:
    """Build a Quote whose offsets are computed from TRANSCRIPT, never hand-counted."""
    start = TRANSCRIPT.index(text)
    return Quote(text=text, speaker=speaker, start_char=start, end_char=start + len(text))


def _analysis(confidence: float, requires_human_review: bool) -> CallAnalysis:
    """Build a minimal valid CallAnalysis for use as a state value."""
    return CallAnalysis(
        call_id="call-001",
        language=Language.EN,
        summary="Agent promised a refund.",
        confidence=confidence,
        requires_human_review=requires_human_review,
    )


def test_agent_state_accepts_partial_keys() -> None:
    """AgentState, being total=False, can be constructed with only a subset of keys.

    This is what lets a node return a partial update dict without having to know
    about keys owned by other nodes.
    """
    state: AgentState = {"call_id": "call-001"}
    assert state["call_id"] == "call-001"
    assert "confidence" not in state

    # A node's update merges in without disturbing what is already there.
    state["confidence"] = 0.82
    assert state == {"call_id": "call-001", "confidence": 0.82}


def test_agent_state_analysis_key_holds_validated_call_analysis() -> None:
    """AgentState['analysis'], once set, holds a validated CallAnalysis instance."""
    state: AgentState = {"analysis": _analysis(confidence=0.9, requires_human_review=False)}

    analysis = state["analysis"]
    assert isinstance(analysis, CallAnalysis)
    assert analysis.language is Language.EN


def test_claim_bearing_keys_hold_pydantic_models_not_dicts() -> None:
    """Commitments in state are validated `Commitment` objects carrying real evidence."""
    commitment = Commitment(
        evidence=_quote("I will send the refund by Sunday."),
        promised_by="Agent",
        description="Send the refund.",
        deadline=datetime(2026, 9, 27, tzinfo=UTC),
    )
    state: AgentState = {"transcript": TRANSCRIPT, "commitments": [commitment]}

    stored = state["commitments"][0]
    assert isinstance(stored, Commitment)
    # The evidence offsets point at the real span in the transcript.
    assert TRANSCRIPT[stored.evidence.start_char : stored.evidence.end_char] == stored.evidence.text


def test_state_declares_every_key_the_pipeline_needs() -> None:
    """The state carries a slot for each stage's output, so no node invents one later."""
    hints = get_type_hints(AgentState)
    for key in (
        "call_id",
        "transcript",
        "language",
        "redacted_transcript",
        "raw_extraction",
        "commitments",
        "compliance_flags",
        "rubric_scores",
        "rejected_claims",
        "grounding_coverage",
        "confidence",
        "requires_human_review",
        "review_status",
        "analysis",
        "error",
    ):
        assert key in hints, f"AgentState is missing the '{key}' slot"


def test_no_state_key_collides_with_a_graph_node_name() -> None:
    """LangGraph forbids a node named like a state key — keep that invariant tested.

    Failure path for the wiring: this is the constraint that forced the
    confidence node to be named `assess_confidence`. If someone later renames a
    node back to a bare state key, `build_graph()` raises at import-time-ish
    call time; this test says *why* before that happens.
    """
    from sawti.agent.graph import NODE_ESCALATE, PIPELINE

    state_keys = set(get_type_hints(AgentState))
    for node_name in (*PIPELINE, NODE_ESCALATE):
        assert node_name not in state_keys, f"node '{node_name}' collides with a state key"
