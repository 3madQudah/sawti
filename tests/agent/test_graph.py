"""Tests for `sawti.agent.graph`.

Phase 2, stage 1: mirrors `src/sawti/agent/graph.py`.

Stage 1 has no business logic to test, so these tests are about *topology* and
*routing* — the two things that must be right before any node does real work.
The failure paths matter most here: a run that loses its confidence keys, or
one that errored, must not slip out through the auto-pass exit.
"""

from __future__ import annotations

from itertools import count, pairwise

import pytest
from langgraph.graph import END

from sawti.agent.graph import (
    NODE_COMPLIANCE,
    NODE_CONFIDENCE,
    NODE_ESCALATE,
    PIPELINE,
    build_graph,
    route_after_confidence,
)
from sawti.agent.nodes import extract as extract_module
from sawti.agent.state import AgentState
from sawti.schemas import Commitment, ExtractionProposal, Quote

TRANSCRIPT = "Agent: I will refund you by Sunday.\nCustomer: Thank you.\n"


class _StubProvider:
    """Provider stub returning one correctly-located commitment, or raising."""

    def __init__(self, error: Exception | None = None, *, grounded: bool = True) -> None:
        """Store the error this stub should raise, and whether its offsets are honest."""
        self.error = error
        self.grounded = grounded

    async def structured_complete(self, prompt: str, **kwargs: object) -> ExtractionProposal:
        """Return a proposal whose quote offsets are computed from the prompt itself."""
        if self.error is not None:
            raise self.error
        # grounded=False mimics a real unsupported claim: the model paraphrases
        # rather than copying, so the text is nowhere in the transcript and
        # `ground` drops it. (Merely wrong offsets are repaired, not rejected.)
        text = "I will refund you by Sunday." if self.grounded else "I will refund you today."
        start = prompt.index(text) if self.grounded else 0
        quote = Quote(text=text, speaker="Agent", start_char=start, end_char=start + len(text))
        return ExtractionProposal(
            summary="Refund promised.",
            commitments=[Commitment(evidence=quote, promised_by="Agent", description="Refund.")],
        )


@pytest.fixture
def stub_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the extraction node's provider so graph tests never hit the network."""
    monkeypatch.setattr(extract_module, "get_llm_provider", lambda: _StubProvider())


# A run that escalates suspends inside `escalate` rather than completing it, so
# the final streamed chunk is LangGraph's interrupt marker, not the node name.
INTERRUPTED = "__interrupt__"

_thread_counter = count()


def _config() -> dict[str, dict[str, str]]:
    """A fresh thread id per call — the graph is checkpointed, so runs must not share one."""
    return {"configurable": {"thread_id": f"test-{next(_thread_counter)}"}}


async def _executed_nodes(state: AgentState) -> list[str]:
    """Run the graph and return the node names that actually executed, in order.

    The final state cannot tell us which branch was taken — a suspended run has
    not written its escalation keys yet. Streaming updates can: each chunk is
    keyed by the node that produced it, or by INTERRUPTED when the run suspends.
    """
    graph = build_graph()
    return [
        next(iter(chunk)) async for chunk in graph.astream(state, _config(), stream_mode="updates")
    ]


def test_build_graph_wires_extract_through_confidence_in_order() -> None:
    """build_graph() connects extract -> ground -> score -> compliance -> confidence in sequence."""
    drawable = build_graph().get_graph()
    unconditional = {(e.source, e.target) for e in drawable.edges if not e.conditional}

    assert ("__start__", PIPELINE[0]) in unconditional
    for source, target in pairwise(PIPELINE):
        assert (source, target) in unconditional, f"missing edge {source} -> {target}"


def test_route_after_confidence_returns_escalate_when_review_required() -> None:
    """route_after_confidence() routes to 'escalate' when requires_human_review is True."""
    assert route_after_confidence({"requires_human_review": True}) == NODE_ESCALATE


def test_route_after_confidence_ends_graph_when_review_not_required() -> None:
    """route_after_confidence() ends the graph (no escalation) when requires_human_review is False."""
    assert route_after_confidence({"requires_human_review": False}) == END


def test_route_after_confidence_escalates_when_the_flag_is_missing() -> None:
    """Failure path: absent routing information is treated as low confidence, not as a pass.

    If the confidence node crashed before writing its keys, the safe default is
    a human, never an automated verdict.
    """
    assert route_after_confidence({}) == NODE_ESCALATE


def test_route_after_confidence_escalates_on_a_recorded_error() -> None:
    """Failure path: a node that recorded an error escalates even if the flag says pass."""
    state: AgentState = {"requires_human_review": False, "error": "extraction timed out"}
    assert route_after_confidence(state) == NODE_ESCALATE


async def test_graph_runs_end_to_end_through_every_pipeline_node(stub_provider: None) -> None:
    """A confident run executes START -> ... -> confidence and exits without escalating."""
    executed = await _executed_nodes(
        {"call_id": "call-001", "transcript": TRANSCRIPT, "requires_human_review": False}
    )
    assert executed == list(PIPELINE)


async def test_graph_carries_grounded_claims_through_to_the_end(stub_provider: None) -> None:
    """End to end, a well-located quote survives extraction and grounding."""
    result = await build_graph().ainvoke(
        {"call_id": "call-001", "transcript": TRANSCRIPT, "requires_human_review": False}, _config()
    )

    assert len(result["commitments"]) == 1
    assert result["rejected_claims"] == []
    assert result["grounding_coverage"] == 1.0


async def test_extraction_failure_routes_the_run_to_a_human(monkeypatch: pytest.MonkeyPatch) -> None:
    """Failure path: a provider error sets `error`, and the graph escalates rather than passing."""
    monkeypatch.setattr(
        extract_module, "get_llm_provider", lambda: _StubProvider(error=RuntimeError("no key"))
    )

    executed = await _executed_nodes(
        {"call_id": "call-001", "transcript": TRANSCRIPT, "requires_human_review": False}
    )

    assert executed[-1] == INTERRUPTED


async def test_low_confidence_run_never_reaches_an_automated_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run whose confidence node sets requires_human_review=True never exits via the automated path.

    Driven through the real mechanism rather than a seeded flag: the model
    paraphrases instead of copying, `ground` drops the claim, coverage falls to
    0.0, and the conditional edge sends the run to a human.
    """
    monkeypatch.setattr(extract_module, "get_llm_provider", lambda: _StubProvider(grounded=False))

    executed = await _executed_nodes({"call_id": "call-001", "transcript": TRANSCRIPT})

    assert executed == [*PIPELINE, INTERRUPTED]
    assert executed[-1] == INTERRUPTED


async def test_unsupported_claims_are_dropped_and_drive_the_escalation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole phase 2 story, end to end, in one assertion chain.

    Paraphrased quote -> claim dropped, not down-weighted -> coverage 0.0 ->
    confidence 0.0 -> requires_human_review. No automated verdict is produced.
    """
    monkeypatch.setattr(extract_module, "get_llm_provider", lambda: _StubProvider(grounded=False))

    result = await build_graph().ainvoke({"call_id": "call-001", "transcript": TRANSCRIPT}, _config())

    assert result["commitments"] == []
    assert len(result["rejected_claims"]) == 1
    assert result["grounding_coverage"] == 0.0
    assert result["confidence"] == 0.0
    assert result["requires_human_review"] is True
    # `confidence` set this; `escalate` has suspended and not yet overwritten it.
    assert result["review_status"] == "awaiting_human"


async def test_fully_grounded_run_auto_passes(stub_provider: None) -> None:
    """The other side: every quote verifies, so the run reaches a verdict without a human."""
    result = await build_graph().ainvoke({"call_id": "call-001", "transcript": TRANSCRIPT}, _config())

    assert result["confidence"] == 1.0
    assert result["requires_human_review"] is False
    assert result["review_status"] == "auto_passed"


def test_escalate_is_reachable_only_from_the_conditional_edge() -> None:
    """Escalation is a branch of the confidence node, not a step everything walks through."""
    drawable = build_graph().get_graph()
    into_escalate = [e for e in drawable.edges if e.target == NODE_ESCALATE]

    assert len(into_escalate) == 1
    assert into_escalate[0].source == NODE_CONFIDENCE
    assert into_escalate[0].conditional is True
    # ...and compliance feeds confidence unconditionally, so the branch is the only exit.
    assert (NODE_COMPLIANCE, NODE_CONFIDENCE) in {
        (e.source, e.target) for e in drawable.edges if not e.conditional
    }
