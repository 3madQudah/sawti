"""Tests for `sawti.agent.nodes.escalate`.

Phase 2, stage 4: mirrors `src/sawti/agent/nodes/escalate.py`.

Escalation is tested through the compiled graph rather than by calling the node
directly: `interrupt()` only means anything inside a checkpointed run, so a
direct call would test nothing real.
"""

from __future__ import annotations

from itertools import count

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from sawti.agent.graph import build_graph
from sawti.agent.nodes import extract as extract_module
from sawti.schemas import Commitment, ExtractionProposal, Quote

TRANSCRIPT = "Agent: I will refund you by Sunday.\nCustomer: Thank you.\n"

_thread_counter = count()


def _config() -> dict[str, dict[str, str]]:
    """A fresh thread id per run — resuming depends on addressing the right thread."""
    return {"configurable": {"thread_id": f"escalate-{next(_thread_counter)}"}}


UNSUPPORTED_TEXT = "I will refund you first thing tomorrow."


class _UngroundedProvider:
    """Returns a paraphrased quote that appears nowhere in the transcript."""

    async def structured_complete(self, prompt: str, **kwargs: object) -> ExtractionProposal:
        """Return one commitment whose evidence is not in the transcript at all."""
        text = UNSUPPORTED_TEXT
        quote = Quote(text=text, speaker="Agent", start_char=0, end_char=len(text))
        return ExtractionProposal(
            summary="Refund promised.",
            commitments=[Commitment(evidence=quote, promised_by="Agent", description="Refund.")],
        )


@pytest.fixture(autouse=True)
def _ungrounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every run in this module down the escalation branch."""
    monkeypatch.setattr(extract_module, "get_llm_provider", lambda: _UngroundedProvider())


async def test_escalate_suspends_the_graph_instead_of_returning_a_verdict() -> None:
    """escalate() suspends the graph via LangGraph's interrupt() rather than returning a verdict."""
    graph = build_graph()
    config = _config()

    await graph.ainvoke({"call_id": "call-001", "transcript": TRANSCRIPT}, config)
    state = await graph.aget_state(config)

    # The run is parked *on* escalate — it has not completed it.
    assert state.next == ("escalate",)
    assert state.values.get("review_status") == "awaiting_human"
    assert "reviewer_input" not in state.values


async def test_escalate_hands_the_reviewer_the_rejected_claims() -> None:
    """The interrupt payload carries why it escalated and what could not be supported."""
    graph = build_graph()
    config = _config()

    await graph.ainvoke({"call_id": "call-001", "transcript": TRANSCRIPT}, config)

    # On langgraph 0.2.x the interrupt payload hangs off the paused task, not the
    # invoke result — `ainvoke` just returns the state as of the suspension.
    state = await graph.aget_state(config)
    payload = state.tasks[0].interrupts[0].value
    assert payload["call_id"] == "call-001"
    assert payload["reason"] == "confidence below threshold"
    assert payload["confidence"] == 0.0
    assert len(payload["rejected_claims"]) == 1
    assert payload["rejected_claims"][0]["evidence"]["text"] == UNSUPPORTED_TEXT


async def test_escalate_resumes_execution_with_the_reviewers_input() -> None:
    """escalate() resumes execution using the reviewer's submitted decision."""
    graph = build_graph()
    config = _config()

    await graph.ainvoke({"call_id": "call-001", "transcript": TRANSCRIPT}, config)
    resumed = await graph.ainvoke(Command(resume={"verdict": "commitment is real"}), config)

    assert resumed["review_status"] == "human_reviewed"
    assert resumed["reviewer_input"] == {"verdict": "commitment is real"}


async def test_resumed_run_is_finished_and_not_rescored() -> None:
    """After a human resumes, the graph ends — it does not loop back through scoring.

    A reviewer's verdict is final; re-running the model over it would let the
    system overrule the human it just escalated to.
    """
    graph = build_graph()
    config = _config()

    await graph.ainvoke({"call_id": "call-001", "transcript": TRANSCRIPT}, config)
    await graph.ainvoke(Command(resume={"verdict": "ok"}), config)
    state = await graph.aget_state(config)

    assert state.next == ()


async def test_interrupt_state_survives_on_an_explicitly_shared_checkpointer() -> None:
    """Failure path for durability: a run resumes only from the checkpoint that holds it.

    Two graph instances sharing one saver can resume each other's threads; a
    graph with its own saver cannot see the thread at all, which is exactly the
    MemorySaver limitation that defers durable review queues to Postgres.
    """
    saver = MemorySaver()
    config = _config()

    await build_graph(checkpointer=saver).ainvoke(
        {"call_id": "call-001", "transcript": TRANSCRIPT}, config
    )
    # A *different* compiled graph, same saver: the paused run is still there.
    resumed = await build_graph(checkpointer=saver).ainvoke(Command(resume={"verdict": "ok"}), config)
    assert resumed["review_status"] == "human_reviewed"

    # A graph with its own saver has never heard of this thread.
    isolated = await build_graph().aget_state(config)
    assert isolated.values == {}
