"""Human-review escalation node — interrupts the graph for supervisor input.

Phase 2, stage 4.

This is the only node that does not return. `interrupt()` raises a control-flow
exception that LangGraph catches: the graph's state is written to the
checkpointer, execution stops, and `ainvoke` returns with an `__interrupt__`
payload describing what the human is being asked. Resuming means invoking the
same graph again with the same `thread_id` and a `Command(resume=...)`; LangGraph
restores the checkpoint and re-runs *this node only*, with `interrupt()` now
returning the reviewer's value instead of raising.

Why the interrupt sits here rather than at the end of the graph: an escalated
call must not produce an automated verdict *at all*. Suspending on the branch
means the auto-pass exit is physically unreachable for a low-confidence run —
there is no "provisional result" that a caller could mistake for a decision.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.types import interrupt

from sawti.agent.state import AgentState

logger = logging.getLogger(__name__)


async def escalate(state: AgentState) -> dict[str, Any]:
    """Pause the graph and hand the call off for human review.

    Role in the graph: the terminal node of the escalation branch. It suspends
    via `interrupt()` and, on resume, records that a human has taken over.

    The payload handed to the reviewer is deliberately small and specific — the
    call id, why it escalated, and the claims that failed grounding — so a
    review queue can render it without reloading the whole analysis.

    Args:
        state: Current agent state, flagged as requiring human review.

    Returns:
        A partial state update marking the call as human-reviewed, carrying
        whatever the reviewer supplied on resume.
    """
    logger.info(
        "escalating %s for human review (confidence %.3f)",
        state.get("call_id"),
        state.get("confidence", 0.0),
    )

    # Execution stops on this line. Everything after it runs only on resume.
    reviewer_input = interrupt(
        {
            "call_id": state.get("call_id"),
            "reason": state.get("error") or "confidence below threshold",
            "confidence": state.get("confidence"),
            "grounding_coverage": state.get("grounding_coverage"),
            "summary": state.get("summary"),
            # What the model claimed but could not support. This is the most
            # useful thing to put in front of a reviewer, and in phase 4 it is
            # what `sawti.memory.induction` learns corrections from.
            "rejected_claims": [claim.model_dump(mode="json") for claim in state.get("rejected_claims", [])],
        }
    )

    return {
        "review_status": "human_reviewed",
        # The reviewer's verdict stands on its own; the graph does not re-score
        # it, which is why `escalate` goes straight to END.
        "reviewer_input": reviewer_input,
    }
