"""Human-review escalation node — interrupts the graph for supervisor input.

Phase 2: agent nodes.
"""

from __future__ import annotations

from sawti.agent.state import AgentState


async def escalate(state: AgentState) -> AgentState:
    """Pause the graph and hand the call off for human review.

    Uses LangGraph's `interrupt()` so the graph suspends until a QA
    reviewer submits a `Correction` via the review API.

    Args:
        state: Current agent state, flagged as requiring human review.

    Returns:
        Updated state once the human review resumes execution.

    Raises:
        NotImplementedError: Until escalation is implemented.
    """
    # TODO(phase-2): call langgraph's interrupt() and resume with reviewer input.
    raise NotImplementedError
