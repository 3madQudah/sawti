"""Assembly of the LangGraph analysis graph: node wiring and conditional edges.

Phase 2: agent graph assembly.
"""

from __future__ import annotations

from langgraph.graph import StateGraph

from sawti.agent.state import AgentState


def build_graph() -> StateGraph:
    """Construct the compiled LangGraph analysis graph.

    Wires extract -> ground -> score -> compliance -> confidence, with a
    conditional edge routing low-confidence results to `escalate` (human
    review) instead of an automated verdict.

    Returns:
        The compiled graph, ready to `.invoke()` or `.stream()`.

    Raises:
        NotImplementedError: Until node wiring is implemented.
    """
    # TODO(phase-2): add_node for extract/ground/score/compliance/confidence/escalate,
    # add conditional_edges from confidence based on AgentState["requires_human_review"].
    raise NotImplementedError


def route_after_confidence(state: AgentState) -> str:
    """Conditional edge: decide whether to end the graph or escalate to human review.

    Args:
        state: Current agent state, expected to contain `requires_human_review`.

    Returns:
        The name of the next node ("escalate" or the graph end).

    Raises:
        NotImplementedError: Until routing logic is implemented.
    """
    # TODO(phase-2): return "escalate" when state["requires_human_review"] is True, else end.
    raise NotImplementedError
