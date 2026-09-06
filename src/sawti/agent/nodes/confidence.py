"""Confidence estimation node — determines automated-verdict vs. human-review routing.

Phase 2: agent nodes.
"""

from __future__ import annotations

from sawti.agent.state import AgentState


async def confidence(state: AgentState) -> AgentState:
    """Compute an overall confidence score for the call analysis so far.

    Per project rule, low-confidence results must route to human review,
    never to an automated verdict — this node sets
    `state["requires_human_review"]` accordingly for the graph's
    conditional edge to act on.

    Args:
        state: Current agent state, containing scores and flags.

    Returns:
        Updated state with `confidence` and `requires_human_review` set.

    Raises:
        NotImplementedError: Until confidence estimation is implemented.
    """
    # TODO(phase-2): derive confidence from score coverage/agreement; compare against
    # sawti.config.get_settings().confidence_threshold to set requires_human_review.
    raise NotImplementedError
