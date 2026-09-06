"""Claim/commitment/sentiment extraction node — first stage of the analysis graph.

Phase 2: agent nodes.
"""

from __future__ import annotations

from sawti.agent.state import AgentState


async def extract(state: AgentState) -> AgentState:
    """Extract candidate claims, commitments, and sentiment points from the transcript.

    Args:
        state: Current agent state; must contain a redacted transcript.

    Returns:
        Updated state with raw (pre-grounding-check) extraction results.

    Raises:
        NotImplementedError: Until extraction is implemented.
    """
    # TODO(phase-2): call LLMProvider.structured_complete against redacted_transcript,
    # informed by any retrieved sawti.memory rules for this call's language/context.
    raise NotImplementedError
