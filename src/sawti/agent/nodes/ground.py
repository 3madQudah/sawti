"""Quote verification node — rejects any claim whose evidence isn't a verbatim transcript match.

Phase 2: agent nodes.
"""

from __future__ import annotations

from sawti.agent.state import AgentState


async def ground(state: AgentState) -> AgentState:
    """Verify every extracted claim's evidence quote against the source transcript.

    Any claim whose `evidence.text` does not appear verbatim at
    `evidence.start_char:evidence.end_char` in the transcript is dropped,
    not downgraded — an unsupported claim never reaches scoring.

    Args:
        state: Current agent state, containing raw extraction output.

    Returns:
        Updated state with only grounded claims retained.

    Raises:
        NotImplementedError: Until grounding verification is implemented.
    """
    # TODO(phase-2): verify each claim's evidence offsets/text against state["transcript"];
    # drop claims that fail verification rather than lowering their confidence.
    raise NotImplementedError
