"""QA rubric scoring node.

Phase 2: agent nodes.
"""

from __future__ import annotations

from sawti.agent.state import AgentState


async def score(state: AgentState) -> AgentState:
    """Score the call against the QA rubric, producing grounded `RubricScore` entries.

    Args:
        state: Current agent state, containing grounded claims and the transcript.

    Returns:
        Updated state with rubric scores populated.

    Raises:
        NotImplementedError: Until rubric scoring is implemented.
    """
    # TODO(phase-2): apply the QA rubric via LLMProvider, requiring evidence per RubricScore.
    raise NotImplementedError
