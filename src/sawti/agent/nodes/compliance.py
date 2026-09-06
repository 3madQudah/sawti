"""Compliance rule-checking node.

Phase 2: agent nodes.
"""

from __future__ import annotations

from sawti.agent.state import AgentState


async def compliance(state: AgentState) -> AgentState:
    """Check the call against compliance rules, producing grounded `ComplianceFlag` entries.

    Args:
        state: Current agent state, containing grounded claims and the transcript.

    Returns:
        Updated state with compliance flags populated.

    Raises:
        NotImplementedError: Until compliance checking is implemented.
    """
    # TODO(phase-2): evaluate the compliance rule set against transcript/claims via LLMProvider.
    raise NotImplementedError
