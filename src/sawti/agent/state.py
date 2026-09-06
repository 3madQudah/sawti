"""LangGraph state definition shared by every node in the analysis graph.

Phase 2: agent graph foundation.
"""

from __future__ import annotations

from typing import Any, TypedDict

from sawti.schemas import CallAnalysis, Language


class AgentState(TypedDict, total=False):
    """Mutable state threaded through the LangGraph analysis graph.

    Nodes read and write named keys here; every value that leaves the graph
    as a final result must still be validated into `CallAnalysis` before
    it is returned — this TypedDict is working state, not the output contract.
    """

    call_id: str
    transcript: str
    language: Language
    redacted_transcript: str
    raw_extraction: dict[str, Any]
    analysis: CallAnalysis
    confidence: float
    requires_human_review: bool
    error: str | None
