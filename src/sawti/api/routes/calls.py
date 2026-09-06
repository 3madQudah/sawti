"""Call analysis endpoints: submit calls, fetch analyses.

Phase 3: API layer.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/calls", tags=["calls"])


@router.post("/")
async def submit_call() -> None:
    """Submit a call recording/transcript for analysis.

    Raises:
        NotImplementedError: Until the endpoint is implemented.
    """
    # TODO(phase-3): accept audio/transcript, enqueue analysis via sawti.api.tasks, return call id.
    raise NotImplementedError


@router.get("/{call_id}")
async def get_call_analysis(call_id: str) -> None:
    """Fetch the `CallAnalysis` for a previously submitted call.

    Args:
        call_id: Identifier of the call to fetch.

    Raises:
        NotImplementedError: Until the endpoint is implemented.
    """
    # TODO(phase-3): load CallAnalysisRecord from db and return it as CallAnalysis.
    raise NotImplementedError
