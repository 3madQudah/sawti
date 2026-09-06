"""Correction submission endpoints for QA reviewers.

Phase 3: API layer.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.post("/{call_id}/corrections")
async def submit_correction(call_id: str) -> None:
    """Submit a QA reviewer's `Correction` for a call analysis.

    Args:
        call_id: Identifier of the call being corrected.

    Raises:
        NotImplementedError: Until the endpoint is implemented.
    """
    # TODO(phase-3): validate reviewer_id/timestamp, persist via sawti.memory.capture,
    # and resume any interrupted graph run for this call.
    raise NotImplementedError
