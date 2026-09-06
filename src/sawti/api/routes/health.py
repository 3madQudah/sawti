"""Liveness/readiness endpoints.

Phase 3: API layer.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health() -> dict[str, str]:
    """Report basic service liveness.

    Returns:
        A minimal status payload.
    """
    return {"status": "ok"}
