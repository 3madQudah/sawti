"""Memory-rule endpoint: what the system has learned from reviews.

Phase 6.2. Read-only. The container e2e polls it to know when the
asynchronous induction after a review has finished; the 6.4 dashboard can
show it.
"""

from __future__ import annotations

from fastapi import APIRouter

from sawti.api.deps import CurrentReviewer, DbSession
from sawti.api.schemas import MemoryRuleListResponse, MemoryRuleView
from sawti.db.models import MemoryRuleRecord

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/rules", response_model=MemoryRuleListResponse)
def list_rules(session: DbSession, _: CurrentReviewer) -> MemoryRuleListResponse:
    """Active (non-retired) rules, newest first."""
    rows = (
        session.query(MemoryRuleRecord)
        .filter_by(retired=False)
        .order_by(MemoryRuleRecord.created_at.desc())
        .all()
    )
    return MemoryRuleListResponse(
        rules=[
            MemoryRuleView(
                id=row.id,
                text=row.rule_text,
                source_corrections=len(row.source_correction_ids),
                created_at=row.created_at,
            )
            for row in rows
        ]
    )
