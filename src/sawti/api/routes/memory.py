"""Memory endpoints: what the system has learned from reviews, and the conflicts a human must settle.

Phase 6.2: `GET /memory/rules` (the container e2e polls it to know when the
asynchronous induction after a review has finished). Phase 6.4: what each rule
is drawn from and how often it is used, the pending conflicts, and resolving
one. Resolution goes through `sawti.memory.persistence.resolve_conflict` (the
rule lifecycle, under the memory write lock), never SQL here
(docs/09-DECISIONS.md, 2026-10-10).
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, status

from sawti.api.deps import CurrentReviewer, DbSession
from sawti.api.queries import active_rules, analyses_used, correction_languages, rule_languages
from sawti.api.schemas import (
    ConflictingRuleView,
    ConflictResolveRequest,
    ConflictResolveResponse,
    ConflictSourceView,
    MemoryConflictListResponse,
    MemoryConflictView,
    MemoryRuleListResponse,
    MemoryRuleView,
    PendingCounts,
)
from sawti.api.tasks import consolidate_rules_task
from sawti.db.models import CallAnalysisRecord, MemoryConflict, MemoryRuleRecord, ReviewerAction
from sawti.memory.persistence import (
    ConflictAlreadyResolvedError,
    ConflictNotFoundError,
    MemoryStoreBusyError,
    resolve_conflict,
)
from sawti.schemas import Language

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/rules", response_model=MemoryRuleListResponse)
def list_rules(session: DbSession, _: CurrentReviewer) -> MemoryRuleListResponse:
    """Active (non-retired) rules, newest first, with their source languages and usage."""
    rows = active_rules(session)
    languages = rule_languages(session, rows)
    used = analyses_used(session)
    return MemoryRuleListResponse(
        rules=[
            MemoryRuleView(
                id=row.id,
                text=row.rule_text,
                source_corrections=len(row.source_correction_ids),
                languages=languages[row.id],
                analyses_used=used[row.id],
                created_at=row.created_at,
            )
            for row in rows
        ]
    )


@router.get("/conflicts", response_model=MemoryConflictListResponse)
def list_conflicts(session: DbSession, _: CurrentReviewer) -> MemoryConflictListResponse:
    """Unresolved conflicts, oldest first: each candidate beside the rules it contradicts."""
    rows = (
        session.query(MemoryConflict, ReviewerAction, CallAnalysisRecord)
        .join(ReviewerAction, ReviewerAction.id == MemoryConflict.reviewer_action_id)
        .join(CallAnalysisRecord, CallAnalysisRecord.id == ReviewerAction.call_analysis_id)
        .filter(MemoryConflict.resolution.is_(None))
        .order_by(MemoryConflict.detected_at, MemoryConflict.id)
        .all()
    )
    rule_ids = {uuid.UUID(entry["rule_id"]) for conflict, _, _ in rows for entry in conflict.conflicting}
    rules = {r.id: r for r in session.query(MemoryRuleRecord).filter(MemoryRuleRecord.id.in_(rule_ids))}
    languages = correction_languages(session, (action.id for _, action, _ in rows))

    conflicts = []
    for conflict, action, analysis in rows:
        conflicting = []
        for entry in conflict.conflicting:
            rule = rules.get(uuid.UUID(entry["rule_id"]))
            conflicting.append(
                ConflictingRuleView(
                    rule_id=entry["rule_id"],
                    text=rule.rule_text if rule is not None else None,
                    active=rule is not None and not rule.retired,
                    similarity=entry["similarity"],
                    reasoning=entry["reasoning"],
                )
            )
        conflicts.append(
            MemoryConflictView(
                id=conflict.id,
                detected_at=conflict.detected_at,
                candidate_rule_id=conflict.candidate_rule_id,
                candidate_text=conflict.candidate_text,
                conflicting=conflicting,
                source=ConflictSourceView(
                    correction_id=action.id,
                    call_id=analysis.call_id,
                    language=languages[action.id],
                    reviewer_id=action.reviewer_id,
                    corrected_at=action.created_at,
                ),
            )
        )
    pending = {language: 0 for language in Language}
    for view in conflicts:
        pending[view.source.language] += 1
    return MemoryConflictListResponse(
        conflicts=conflicts, pending=PendingCounts(**{lang.value: n for lang, n in pending.items()})
    )


@router.post("/conflicts/{conflict_id}/resolve", response_model=ConflictResolveResponse)
def resolve(
    conflict_id: uuid.UUID, body: ConflictResolveRequest, reviewer: CurrentReviewer
) -> ConflictResolveResponse:
    """Apply a reviewer's decision. 404 unknown, 409 already resolved, 503 while the store is being written.

    An action that activates the candidate queues consolidation on the worker
    (the API has no embedding model). The resolution itself is already
    committed if that enqueue fails; `consolidation_queued` then says so, and
    the next induction run consolidates the whole store anyway.
    """
    try:
        outcome = resolve_conflict(conflict_id, body.action, reviewer_id=reviewer.reviewer_id)
    except ConflictNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such conflict") from exc
    except ConflictAlreadyResolvedError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except MemoryStoreBusyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    queued = False
    if outcome.needs_consolidation:
        try:
            consolidate_rules_task.delay()
            queued = True
        except Exception:
            logger.exception("conflict %s resolved, but consolidation could not be queued", conflict_id)
    return ConflictResolveResponse(
        conflict_id=outcome.conflict_id,
        action=outcome.action,
        resolved_by=outcome.resolved_by,
        resolved_at=outcome.resolved_at,
        activated_rule_id=outcome.activated_rule_id,
        retired_rule_ids=outcome.retired_rule_ids,
        consolidation_queued=queued,
    )
