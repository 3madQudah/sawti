"""Review endpoints: the queue of escalated calls, and submitting a reviewer's verdicts.

Phase 6.2. Reviewer identity comes only from `get_current_reviewer()`.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated

import sqlalchemy as sa
from fastapi import APIRouter, HTTPException, Query, status

from sawti.api.deps import CurrentReviewer, DbSession
from sawti.api.records import InvalidVerdictError, plan_corrections
from sawti.api.schemas import (
    PendingCounts,
    ReviewAccepted,
    ReviewQueueItem,
    ReviewQueueResponse,
    ReviewSubmission,
)
from sawti.api.tasks import induce_rules_task, resume_call_task
from sawti.api.views import grounded_claims, split_turns
from sawti.db.models import Agent, Call, CallAnalysisRecord
from sawti.db.models import ReviewSubmission as ReviewSubmissionRow
from sawti.memory.capture import capture_correction
from sawti.schemas import CallAnalysis, CallStatus, Language

router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.get("", response_model=ReviewQueueResponse)
def review_queue(
    session: DbSession,
    _: CurrentReviewer,
    call_status: Annotated[CallStatus, Query(alias="status")] = CallStatus.AWAITING_REVIEW,
    language: Language | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> ReviewQueueResponse:
    """Calls in `status` (default `awaiting_review`), oldest first, plus per-language pending counts."""
    query = (
        session.query(Call, Agent, CallAnalysisRecord)
        .join(Agent, Agent.id == Call.agent_id)
        .outerjoin(CallAnalysisRecord, CallAnalysisRecord.call_id == Call.id)
        .filter(Call.status == call_status)
    )
    if language is not None:
        query = query.filter(Call.language == language.value)
    rows = query.order_by(sa.func.coalesce(Call.finished_at, Call.queued_at).asc()).limit(limit).all()

    counts: dict[str, int] = {
        lang: count
        for lang, count in session.query(Call.language, sa.func.count())
        .filter(Call.status == CallStatus.AWAITING_REVIEW)
        .group_by(Call.language)
        .all()
    }
    return ReviewQueueResponse(
        items=[
            ReviewQueueItem(
                call_id=call.id,
                agent_name=agent.name,
                language=call.language,
                grounding_coverage=record.grounding_coverage if record is not None else None,
                escalation_reason=record.escalation_reason if record is not None else None,
                waiting_since=call.finished_at or call.queued_at,
            )
            for call, agent, record in rows
        ],
        pending=PendingCounts(**{lang.value: counts.get(lang.value, 0) for lang in Language}),
    )


@router.post(
    "/{call_id}/corrections",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ReviewAccepted,
)
def submit_review(
    call_id: uuid.UUID, body: ReviewSubmission, session: DbSession, reviewer: CurrentReviewer
) -> ReviewAccepted:
    """Record a reviewer's verdicts, capture the corrections, and queue the graph's resume.

    409 if the call is not `awaiting_review` or already has a submission;
    422 unless there is exactly one verdict for every grounded claim. The
    submission, its `Correction`s and the `ReviewerAction` rows commit as one
    unit before the resume is queued. If the review produced corrections, rule
    induction over them is queued too (`induce_rules_task`).
    """
    # Row lock: two reviewers submitting at once cannot both pass the checks below.
    call = session.query(Call).filter(Call.id == call_id).with_for_update().one_or_none()
    if call is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such call")
    if call.status != CallStatus.AWAITING_REVIEW:
        raise HTTPException(status.HTTP_409_CONFLICT, f"call is {call.status.value}, not awaiting_review")
    record = session.query(CallAnalysisRecord).filter_by(call_id=call_id).one()
    if session.query(ReviewSubmissionRow).filter_by(call_analysis_id=record.id).first() is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "call already has a review submission")

    original = CallAnalysis.model_validate(record.payload)
    expected = {
        claim.claim_id
        for claim in grounded_claims(original, split_turns(call.redacted_transcript or "", str(call.id)))
    }
    given = {verdict.claim_id for verdict in body.verdicts}
    if given != expected:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            {"missing_verdicts": sorted(expected - given), "unknown_claims": sorted(given - expected)},
        )

    try:
        planned = plan_corrections(body, original)
    except InvalidVerdictError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    submission = ReviewSubmissionRow(
        call_analysis_id=record.id,
        reviewer_id=reviewer.reviewer_id,
        # exclude_unset: "field not changed" and "deadline cleared (null)" must stay distinct.
        verdicts=body.model_dump(mode="json", exclude_unset=True),
        created_at=datetime.now(UTC),
    )
    session.add(submission)
    session.flush()
    for item in planned:
        capture_correction(
            original,
            item.corrected,
            error_location=item.error_location,
            reviewer_id=reviewer.reviewer_id,
            note=item.note,
            session=session,
            review_submission_id=submission.id,
        )
    session.commit()

    resume_call_task.delay(str(call_id))
    if planned:
        # The memory loop: rules learned from these corrections reach later calls.
        induce_rules_task.delay(str(submission.id))
    return ReviewAccepted(call_id=call_id, submission_id=submission.id, corrections_recorded=len(planned))
