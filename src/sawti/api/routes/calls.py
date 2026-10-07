"""Call endpoints: submit a transcript for analysis, read back everything the review screen needs.

Phase 6.2. These routes only read and write the database and enqueue tasks;
the graph runs in the worker (`sawti.api.tasks`).
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, status
from sqlalchemy.exc import IntegrityError

from sawti.api.deps import ApiKey, CurrentReviewer, DbSession
from sawti.api.schemas import (
    CallDetail,
    CallSubmitRequest,
    CallSubmitResponse,
    MemoryRuleUsed,
    ReviewSubmission,
)
from sawti.api.tasks import analyze_call_task
from sawti.api.views import grounded_claims, rejected_claims_view, split_turns
from sawti.data.transcript_parser import UnattributableLineError
from sawti.db.models import Agent, Call, CallAnalysisRecord, MemoryRuleRecord
from sawti.db.models import ReviewSubmission as ReviewSubmissionRow
from sawti.memory.capture import get_or_create_placeholder_agent
from sawti.privacy.redaction import redact
from sawti.schemas import CallAnalysis, CallStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/calls", tags=["calls"])


def _agent_id(session: DbSession, external_id: str) -> uuid.UUID:
    """The `Agent` for `external_id`, created on first sight (named by its external id).

    Two first-time submissions for one agent can race on the unique
    `external_id`; the loser re-reads the winner's row.
    """
    try:
        with session.begin_nested():
            return get_or_create_placeholder_agent(session, external_id=external_id, name=external_id)
    except IntegrityError:
        return get_or_create_placeholder_agent(session, external_id=external_id, name=external_id)


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=CallSubmitResponse)
def submit_call(body: CallSubmitRequest, session: DbSession, _: ApiKey) -> CallSubmitResponse:
    """Redact the transcript, store the redacted text only, and queue the call for analysis.

    The raw transcript exists only inside this function; it is neither stored
    nor put into graph state (docs/09-DECISIONS.md, 2026-10-06, decision 1).

    Returns 422 if a line has no `Agent:`/`Customer:` prefix (the review
    screen needs speaker turns, and the parser does not guess), and 503 if
    the task could not be queued — the call is then marked `failed`.
    """
    call_id = uuid.uuid4()
    redaction = redact(body.transcript)
    try:
        split_turns(redaction.redacted_text, str(call_id))
    except UnattributableLineError as exc:
        detail = str(exc).replace(str(call_id), "transcript")
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail) from exc

    now = datetime.now(UTC)
    session.add(
        Call(
            id=call_id,
            agent_id=_agent_id(session, body.agent_external_id),
            redacted_transcript=redaction.redacted_text,
            pii_redacted_count=redaction.redaction_count,
            language=body.language.value,
            occurred_at=now,
            status=CallStatus.QUEUED,
            queued_at=now,
            enqueued_at=now,
        )
    )
    session.commit()

    try:
        analyze_call_task.delay(str(call_id))
    except Exception as exc:
        logger.exception("could not enqueue %s", call_id)
        call = session.get(Call, call_id)
        if call is not None:
            call.status, call.finished_at = CallStatus.FAILED, datetime.now(UTC)
            session.commit()
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "could not queue the call") from exc
    return CallSubmitResponse(call_id=call_id)


@router.get("/{call_id}", response_model=CallDetail)
def get_call(call_id: uuid.UUID, session: DbSession, _: CurrentReviewer) -> CallDetail:
    """Everything the call review screen renders. Analysis fields are empty until analysis finishes."""
    row = (
        session.query(Call, Agent)
        .join(Agent, Agent.id == Call.agent_id)
        .filter(Call.id == call_id)
        .one_or_none()
    )
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such call")
    call, agent = row
    transcript = call.redacted_transcript or ""
    turns = split_turns(transcript, str(call.id))
    detail = CallDetail(
        call_id=call.id,
        status=call.status,
        language=call.language,
        agent_name=agent.name,
        queued_at=call.queued_at,
        started_at=call.started_at,
        finished_at=call.finished_at,
        pii_redacted_count=call.pii_redacted_count,
        attempts=call.attempts,
        redacted_transcript=transcript,
        turns=turns,
    )

    record = session.query(CallAnalysisRecord).filter_by(call_id=call.id).one_or_none()
    if record is None:
        return detail
    analysis = CallAnalysis.model_validate(record.payload)
    rule_ids = [uuid.UUID(rule_id) for rule_id in record.retrieved_rule_ids]
    rules = {r.id: r for r in session.query(MemoryRuleRecord).filter(MemoryRuleRecord.id.in_(rule_ids))}
    submission = session.query(ReviewSubmissionRow).filter_by(call_analysis_id=record.id).one_or_none()

    detail.summary = analysis.summary
    detail.confidence = record.confidence
    detail.grounding_coverage = record.grounding_coverage
    detail.confidence_threshold = record.confidence_threshold
    detail.escalation_reason = record.escalation_reason
    detail.claims = grounded_claims(analysis, turns)
    detail.rejected_claims = rejected_claims_view(record.rejected_claims)
    # Best match first, as retrieved; a rule deleted since is simply absent.
    detail.memory_rules = [MemoryRuleUsed(id=i, text=rules[i].rule_text) for i in rule_ids if i in rules]
    detail.review = ReviewSubmission.model_validate(submission.verdicts) if submission is not None else None
    return detail
