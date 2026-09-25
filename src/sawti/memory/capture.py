"""Capture a supervisor correction with its full surrounding context.

Phase 4: agent memory.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from sawti.db.models import Agent, Call, CallAnalysisRecord, ReviewerAction
from sawti.db.session import get_session
from sawti.schemas import CallAnalysis, Correction


def get_or_create_placeholder_agent(session: Session, *, external_id: str, name: str) -> uuid.UUID:
    """Return the id of a shared placeholder `Agent`, creating it on first call.

    Shared by every caller that needs to satisfy `capture_correction()`'s FK
    for analyses that were never produced by a real contact-center agent —
    `scripts/generate_synthetic_corrections.py` (Stage 2) and
    `sawti.eval.experiments.five_batch` (Stage 8) each use their own
    `external_id` so their placeholder data stays distinguishable in the DB.

    Args:
        session: An open session (the caller owns its lifecycle/commit).
        external_id: Unique identity for this placeholder — reused across
            calls to find the same row rather than minting a fresh one.
        name: Human-readable label, ideally saying plainly that this is not
            a real agent.

    Returns:
        The placeholder `Agent`'s id.
    """
    existing = session.query(Agent).filter_by(external_id=external_id).one_or_none()
    if existing is not None:
        return existing.id
    agent = Agent(name=name, external_id=external_id, created_at=datetime.now(UTC))
    session.add(agent)
    # `Agent.id`'s `default=uuid.uuid4` is a SQLAlchemy flush-time default,
    # not a construction-time one — `agent.id` is None until flushed.
    session.flush()
    return agent.id


def persist_analysis_record(
    session: Session, analysis: CallAnalysis, *, agent_id: uuid.UUID, transcript: str
) -> None:
    """Persist a `Call` + `CallAnalysisRecord` for `analysis`, satisfying `ReviewerAction`'s FK.

    `capture_correction()` requires a `CallAnalysisRecord` row matching
    `original.id` to already exist — this is how a caller creates one for an
    analysis that has nowhere else it would get persisted (no real Phase 6
    service layer yet). `CallAnalysisRecord.id` is set to `analysis.id`, the
    same id `capture_correction()` uses as `ReviewerAction.call_analysis_id`,
    so the two rows line up by construction.

    Args:
        session: An open session (the caller owns its lifecycle/commit).
        analysis: The analysis to persist a record for — typically the
            `original` a `Correction` will be captured against.
        agent_id: id of the `Agent` the new `Call` row attaches to (see
            `get_or_create_placeholder_agent`).
        transcript: The call's transcript text, stored on the `Call` row.
    """
    now = datetime.now(UTC)
    call_row_id = uuid.uuid4()
    session.add(
        Call(
            id=call_row_id,
            agent_id=agent_id,
            transcript=transcript,
            language=analysis.language.value,
            occurred_at=now,
        )
    )
    session.add(
        CallAnalysisRecord(
            id=analysis.id,
            call_id=call_row_id,
            payload=analysis.model_dump(mode="json"),
            confidence=analysis.confidence,
            requires_human_review=analysis.requires_human_review,
            created_at=now,
        )
    )


def capture_correction(
    original: CallAnalysis,
    corrected: CallAnalysis,
    *,
    error_location: str,
    reviewer_id: str,
    note: str | None = None,
) -> Correction:
    """Build a `Correction` record capturing a reviewer's edit and its full context.

    Persists the correction as a `ReviewerAction` row keyed on `original.id` —
    which is `sawti.db.models.CallAnalysisRecord.id`, the analysis being
    corrected. That row must already exist: `ReviewerAction.call_analysis_id`
    is a foreign key, and this function only records a correction against an
    existing analysis, it does not create one.

    Args:
        original: The unreviewed agent output.
        corrected: The reviewer's corrected version.
        error_location: Field path or artifact id of what was wrong.
        reviewer_id: Identity of the QA reviewer making the correction.
        note: Optional free-text explanation from the reviewer.

    Returns:
        A validated `Correction` record, ready for induction.

    Raises:
        ValueError: If `original` and `corrected` disagree on `call_id` —
            a correction must be about one call.
        sqlalchemy.exc.IntegrityError: If no `CallAnalysisRecord` row exists
            for `original.id`.
    """
    if original.call_id != corrected.call_id:
        raise ValueError(
            f"original.call_id ({original.call_id!r}) != corrected.call_id ({corrected.call_id!r}) "
            "— a correction must be about one call."
        )

    correction = Correction(
        call_id=original.call_id,
        original=original,
        corrected=corrected,
        error_location=error_location,
        note=note,
        reviewer_id=reviewer_id,
        timestamp=datetime.now(UTC),
    )

    with get_session() as session:
        session.add(
            ReviewerAction(
                id=correction.id,
                call_analysis_id=original.id,
                reviewer_id=reviewer_id,
                payload=correction.model_dump(mode="json"),
                created_at=correction.timestamp,
            )
        )

    return correction
