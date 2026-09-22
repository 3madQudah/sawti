"""Capture a supervisor correction with its full surrounding context.

Phase 4: agent memory.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sawti.db.models import ReviewerAction
from sawti.db.session import get_session
from sawti.schemas import CallAnalysis, Correction


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
