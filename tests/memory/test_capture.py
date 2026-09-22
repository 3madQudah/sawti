"""Tests for `sawti.memory.capture`.

Phase 4: mirrors `src/sawti/memory/capture.py`.

Integration tests against a real Postgres instance — `capture_correction()`
persists via `sawti.db.session`, and `ReviewerAction.call_analysis_id` is a
real foreign key, so these need an actual `CallAnalysisRecord` row to point
at. Run `docker compose up -d` first; see `docs/09-DECISIONS.md`.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from sawti.db.models import Agent, Base, Call, CallAnalysisRecord, ReviewerAction
from sawti.db.session import get_engine, get_session
from sawti.memory.capture import capture_correction
from sawti.schemas import CallAnalysis, Language


def _sample_analysis(call_id: str, *, summary: str) -> CallAnalysis:
    """A minimal, valid `CallAnalysis` for a given call_id — no commitments/flags needed here."""
    return CallAnalysis(
        call_id=call_id,
        language=Language.EN,
        summary=summary,
        confidence=0.9,
        requires_human_review=False,
    )


@pytest.fixture(scope="module", autouse=True)
def _schema() -> None:
    """Create the ORM schema once for this test module, against the real database."""
    Base.metadata.create_all(get_engine())


@pytest.fixture
def _persisted_original() -> Iterator[CallAnalysis]:
    """A `CallAnalysis` with a matching `CallAnalysisRecord` row, satisfying the FK.

    `capture_correction()` only records a correction against an *existing*
    analysis — it does not create one — so every test needs one seeded first.
    Cleaned up afterward, including any `ReviewerAction` the test created
    (its own FK to this row would otherwise block the teardown delete).
    """
    original = _sample_analysis("call_test_capture", summary="Customer called about a billing dispute.")
    agent_id = uuid.uuid4()
    call_row_id = uuid.uuid4()
    now = datetime.now(UTC)
    with get_session() as session:
        session.add(Agent(id=agent_id, name="test-agent", created_at=now))
        session.add(Call(id=call_row_id, agent_id=agent_id, language="en", occurred_at=now))
        session.add(
            CallAnalysisRecord(
                id=original.id,
                call_id=call_row_id,
                payload=original.model_dump(mode="json"),
                confidence=original.confidence,
                requires_human_review=original.requires_human_review,
                created_at=now,
            )
        )

    yield original

    with get_session() as session:
        for action in session.query(ReviewerAction).filter_by(call_analysis_id=original.id):
            session.delete(action)
        for model, row_id in (
            (CallAnalysisRecord, original.id),
            (Call, call_row_id),
            (Agent, agent_id),
        ):
            row = session.get(model, row_id)
            if row is not None:
                session.delete(row)


def test_capture_correction_stamps_timestamp_and_reviewer_id(_persisted_original: CallAnalysis) -> None:
    """capture_correction() produces a Correction with timestamp and reviewer_id set."""
    corrected = _sample_analysis(
        _persisted_original.call_id,
        summary="Customer called about a billing dispute; a refund was promised by Friday.",
    )
    before = datetime.now(UTC)

    correction = capture_correction(
        _persisted_original,
        corrected,
        error_location="summary",
        reviewer_id="qa-reviewer-1",
        note="missed the refund commitment",
    )

    assert correction.reviewer_id == "qa-reviewer-1"
    assert correction.timestamp >= before
    assert correction.timestamp <= datetime.now(UTC)


def test_capture_correction_preserves_original_and_corrected_analyses(
    _persisted_original: CallAnalysis,
) -> None:
    """capture_correction() retains both the original and corrected CallAnalysis unmodified."""
    corrected = _sample_analysis(_persisted_original.call_id, summary="A materially different summary.")

    correction = capture_correction(
        _persisted_original,
        corrected,
        error_location="summary",
        reviewer_id="qa-reviewer-2",
    )

    assert correction.original == _persisted_original
    assert correction.corrected == corrected
    assert correction.original.summary != correction.corrected.summary
