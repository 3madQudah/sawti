"""Tests for `scripts/reviewer_agreement.py`: it reads submissions from the database correctly.

Phase 6.2.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import sqlalchemy as sa
from reviewer_agreement import load_reviews

from sawti.db.models import Agent, Call, CallAnalysisRecord, ReviewSubmission
from sawti.db.session import get_engine, get_session
from sawti.schemas import Language


def test_load_reviews_pairs_each_submission_with_its_calls_language() -> None:
    with get_engine().begin() as conn:
        conn.execute(
            sa.text("TRUNCATE reviewer_actions, review_submissions, call_analyses, calls, agents CASCADE")
        )
    now = datetime.now(UTC)
    with get_session() as session:
        agent = Agent(id=uuid.uuid4(), name="a", created_at=now)
        call = Call(id=uuid.uuid4(), agent=agent, language="mixed", occurred_at=now, redacted_transcript="x")
        record = CallAnalysisRecord(
            id=uuid.uuid4(), call=call, payload={}, confidence=0.5, requires_human_review=True, created_at=now
        )
        session.add_all([agent, call, record])
        session.flush()
        session.add(
            ReviewSubmission(
                call_analysis_id=record.id,
                reviewer_id="qa",
                verdicts={"verdicts": [{"claim_id": "commitments[0]", "verdict": "reject"}]},
                created_at=now,
            )
        )
    assert load_reviews() == [(Language.MIXED, ["reject"])]
