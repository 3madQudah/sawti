"""Tests for `scripts/induce_rules_from_corrections.py`'s report: the accounting reconciles.

Phase 6.2.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import sqlalchemy as sa
from induce_rules_from_corrections import report

from sawti.db.models import Agent, Call, CallAnalysisRecord, MemoryRuleRecord, ReviewerAction
from sawti.db.session import get_engine, get_session


def _correction(session: object, language: str) -> str:
    now = datetime.now(UTC)
    agent = Agent(id=uuid.uuid4(), name="a", created_at=now)
    call = Call(id=uuid.uuid4(), agent=agent, language=language, occurred_at=now)
    record = CallAnalysisRecord(
        id=uuid.uuid4(), call=call, payload={}, confidence=1.0, requires_human_review=False, created_at=now
    )
    action = ReviewerAction(
        id=uuid.uuid4(),
        call_analysis=record,
        reviewer_id="r",
        payload={"original": {"language": language}},
        created_at=now,
        induced_at=now,
        induction_outcome="inserted",
    )
    session.add_all([agent, call, record, action])  # type: ignore[attr-defined]
    return str(action.id)


def test_report_reconciles_rows_and_explains_multi_language_rules() -> None:
    """2 induced + 1 merge spanning ar and mixed: 3 rows, 2 retired, 1 active counted in 2 languages."""
    with get_engine().begin() as conn:
        tables = "reviewer_actions, review_submissions, call_analyses, calls, agents, memory_rules"
        conn.execute(sa.text(f"TRUNCATE {tables} CASCADE"))
    now = datetime.now(UTC)
    with get_session() as session:
        ar, mixed = _correction(session, "ar"), _correction(session, "mixed")
        session.add_all(
            [
                MemoryRuleRecord(rule_text="a", source_correction_ids=[ar], created_at=now, retired=True),
                MemoryRuleRecord(rule_text="b", source_correction_ids=[mixed], created_at=now, retired=True),
                MemoryRuleRecord(rule_text="a and b", source_correction_ids=[ar, mixed], created_at=now),
            ]
        )
    out = report()
    rows = out["rule_rows"]
    assert (
        rows["inserted_by_induction"],
        rows["created_by_consolidation"],
        rows["retired_by_consolidation"],
    ) == (2, 1, 2)
    assert (
        rows["active"]
        == rows["inserted_by_induction"] + rows["created_by_consolidation"] - rows["retired_by_consolidation"]
    )
    assert out["active_rules"] == 1
    assert out["active_rules_per_language"] == {"ar": 1, "mixed": 1}  # sums to 2 > 1 active, by design
    assert out["active_rules_by_language_set"] == {"ar+mixed": 1}
    assert out["active_rules_detail"][0]["source_corrections_per_language"] == {"ar": 1, "mixed": 1}
