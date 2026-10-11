"""`scripts/recheck_legacy_conflicts.py`: which corrections it re-opens, and when it refuses (phase 6.4)."""

from __future__ import annotations

import sys
import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
import recheck_legacy_conflicts as recheck
import sqlalchemy as sa

from sawti.db.models import Agent, Call, CallAnalysisRecord, MemoryConflict, ReviewerAction
from sawti.db.session import get_engine, get_session


@pytest.fixture(autouse=True)
def _clean() -> None:
    with get_engine().begin() as conn:
        conn.execute(
            sa.text(
                "TRUNCATE memory_conflicts, reviewer_actions, review_submissions, call_analyses, calls, "
                "agents, memory_rules CASCADE"
            )
        )


def _action(session: Any, *, outcome: str | None, language: str = "ar", recorded: bool = False) -> uuid.UUID:
    now = datetime.now(UTC)
    agent = Agent(id=uuid.uuid4(), name="a", created_at=now)
    call = Call(id=uuid.uuid4(), agent_id=agent.id, language=language, occurred_at=now)
    analysis = CallAnalysisRecord(
        id=uuid.uuid4(),
        call_id=call.id,
        payload={},
        confidence=0.5,
        requires_human_review=True,
        created_at=now,
    )
    action = ReviewerAction(
        id=uuid.uuid4(),
        call_analysis_id=analysis.id,
        reviewer_id="r",
        payload={"original": {"language": language}},
        created_at=now,
        induced_at=now if outcome else None,
        induction_outcome=outcome,
    )
    session.add_all([agent, call, analysis, action])
    session.flush()
    if recorded:
        session.add(
            MemoryConflict(
                reviewer_action_id=action.id,
                candidate_rule_id=uuid.uuid4(),
                candidate_text="t",
                conflicting=[],
                detected_at=now,
            )
        )
    return action.id


def test_only_unrecorded_conflicts_are_legacy() -> None:
    with get_session() as session:
        legacy = _action(session, outcome="conflict", language="mixed")
        _action(session, outcome="conflict", recorded=True)
        _action(session, outcome="inserted")
    with get_session() as session:
        found = recheck.legacy_conflicts(session)
        assert [a.id for a in found] == [legacy]
        assert recheck.describe(found)[0]["language"] == "mixed"


def test_dry_run_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    with get_session() as session:
        legacy = _action(session, outcome="conflict")
    monkeypatch.setattr(sys, "argv", ["recheck_legacy_conflicts.py"])
    assert recheck.main() == 0
    with get_session() as session:
        assert session.get(ReviewerAction, legacy).induction_outcome == "conflict"  # type: ignore[union-attr]


def test_apply_reopens_them_and_runs_the_normal_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    with get_session() as session:
        legacy = _action(session, outcome="conflict")
        inserted = _action(session, outcome="inserted")
    ran: list[list[str]] = []
    fake_backfill = type(sys)("induce_rules_from_corrections")
    fake_backfill.main = lambda: ran.append(list(sys.argv)) or 0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "induce_rules_from_corrections", fake_backfill)
    monkeypatch.setattr(sys, "argv", ["recheck_legacy_conflicts.py", "--apply"])

    assert recheck.main() == 0

    assert len(ran) == 1 and "--min-interval" in ran[0]
    with get_session() as session:
        reopened = session.get(ReviewerAction, legacy)
        untouched = session.get(ReviewerAction, inserted)
        assert reopened is not None and reopened.induced_at is None and reopened.induction_outcome is None
        assert untouched is not None and untouched.induction_outcome == "inserted"


def test_apply_refuses_while_other_corrections_are_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    with get_session() as session:
        legacy = _action(session, outcome="conflict")
        _action(session, outcome=None)
    monkeypatch.setattr(sys, "argv", ["recheck_legacy_conflicts.py", "--apply"])

    assert recheck.main() == 2
    with get_session() as session:
        assert session.get(ReviewerAction, legacy).induction_outcome == "conflict"  # type: ignore[union-attr]
