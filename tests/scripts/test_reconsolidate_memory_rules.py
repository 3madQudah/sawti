"""Tests for `scripts/reconsolidate_memory_rules.py`: undo a chain merge, re-cluster, resume.

Phase 6.2. Fake embedder (unit vectors at chosen angles) and a fake merge, so
no model and no LLM; the database is the suite's real test Postgres.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import reconsolidate_memory_rules as script
import sqlalchemy as sa

from sawti.db.models import Agent, Call, CallAnalysisRecord, MemoryRuleRecord, ReviewerAction
from sawti.db.session import get_engine, get_session
from sawti.memory.rule_schema import MemoryRule

ANGLE = {"A": 0.0, "B": 40.0, "C": 80.0}  # A~B, B~C >= 0.7; A~C = 0.17


def _embed(text: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    radians = np.radians(ANGLE[text[0]])
    return np.array([np.cos(radians), np.sin(radians)])


def _merge(rules: list[MemoryRule]) -> MemoryRule:
    sources = [s for r in rules for s in r.source_correction_ids]
    return MemoryRule(
        rule_text="".join(sorted(r.rule_text[0] for r in rules)) + " merged",
        source_correction_ids=sources,
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def chained_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """A, B, C induced from one correction each, all retired into one single-linkage merge."""
    monkeypatch.setattr(script, "_default_embed", _embed)
    monkeypatch.setattr(script, "_merge_cluster", _merge)
    monkeypatch.setattr(script, "PLAN_PATH", tmp_path / "plan.json")
    with get_engine().begin() as conn:
        tables = "reviewer_actions, review_submissions, call_analyses, calls, agents, memory_rules"
        conn.execute(sa.text(f"TRUNCATE {tables} CASCADE"))
    now = datetime.now(UTC)
    ids: dict[str, str] = {}
    sources = []
    with get_session() as session:
        for name, language in (("A", "ar"), ("B", "en"), ("C", "mixed")):
            agent = Agent(id=uuid.uuid4(), name="a", created_at=now)
            call = Call(id=uuid.uuid4(), agent=agent, language=language, occurred_at=now)
            record = CallAnalysisRecord(
                id=uuid.uuid4(),
                call=call,
                payload={},
                confidence=1.0,
                requires_human_review=False,
                created_at=now,
            )
            action = ReviewerAction(
                id=uuid.uuid4(),
                call_analysis=record,
                reviewer_id="r",
                payload={"original": {"language": language}},
                created_at=now,
                induction_outcome="inserted",
                induced_at=now,
            )
            rule = MemoryRuleRecord(
                id=uuid.uuid4(),
                rule_text=f"{name} rule",
                source_correction_ids=[str(action.id)],
                created_at=now,
                retired=True,
            )
            session.add_all([agent, call, record, action, rule])
            ids[name] = str(rule.id)
            sources.append(str(action.id))
        session.add(MemoryRuleRecord(rule_text="ABC chained", source_correction_ids=sources, created_at=now))
    return ids


def _active_texts() -> set[str]:
    with get_session() as session:
        return {r.rule_text for r in session.query(MemoryRuleRecord).filter_by(retired=False)}


def test_reconsolidation_undoes_the_chain_and_never_joins_a_with_c(chained_store: dict[str, str]) -> None:
    plan = script.prepare()
    assert plan["pre_fix"]["size_distribution"] == {3: 1}
    assert plan["pre_fix"]["multi_member_clusters"][0]["min_pairwise_cosine"] < 0.7
    assert not any({chained_store["A"], chained_store["C"]} <= set(c) for c in plan["clusters"])

    assert script.merge_pending(plan) == 1
    active = _active_texts()
    assert "ABC chained" not in active
    assert active in ({"AB merged", "C rule"}, {"A rule", "BC merged"})

    post = script.post_fix_stats(plan)
    assert post["size_distribution"] == {1: 1, 2: 1}
    assert post["multi_member_clusters"][0]["min_pairwise_cosine"] >= 0.7


def test_reconsolidation_resumes_without_redoing_merges(chained_store: dict[str, str]) -> None:
    plan = script.prepare()
    script.merge_pending(plan)
    before = _active_texts()
    assert script.merge_pending(plan) == 0
    assert _active_texts() == before
