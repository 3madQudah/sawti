"""The memory loop in the service, end to end in-process (phase 6.2).

review with a `correct` verdict -> induction task -> rule in `memory_rules`
-> a new, similar call retrieves that rule's id.

Real Postgres, eager Celery, the fake LLM (which answers induction and the
conflict check too), and a bag-of-words embedder so "similar" means
something: an unrelated rule is already stored and `top_k` is 1, so the new
call must pick the induced rule over it on similarity.
"""

from __future__ import annotations

import re
import uuid
import zlib
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sawti.api import tasks
from sawti.config import get_settings
from sawti.db.models import MemoryRuleRecord, ReviewerAction
from sawti.db.session import get_session
from sawti.memory import persistence

SIMILAR_CALL = (
    "Agent: Hello, how can I help you today?\n"
    "Customer: Nobody came to fix my line.\n"
    "Agent: I will send a technician tomorrow before noon.\n"
    "Customer: Fine.\n"
)


def _bag_of_words(text: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    vector = np.zeros(256)
    for word in re.findall(r"[a-z]+", text.lower()):
        vector[zlib.crc32(word.encode()) % 256] += 1.0
    return vector


@pytest.fixture(autouse=True)
def _bag_of_words_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tasks, "_induction_embed", _bag_of_words)


@pytest.fixture
def unrelated_rule() -> uuid.UUID:
    rule_id = uuid.uuid4()
    with get_session() as session:
        session.add(
            MemoryRuleRecord(
                id=rule_id,
                rule_text="Score empathy low when the agent interrupts or talks over the customer.",
                created_at=datetime.now(UTC),
            )
        )
    return rule_id


def _active_rules() -> list[MemoryRuleRecord]:
    with get_session() as session:
        return session.query(MemoryRuleRecord).filter_by(retired=False).all()


def test_a_correct_verdict_becomes_a_rule_that_a_similar_call_retrieves(
    client: TestClient,
    headers: dict[str, str],
    submit: Callable[..., str],
    llm: Any,
    unrelated_rule: uuid.UUID,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_id = submit()
    body = {
        "verdicts": [
            {
                "claim_id": "commitments[0]",
                "verdict": "correct",
                "changes": {"description": "Send a technician tomorrow before noon."},
            },
            {"claim_id": "commitments[1]", "verdict": "confirm"},
        ]
    }
    assert client.post(f"/reviews/{call_id}/corrections", json=body, headers=headers).status_code == 202

    # The eager induction task has run: one new rule, sourced from the one correction.
    with get_session() as session:
        action = session.query(ReviewerAction).one()
        assert action.induced_at is not None and action.induction_outcome == "inserted"
    induced = [rule for rule in _active_rules() if rule.id != unrelated_rule]
    assert len(induced) == 1
    assert induced[0].source_correction_ids == [str(action.id)]
    assert "technician" in induced[0].rule_text.lower()

    monkeypatch.setenv("SAWTI_MEMORY_TOP_K", "1")
    get_settings.cache_clear()
    detail = client.get(f"/calls/{submit(SIMILAR_CALL)}", headers=headers).json()
    assert [rule["id"] for rule in detail["memory_rules"]] == [str(induced[0].id)]


def test_induction_runs_once_per_correction(
    client: TestClient, headers: dict[str, str], submit: Callable[..., str], llm: Any
) -> None:
    """A redelivered induction task finds nothing pending and writes nothing."""
    call_id = submit()
    body = {
        "verdicts": [
            {"claim_id": "commitments[0]", "verdict": "reject"},
            {"claim_id": "commitments[1]", "verdict": "confirm"},
        ]
    }
    accepted = client.post(f"/reviews/{call_id}/corrections", json=body, headers=headers).json()
    assert len(_active_rules()) == 1
    calls_before = llm.calls

    again = tasks.induce_rules_task.apply(args=[accepted["submission_id"]]).get()
    assert again == {"processed": 0, "inserted": 0, "conflicts": 0, "merged_away": 0}
    assert llm.calls == calls_before
    assert len(_active_rules()) == 1


def test_a_confirm_only_review_queues_no_induction(
    client: TestClient, headers: dict[str, str], submit: Callable[..., str], llm: Any
) -> None:
    call_id = submit()
    body = {
        "verdicts": [
            {"claim_id": "commitments[0]", "verdict": "confirm"},
            {"claim_id": "commitments[1]", "verdict": "confirm"},
        ]
    }
    client.post(f"/reviews/{call_id}/corrections", json=body, headers=headers)
    assert _active_rules() == []


def test_a_contradicting_rule_is_not_stored_but_the_correction_is_marked(
    client: TestClient,
    headers: dict[str, str],
    submit: Callable[..., str],
    llm: Any,
    unrelated_rule: uuid.UUID,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`insert_rule`'s phase 4 policy holds: a conflict leaves both rules for a human, inserts nothing."""
    from sawti.memory import conflict

    monkeypatch.setattr(
        conflict,
        "_judges_contradiction",
        lambda candidate, existing: conflict._ConflictJudgment(contradicts=True),
    )
    call_id = submit()
    body = {
        "verdicts": [
            {"claim_id": "commitments[0]", "verdict": "reject"},
            {"claim_id": "commitments[1]", "verdict": "confirm"},
        ]
    }
    client.post(f"/reviews/{call_id}/corrections", json=body, headers=headers)

    assert [rule.id for rule in _active_rules()] == [unrelated_rule]
    with get_session() as session:
        assert session.query(ReviewerAction).one().induction_outcome == "conflict"


def test_rule_text_is_redacted_before_it_is_stored(monkeypatch: pytest.MonkeyPatch, llm: Any) -> None:
    """Rules go into prompts; an induced rule echoing a phone number is stored without it."""
    from sawti.memory.rule_schema import MemoryRule

    def _leaky(corrections: list[Any]) -> MemoryRule:
        return MemoryRule(rule_text="Call the customer back on 0791234567.", created_at=datetime.now(UTC))

    monkeypatch.setattr(persistence, "induce_rule", _leaky)
    monkeypatch.setattr(persistence, "pending_corrections", lambda session, **_: [_FakeCorrection()])
    monkeypatch.setattr(persistence, "_mark_induced", lambda *a, **k: None)
    persistence.induce_and_store(embed=_bag_of_words)
    [rule] = _active_rules()
    assert "0791234567" not in rule.rule_text


class _FakeCorrection:
    """Just enough of a `Correction` for `induce_and_store`'s loop."""

    id = uuid.uuid4()

    class original:  # noqa: N801 - mimics the attribute path correction.original.language.value
        class language:  # noqa: N801
            value = "en"


def test_memory_rules_endpoint_lists_active_rules_for_reviewers_only(
    client: TestClient, headers: dict[str, str], unrelated_rule: uuid.UUID
) -> None:
    assert client.get("/memory/rules").status_code == 401
    rules = client.get("/memory/rules", headers=headers).json()["rules"]
    assert [(r["id"], r["source_corrections"]) for r in rules] == [(str(unrelated_rule), 0)]
