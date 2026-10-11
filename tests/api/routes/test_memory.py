"""`/memory/rules`, `/memory/conflicts` and resolving a conflict (phase 6.4).

Conflicts come from the real loop: a review stores a rule, later reviews'
candidates are judged to contradict it (`contradicting_judge`), and
`induce_and_store` records them. Resolution is checked through the API and in
the rule table it changed.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from sawti.api import tasks
from sawti.db.models import MemoryConflict, MemoryRuleRecord, ReviewerAction
from sawti.db.session import get_engine, get_session
from sawti.memory import persistence
from sawti.memory.persistence import _INDUCTION_LOCK_KEY
from sawti.privacy.redaction import redact

JUDGE_PHONE = "0795551234"  # tests/api/conftest.py
#: What the stored reasoning must be: the judge's sentence, redacted.
REDACTED_REASONING = redact(f"Opposite guidance for the same case; see {JUDGE_PHONE}.").redacted_text


def _rules() -> dict[uuid.UUID, MemoryRuleRecord]:
    with get_session() as session:
        rows = session.query(MemoryRuleRecord).all()
        session.expunge_all()
        return {row.id: row for row in rows}


def _conflicts() -> list[MemoryConflict]:
    with get_session() as session:
        rows = session.query(MemoryConflict).order_by(MemoryConflict.detected_at).all()
        session.expunge_all()
        return rows


@pytest.fixture
def consolidations(monkeypatch: pytest.MonkeyPatch) -> list[None]:
    queued: list[None] = []
    monkeypatch.setattr(tasks.consolidate_rules_task, "delay", lambda: queued.append(None))
    return queued


# --- Recording -------------------------------------------------------------------------


def test_a_refused_candidate_is_recorded_with_similarity_and_redacted_reasoning(
    make_conflicts: Callable[..., list[str]],
) -> None:
    make_conflicts("ar")

    [existing] = [r for r in _rules().values() if not r.retired]
    [recorded] = _conflicts()
    assert recorded.resolution is None and recorded.resolved_by is None
    assert recorded.candidate_rule_id not in _rules(), "a refused candidate is not a rule"
    assert recorded.candidate_text
    [entry] = recorded.conflicting
    assert entry["rule_id"] == str(existing.id)
    assert 0.0 <= entry["similarity"] <= 1.0
    assert entry["reasoning"] == REDACTED_REASONING and JUDGE_PHONE not in REDACTED_REASONING
    with get_session() as session:
        action = session.get(ReviewerAction, recorded.reviewer_action_id)
        assert action is not None and action.induction_outcome == "conflict"


# --- GET /memory/rules -------------------------------------------------------------------


def test_rules_report_their_source_languages_and_how_often_analyses_used_them(
    client: TestClient,
    headers: dict[str, str],
    submit: Callable[..., str],
    review_call: Callable[[str], None],
    transcripts: dict[str, str],
    contradicting_judge: None,
) -> None:
    review_call(submit(transcripts["mixed"], "mixed"))  # stores the first rule, from a mixed call
    submit(transcripts["en"], "en")
    submit(transcripts["ar"], "ar")  # both analyses retrieve it (top-k 5, one rule)
    with get_session() as session:
        session.add(
            MemoryRuleRecord(
                rule_text="retired",
                created_at=sa.func.now(),
                retired=True,
                source_correction_ids=[],
            )
        )

    response = client.get("/memory/rules", headers=headers)

    assert response.status_code == 200
    [rule] = response.json()["rules"]
    assert rule["languages"] == ["mixed"]
    assert rule["source_corrections"] == 1
    assert rule["analyses_used"] == 2


def test_a_merged_rule_lists_every_source_language_once(
    client: TestClient, headers: dict[str, str], submit: Callable[..., str], review_call: Any, llm: Any
) -> None:
    review_call(submit())
    with get_session() as session:
        [action] = session.query(ReviewerAction).all()
        [rule] = session.query(MemoryRuleRecord).all()
        rule.source_correction_ids = [str(action.id), str(action.id), str(uuid.uuid4())]
    [view] = client.get("/memory/rules", headers=headers).json()["rules"]
    assert view["languages"] == ["en"] and view["source_corrections"] == 3


# --- GET /memory/conflicts -------------------------------------------------------------------


def test_conflicts_list_both_rules_side_by_side_with_their_source(
    client: TestClient, headers: dict[str, str], make_conflicts: Callable[..., list[str]]
) -> None:
    ar_call, mixed_call, en_call = make_conflicts("ar", "mixed", "en")
    [existing] = [r for r in _rules().values() if not r.retired]

    response = client.get("/memory/conflicts", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["pending"] == {"ar": 1, "en": 1, "mixed": 1}
    assert [c["source"]["call_id"] for c in body["conflicts"]] == [ar_call, mixed_call, en_call]
    first = body["conflicts"][0]
    assert first["source"]["language"] == "ar"
    assert first["source"]["reviewer_id"] == "qa-reviewer-7"
    assert first["candidate_text"]
    [side] = first["conflicting"]
    assert side["rule_id"] == str(existing.id)
    assert side["text"] == existing.rule_text and side["active"] is True
    assert isinstance(side["similarity"], float) and side["reasoning"] == REDACTED_REASONING


def test_resolved_conflicts_leave_the_pending_list(
    client: TestClient, headers: dict[str, str], make_conflicts: Callable[..., list[str]], consolidations: Any
) -> None:
    make_conflicts("ar", "en")
    first = client.get("/memory/conflicts", headers=headers).json()["conflicts"][0]["id"]
    client.post(f"/memory/conflicts/{first}/resolve", json={"action": "keep_existing"}, headers=headers)

    body = client.get("/memory/conflicts", headers=headers).json()
    assert [c["source"]["language"] for c in body["conflicts"]] == ["en"]
    assert body["pending"] == {"ar": 0, "en": 1, "mixed": 0}


# --- POST /memory/conflicts/{id}/resolve -------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "existing_active", "candidate_active", "consolidation"),
    [
        ("keep_existing", True, False, False),
        ("keep_new", False, True, True),
        ("discard", False, False, False),
        ("not_a_conflict", True, True, True),
    ],
)
def test_each_action_changes_the_store_as_defined(
    client: TestClient,
    headers: dict[str, str],
    make_conflicts: Callable[..., list[str]],
    consolidations: list[None],
    action: str,
    existing_active: bool,
    candidate_active: bool,
    consolidation: bool,
) -> None:
    make_conflicts("mixed")
    [existing] = [r for r in _rules().values() if not r.retired]
    [conflict] = _conflicts()

    response = client.post(
        f"/memory/conflicts/{conflict.id}/resolve", json={"action": action}, headers=headers
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["action"] == action and body["resolved_by"] == "qa-reviewer-7"
    assert body["consolidation_queued"] is consolidation and len(consolidations) == int(consolidation)
    assert body["retired_rule_ids"] == ([] if existing_active else [str(existing.id)])
    assert body["activated_rule_id"] == (str(conflict.candidate_rule_id) if candidate_active else None)

    rules = _rules()
    assert rules[existing.id].retired is not existing_active
    candidate = rules.get(conflict.candidate_rule_id)
    assert (candidate is not None and not candidate.retired) is candidate_active
    if candidate is not None:
        assert candidate.rule_text == conflict.candidate_text
        assert candidate.source_correction_ids == [str(conflict.reviewer_action_id)]
    [stored] = _conflicts()
    assert stored.resolution == action and stored.resolved_by == "qa-reviewer-7"
    assert stored.resolved_at is not None


def test_activating_a_candidate_runs_consolidation_on_the_worker(
    client: TestClient, headers: dict[str, str], make_conflicts: Callable[..., list[str]]
) -> None:
    """Eager Celery: the queued consolidation really runs (fake embedder and LLM) and succeeds."""
    make_conflicts("en")
    [conflict] = _conflicts()
    response = client.post(
        f"/memory/conflicts/{conflict.id}/resolve", json={"action": "not_a_conflict"}, headers=headers
    )
    assert response.status_code == 200 and response.json()["consolidation_queued"] is True
    assert sum(not r.retired for r in _rules().values()) >= 1


def test_a_conflict_is_resolved_once(
    client: TestClient, headers: dict[str, str], make_conflicts: Callable[..., list[str]], consolidations: Any
) -> None:
    make_conflicts("en")
    [conflict] = _conflicts()
    url = f"/memory/conflicts/{conflict.id}/resolve"
    assert client.post(url, json={"action": "keep_existing"}, headers=headers).status_code == 200

    second = client.post(url, json={"action": "keep_new"}, headers=headers)

    assert second.status_code == 409
    [stored] = _conflicts()
    assert stored.resolution == "keep_existing"
    assert conflict.candidate_rule_id not in _rules()


# --- Consolidation leaves rules held by an open conflict alone (phase 7.1) ---------------


def _duplicate_of(rule: MemoryRuleRecord) -> uuid.UUID:
    """Store a second active rule with the same text: consolidation would merge the two."""
    twin = uuid.uuid4()
    with get_session() as session:
        session.add(
            MemoryRuleRecord(
                id=twin,
                rule_text=rule.rule_text,
                source_correction_ids=list(rule.source_correction_ids),
                created_at=rule.created_at,
            )
        )
    return twin


def test_a_conflict_recorded_in_a_run_still_names_active_rules_after_that_runs_consolidation(
    submit: Callable[..., str],
    review_call: Callable[[str], None],
    transcripts: dict[str, str],
    contradicting_judge: None,
) -> None:
    """The 2026-10-11 bug: the run that records a conflict then consolidates the whole store.

    Before 7.1 the conflicted rule and its duplicate merged in that same run, so
    the open conflict named only retired rules and `keep_new` / `discard` were no-ops.
    """
    review_call(submit(transcripts["en"], "en"))
    [existing] = [r for r in _rules().values() if not r.retired]
    twin = _duplicate_of(existing)

    review_call(submit(transcripts["ar"], "ar"))

    [conflict] = _conflicts()
    rules = _rules()
    named = [uuid.UUID(entry["rule_id"]) for entry in conflict.conflicting]
    assert existing.id in named
    assert all(not rules[rule_id].retired for rule_id in named), "an open conflict names a retired rule"
    assert not rules[existing.id].retired and not rules[twin].retired, "held rules must not be merged"


def test_held_rules_are_merged_once_their_conflict_is_resolved(
    client: TestClient,
    headers: dict[str, str],
    make_conflicts: Callable[..., list[str]],
    consolidations: list[None],
) -> None:
    make_conflicts("mixed")
    [existing] = [r for r in _rules().values() if not r.retired]
    twin = _duplicate_of(existing)
    [conflict] = _conflicts()

    # Open conflict: consolidation leaves both alone.
    assert persistence.consolidate_store(embed=tasks._induction_embed) == 0
    assert not _rules()[existing.id].retired and not _rules()[twin].retired

    resolved = client.post(
        f"/memory/conflicts/{conflict.id}/resolve", json={"action": "keep_existing"}, headers=headers
    )
    assert resolved.status_code == 200

    # Resolved: they are ordinary rules again, and the duplicates merge.
    assert persistence.consolidate_store(embed=tasks._induction_embed) == 2
    rules = _rules()
    assert rules[existing.id].retired and rules[twin].retired
    [merged] = [r for r in rules.values() if not r.retired]
    assert set(merged.source_correction_ids) == set(existing.source_correction_ids)


def test_rules_held_by_open_conflicts_lists_only_unresolved_conflicts(
    client: TestClient,
    headers: dict[str, str],
    make_conflicts: Callable[..., list[str]],
    consolidations: list[None],
) -> None:
    make_conflicts("en", "ar")
    first, second = _conflicts()
    named = {uuid.UUID(entry["rule_id"]) for c in (first, second) for entry in c.conflicting}
    with get_session() as session:
        assert persistence.rules_held_by_open_conflicts(session) == named

    for c in (first, second):
        client.post(f"/memory/conflicts/{c.id}/resolve", json={"action": "keep_existing"}, headers=headers)
    with get_session() as session:
        assert persistence.rules_held_by_open_conflicts(session) == set()


@pytest.mark.parametrize(
    "body",
    [{"action": "keep_both"}, {"action": "KEEP_NEW"}, {"action": ""}, {}, {"action": None}],
)
def test_resolve_rejects_anything_but_the_four_actions(
    client: TestClient,
    headers: dict[str, str],
    make_conflicts: Callable[..., list[str]],
    body: dict[str, Any],
) -> None:
    make_conflicts("en")
    [conflict] = _conflicts()

    response = client.post(f"/memory/conflicts/{conflict.id}/resolve", json=body, headers=headers)

    assert response.status_code == 422
    assert _conflicts()[0].resolution is None


def test_resolve_unknown_or_malformed_id(client: TestClient, headers: dict[str, str]) -> None:
    body = {"action": "discard"}
    assert (
        client.post(f"/memory/conflicts/{uuid.uuid4()}/resolve", json=body, headers=headers).status_code
        == 404
    )
    assert client.post("/memory/conflicts/not-a-uuid/resolve", json=body, headers=headers).status_code == 422


@contextlib.contextmanager
def store_locked() -> Iterator[None]:
    """Another writer (an induction run) holds the memory write lock."""
    with get_engine().connect() as conn:
        conn.execute(sa.text("SELECT pg_advisory_lock(:k)"), {"k": _INDUCTION_LOCK_KEY})
        try:
            yield
        finally:
            conn.execute(sa.text("SELECT pg_advisory_unlock(:k)"), {"k": _INDUCTION_LOCK_KEY})


def test_resolve_answers_503_instead_of_waiting_while_the_store_is_written(
    client: TestClient, headers: dict[str, str], make_conflicts: Callable[..., list[str]]
) -> None:
    make_conflicts("en")
    [conflict] = _conflicts()
    url = f"/memory/conflicts/{conflict.id}/resolve"

    with store_locked():
        response = client.post(url, json={"action": "discard"}, headers=headers)

    assert response.status_code == 503
    assert _conflicts()[0].resolution is None
    assert client.post(url, json={"action": "discard"}, headers=headers).status_code == 200


# --- Auth --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/memory/rules"),
        ("get", "/memory/conflicts"),
        ("post", f"/memory/conflicts/{uuid.uuid4()}/resolve"),
    ],
)
@pytest.mark.parametrize(
    "sent",
    [
        {},
        {"X-API-Key": "", "X-Reviewer-Id": "qa-reviewer-7"},
        {"X-API-Key": "wrong", "X-Reviewer-Id": "qa-reviewer-7"},
        {"X-API-Key": "test-api-key"},
        {"X-API-Key": "test-api-key", "X-Reviewer-Id": "  "},
    ],
)
def test_memory_routes_need_a_key_and_a_reviewer(
    client: TestClient, method: str, path: str, sent: dict[str, str]
) -> None:
    kwargs: dict[str, Any] = {"headers": sent}
    if method == "post":
        kwargs["json"] = {"action": "discard"}
    assert getattr(client, method)(path, **kwargs).status_code == 401
