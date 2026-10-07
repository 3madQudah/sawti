"""Tests for `sawti.api.routes.reviews` and `sawti.api.deps.get_current_reviewer`.

Phase 6.2: the queue, per-language counts, verdict validation, the
Correction mapping, and the resume that ends in `reviewed`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sawti.db.models import ReviewerAction, ReviewSubmission
from sawti.db.session import get_session
from sawti.schemas import Correction

ALL_CONFIRM = {
    "verdicts": [
        {"claim_id": "commitments[0]", "verdict": "confirm"},
        {"claim_id": "commitments[1]", "verdict": "confirm"},
    ]
}


def _review(client: TestClient, headers: dict[str, str], call_id: str, body: dict[str, Any]) -> Any:
    return client.post(f"/reviews/{call_id}/corrections", json=body, headers=headers)


# --- Identity ---------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_headers",
    [
        {},  # nothing
        {"X-API-Key": "test-api-key"},  # key but no reviewer
        {"X-Reviewer-Id": "qa-1"},  # reviewer but no key
        {"X-API-Key": "wrong", "X-Reviewer-Id": "qa-1"},  # wrong key
        {"X-API-Key": "test-api-key", "X-Reviewer-Id": "   "},  # blank reviewer
    ],
)
def test_reviewer_routes_401_without_valid_identity(client: TestClient, bad_headers: dict[str, str]) -> None:
    assert client.get("/reviews", headers=bad_headers).status_code == 401


def test_submit_call_401s_on_a_wrong_key(client: TestClient) -> None:
    body = {"transcript": "Agent: hi\n", "language": "en", "agent_external_id": "a"}
    assert client.post("/calls", json=body, headers={"X-API-Key": "wrong"}).status_code == 401


def test_protected_routes_fail_closed_when_no_key_is_configured(
    client: TestClient, headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from sawti.config import get_settings

    # Empty is how "unset" is expressed here (an unset env var would fall back
    # to .env); it must also not be matched by an empty header.
    monkeypatch.setenv("SAWTI_API_KEY", "")
    get_settings.cache_clear()
    assert client.get("/reviews", headers=headers).status_code == 503
    assert client.get("/reviews", headers={**headers, "X-API-Key": ""}).status_code == 503


def test_reviewer_identity_reaches_reviewer_action(
    client: TestClient, headers: dict[str, str], submit: Callable[..., str], llm: object
) -> None:
    call_id = submit()
    body = {
        "verdicts": [
            {"claim_id": "commitments[0]", "verdict": "reject"},
            {"claim_id": "commitments[1]", "verdict": "confirm"},
        ]
    }
    assert _review(client, {**headers, "X-Reviewer-Id": "qa-lina"}, call_id, body).status_code == 202
    with get_session() as session:
        actions = session.query(ReviewerAction).all()
        submission = session.query(ReviewSubmission).one()
        assert [a.reviewer_id for a in actions] == ["qa-lina"]
        assert Correction.model_validate(actions[0].payload).reviewer_id == "qa-lina"
        assert submission.reviewer_id == "qa-lina"


# --- Queue --------------------------------------------------------------------


def test_queue_lists_escalated_calls_with_per_language_counts(
    client: TestClient,
    headers: dict[str, str],
    submit: Callable[..., str],
    transcripts: dict[str, str],
    llm: object,
) -> None:
    ids = {lang: submit(text, lang) for lang, text in transcripts.items()}
    submit(transcripts["ar"], "ar")

    queue = client.get("/reviews", headers=headers).json()
    assert queue["pending"] == {"ar": 2, "en": 1, "mixed": 1}
    assert len(queue["items"]) == 4
    item = next(i for i in queue["items"] if i["call_id"] == ids["en"])
    assert item["agent_name"] == "agent-42"
    assert item["grounding_coverage"] == 2 / 3
    assert item["escalation_reason"].startswith("1 of 3 claims")
    assert item["waiting_since"] is not None

    filtered = client.get("/reviews", params={"language": "ar"}, headers=headers).json()
    assert {i["language"] for i in filtered["items"]} == {"ar"}
    # Counts ignore the filter, so the screen's counters do not jump.
    assert filtered["pending"] == {"ar": 2, "en": 1, "mixed": 1}


def test_queue_omits_calls_that_auto_passed(
    client: TestClient, headers: dict[str, str], submit: Callable[..., str], llm: Any
) -> None:
    llm.ungrounded = 0
    submit()
    queue = client.get("/reviews", headers=headers).json()
    assert queue["items"] == [] and queue["pending"] == {"ar": 0, "en": 0, "mixed": 0}
    completed = client.get("/reviews", params={"status": "completed"}, headers=headers).json()
    assert len(completed["items"]) == 1


# --- Submission validation ------------------------------------------------------


def test_submission_missing_a_grounded_claim_is_rejected(
    client: TestClient, headers: dict[str, str], submit: Callable[..., str], llm: object
) -> None:
    call_id = submit()
    response = _review(
        client, headers, call_id, {"verdicts": [{"claim_id": "commitments[0]", "verdict": "confirm"}]}
    )
    assert response.status_code == 422
    assert response.json()["detail"] == {"missing_verdicts": ["commitments[1]"], "unknown_claims": []}
    with get_session() as session:
        assert session.query(ReviewSubmission).count() == 0


def test_submission_for_an_unknown_claim_is_rejected(
    client: TestClient, headers: dict[str, str], submit: Callable[..., str], llm: object
) -> None:
    body = {"verdicts": [*ALL_CONFIRM["verdicts"], {"claim_id": "rubric_scores[0]", "verdict": "confirm"}]}
    response = _review(client, headers, submit(), body)
    assert response.status_code == 422
    assert response.json()["detail"]["unknown_claims"] == ["rubric_scores[0]"]


@pytest.mark.parametrize(
    "verdict",
    [
        {"claim_id": "commitments[0]", "verdict": "correct"},  # correct without changes
        {"claim_id": "commitments[0]", "verdict": "correct", "note": "free text only"},  # still no changes
        {
            "claim_id": "commitments[0]",
            "verdict": "confirm",
            "changes": {"description": "x"},
        },  # changes w/o correct
        {"claim_id": "commitments[0]", "verdict": "correct", "changes": {}},  # empty changes
        {
            "claim_id": "commitments[0]",
            "verdict": "correct",
            "changes": {"score": 0.5},
        },  # not a commitment field
        {
            "claim_id": "commitments[0]",
            "verdict": "correct",
            "changes": {"description": "Follow up."},
        },  # no-op
        {
            "claim_id": "commitments[0]",
            "verdict": "correct",
            "changes": {"description": None},
        },  # only deadline may be null
    ],
)
def test_correct_needs_real_structured_changes(
    client: TestClient,
    headers: dict[str, str],
    submit: Callable[..., str],
    llm: object,
    verdict: dict[str, str],
) -> None:
    body = {"verdicts": [verdict, {"claim_id": "commitments[1]", "verdict": "confirm"}]}
    assert _review(client, headers, submit(), body).status_code == 422


def test_duplicate_verdicts_for_one_claim_are_rejected(
    client: TestClient, headers: dict[str, str], submit: Callable[..., str], llm: object
) -> None:
    body = {"verdicts": [*ALL_CONFIRM["verdicts"], {"claim_id": "commitments[0]", "verdict": "reject"}]}
    assert _review(client, headers, submit(), body).status_code == 422


def test_submission_for_a_call_not_awaiting_review_is_409(
    client: TestClient, headers: dict[str, str], submit: Callable[..., str], llm: Any
) -> None:
    llm.ungrounded = 0  # auto-passes -> completed
    assert _review(client, headers, submit(), {"verdicts": []}).status_code == 409


def test_second_submission_is_409(
    client: TestClient, headers: dict[str, str], submit: Callable[..., str], llm: object
) -> None:
    call_id = submit()
    assert _review(client, headers, call_id, ALL_CONFIRM).status_code == 202
    assert _review(client, headers, call_id, ALL_CONFIRM).status_code == 409


# --- Mapping and resume ---------------------------------------------------------


def test_review_resumes_the_graph_and_ends_reviewed(
    client: TestClient, headers: dict[str, str], submit: Callable[..., str], llm: object
) -> None:
    call_id = submit()
    response = _review(client, headers, call_id, ALL_CONFIRM)
    assert response.status_code == 202
    assert response.json()["corrections_recorded"] == 0

    detail = client.get(f"/calls/{call_id}", headers=headers).json()
    assert detail["status"] == "reviewed"
    assert detail["review"]["verdicts"] == [
        {**v, "changes": None, "note": None} for v in ALL_CONFIRM["verdicts"]
    ]
    assert client.get("/reviews", headers=headers).json()["pending"]["en"] == 0


def test_verdicts_map_onto_corrections(
    client: TestClient, headers: dict[str, str], submit: Callable[..., str], llm: object
) -> None:
    """reject -> claim removed; correct -> changes applied; confirm -> nothing. corrected != original."""
    call_id = submit()
    body = {
        "verdicts": [
            {"claim_id": "commitments[0]", "verdict": "reject"},
            {
                "claim_id": "commitments[1]",
                "verdict": "correct",
                "changes": {"description": "Send a technician.", "deadline": "2026-10-08T12:00:00+03:00"},
                "note": "The agent gave a time.",
            },
        ],
        "note": "Agent greeting is fine.",
    }
    assert _review(client, headers, call_id, body).json()["corrections_recorded"] == 2

    with get_session() as session:
        submission = session.query(ReviewSubmission).one()
        corrections = {
            c.error_location: c
            for c in (Correction.model_validate(a.payload) for a in session.query(ReviewerAction))
        }
        assert {a.review_submission_id for a in session.query(ReviewerAction)} == {submission.id}

    rejected = corrections["commitments"]
    assert rejected.corrected.commitments == rejected.original.commitments[1:]
    assert rejected.note == "Agent greeting is fine."

    corrected = corrections["commitments[1]"]
    assert corrected.corrected != corrected.original
    fixed = corrected.corrected.commitments[1]
    assert fixed.description == "Send a technician."
    assert fixed.deadline is not None and fixed.deadline.hour == 12
    assert fixed.evidence == corrected.original.commitments[1].evidence  # the quote is not editable
    assert corrected.note == "The agent gave a time.\n\nAgent greeting is fine."

    # Stored with only the fields the reviewer set ("unset" stays distinct from
    # "cleared"), and it still validates when the detail endpoint reads it back.
    assert submission.verdicts["verdicts"][1]["changes"] == {
        "description": "Send a technician.",
        "deadline": "2026-10-08T12:00:00+03:00",
    }
    detail = client.get(f"/calls/{call_id}", headers=headers).json()
    assert detail["review"]["verdicts"][1]["changes"]["description"] == "Send a technician."


def test_an_unset_api_key_rejects_every_protected_route(
    client: TestClient,
    headers: dict[str, str],
    submit: Callable[..., str],
    llm: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truly unset — no env var and no .env file — not just empty: every review route answers 503.

    `delenv` alone is not enough (Settings would fall back to the key in .env),
    so the dependency is handed settings built with no env file at all.
    """
    from sawti.api import deps
    from sawti.config import Settings

    call_id = submit()  # created while the key is still configured
    monkeypatch.delenv("SAWTI_API_KEY")
    unset = Settings(_env_file=None)  # type: ignore[call-arg]
    assert unset.api_key is None
    monkeypatch.setattr(deps, "get_settings", lambda: unset)

    protected = [
        ("get", "/reviews", None),
        ("get", f"/calls/{call_id}", None),
        ("post", f"/reviews/{call_id}/corrections", ALL_CONFIRM),
        ("post", "/calls", {"transcript": "Agent: hi\n", "language": "en", "agent_external_id": "a"}),
    ]
    for method, path, body in protected:
        response = getattr(client, method)(path, headers=headers, **({"json": body} if body else {}))
        assert response.status_code == 503, (method, path, response.status_code)
    assert client.get("/health").status_code == 200  # the one open route stays open
