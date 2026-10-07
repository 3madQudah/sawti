"""Tests for `sawti.api.routes.calls`.

Phase 6.2: submit, redaction at the boundary, and `CallDetail`.
"""

from __future__ import annotations

from collections.abc import Callable

import sqlalchemy as sa
from fastapi.testclient import TestClient

from sawti.db.session import get_engine


def test_submit_returns_202_and_analysis_escalates(
    client: TestClient, headers: dict[str, str], submit: Callable[..., str], llm: object
) -> None:
    """The fake LLM proposes one unsupported claim of three -> coverage 2/3 < 0.7 -> awaiting_review."""
    call_id = submit()
    detail = client.get(f"/calls/{call_id}", headers=headers).json()
    assert detail["status"] == "awaiting_review"
    assert detail["grounding_coverage"] == 2 / 3
    assert detail["confidence_threshold"] == 0.7
    assert detail["attempts"] == 1
    assert detail["escalation_reason"] == (
        "1 of 3 claims had no verbatim evidence (grounding coverage 67%, threshold 70%)"
    )


def test_submit_stores_only_the_redacted_transcript(
    client: TestClient, headers: dict[str, str], submit: Callable[..., str], llm: object
) -> None:
    call_id = submit()
    with get_engine().connect() as conn:
        stored, count = conn.execute(
            sa.text("SELECT redacted_transcript, pii_redacted_count FROM calls WHERE id = :id"),
            {"id": call_id},
        ).one()
    assert "0791234567" not in stored
    assert "[NATIONAL_ID]" in stored
    assert count == 1


def test_submit_rejects_a_line_without_a_speaker_prefix(client: TestClient, headers: dict[str, str]) -> None:
    response = client.post(
        "/calls",
        json={"transcript": "Agent: hello\nno prefix here\n", "language": "en", "agent_external_id": "a"},
        headers=headers,
    )
    assert response.status_code == 422
    assert "line 2" in response.json()["detail"]
    with get_engine().connect() as conn:
        assert conn.execute(sa.text("SELECT count(*) FROM calls")).scalar_one() == 0


def test_submit_rejects_an_unknown_language(client: TestClient, headers: dict[str, str]) -> None:
    response = client.post(
        "/calls",
        json={"transcript": "Agent: hi\n", "language": "fr", "agent_external_id": "a"},
        headers=headers,
    )
    assert response.status_code == 422


def test_call_detail_offsets_index_the_redacted_transcript(
    client: TestClient,
    headers: dict[str, str],
    submit: Callable[..., str],
    transcripts: dict[str, str],
    llm: object,
) -> None:
    """For every grounded claim and every turn, the offsets slice out exactly the quoted text."""
    for language, transcript in transcripts.items():
        detail = client.get(f"/calls/{submit(transcript, language)}", headers=headers).json()
        text = detail["redacted_transcript"]
        assert detail["claims"], language
        for claim in detail["claims"]:
            assert text[claim["start_char"] : claim["end_char"]] == claim["quote"]
            turn = detail["turns"][claim["turn_index"]]
            assert turn["start_char"] <= claim["start_char"] < turn["end_char"]
        for turn in detail["turns"]:
            assert text[turn["start_char"] : turn["end_char"]] == turn["text"]


def test_call_detail_turns_carry_speaker_and_per_turn_language(
    client: TestClient,
    headers: dict[str, str],
    submit: Callable[..., str],
    transcripts: dict[str, str],
    llm: object,
) -> None:
    detail = client.get(f"/calls/{submit(transcripts['mixed'], 'mixed')}", headers=headers).json()
    assert [t["speaker"] for t in detail["turns"]] == ["Agent", "Customer", "Agent", "Customer"]
    assert {t["language"] for t in detail["turns"]} == {"mixed"}
    detail = client.get(f"/calls/{submit(transcripts['ar'], 'ar')}", headers=headers).json()
    assert {t["language"] for t in detail["turns"]} == {"ar"}


def test_call_detail_lists_rejected_claims_with_their_attempted_quote(
    client: TestClient, headers: dict[str, str], submit: Callable[..., str], llm: object
) -> None:
    detail = client.get(f"/calls/{submit()}", headers=headers).json()
    assert detail["rejected_claims"] == [
        {
            "kind": "commitment",
            "attempted_quote": "(fake provider: unsupported claim 1)",
            "speaker": "Agent",
            "description": "An unsupported promise.",
        }
    ]
    assert [c["claim_id"] for c in detail["claims"]] == ["commitments[0]", "commitments[1]"]


def test_call_detail_404s_for_an_unknown_call(client: TestClient, headers: dict[str, str]) -> None:
    response = client.get("/calls/00000000-0000-0000-0000-000000000000", headers=headers)
    assert response.status_code == 404
