"""End-to-end over HTTP against the running containers.

submit -> escalate -> queue -> review (a `correct` verdict) -> resumed ->
`reviewed`; then the memory loop: the review's correction is induced into a
rule, and a new, similar call retrieves it.

Phase 6.2 done-when. Marked `integration` and excluded from the default run;
`make e2e` starts the stack on a fresh scratch database (`sawti_e2e`, never
the dev data) with the deterministic `fake` LLM — every call escalates, and
induction answers offline — while embeddings, the checkpointer, Redis and
Postgres are the real ones. Nothing here touches the database directly.

Environment: `SAWTI_E2E_BASE_URL` (default http://localhost:8000) and
`SAWTI_API_KEY` (from the environment or `.env`, as the stack reads it).
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
import pytest

from sawti.config import Settings

pytestmark = pytest.mark.integration

BASE_URL = os.environ.get("SAWTI_E2E_BASE_URL", "http://localhost:8000")
TRANSCRIPT = (
    "Agent: Good morning, this is Omar from Orange, how can I help?\n"
    "Customer: My bill doubled this month, my number is 0795551234.\n"
    "Agent: I will refund the extra charge within three days.\n"
    "Customer: الله يعطيك العافية، thank you.\n"
)
SIMILAR = (
    "Agent: Hello, this is Omar from Orange.\n"
    "Customer: I was charged twice this month.\n"
    "Agent: I will refund the extra charge within three days.\n"
    "Customer: Thanks.\n"
)


def _headers() -> dict[str, str]:
    key = Settings().api_key
    assert key is not None and key.get_secret_value(), "SAWTI_API_KEY must be set for the e2e"
    return {"X-API-Key": key.get_secret_value(), "X-Reviewer-Id": "e2e-reviewer"}


def _wait_for(client: httpx.Client, call_id: str, status: str, timeout: float = 120.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    detail: dict[str, Any] = {}
    while time.monotonic() < deadline:
        detail = client.get(f"/calls/{call_id}", headers=_headers()).json()
        if detail["status"] == status:
            return detail
        assert detail["status"] != "failed", detail
        time.sleep(0.5)
    raise AssertionError(f"{call_id} never reached {status}; last: {detail.get('status')}")


def _wait_for_rule(client: httpx.Client, timeout: float = 120.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rules = client.get("/memory/rules", headers=_headers()).json()["rules"]
        if rules:
            return dict(rules[0])
        time.sleep(1.0)
    raise AssertionError("no memory rule was induced from the review")


def _submit(client: httpx.Client, transcript: str, language: str) -> str:
    response = client.post(
        "/calls",
        json={"transcript": transcript, "language": language, "agent_external_id": "e2e-agent"},
        headers=_headers(),
    )
    assert response.status_code == 202, response.text
    return str(response.json()["call_id"])


def _review(client: httpx.Client, call_id: str, verdicts: list[dict[str, Any]]) -> httpx.Response:
    return client.post(f"/reviews/{call_id}/corrections", json={"verdicts": verdicts}, headers=_headers())


def test_submit_escalate_review_resume_then_a_similar_call_uses_the_learned_rule() -> None:
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        assert client.get("/health").json() == {"status": "ok", "database": "ok", "redis": "ok"}
        assert client.get("/memory/rules", headers=_headers()).json()["rules"] == []  # a fresh database

        call_id = _submit(client, TRANSCRIPT, "mixed")
        detail = _wait_for(client, call_id, "awaiting_review")
        assert "0795551234" not in detail["redacted_transcript"]
        assert detail["pii_redacted_count"] == 1
        assert detail["escalation_reason"] and detail["rejected_claims"]
        text = detail["redacted_transcript"]
        for claim in detail["claims"]:
            assert text[claim["start_char"] : claim["end_char"]] == claim["quote"]

        queue = client.get("/reviews", params={"language": "mixed"}, headers=_headers()).json()
        assert call_id in {item["call_id"] for item in queue["items"]}
        assert queue["pending"]["mixed"] >= 1

        claims = [c["claim_id"] for c in detail["claims"]]
        verdicts: list[dict[str, Any]] = [
            {
                "claim_id": claims[0],
                "verdict": "correct",
                "changes": {"description": "Refund the extra charge within three days."},
            }
        ] + [{"claim_id": c, "verdict": "confirm"} for c in claims[1:]]
        assert _review(client, call_id, verdicts[:-1]).status_code == 422  # a claim left unjudged
        accepted = _review(client, call_id, verdicts)
        assert accepted.status_code == 202, accepted.text
        assert accepted.json()["corrections_recorded"] == 1  # the `correct` verdict

        reviewed = _wait_for(client, call_id, "reviewed")
        assert reviewed["review"]["verdicts"][0]["verdict"] == "correct"
        assert _review(client, call_id, verdicts).status_code == 409
        pending = client.get("/reviews", headers=_headers()).json()["items"]
        assert call_id not in {item["call_id"] for item in pending}

        # The memory loop: induction runs asynchronously after the review.
        rule = _wait_for_rule(client)
        assert rule["source_corrections"] == 1
        similar = _submit(client, SIMILAR, "en")
        retrieved = _wait_for(client, similar, "awaiting_review")["memory_rules"]
        assert rule["id"] in {r["id"] for r in retrieved}
