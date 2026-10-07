"""The whole service pipeline, traced: every node and LLM call, and no PII anywhere (phase 6.3).

Submits a transcript with known PII over HTTP; the eager worker runs the
graph; a fake Langfuse client records every payload the tracer sends.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sawti.agent.nodes import extract as extract_module
from sawti.observability import tracing

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "observability"))
from fake_langfuse import FakeLangfuse

PII = ["0791234567", "lina.haddad@example.com", "+962 79 555 1234", "JO94CBJO0010000000000131000302"]
TRANSCRIPT = (
    f"Agent: Good morning, can I have your number?\nCustomer: It is {PII[0]}, email {PII[1]}.\n"
    f"Agent: I will send a technician tomorrow before noon.\nCustomer: Backup {PII[2]}, IBAN {PII[3]}.\n"
)
NODES = {"extract", "ground", "score", "compliance", "assess_confidence", "escalate"}


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, llm: Any) -> FakeLangfuse:
    client = FakeLangfuse()
    monkeypatch.setattr(tracing, "get_client", lambda: client)

    # The fake provider reports usage itself (word counts; there is no model).
    traced = tracing.TracedProvider(llm, provider_name="fake")
    monkeypatch.setattr(extract_module, "get_llm_provider", lambda: traced)
    return client


def test_every_node_and_llm_call_is_traced_and_no_pii_reaches_langfuse(
    client: TestClient, headers: dict[str, str], submit: Callable[..., str], fake: FakeLangfuse
) -> None:
    call_id = submit(TRANSCRIPT, "en")
    detail = client.get(f"/calls/{call_id}", headers=headers).json()
    assert detail["status"] == "awaiting_review" and detail["pii_redacted_count"] == 4

    sent = fake.everything_sent()
    for value in PII:
        assert value not in sent, value

    [trace] = fake.of("trace")
    assert trace.payload["session_id"] == call_id and trace.payload["name"] == "analyze_call"
    assert {s.payload["name"] for s in fake.of("span")} == {f"node:{n}" for n in NODES}
    for span in fake.of("span"):
        assert span.payload["metadata"] == {
            "node": span.payload["name"].removeprefix("node:"),
            "call_id": call_id,
            "language": "en",
        }
        assert span.ended is not None
    escalate = next(s for s in fake.of("span") if s.payload["name"] == "node:escalate")
    assert escalate.ended["status_message"] == "interrupted for human review"
    ground = next(s for s in fake.of("span") if s.payload["name"] == "node:ground")
    assert ground.ended["output"]["rejected_claims_count"] == 1

    [generation] = fake.of("generation")
    usage = generation.payload["usage"]
    assert usage["unit"] == "TOKENS" and usage["input"] > 0 and usage["output"] > 0
    assert generation.payload["model"] == "fake"
    assert generation.payload["metadata"]["call_id"] == call_id
    assert generation.payload["metadata"]["language"] == "en"
    assert "[NATIONAL_ID]" in str(generation.payload["input"])  # it saw the redacted text
    assert fake.flushed >= 1


def test_the_resume_is_its_own_trace_in_the_same_session(
    client: TestClient, headers: dict[str, str], submit: Callable[..., str], fake: FakeLangfuse
) -> None:
    call_id = submit(TRANSCRIPT, "en")
    body = {
        "verdicts": [
            {"claim_id": "commitments[0]", "verdict": "confirm"},
            {"claim_id": "commitments[1]", "verdict": "confirm"},
        ]
    }
    assert client.post(f"/reviews/{call_id}/corrections", json=body, headers=headers).status_code == 202
    traces = fake.of("trace")
    assert [t.payload["name"] for t in traces] == ["analyze_call", "resume_call"]
    assert {t.payload["session_id"] for t in traces} == {call_id}
    for value in PII:
        assert value not in fake.everything_sent()
