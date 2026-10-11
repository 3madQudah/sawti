"""Upload → ASR (mocked) → redaction → analysis, end to end in-process (phase 6.4).

Celery is eager, so one `POST /calls/audio` runs the ASR task, which enqueues
the analysis task, which runs the real graph with the fake LLM. Langfuse is a
recording fake. What arrives at the review screen must be indistinguishable
from a `POST /calls` of the same transcript, and the raw PII must be nowhere:
not in the database, not in a trace.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from fastapi.testclient import TestClient

from sawti.agent.nodes import extract as extract_module
from sawti.db.session import get_engine
from sawti.observability import tracing

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "observability"))
from fake_langfuse import FakeLangfuse

PII = ["0791234567", "lina.haddad@example.com"]
RAW = (
    "Agent: Good morning, can I have your number?\n"
    f"Customer: It is {PII[0]}, email {PII[1]}.\n"
    "Agent: I will send a technician tomorrow before noon.\n"
    "Customer: Thank you.\n"
)


def _everything_stored() -> str:
    with get_engine().connect() as conn:
        tables = list(
            conn.execute(sa.text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")).scalars()
        )
        return "\n".join(str(row) for t in tables for row in conn.execute(sa.text(f'SELECT * FROM "{t}"')))


def test_uploaded_audio_reaches_the_review_queue_redacted(
    client: TestClient,
    headers: dict[str, str],
    upload: Callable[..., Any],
    submit: Callable[..., str],
    fake_asr: Any,
    llm: Any,
    audio_dir: Path,
    monkeypatch: Any,
) -> None:
    langfuse = FakeLangfuse()
    monkeypatch.setattr(tracing, "get_client", lambda: langfuse)
    traced = tracing.TracedProvider(llm, provider_name="fake")
    monkeypatch.setattr(extract_module, "get_llm_provider", lambda: traced)
    fake_asr.transcript = RAW

    response = upload(filename="call.m4a", language_hint="en", agent_external_id="agent-42")
    assert response.status_code == 202, response.text
    call_id = response.json()["call_id"]

    detail = client.get(f"/calls/{call_id}", headers=headers).json()
    # The fake LLM proposes one ungrounded claim, so the call escalates.
    assert detail["status"] == "awaiting_review"
    assert detail["pii_redacted_count"] == 2
    assert detail["agent_name"] == "agent-42"
    assert [t["speaker"] for t in detail["turns"]] == ["Agent", "Customer", "Agent", "Customer"]
    assert detail["claims"], "the analysis ran on the transcribed call"
    assert detail["failure_reason"] is None
    queue = client.get("/reviews", headers=headers).json()
    assert [item["call_id"] for item in queue["items"]] == [call_id]

    # The same transcript submitted as text is stored identically: one redaction path.
    text_call = client.get(f"/calls/{submit(RAW, 'en')}", headers=headers).json()
    assert text_call["redacted_transcript"] == detail["redacted_transcript"]

    assert list(audio_dir.iterdir()) == [], "raw audio deleted after transcription"
    stored, sent = _everything_stored(), langfuse.everything_sent()
    for value in PII:
        assert value not in detail["redacted_transcript"]
        assert value not in stored, value
        assert value not in sent, value
    assert {trace.payload["name"] for trace in langfuse.of("trace")} == {"analyze_call"}
