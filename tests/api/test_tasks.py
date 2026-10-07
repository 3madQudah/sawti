"""Tests for `sawti.api.tasks`: the worker side of the service.

Phase 6.2: idempotency, transient-error retries, the error -> escalate path,
memory-rule retrieval, and that no PII reaches any table — checkpoints included.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest
import sqlalchemy as sa

from sawti.api import tasks
from sawti.db.models import Agent, Call, CallAnalysisRecord, MemoryRuleRecord
from sawti.db.session import get_engine, get_session
from sawti.llm.provider import TransientProviderError
from sawti.memory.store import MemoryStore
from sawti.privacy.redaction import redact
from sawti.schemas import CallStatus


class _Http429Error(Exception):
    """Shaped like google-genai's `APIError`: an HTTP status on `.code`."""

    code = 429


@pytest.fixture
def queued_call(transcripts: dict[str, str]) -> Callable[..., str]:
    """Insert a `queued` call directly (as `POST /calls` would) and return its id."""

    def _make(language: str = "en", transcript: str | None = None) -> str:
        call_id = uuid.uuid4()
        now = datetime.now(UTC)
        redaction = redact(transcript or transcripts[language])
        with get_session() as session:
            agent = Agent(id=uuid.uuid4(), name="a", created_at=now)
            session.add(agent)
            session.flush()
            session.add(
                Call(
                    id=call_id,
                    agent_id=agent.id,
                    redacted_transcript=redaction.redacted_text,
                    pii_redacted_count=redaction.redaction_count,
                    language=language,
                    occurred_at=now,
                    status=CallStatus.QUEUED,
                    queued_at=now,
                )
            )
        return str(call_id)

    return _make


def _call(call_id: str) -> Call:
    with get_session() as session:
        call = session.get(Call, uuid.UUID(call_id))
        assert call is not None
        return call


def _records(call_id: str) -> list[CallAnalysisRecord]:
    with get_session() as session:
        return session.query(CallAnalysisRecord).filter_by(call_id=uuid.UUID(call_id)).all()


def test_analyze_twice_makes_one_record_and_one_llm_call(queued_call: Callable[..., str], llm: Any) -> None:
    """A redelivered message is a no-op: the guard only claims a `queued` call."""
    call_id = queued_call()
    assert tasks.analyze_call_task.apply(args=[call_id]).get() == "awaiting_review"
    assert tasks.analyze_call_task.apply(args=[call_id]).get() == "skipped"
    assert llm.calls == 1
    assert len(_records(call_id)) == 1
    assert _call(call_id).attempts == 1


def test_auto_passing_call_completes(queued_call: Callable[..., str], llm: Any) -> None:
    llm.ungrounded = 0
    call_id = queued_call()
    assert tasks.analyze_call_task.apply(args=[call_id]).get() == "completed"
    call = _call(call_id)
    assert call.started_at is not None and call.finished_at is not None
    assert call.queued_at <= call.started_at <= call.finished_at
    assert _records(call_id)[0].escalation_reason is None


def test_transient_error_requeues_and_the_retry_succeeds(queued_call: Callable[..., str], llm: Any) -> None:
    """429 -> the call goes back to `queued` and is retried; the retry claims it again and finishes.

    With eager propagation off, Celery runs the retry inline, which is what a
    broker redelivery does in production. `attempts == 2` shows the call was
    put back to `queued` (or the guard would have skipped the retry).
    """
    tasks.celery_app.conf.task_eager_propagates = False
    llm.errors = [_Http429Error("RESOURCE_EXHAUSTED")]
    call_id = queued_call()

    assert tasks.analyze_call_task.apply(args=[call_id]).get() == "awaiting_review"
    assert _call(call_id).attempts == 2
    assert llm.calls == 2
    assert len(_records(call_id)) == 1


def test_retries_exhausted_marks_the_call_failed(queued_call: Callable[..., str], llm: Any) -> None:
    """With max_retries=2 (fixture), a third transient failure is final."""
    llm.errors = [TransientProviderError("503")] * 5
    call_id = queued_call()
    assert tasks.analyze_call_task.apply(args=[call_id], retries=2).get() == "failed"
    call = _call(call_id)
    assert call.status == CallStatus.FAILED and call.finished_at is not None
    assert _records(call_id) == []


def test_non_transient_provider_error_escalates_to_a_human(queued_call: Callable[..., str], llm: Any) -> None:
    """Not retried: the graph's error -> escalate path puts it in the review queue."""
    llm.errors = [ValueError("response failed schema validation")]
    call_id = queued_call()
    assert tasks.analyze_call_task.apply(args=[call_id]).get() == "awaiting_review"
    assert llm.calls == 1
    record = _records(call_id)[0]
    assert record.escalation_reason.startswith("Analysis error: extract: ValueError")
    assert record.confidence == 0.0 and record.requires_human_review
    assert record.payload["commitments"] == []


def test_retrieved_memory_rules_reach_the_prompt_and_are_recorded(
    queued_call: Callable[..., str], llm: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rules come from `memory_rules`, retired ones excluded; their ids land on the record."""
    monkeypatch.setattr(tasks, "_memory_store_factory", lambda: MemoryStore(embed=lambda t: np.ones(3)))
    now = datetime.now(UTC)
    active, retired = uuid.uuid4(), uuid.uuid4()
    with get_session() as session:
        session.add(MemoryRuleRecord(id=active, rule_text="Quote callbacks with their time.", created_at=now))
        session.add(MemoryRuleRecord(id=retired, rule_text="Old rule.", created_at=now, retired=True))

    prompts: list[str] = []
    original = llm.structured_complete

    async def _capture(prompt: str, **kwargs: Any) -> Any:
        prompts.append(kwargs.get("system") or "")
        return await original(prompt, **kwargs)

    llm.structured_complete = _capture
    call_id = queued_call()
    tasks.analyze_call_task.apply(args=[call_id]).get()

    assert _records(call_id)[0].retrieved_rule_ids == [str(active)]
    assert "Quote callbacks with their time." in prompts[0]
    assert "Old rule." not in prompts[0]


def test_no_pii_reaches_any_table_including_checkpoints(queued_call: Callable[..., str], llm: Any) -> None:
    """Known PII in, then every row of every app table and every checkpoint byte searched for it."""
    pii = ["0791234567", "lina.haddad@example.com", "+962 79 555 1234", "JO94CBJO0010000000000131000302"]
    transcript = (
        f"Agent: Can I have your number?\nCustomer: It is {pii[0]}, email {pii[1]}.\n"
        f"Agent: And a backup?\nCustomer: {pii[2]}, IBAN {pii[3]}.\n"
    )
    call_id = queued_call("en", transcript)
    assert _call(call_id).pii_redacted_count == len(pii)  # all four were really there to leak
    tasks.analyze_call_task.apply(args=[call_id]).get()
    with get_engine().connect() as conn:
        assert conn.execute(
            sa.text("SELECT count(*) FROM checkpoints WHERE thread_id = :t"), {"t": call_id}
        ).scalar_one()
        text_rows = [
            str(row)
            for table in ("calls", "call_analyses", "reviewer_actions", "review_submissions", "checkpoints")
            for row in conn.execute(sa.text(f"SELECT * FROM {table}"))
        ]
        byte_rows = [
            bytes(blob or b"")
            for query in (
                "SELECT blob FROM checkpoint_blobs",
                "SELECT blob FROM checkpoint_writes",
            )
            for (blob,) in conn.execute(sa.text(query))
        ]
    for value in pii:
        assert not any(value in row for row in text_rows), value
        assert not any(value.encode() in blob for blob in byte_rows), value
