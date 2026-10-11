"""The ASR task, with Whisper and pyannote replaced by `FakeAsr` (phase 6.4).

What it must guarantee (docs/09-DECISIONS.md, 2026-10-10): the transcript is
redacted by the same function as `POST /calls`; the raw text is never stored,
logged or traced; raw audio is deleted after the attempt unless
`SAWTI_KEEP_AUDIO=true`; a failure is `failed` with a reason that carries no
transcript text.
"""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from sawti.api import asr_tasks, tasks
from sawti.api.ingest import redact_transcript
from sawti.config import get_settings
from sawti.data.transcript_parser import UnattributableLineError
from sawti.db.models import Call
from sawti.db.session import get_engine, get_session
from sawti.memory.capture import get_or_create_placeholder_agent
from sawti.observability import tracing
from sawti.schemas import CallStatus

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "observability"))
from fake_langfuse import FakeLangfuse

PHONE = "0791234567"
EMAIL = "lina.haddad@example.com"
RAW = (
    "Agent: Good morning, this is Sara from Zain.\n"
    f"Customer: Call me on {PHONE} or write to {EMAIL}.\n"
    "Agent: I will send a technician tomorrow.\n"
)


@pytest.fixture
def analysis_enqueued(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(tasks.analyze_call_task, "delay", lambda call_id: calls.append(call_id))
    return calls


@pytest.fixture
def uploaded(audio_dir: Path, make_audio: Callable[..., bytes]) -> Callable[..., str]:
    """A `transcribing` call with its audio stored, as `POST /calls/audio` leaves it."""

    def _uploaded(language: str = "en") -> str:
        call_id = uuid.uuid4()
        audio_dir.mkdir(parents=True, exist_ok=True)
        (audio_dir / f"{call_id}.wav").write_bytes(make_audio(".wav"))
        with get_session() as session:
            agent_id = get_or_create_placeholder_agent(session, external_id="a", name="a")
            session.add(
                Call(
                    id=call_id,
                    agent_id=agent_id,
                    audio_path=f"{call_id}.wav",
                    language=language,
                    occurred_at=datetime.now(UTC),
                    status=CallStatus.TRANSCRIBING,
                )
            )
        return str(call_id)

    return _uploaded


def _call(call_id: str) -> Call:
    with get_session() as session:
        call = session.get(Call, uuid.UUID(call_id))
        assert call is not None
        session.expunge(call)
        return call


def _keep_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAWTI_KEEP_AUDIO", "true")
    get_settings.cache_clear()


def _everything_stored() -> str:
    """Every row of every table in the test database, as text: where raw PII must not be."""
    with get_engine().connect() as conn:
        tables = conn.execute(
            sa.text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        ).scalars()
        return "\n".join(
            str(row) for table in list(tables) for row in conn.execute(sa.text(f'SELECT * FROM "{table}"'))
        )


def test_success_stores_only_redacted_text_queues_analysis_and_deletes_audio(
    fake_asr: Any, uploaded: Callable[..., str], analysis_enqueued: list[str], audio_dir: Path
) -> None:
    fake_asr.transcript = RAW
    call_id = uploaded("mixed")

    assert asr_tasks.transcribe_audio_task.delay(call_id).get() == "queued"

    call = _call(call_id)
    assert call.status == "queued"
    assert call.redacted_transcript == redact_transcript(RAW).redacted_text
    assert PHONE not in call.redacted_transcript and EMAIL not in call.redacted_transcript
    assert call.pii_redacted_count == 2
    assert call.queued_at is not None and call.enqueued_at == call.queued_at
    assert call.audio_path is None and list(audio_dir.iterdir()) == []
    assert analysis_enqueued == [call_id]
    # Whisper's language is forced from the call's category.
    assert [category for _, category in fake_asr.calls] == ["mixed"]
    stored = _everything_stored()
    assert PHONE not in stored and EMAIL not in stored


def test_keep_audio_keeps_the_file_and_its_path(
    fake_asr: Any,
    uploaded: Callable[..., str],
    analysis_enqueued: list[str],
    audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _keep_audio(monkeypatch)
    call_id = uploaded()

    asr_tasks.transcribe_audio_task.delay(call_id)

    call = _call(call_id)
    assert call.status == "queued"
    assert call.audio_path == f"{call_id}.wav" and (audio_dir / call.audio_path).is_file()


def test_failure_marks_the_call_failed_without_transcript_text_and_deletes_audio(
    fake_asr: Any, uploaded: Callable[..., str], analysis_enqueued: list[str], audio_dir: Path
) -> None:
    fake_asr.error = RuntimeError(f"decoder choked on 'Customer: call me on {PHONE}'")
    call_id = uploaded()

    assert asr_tasks.transcribe_audio_task.delay(call_id).get() == "failed"

    call = _call(call_id)
    assert call.status == "failed" and call.finished_at is not None
    # An unknown error's message could quote anything; only its type is kept.
    assert call.failure_reason == "asr: RuntimeError"
    assert call.redacted_transcript is None
    assert call.audio_path is None and list(audio_dir.iterdir()) == []
    assert analysis_enqueued == []
    assert PHONE not in _everything_stored()


def test_failure_keeps_the_audio_for_a_retry_only_with_keep_audio(
    fake_asr: Any,
    uploaded: Callable[..., str],
    analysis_enqueued: list[str],
    audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _keep_audio(monkeypatch)
    fake_asr.error = RuntimeError("boom")
    call_id = uploaded()

    asr_tasks.transcribe_audio_task.delay(call_id)

    call = _call(call_id)
    assert call.status == "failed"
    assert call.audio_path == f"{call_id}.wav" and (audio_dir / call.audio_path).is_file()


@pytest.mark.parametrize(
    ("transcript", "reason"),
    [
        ("", "asr: NoSpeechError: no speech recognised in the audio"),
        ("   \n", "asr: NoSpeechError: no speech recognised in the audio"),
        (f"Agent: hi\nsomebody said {PHONE}\n", "asr: the transcript has a line without a speaker prefix"),
    ],
)
def test_unusable_transcripts_fail_with_a_reason_that_quotes_nothing(
    fake_asr: Any,
    uploaded: Callable[..., str],
    analysis_enqueued: list[str],
    transcript: str,
    reason: str,
) -> None:
    fake_asr.transcript = transcript
    call_id = uploaded()

    asr_tasks.transcribe_audio_task.delay(call_id)

    call = _call(call_id)
    assert call.status == "failed" and call.failure_reason == reason
    assert analysis_enqueued == []


def test_known_errors_keep_their_message_and_parse_errors_do_not() -> None:
    assert asr_tasks.failure_reason(FileNotFoundError("Audio file not found: x.wav")) == (
        "asr: FileNotFoundError: Audio file not found: x.wav"
    )
    parse_error = UnattributableLineError(f"transcript line 2 has no speaker: 'call {PHONE}'")
    assert PHONE not in asr_tasks.failure_reason(parse_error)


def test_raw_transcript_is_never_logged_or_traced(
    fake_asr: Any,
    uploaded: Callable[..., str],
    analysis_enqueued: list[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    langfuse = FakeLangfuse()
    monkeypatch.setattr(tracing, "get_client", lambda: langfuse)
    fake_asr.transcript = RAW
    caplog.set_level(logging.DEBUG)

    asr_tasks.transcribe_audio_task.delay(uploaded())
    fake_asr.error = RuntimeError(f"text {PHONE}")
    asr_tasks.transcribe_audio_task.delay(uploaded())

    logged = caplog.text + "".join(str(r.args) for r in caplog.records)
    assert "transcribe_audio" in logged, "the task does log, just never the text"
    for value in (PHONE, EMAIL, "Good morning, this is Sara"):
        assert value not in logged
        assert value not in langfuse.everything_sent()
    assert langfuse.of("trace") == [], "the ASR task opens no trace"


@pytest.mark.parametrize("state", ["missing", "queued", "no_audio"])
def test_a_call_not_awaiting_transcription_is_skipped(
    fake_asr: Any, uploaded: Callable[..., str], analysis_enqueued: list[str], state: str
) -> None:
    call_id = str(uuid.uuid4()) if state == "missing" else uploaded()
    if state != "missing":
        values: dict[str, Any] = {"status": CallStatus.QUEUED} if state == "queued" else {"audio_path": None}
        with get_session() as session:
            session.execute(sa.update(Call).where(Call.id == uuid.UUID(call_id)).values(**values))

    assert asr_tasks.transcribe_audio_task.delay(call_id).get() == "skipped"
    assert fake_asr.calls == [] and analysis_enqueued == []


def test_an_analysis_enqueue_failure_leaves_the_call_queued_for_the_reaper(
    fake_asr: Any, uploaded: Callable[..., str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broker_down(call_id: str) -> None:
        raise ConnectionError("redis down")

    monkeypatch.setattr(tasks.analyze_call_task, "delay", broker_down)
    call_id = uploaded()

    assert asr_tasks.transcribe_audio_task.delay(call_id).get() == "queued"
    call = _call(call_id)
    assert call.status == "queued" and call.enqueued_at is not None
