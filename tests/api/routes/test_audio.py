"""`POST /calls/audio`: validation, storage, the `transcribing` status, auth (phase 6.4).

The ASR task itself is not run here: `transcribe_audio_task.delay` is replaced
by a recorder, so each call stays `transcribing` the way it would while the
native worker is busy. The task is tested in `tests/api/test_asr_tasks.py`,
and the whole chain in `tests/api/test_audio_pipeline.py`.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sawti.api import asr_tasks
from sawti.config import Settings
from sawti.db.models import Agent, Call
from sawti.db.session import get_session

WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 64
MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 64


@pytest.fixture
def enqueued(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record ASR enqueues instead of running the task."""
    calls: list[str] = []
    monkeypatch.setattr(asr_tasks.transcribe_audio_task, "delay", lambda call_id: calls.append(call_id))
    return calls


def _call(call_id: str) -> Call:
    with get_session() as session:
        call = session.get(Call, uuid.UUID(call_id))
        assert call is not None
        session.expunge(call)
        return call


def _call_count() -> int:
    with get_session() as session:
        return session.query(Call).count()


@pytest.mark.parametrize("suffix", [".wav", ".mp3", ".m4a"])
def test_upload_stores_the_file_and_queues_transcription(
    upload: Callable[..., Any], enqueued: list[str], audio_dir: Path, suffix: str, make_audio: Any
) -> None:
    content = make_audio(suffix)
    response = upload(content, filename=f"Recording 1{suffix.upper()}", language_hint="mixed")

    assert response.status_code == 202, response.text
    call_id = response.json()["call_id"]
    assert enqueued == [call_id]
    call = _call(call_id)
    assert call.status == "transcribing"
    assert call.language == "mixed"
    assert call.redacted_transcript is None
    # Relative to SAWTI_AUDIO_DIR, so the API container and the host worker agree.
    assert call.audio_path == f"{call_id}{suffix}"
    assert (audio_dir / call.audio_path).read_bytes() == content
    assert [p.name for p in audio_dir.iterdir()] == [call.audio_path], "no partial file left behind"


def test_upload_files_the_call_under_the_given_agent_or_a_placeholder(
    upload: Callable[..., Any], enqueued: list[str]
) -> None:
    named = upload(agent_external_id="agent-42").json()["call_id"]
    anonymous = upload().json()["call_id"]
    with get_session() as session:
        names = {
            str(call.id): session.get(Agent, call.agent_id).external_id  # type: ignore[union-attr]
            for call in session.query(Call)
        }
    assert names == {named: "agent-42", anonymous: "unattributed"}


def test_get_call_reports_transcribing_with_no_transcript_yet(
    client: TestClient, headers: dict[str, str], upload: Callable[..., Any], enqueued: list[str]
) -> None:
    call_id = upload(language_hint="ar").json()["call_id"]

    response = client.get(f"/calls/{call_id}", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "transcribing"
    assert body["language"] == "ar"
    assert body["redacted_transcript"] == "" and body["turns"] == []
    assert body["failure_reason"] is None and body["claims"] == []


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("call.txt", b"Agent: hello\n"),
        ("call", WAV),
        ("call.flac", b"fLaC\x00\x00\x00\x22" + b"\x00" * 64),
        # Right extension, wrong bytes: the content check, not the client's word.
        ("call.wav", MP3),
        ("call.m4a", b"not audio at all, just text"),
    ],
)
def test_upload_rejects_other_types_with_415_and_keeps_nothing(
    upload: Callable[..., Any], enqueued: list[str], audio_dir: Path, filename: str, content: bytes
) -> None:
    response = upload(content, filename=filename)

    assert response.status_code == 415, response.text
    assert enqueued == [] and _call_count() == 0
    assert not audio_dir.exists() or list(audio_dir.iterdir()) == []


def test_upload_rejects_a_file_over_the_limit_with_413(
    upload: Callable[..., Any],
    enqueued: list[str],
    audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_audio: Any,
) -> None:
    from sawti.config import get_settings

    monkeypatch.setenv("SAWTI_MAX_AUDIO_BYTES", "3000")
    get_settings.cache_clear()

    assert upload(make_audio(".wav", 3000)).status_code == 202
    response = upload(make_audio(".wav", 3001))

    assert response.status_code == 413, response.text
    assert len(enqueued) == 1 and _call_count() == 1
    assert len(list(audio_dir.iterdir())) == 1, "the oversized upload left nothing on disk"


def test_the_default_upload_limit_is_50_mb() -> None:
    assert Settings().max_audio_bytes == 50 * 1024 * 1024


def test_upload_rejects_an_empty_file_with_422(upload: Callable[..., Any], enqueued: list[str]) -> None:
    response = upload(b"")
    assert response.status_code == 422
    assert enqueued == [] and _call_count() == 0


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"language_hint": None}, "language_hint"),
        ({"language_hint": "fr"}, "language_hint"),
        ({"agent_external_id": "a" * 256}, "agent_external_id"),
    ],
)
def test_upload_validates_form_fields_with_422(
    upload: Callable[..., Any], enqueued: list[str], kwargs: dict[str, Any], field: str
) -> None:
    response = upload(**kwargs)
    assert response.status_code == 422
    assert field in response.text
    assert enqueued == [] and _call_count() == 0


def test_upload_without_a_file_is_422(client: TestClient, enqueued: list[str], api_key: str) -> None:
    response = client.post("/calls/audio", data={"language_hint": "en"}, headers={"X-API-Key": api_key})
    assert response.status_code == 422
    assert "file" in response.text


@pytest.mark.parametrize("sent", [{}, {"X-API-Key": ""}, {"X-API-Key": "wrong"}])
def test_upload_needs_the_api_key(
    upload: Callable[..., Any], enqueued: list[str], audio_dir: Path, sent: dict[str, str]
) -> None:
    response = upload(headers=sent)
    assert response.status_code == 401
    assert enqueued == [] and _call_count() == 0
    assert not audio_dir.exists() or list(audio_dir.iterdir()) == []


def test_upload_is_503_and_the_call_failed_when_the_asr_queue_is_down(
    upload: Callable[..., Any], audio_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broker_down(call_id: str) -> None:
        raise ConnectionError("redis down")

    monkeypatch.setattr(asr_tasks.transcribe_audio_task, "delay", broker_down)

    response = upload()

    assert response.status_code == 503
    [call] = _all_calls()
    assert call.status == "failed"
    assert call.failure_reason == "asr: could not queue the call for transcription"
    assert call.audio_path is None
    assert list(audio_dir.iterdir()) == [], "raw audio is deleted when the call cannot proceed"


def _all_calls() -> list[Call]:
    with get_session() as session:
        calls = session.query(Call).all()
        session.expunge_all()
        return calls
