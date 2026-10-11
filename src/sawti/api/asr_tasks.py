"""The ASR task: uploaded audio to a redacted transcript, then onto the analysis queue.

Phase 6.4. Runs **only** in the native worker on the Mac (`make asr-worker`:
`celery -A sawti.api.asr_tasks worker -Q asr --pool=solo`). mlx-whisper needs
Metal, which a Linux container cannot reach. The Docker workers consume only
`DEFAULT_QUEUE` and never import this module. Routing lives in
`sawti.api.tasks.celery_app.conf.task_routes`.

    transcribing --transcribe_audio_task--> queued  (analysis enqueued on the default queue)
                                       \\-> failed  (`failure_reason` set)

**PII** (docs/09-DECISIONS.md, 2026-10-10):

* The raw transcript lives only in a local variable here, on its way to
  `sawti.api.ingest.redact_transcript`, the same function `POST /calls`
  uses. It is never stored, never logged, and never traced (this task opens no
  Langfuse trace; the analysis it enqueues sees the redacted text only). Log
  lines and `failure_reason` carry ids, timings and error types, never text.
* Raw audio is deleted after the attempt, whether it succeeded or failed,
  and `audio_path` cleared, unless `SAWTI_KEEP_AUDIO=true`. On failure a kept
  file stays available for a retry.

Heavy imports (Whisper, pyannote) happen inside the task, so the API can import
this module for `.delay()` without them.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from celery import Task

from sawti.api.tasks import TRANSCRIBE_TASK, analyze_call_task, celery_app
from sawti.config import get_settings
from sawti.db.models import Call
from sawti.db.session import get_session
from sawti.schemas import CallStatus

logger = logging.getLogger(__name__)

#: The pyannote pipeline, loaded on the first task and reused: loading it takes seconds.
_diarization_pipeline: Any | None = None


class NoSpeechError(RuntimeError):
    """Whisper recognised nothing in the file, so there is no call to analyze."""


def audio_file(name: str) -> Path:
    """The stored upload `name` (`calls.audio_path`) under this process's `SAWTI_AUDIO_DIR`."""
    return get_settings().audio_dir / name


def delete_audio(name: str) -> None:
    """Delete an upload; a file already gone is fine (a redelivered task, a manual cleanup)."""
    audio_file(name).unlink(missing_ok=True)


def failure_reason(exc: BaseException) -> str:
    """A reviewer-facing reason for an ASR failure. Never transcript text.

    The type alone for `UnattributableLineError` (its message quotes a
    transcript line) and for anything not from this pipeline's known stages; the
    message too for errors whose messages are about files, models or speakers.
    """
    from sawti.asr.diarization import DiarizationUnavailableError
    from sawti.data.transcript_parser import UnattributableLineError

    if isinstance(exc, UnattributableLineError):
        return "asr: the transcript has a line without a speaker prefix"
    if isinstance(exc, NoSpeechError | FileNotFoundError | DiarizationUnavailableError):
        return f"asr: {type(exc).__name__}: {exc}"[:1000]
    return f"asr: {type(exc).__name__}"


def _pipeline() -> Any:
    global _diarization_pipeline
    if _diarization_pipeline is None:
        from sawti.asr.diarization import load_pipeline

        _diarization_pipeline = load_pipeline()
    return _diarization_pipeline


def _transcribe(path: Path, category: str) -> tuple[str, int]:
    """ASR + redaction. Returns only the redacted text and the redaction count."""
    from sawti.api.ingest import redact_transcript
    from sawti.asr.pipeline import transcribe_call

    result = transcribe_call(path, category=category, diarization_pipeline=_pipeline())
    if not result.raw_transcript.strip():
        raise NoSpeechError("no speech recognised in the audio")
    logger.info(
        "transcribe_audio: %.1fs of %s audio in %.1fs (RTF %.3f; whisper %.1fs, diarization %.1fs)",
        result.audio_seconds,
        category,
        result.timings.total_s,
        result.realtime_factor,
        result.timings.transcribe_s,
        result.timings.diarize_s,
    )
    redaction = redact_transcript(result.raw_transcript)
    return redaction.redacted_text, redaction.redaction_count


@celery_app.task(bind=True, name=TRANSCRIBE_TASK)
def transcribe_audio_task(self: Task, call_id: str) -> str:
    """Transcribe one uploaded call, store the redacted transcript, queue its analysis.

    Safe to deliver more than once: it acts only on a call still
    `transcribing` that still has its audio, and the final update is guarded
    on that status.

    Returns:
        `"queued"`, `"failed"`, or `"skipped"`.
    """
    keep_audio = get_settings().keep_audio
    with get_session() as session:
        call = session.get(Call, uuid.UUID(call_id))
        if call is None or call.status != CallStatus.TRANSCRIBING or call.audio_path is None:
            logger.info("transcribe_audio %s: not transcribing, skipping", call_id)
            return "skipped"
        audio_name, category = call.audio_path, call.language

    try:
        redacted_text, redaction_count = _transcribe(audio_file(audio_name), category)
    except Exception as exc:
        reason = failure_reason(exc)
        logger.error("transcribe_audio %s failed: %s", call_id, reason)
        with get_session() as session:
            session.execute(
                sa.update(Call)
                .where(Call.id == uuid.UUID(call_id), Call.status == CallStatus.TRANSCRIBING)
                .values(
                    status=CallStatus.FAILED,
                    finished_at=datetime.now(UTC),
                    failure_reason=reason,
                    audio_path=audio_name if keep_audio else None,
                )
            )
        if not keep_audio:
            delete_audio(audio_name)
        return CallStatus.FAILED.value

    now = datetime.now(UTC)
    with get_session() as session:
        updated = session.execute(
            sa.update(Call)
            .where(Call.id == uuid.UUID(call_id), Call.status == CallStatus.TRANSCRIBING)
            .values(
                redacted_transcript=redacted_text,
                pii_redacted_count=redaction_count,
                status=CallStatus.QUEUED,
                queued_at=now,
                enqueued_at=now,
                audio_path=audio_name if keep_audio else None,
            )
            .returning(Call.id)
        ).first()
    if updated is None:
        return "skipped"
    if not keep_audio:
        delete_audio(audio_name)
    try:
        analyze_call_task.delay(call_id)
    except Exception:
        # The call is `queued` with a fresh `enqueued_at`; the reaper re-sends
        # it after `reaper_queued_timeout_seconds`.
        logger.exception("transcribe_audio %s: could not enqueue analysis; the reaper will", call_id)
    return CallStatus.QUEUED.value
