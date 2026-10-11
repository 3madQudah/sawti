"""Call endpoints: submit a transcript or a recording, read back everything the review screen needs.

Phase 6.2; audio upload added in 6.4. These routes only read and write the database and enqueue tasks;
the graph runs in the worker (`sawti.api.tasks`).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from sqlalchemy.exc import IntegrityError

from sawti.api.asr_tasks import delete_audio, transcribe_audio_task
from sawti.api.deps import ApiKey, CurrentReviewer, DbSession
from sawti.api.ingest import redact_transcript
from sawti.api.schemas import (
    CallDetail,
    CallSubmitRequest,
    CallSubmitResponse,
    MemoryRuleUsed,
    ReviewSubmission,
)
from sawti.api.tasks import analyze_call_task
from sawti.api.views import grounded_claims, rejected_claims_view, split_turns
from sawti.config import get_settings
from sawti.data.transcript_parser import UnattributableLineError
from sawti.db.models import Agent, Call, CallAnalysisRecord, MemoryRuleRecord
from sawti.db.models import ReviewSubmission as ReviewSubmissionRow
from sawti.memory.capture import get_or_create_placeholder_agent
from sawti.schemas import CallAnalysis, CallStatus, Language

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/calls", tags=["calls"])


def _agent_id(session: DbSession, external_id: str) -> uuid.UUID:
    """The `Agent` for `external_id`, created on first sight (named by its external id).

    Two first-time submissions for one agent can race on the unique
    `external_id`; the loser re-reads the winner's row.
    """
    try:
        with session.begin_nested():
            return get_or_create_placeholder_agent(session, external_id=external_id, name=external_id)
    except IntegrityError:
        return get_or_create_placeholder_agent(session, external_id=external_id, name=external_id)


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=CallSubmitResponse)
def submit_call(body: CallSubmitRequest, session: DbSession, _: ApiKey) -> CallSubmitResponse:
    """Redact the transcript, store the redacted text only, and queue the call for analysis.

    The raw transcript exists only inside this function; it is neither stored
    nor put into graph state (docs/09-DECISIONS.md, 2026-10-06, decision 1).

    Returns 422 if a line has no `Agent:`/`Customer:` prefix (the review
    screen needs speaker turns, and the parser does not guess), and 503 if
    the task could not be queued — the call is then marked `failed`.
    """
    call_id = uuid.uuid4()
    try:
        redaction = redact_transcript(body.transcript)
    except UnattributableLineError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    now = datetime.now(UTC)
    session.add(
        Call(
            id=call_id,
            agent_id=_agent_id(session, body.agent_external_id),
            redacted_transcript=redaction.redacted_text,
            pii_redacted_count=redaction.redaction_count,
            language=body.language.value,
            occurred_at=now,
            status=CallStatus.QUEUED,
            queued_at=now,
            enqueued_at=now,
        )
    )
    session.commit()

    try:
        analyze_call_task.delay(str(call_id))
    except Exception as exc:
        logger.exception("could not enqueue %s", call_id)
        call = session.get(Call, call_id)
        if call is not None:
            call.status, call.finished_at = CallStatus.FAILED, datetime.now(UTC)
            session.commit()
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "could not queue the call") from exc
    return CallSubmitResponse(call_id=call_id)


#: Accepted uploads: extension -> a check of the file's first bytes. The
#: extension alone is the client's word; the bytes are what ffmpeg will see.
AUDIO_FORMATS: dict[str, Callable[[bytes], bool]] = {
    ".wav": lambda head: head[:4] == b"RIFF" and head[8:12] == b"WAVE",
    # An ID3 tag, or straight into an MPEG audio frame (11 sync bits set).
    ".mp3": lambda head: head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0),
    # ISO base media: a box size, then `ftyp`.
    ".m4a": lambda head: head[4:8] == b"ftyp",
}

#: The `Agent` an upload without `agent_external_id` is filed under.
UNATTRIBUTED_AGENT = "unattributed"

_CHUNK_BYTES = 1024 * 1024


def _too_large(limit: int) -> HTTPException:
    return HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"audio is larger than {limit} bytes")


def _store_upload(upload: UploadFile, suffix: str, call_id: uuid.UUID) -> str:
    """Stream the upload into `SAWTI_AUDIO_DIR` as `<call_id><suffix>`; return that name.

    Written to a temporary name first and renamed once complete and valid, so
    the ASR worker never sees a partial file. Raises 413 past
    `SAWTI_MAX_AUDIO_BYTES`, 415 if the bytes are not the claimed format, and
    422 if the file is empty. Nothing is left on disk after an error.
    """
    settings = get_settings()
    settings.audio_dir.mkdir(parents=True, exist_ok=True)
    name = f"{call_id}{suffix}"
    partial = settings.audio_dir / f".{name}.part"
    size, head = 0, b""
    try:
        with partial.open("wb") as out:
            while chunk := upload.file.read(_CHUNK_BYTES):
                size += len(chunk)
                if size > settings.max_audio_bytes:
                    raise _too_large(settings.max_audio_bytes)
                if len(head) < 12:
                    head = (head + chunk)[:12]
                out.write(chunk)
        if size == 0:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "the audio file is empty")
        if not AUDIO_FORMATS[suffix](head):
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"the file's content is not {suffix[1:]} audio"
            )
        partial.rename(settings.audio_dir / name)
    finally:
        partial.unlink(missing_ok=True)
    return name


@router.post("/audio", status_code=status.HTTP_202_ACCEPTED, response_model=CallSubmitResponse)
def submit_audio(
    session: DbSession,
    _: ApiKey,
    file: Annotated[UploadFile, File(description="The call recording: wav, mp3 or m4a, at most 50 MB.")],
    language_hint: Annotated[
        Language,
        Form(description="The call's language category. Picks Whisper's forced language. Required for now."),
    ],
    agent_external_id: Annotated[str | None, Form(min_length=1, max_length=255)] = None,
) -> CallSubmitResponse:
    """Store an uploaded recording and queue it for the native ASR worker.

    The call is `transcribing` until the worker has transcribed, diarized and
    redacted it (the same redaction as `POST /calls`); it is then `queued` for
    analysis like any submitted transcript. Poll `GET /calls/{id}`.

    Raw audio is PII: it stays only in `SAWTI_AUDIO_DIR` until transcription
    ends, and is then deleted unless `SAWTI_KEEP_AUDIO=true`.

    Returns 413 over the size limit, 415 for anything but wav/mp3/m4a (by
    extension *and* content), 422 for an empty file or missing form fields,
    and 503 if the ASR task could not be queued (the call is then `failed`).

    The size limit is enforced while copying into `SAWTI_AUDIO_DIR`, after
    Starlette has spooled the multipart body to a temporary file. A huge
    upload is therefore received before it is refused. That is acceptable for
    a localhost-only service (docs/09-DECISIONS.md, 2026-10-10); a deployed
    one would cap the body size in front of the app.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in AUDIO_FORMATS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported audio type {suffix or '(none)'}; expected one of {', '.join(AUDIO_FORMATS)}",
        )
    call_id = uuid.uuid4()
    audio_name = _store_upload(file, suffix, call_id)

    try:
        session.add(
            Call(
                id=call_id,
                agent_id=_agent_id(session, agent_external_id or UNATTRIBUTED_AGENT),
                audio_path=audio_name,
                language=language_hint.value,
                occurred_at=datetime.now(UTC),
                status=CallStatus.TRANSCRIBING,
            )
        )
        session.commit()
    except Exception:
        delete_audio(audio_name)
        raise

    try:
        transcribe_audio_task.delay(str(call_id))
    except Exception as exc:
        logger.exception("could not enqueue %s for transcription", call_id)
        call = session.get(Call, call_id)
        if call is not None:
            call.status, call.finished_at = CallStatus.FAILED, datetime.now(UTC)
            call.failure_reason = "asr: could not queue the call for transcription"
            if not get_settings().keep_audio:
                call.audio_path = None
                delete_audio(audio_name)
            session.commit()
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "could not queue the call") from exc
    return CallSubmitResponse(call_id=call_id)


@router.get("/{call_id}", response_model=CallDetail)
def get_call(call_id: uuid.UUID, session: DbSession, _: CurrentReviewer) -> CallDetail:
    """Everything the call review screen renders. Analysis fields are empty until analysis finishes."""
    row = (
        session.query(Call, Agent)
        .join(Agent, Agent.id == Call.agent_id)
        .filter(Call.id == call_id)
        .one_or_none()
    )
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such call")
    call, agent = row
    transcript = call.redacted_transcript or ""
    turns = split_turns(transcript, str(call.id))
    detail = CallDetail(
        call_id=call.id,
        status=call.status,
        language=call.language,
        agent_name=agent.name,
        queued_at=call.queued_at,
        started_at=call.started_at,
        finished_at=call.finished_at,
        pii_redacted_count=call.pii_redacted_count,
        attempts=call.attempts,
        failure_reason=call.failure_reason,
        redacted_transcript=transcript,
        turns=turns,
    )

    record = session.query(CallAnalysisRecord).filter_by(call_id=call.id).one_or_none()
    if record is None:
        return detail
    analysis = CallAnalysis.model_validate(record.payload)
    rule_ids = [uuid.UUID(rule_id) for rule_id in record.retrieved_rule_ids]
    rules = {r.id: r for r in session.query(MemoryRuleRecord).filter(MemoryRuleRecord.id.in_(rule_ids))}
    submission = session.query(ReviewSubmissionRow).filter_by(call_analysis_id=record.id).one_or_none()

    detail.summary = analysis.summary
    detail.confidence = record.confidence
    detail.grounding_coverage = record.grounding_coverage
    detail.confidence_threshold = record.confidence_threshold
    detail.escalation_reason = record.escalation_reason
    detail.claims = grounded_claims(analysis, turns)
    detail.rejected_claims = rejected_claims_view(record.rejected_claims)
    # Best match first, as retrieved; a rule deleted since is simply absent.
    detail.memory_rules = [MemoryRuleUsed(id=i, text=rules[i].rule_text) for i in rule_ids if i in rules]
    detail.review = ReviewSubmission.model_validate(submission.verdicts) if submission is not None else None
    return detail
