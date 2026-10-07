"""Celery tasks: the only place the analysis graph runs in the service.

Phase 6.2. The API process enqueues these and never runs the graph itself.
To keep that true *and* keep the API image free of torch, everything heavy —
the graph, the Postgres checkpointer, the memory store — is imported inside
the task bodies, not at module import. The API imports this module only for
`.delay()`; `tests/api/test_main.py` checks that importing the app does not
pull in `sawti.agent`.

Lifecycle (`sawti.schemas.CallStatus`):

    queued --analyze_call_task--> processing --> awaiting_review --resume_call_task--> reviewed
                                             \\-> completed
                                             \\-> failed  (non-retryable crash, or retries exhausted)

Retries: a transient provider error (`is_transient_provider_error`: 429,
quota, 5xx, timeout) propagates out of `extract`, and the task puts the call
back to `queued` and retries with exponential backoff — so the idempotency
guard below stays a plain `queued -> processing` claim. Every other provider
failure is handled inside the graph (`error` -> `escalate`), so the call
reaches a human. After `task_max_retries`, the call is `failed`.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from celery import Celery, Task
from celery.signals import worker_process_init
from sqlalchemy.exc import OperationalError

from sawti.config import get_settings
from sawti.db.models import Call, CallAnalysisRecord, ReviewSubmission
from sawti.db.session import get_session
from sawti.llm.provider import is_transient_provider_error
from sawti.schemas import CallStatus, Language

if TYPE_CHECKING:
    from sawti.memory.rule_schema import MemoryRule
    from sawti.memory.store import Embedder, MemoryStore

logger = logging.getLogger(__name__)

_settings = get_settings()

celery_app = Celery(
    "sawti",
    broker=_settings.celery_broker_url,
    backend=_settings.celery_result_backend,
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # Acknowledge after the task finishes: a worker killed mid-task leaves the
    # message for another worker, and the idempotency guard makes the redelivery
    # safe. One message at a time per process keeps queue-wait honest.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    # Run by the `beat` service (docker-compose.yml); see `reap_stuck_calls`.
    beat_schedule={
        "reap-stuck-calls": {"task": "sawti.reap_stuck_calls", "schedule": _settings.reaper_interval_seconds}
    },
)


# --- Worker start-up --------------------------------------------------------


@worker_process_init.connect
def _preload_embeddings(**_: Any) -> None:
    """Load the embedding model as each worker process starts, and log how long it took.

    That load is the worker's cold start; doing it here keeps it out of the
    first task's processing time and makes it measurable on its own.
    """
    if not get_settings().worker_preload_embeddings:
        return
    from sawti.memory.store import DEFAULT_EMBEDDING_MODEL, _load_model

    started = time.perf_counter()
    _load_model(DEFAULT_EMBEDDING_MODEL)
    logger.warning("sawti worker cold start: embedding model loaded in %.2fs", time.perf_counter() - started)


# --- Memory rules -------------------------------------------------------------

_store: MemoryStore | None = None
_store_rule_ids: frozenset[uuid.UUID] = frozenset()


#: Embedder for induction's conflict check and consolidation. None = the real
#: local model (already loaded by `_preload_embeddings`). A seam for tests.
_induction_embed: Embedder | None = None


def _memory_store_factory() -> MemoryStore:
    """Build an empty retrieval store. A seam for tests, which inject a fake embedder."""
    from sawti.memory.store import MemoryStore

    return MemoryStore(embed=_induction_embed)


def retrieve_rules(query: str) -> list[MemoryRule]:
    """Top-k active memory rules for `query`, from the `memory_rules` table.

    The store is rebuilt only when the set of active rules changes (a new
    rule, a consolidation), so rules are embedded once per change rather than
    once per call. No active rules means no store and no model use — the
    phase 2 no-memory behavior.
    """
    global _store, _store_rule_ids
    from sawti.memory.persistence import load_active_rules

    with get_session() as session:
        rules = load_active_rules(session)
    if not rules:
        return []
    ids = frozenset(rule.id for rule in rules)
    if _store is None or ids != _store_rule_ids:
        store = _memory_store_factory()
        store.replace_all(rules)
        _store, _store_rule_ids = store, ids
    return _store.retrieve(query, top_k=get_settings().memory_top_k)


# --- Status transitions -------------------------------------------------------


def _now() -> datetime:
    return datetime.now(UTC)


def claim_call(call_id: str) -> bool:
    """The idempotency guard: `queued -> processing`, atomically. False if someone else has it."""
    with get_session() as session:
        claimed = session.execute(
            sa.update(Call)
            .where(Call.id == uuid.UUID(call_id), Call.status == CallStatus.QUEUED)
            .values(status=CallStatus.PROCESSING, started_at=_now(), attempts=Call.attempts + 1)
            .returning(Call.id)
        ).first()
    return claimed is not None


def _set_status(call_id: str, new: CallStatus, *, expected: CallStatus, **values: Any) -> None:
    with get_session() as session:
        session.execute(
            sa.update(Call)
            .where(Call.id == uuid.UUID(call_id), Call.status == expected)
            .values(status=new, **values)
        )


# --- Analysis ---------------------------------------------------------------


async def _run_graph(call_id: str, state: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Run the graph on the Postgres checkpointer; return (final values, interrupted)."""
    from langchain_core.runnables import RunnableConfig

    from sawti.agent.graph import NODE_ESCALATE, build_graph
    from sawti.db.checkpointer import postgres_checkpointer

    config: RunnableConfig = {"configurable": {"thread_id": call_id}}
    async with postgres_checkpointer() as saver:
        graph = build_graph(checkpointer=saver)
        await graph.ainvoke(state, config)
        snapshot = await graph.aget_state(config)
    return dict(snapshot.values), NODE_ESCALATE in snapshot.next


def analyze(call_id: str) -> CallStatus:
    """One analysis attempt for an already-claimed call. Raises on any failure."""
    from sawti.api.records import analysis_from_state, escalation_reason
    from sawti.api.views import rejected_claims_payload

    with get_session() as session:
        call = session.get(Call, uuid.UUID(call_id))
        if call is None or call.redacted_transcript is None:
            raise RuntimeError(f"call {call_id} has no transcript")
        transcript, language = call.redacted_transcript, Language(call.language)

    rules = retrieve_rules(transcript)
    # Only the redacted transcript enters state: the checkpointer persists the
    # whole state, so a raw `transcript` key would be written to Postgres.
    state = {
        "call_id": call_id,
        "redacted_transcript": transcript,
        "language": language,
        "retrieved_rules": [rule.rule_text for rule in rules],
    }
    threshold = get_settings().confidence_threshold
    values, interrupted = asyncio.run(_run_graph(call_id, state))

    analysis = analysis_from_state(values, call_id=call_id, language=language)
    final = CallStatus.AWAITING_REVIEW if interrupted else CallStatus.COMPLETED
    with get_session() as session:
        session.add(
            CallAnalysisRecord(
                id=analysis.id,
                call_id=uuid.UUID(call_id),
                payload=analysis.model_dump(mode="json"),
                confidence=analysis.confidence,
                requires_human_review=analysis.requires_human_review,
                created_at=_now(),
                grounding_coverage=values.get("grounding_coverage"),
                confidence_threshold=threshold,
                escalation_reason=escalation_reason(values, threshold=threshold),
                rejected_claims=rejected_claims_payload(values.get("rejected_claims", [])),
                retrieved_rule_ids=[str(rule.id) for rule in rules],
            )
        )
        session.execute(
            sa.update(Call)
            .where(Call.id == uuid.UUID(call_id), Call.status == CallStatus.PROCESSING)
            .values(status=final, finished_at=_now())
        )
    return final


@celery_app.task(bind=True, name="sawti.analyze_call", max_retries=None)
def analyze_call_task(self: Task, call_id: str) -> str:
    """Analyze one queued call. Safe to deliver more than once.

    Returns:
        The call's resulting status value, or `"skipped"` if the guard found
        it not `queued` (already taken, or already done).
    """
    if not claim_call(call_id):
        logger.info("analyze_call %s: not queued, skipping", call_id)
        return "skipped"
    settings = get_settings()
    try:
        return analyze(call_id).value
    except Exception as exc:
        retries = self.request.retries
        if is_transient_provider_error(exc) and retries < settings.task_max_retries:
            countdown = min(
                settings.task_retry_backoff_seconds * 2**retries, settings.task_retry_backoff_max_seconds
            )
            logger.warning(
                "analyze_call %s: transient %r, retry %d in %.0fs", call_id, exc, retries + 1, countdown
            )
            _set_status(call_id, CallStatus.QUEUED, expected=CallStatus.PROCESSING, enqueued_at=_now())
            raise self.retry(exc=exc, countdown=countdown) from exc
        logger.exception("analyze_call %s failed after %d retries", call_id, retries)
        _set_status(call_id, CallStatus.FAILED, expected=CallStatus.PROCESSING, finished_at=_now())
        return CallStatus.FAILED.value


# --- Resume -------------------------------------------------------------------


async def _resume_graph(call_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    from langchain_core.runnables import RunnableConfig
    from langgraph.types import Command

    from sawti.agent.graph import NODE_ESCALATE, build_graph
    from sawti.db.checkpointer import postgres_checkpointer

    config: RunnableConfig = {"configurable": {"thread_id": call_id}}
    async with postgres_checkpointer() as saver:
        graph = build_graph(checkpointer=saver)
        snapshot = await graph.aget_state(config)
        # Redelivery after a successful resume finds nothing pending; do not resume twice.
        if NODE_ESCALATE in snapshot.next:
            await graph.ainvoke(Command(resume=payload), config)
            snapshot = await graph.aget_state(config)
    return dict(snapshot.values)


@celery_app.task(
    bind=True,
    name="sawti.resume_call",
    autoretry_for=(OperationalError,),
    retry_backoff=True,
    max_retries=5,
)
def resume_call_task(self: Task, call_id: str) -> str:
    """Resume a reviewed call's suspended graph with the reviewer's verdicts; mark it `reviewed`.

    Returns:
        `"reviewed"`, or `"skipped"` if the call is not awaiting review or has
        no submission.
    """
    with get_session() as session:
        call = session.get(Call, uuid.UUID(call_id))
        record = session.query(CallAnalysisRecord).filter_by(call_id=uuid.UUID(call_id)).one_or_none()
        submission = (
            session.query(ReviewSubmission).filter_by(call_analysis_id=record.id).one_or_none()
            if record is not None
            else None
        )
        if call is None or call.status != CallStatus.AWAITING_REVIEW or submission is None:
            return "skipped"
        payload = {
            "submission_id": str(submission.id),
            "reviewer_id": submission.reviewer_id,
            **submission.verdicts,
        }

    values = asyncio.run(_resume_graph(call_id, payload))
    if values.get("review_status") != "human_reviewed":
        raise RuntimeError(f"resume_call {call_id}: graph did not finish as human_reviewed")
    _set_status(call_id, CallStatus.REVIEWED, expected=CallStatus.AWAITING_REVIEW)
    return CallStatus.REVIEWED.value


# --- Memory loop ----------------------------------------------------------------


@celery_app.task(bind=True, name="sawti.induce_rules", max_retries=None)
def induce_rules_task(self: Task, submission_id: str) -> dict[str, int]:
    """Induce rules from one review submission's corrections and persist them.

    Queued by `POST /reviews/{call_id}/corrections`. Runs the phase 4 loop
    (`sawti.memory.persistence.induce_and_store`): induce, conflict-check,
    store, consolidate. Each correction is induced at most once
    (`reviewer_actions.induced_at`), so a redelivery or a retry only picks up
    what is left. Transient provider errors retry with the same backoff as
    analysis; anything else is logged and the remaining corrections stay
    pending for the next run.

    Returns:
        Counts: corrections processed, rules inserted, conflicts, rules merged away.
    """
    from sawti.memory.persistence import induce_and_store

    settings = get_settings()
    try:
        report = induce_and_store(submission_id=uuid.UUID(submission_id), embed=_induction_embed)
    except Exception as exc:
        retries = self.request.retries
        if is_transient_provider_error(exc) and retries < settings.task_max_retries:
            countdown = min(
                settings.task_retry_backoff_seconds * 2**retries, settings.task_retry_backoff_max_seconds
            )
            raise self.retry(exc=exc, countdown=countdown) from exc
        logger.exception("induce_rules %s failed; its pending corrections stay pending", submission_id)
        return {"processed": 0, "inserted": 0, "conflicts": 0, "merged_away": 0}
    return {
        "processed": sum(report.processed.values()),
        "inserted": sum(report.inserted.values()),
        "conflicts": sum(report.conflicts.values()),
        "merged_away": report.merged_away,
    }


# --- Reaper ---------------------------------------------------------------------


@celery_app.task(name="sawti.reap_stuck_calls")
def reap_stuck_calls() -> dict[str, int]:
    """Re-enqueue calls that fell out of the pipeline; give up on ones that keep failing.

    Run every `SAWTI_REAPER_INTERVAL_SECONDS` by Celery beat. Two cases:

    * `queued` with no message enqueued for `reaper_queued_timeout_seconds`
      — the message was lost (broker flushed, the enqueue after commit never
      reached Redis). Re-enqueued; `enqueued_at` is bumped so a call is
      re-sent at most once per timeout, never once per interval.
    * `processing` for longer than `reaper_processing_timeout_seconds` — the
      worker died mid-task and the message was not redelivered. Put back to
      `queued` and re-enqueued, unless it has already been claimed more than
      `task_max_retries` + 1 times, in which case it is `failed`.

    Re-enqueuing is safe for a call that was merely slow: the analysis task's
    `queued -> processing` guard lets only one message do the work.

    Returns:
        Counts: re-enqueued queued calls, recovered processing calls, failed.
    """
    settings = get_settings()
    now = _now()
    queued_cutoff = now - timedelta(seconds=settings.reaper_queued_timeout_seconds)
    processing_cutoff = now - timedelta(seconds=settings.reaper_processing_timeout_seconds)
    max_attempts = settings.task_max_retries + 1
    with get_session() as session:
        lost = (
            session.execute(
                sa.update(Call)
                .where(
                    Call.status == CallStatus.QUEUED,
                    sa.func.coalesce(Call.enqueued_at, Call.queued_at) < queued_cutoff,
                )
                .values(enqueued_at=now)
                .returning(Call.id)
            )
            .scalars()
            .all()
        )
        exhausted = (
            session.execute(
                sa.update(Call)
                .where(
                    Call.status == CallStatus.PROCESSING,
                    Call.started_at < processing_cutoff,
                    Call.attempts >= max_attempts,
                )
                .values(status=CallStatus.FAILED, finished_at=now)
                .returning(Call.id)
            )
            .scalars()
            .all()
        )
        stalled = (
            session.execute(
                sa.update(Call)
                .where(Call.status == CallStatus.PROCESSING, Call.started_at < processing_cutoff)
                .values(status=CallStatus.QUEUED, enqueued_at=now)
                .returning(Call.id)
            )
            .scalars()
            .all()
        )
    for call_id in [*lost, *stalled]:
        analyze_call_task.delay(str(call_id))
    if lost or stalled or exhausted:
        logger.warning(
            "reaper: re-enqueued %d queued, recovered %d processing, failed %d",
            len(lost),
            len(stalled),
            len(exhausted),
        )
    return {"requeued": len(lost), "recovered": len(stalled), "failed": len(exhausted)}
