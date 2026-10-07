"""Serving benchmark core: run calls through the service's own pipeline under any provider.

Phase 6.3. One implementation for every arm of the cloud vs self-hosted
comparison and for the memory comparison:

  * the Kaggle notebook (vLLM adapter on/off, the HF transformers baseline),
  * `scripts/run_cloud_arm.py` (Gemini),
  * `scripts/run_memory_comparison.py` (Gemini, memory off vs on).

**Same code as the service.** Each call runs `sawti.agent.graph.build_graph()`
— the extraction prompt, the `_LaxExtractionProposal` schema, `ground`,
`score`, `compliance`, `assess_confidence` — on the *redacted* transcript,
with the provider injected through `sawti.llm.provider.provider_override`,
and the final state becomes a `CallAnalysis` through the service's own
`sawti.api.records.analysis_from_state`. Nothing is reimplemented here;
scoring (`score_records`) calls the existing `sawti.eval.metrics` functions.

**Per call** a `CallRecord`: end-to-end and LLM latency, tokens in/out as the
provider reported them, whether the output was schema-valid, the bytes of
transcript text sent to the provider, and the prediction itself (so any run
can be re-scored later without re-running a model).

**Resumable.** Records append to a JSONL checkpoint the moment each call
finishes (flushed and fsynced), so an interruption loses only calls in
flight; re-running skips every (arm, repeat, call_id) already recorded. A
provider's *daily* quota (`...PerDay...`) stops the run cleanly with
`QuotaExhausted`; other transient errors (429, 5xx, timeouts) back off and
retry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from sawti.llm.provider import LLMProvider, ResponseModelT, is_transient_provider_error, provider_override
from sawti.llm.usage import capture_usage, total
from sawti.schemas import CallAnalysis, Language

logger = logging.getLogger(__name__)

Outcome = Literal["ok", "schema_invalid", "error"]

#: `rules_for(call)` -> (rule texts for the prompt, their ids for the record).
RuleSource = Callable[["CallInput"], tuple[list[str], list[str]]]


class CallInput(BaseModel):
    """One call to analyze: only the redacted transcript ever leaves this machine."""

    call_id: str
    language: Language
    redacted_transcript: str


class CallRecord(BaseModel):
    """Everything measured for one call in one arm."""

    type: Literal["record"] = "record"
    arm: str
    repeat: int = 0
    concurrency: int
    call_id: str
    language: Language
    started_at: datetime
    latency_s: float = Field(..., description="End to end: the whole graph run.")
    llm_latency_s: float | None = Field(None, description="The provider request(s) only.")
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model: str | None = None
    llm_requests: int = 0
    outcome: Outcome
    error: str | None = None
    transcript_bytes_sent: int = Field(0, description="UTF-8 bytes of transcript text in the request.")
    request_bytes_sent: int = Field(0, description="UTF-8 bytes of prompt + system prompt.")
    retrieved_rule_ids: list[str] = Field(default_factory=list)
    prediction: dict[str, Any] | None = None

    @property
    def schema_valid(self) -> bool:
        return self.outcome == "ok"


class Segment(BaseModel):
    """One uninterrupted stretch of an arm's run — throughput is calls / summed segment time."""

    type: Literal["segment"] = "segment"
    arm: str
    repeat: int = 0
    concurrency: int
    started_at: datetime
    finished_at: datetime
    calls_completed: int
    #: Provenance for the run, e.g. provider, model, a fingerprint of the API key (never the key).
    note: dict[str, str] = Field(default_factory=dict)


class QuotaExhaustedError(RuntimeError):
    """The provider's daily quota is spent; re-run after it resets to resume."""


def is_daily_quota(exc: BaseException) -> bool:
    """Gemini's free-tier daily cap names its metric `...PerDay...` (docs/09-DECISIONS.md, 2026-09-25)."""
    return "PerDay" in str(exc)


def is_schema_failure(exc: BaseException) -> bool:
    """The model answered, but not with a valid instance of the requested schema."""
    if isinstance(exc, ValidationError | json.JSONDecodeError):
        return True
    return type(exc).__name__ == "LocalHFProviderError"  # retries exhausted without valid JSON


class _RateLimiter:
    """At most one request start per `min_interval` seconds, across all tasks (free-tier RPM)."""

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self) -> None:
        if self._min_interval <= 0:
            return
        async with self._lock:
            delay = self._last + self._min_interval - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._last = time.monotonic()


class _Measuring(LLMProvider):
    """Wraps the arm's provider for one call: times it, counts bytes, classifies the outcome."""

    def __init__(self, inner: LLMProvider, limiter: _RateLimiter) -> None:
        self.inner = inner
        self.limiter = limiter
        self.llm_seconds = 0.0
        self.transcript_bytes = 0
        self.request_bytes = 0
        self.failure: Outcome | None = None
        self.failure_text: str | None = None

    async def _measure(self, prompt: str, system: str | None, call: Any) -> Any:
        await self.limiter.wait()
        self.transcript_bytes += len(prompt.encode("utf-8"))
        self.request_bytes += len(prompt.encode("utf-8")) + len((system or "").encode("utf-8"))
        started = time.perf_counter()
        try:
            return await call()
        except Exception as exc:
            if not is_transient_provider_error(exc):
                self.failure = "schema_invalid" if is_schema_failure(exc) else "error"
                self.failure_text = repr(exc)[:1000]
            raise
        finally:
            self.llm_seconds += time.perf_counter() - started

    async def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        result: str = await self._measure(
            prompt, system, lambda: self.inner.complete(prompt, system=system, **kwargs)
        )
        return result

    async def structured_complete(
        self, prompt: str, *, response_model: type[ResponseModelT], system: str | None = None, **kwargs: Any
    ) -> ResponseModelT:
        result: ResponseModelT = await self._measure(
            prompt,
            system,
            lambda: self.inner.structured_complete(
                prompt, response_model=response_model, system=system, **kwargs
            ),
        )
        return result


async def run_call(
    provider: LLMProvider,
    call: CallInput,
    *,
    arm: str,
    concurrency: int = 1,
    repeat: int = 0,
    rules_for: RuleSource | None = None,
    limiter: _RateLimiter | None = None,
) -> CallRecord:
    """Analyze one call through the service graph with `provider`; never raises for a model failure.

    Raises:
        Transient provider errors (the caller retries), `QuotaExhaustedError`.
    """
    from langchain_core.runnables import RunnableConfig

    from sawti.agent.graph import build_graph
    from sawti.api.records import analysis_from_state
    from sawti.observability import tracing

    rule_texts, rule_ids = rules_for(call) if rules_for is not None else ([], [])
    measuring = _Measuring(provider, limiter or _RateLimiter(0.0))
    state: dict[str, Any] = {
        "call_id": call.call_id,
        "redacted_transcript": call.redacted_transcript,
        "language": call.language,
        "retrieved_rules": rule_texts,
    }
    config: RunnableConfig = {"configurable": {"thread_id": f"{arm}:{repeat}:{call.call_id}"}}
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    with (
        capture_usage() as usages,
        provider_override(measuring),
        tracing.call_trace(call.call_id, call.language.value, name=f"benchmark:{arm}"),
    ):
        try:
            graph = build_graph()  # MemorySaver: a benchmark run needs no durable checkpoint
            await graph.ainvoke(state, config)
            values = dict((await graph.aget_state(config)).values)
        except Exception as exc:
            if is_daily_quota(exc):
                raise QuotaExhaustedError(str(exc)) from exc
            raise
    latency = time.perf_counter() - started
    used = total(usages)

    error = values.get("error")
    outcome: Outcome = "ok"
    if error:
        outcome = measuring.failure or "error"
    prediction = None
    if outcome == "ok":
        prediction = analysis_from_state(values, call_id=call.call_id, language=call.language).model_dump(
            mode="json"
        )
    return CallRecord(
        arm=arm,
        repeat=repeat,
        concurrency=concurrency,
        call_id=call.call_id,
        language=call.language,
        started_at=started_at,
        latency_s=round(latency, 3),
        llm_latency_s=round(measuring.llm_seconds, 3) if usages or measuring.llm_seconds else None,
        prompt_tokens=used.prompt_tokens,
        completion_tokens=used.completion_tokens,
        model=used.model or getattr(provider, "model_name", None),
        llm_requests=len(usages),
        outcome=outcome,
        error=(measuring.failure_text or str(error))[:1000] if error else None,
        transcript_bytes_sent=measuring.transcript_bytes,
        request_bytes_sent=measuring.request_bytes,
        retrieved_rule_ids=rule_ids,
        prediction=prediction,
    )


# --- Checkpointed arm runs ----------------------------------------------------


def load_checkpoint(path: Path) -> tuple[list[CallRecord], list[Segment]]:
    """Every record and segment written so far (a torn last line is ignored)."""
    records: list[CallRecord] = []
    segments: list[Segment] = []
    if not path.exists():
        return records, segments
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("%s: ignoring an incomplete line (interrupted write)", path)
            continue
        if data.get("type") == "segment":
            segments.append(Segment.model_validate(data))
        else:
            records.append(CallRecord.model_validate(data))
    return records, segments


def _append(path: Path, item: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(item.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


async def run_arm(
    provider: LLMProvider,
    calls: Sequence[CallInput],
    *,
    arm: str,
    checkpoint: Path,
    concurrency: int = 1,
    repeat: int = 0,
    rules_for: RuleSource | None = None,
    min_interval_s: float = 0.0,
    max_transient_retries: int = 5,
    retry_wait_s: float = 65.0,
    note: dict[str, str] | None = None,
    deadline_epoch: float | None = None,
) -> list[CallRecord]:
    """Run every not-yet-recorded call of `calls` for (arm, repeat), `concurrency` at a time.

    `deadline_epoch` (Unix time): no new call starts after it — calls already
    running finish — so a GPU session stays inside its time budget. Calls
    not started are simply unrecorded; a later run picks them up.

    Returns:
        The records written by this invocation.

    Raises:
        QuotaExhaustedError: The daily quota ran out; everything finished so
            far is in the checkpoint, so re-running resumes.
    """
    done = {(r.arm, r.repeat, r.call_id) for r in load_checkpoint(checkpoint)[0]}
    todo = [c for c in calls if (arm, repeat, c.call_id) not in done]
    if not todo:
        return []
    limiter = _RateLimiter(min_interval_s)
    semaphore = asyncio.Semaphore(concurrency)
    written: list[CallRecord] = []
    started_at = datetime.now(UTC)
    quota: QuotaExhaustedError | None = None

    async def one(call: CallInput) -> None:
        nonlocal quota
        async with semaphore:
            if deadline_epoch is not None and time.time() >= deadline_epoch:
                return  # time budget spent: leave it unrecorded, a later run picks it up
            for attempt in range(max_transient_retries + 1):
                if quota is not None:
                    return
                try:
                    record = await run_call(
                        provider,
                        call,
                        arm=arm,
                        concurrency=concurrency,
                        repeat=repeat,
                        rules_for=rules_for,
                        limiter=limiter,
                    )
                except QuotaExhaustedError as exc:
                    quota = exc
                    return
                except Exception as exc:
                    if is_transient_provider_error(exc) and attempt < max_transient_retries:
                        logger.warning(
                            "%s %s: transient %r, retry in %.0fs", arm, call.call_id, exc, retry_wait_s
                        )
                        await asyncio.sleep(retry_wait_s)
                        continue
                    logger.error("%s %s: giving up for now (%r); a re-run retries it", arm, call.call_id, exc)
                    return
                _append(checkpoint, record)
                written.append(record)
                logger.info(
                    "%s %s [%s]: %s in %.1fs",
                    arm,
                    call.call_id,
                    call.language.value,
                    record.outcome,
                    record.latency_s,
                )
                return

    try:
        await asyncio.gather(*(one(call) for call in todo))
    finally:
        _append(
            checkpoint,
            Segment(
                arm=arm,
                repeat=repeat,
                concurrency=concurrency,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                calls_completed=len(written),
                note=note or {},
            ),
        )
    if quota is not None:
        raise quota
    return written


# --- Scoring ---------------------------------------------------------------------


def _pct(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def score_records(
    records: Sequence[CallRecord],
    ground_truth: Sequence[CallAnalysis],
    *,
    transcript_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Per language: the project's accuracy metrics plus serving measurements, never blended.

    Accuracy metrics (`sawti.eval.metrics`, unchanged) score schema-valid
    predictions only; failures count against `schema_valid_rate` instead and
    are never zero-filled. Latency percentiles cover every call that
    finished, with `n` stated.
    """
    from sawti.eval.metrics import (
        accuracy_by_category,
        grounding_precision_by_category,
        rubric_agreement_by_category,
        unsupported_claim_rate_by_category,
    )

    references = {r.call_id: r for r in ground_truth}
    out: dict[str, dict[str, Any]] = {}
    for language in Language:
        rows = [r for r in records if r.language == language]
        if not rows:
            continue
        predictions = [CallAnalysis.model_validate(r.prediction) for r in rows if r.prediction is not None]
        refs = [references[p.call_id] for p in predictions if p.call_id in references]
        latencies = [r.latency_s for r in rows]
        tokens_out = [r.completion_tokens for r in rows if r.completion_tokens is not None]
        tokens_in = [r.prompt_tokens for r in rows if r.prompt_tokens is not None]

        def metric(
            fn: Callable[..., dict[Language, float]],
            *args: Any,
            language: Language = language,
            any_predictions: bool = bool(predictions),
            **kwargs: Any,
        ) -> float | None:
            value = fn(*args, **kwargs).get(language) if any_predictions else None
            return round(value, 4) if value is not None else None

        out[language.value] = {
            "n": len(rows),
            "schema_valid": sum(r.schema_valid for r in rows),
            "schema_valid_rate": round(sum(r.schema_valid for r in rows) / len(rows), 4),
            "errors": sum(r.outcome == "error" for r in rows),
            "accuracy": metric(accuracy_by_category, predictions, refs),
            "rubric_agreement": metric(rubric_agreement_by_category, predictions, refs),
            "grounding_precision": metric(
                grounding_precision_by_category, predictions, transcript_dir=transcript_dir
            ),
            "unsupported_claim_rate": metric(
                unsupported_claim_rate_by_category, predictions, transcript_dir=transcript_dir
            ),
            "latency_p50_s": _pct(latencies, 50),
            "latency_p95_s": _pct(latencies, 95),
            "prompt_tokens_mean": round(float(np.mean(tokens_in)), 1) if tokens_in else None,
            "completion_tokens_mean": round(float(np.mean(tokens_out)), 1) if tokens_out else None,
            "transcript_bytes_sent_mean": round(float(np.mean([r.transcript_bytes_sent for r in rows])), 1),
            "request_bytes_sent_mean": round(float(np.mean([r.request_bytes_sent for r in rows])), 1),
        }
    return out


def throughput(
    records: Sequence[CallRecord], segments: Sequence[Segment], *, arm: str, repeat: int = 0
) -> dict[str, Any]:
    """Calls per minute for one arm: completed calls / summed uninterrupted run time."""
    seconds = sum(
        (s.finished_at - s.started_at).total_seconds()
        for s in segments
        if s.arm == arm and s.repeat == repeat
    )
    calls = sum(1 for r in records if r.arm == arm and r.repeat == repeat)
    return {
        "calls": calls,
        "seconds": round(seconds, 1),
        "calls_per_min": round(calls / seconds * 60, 3) if seconds > 0 else None,
        "segments": sum(1 for s in segments if s.arm == arm and s.repeat == repeat),
    }


# --- Inputs ---------------------------------------------------------------------


def load_call_inputs(call_ids: dict[str, list[str]], transcript_dir: Path) -> list[CallInput]:
    """The calls to run, interleaved ar / en / mixed, each transcript redacted.

    Interleaved so a run cut short by a quota still covers every language
    roughly equally. `redact()` is idempotent on already-redacted text, so the
    same loader reads the raw local corpus and the redacted Kaggle bundle.
    """
    from sawti.privacy.redaction import redact

    groups = [
        [
            CallInput(
                call_id=call_id,
                language=Language(language),
                redacted_transcript=redact(
                    (transcript_dir / f"{call_id}.txt").read_text(encoding="utf-8")
                ).redacted_text,
            )
            for call_id in call_ids.get(language.value, [])
        ]
        for language in Language
    ]
    interleaved: list[CallInput] = []
    for index in range(max((len(g) for g in groups), default=0)):
        interleaved.extend(group[index] for group in groups if index < len(group))
    return interleaved
