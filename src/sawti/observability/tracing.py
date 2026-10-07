"""Langfuse tracing for the analysis pipeline: a span per graph node, a generation per LLM call.

Phase 6.3. Self-hosted Langfuse v2 only (docker-compose.yml); see
`sawti.config.Settings.langfuse_host`.

What a trace holds. One trace per graph run — `call_trace()` opens it around
`graph.ainvoke` (the analysis task, the resume task, the benchmark); its
`session_id` is the call id, so a call's analysis and its later resume group
together. Inside it:

  * a **span per graph node** (`traced_node`, wired in `build_graph`): node
    name, duration, call_id, language, and a numeric summary of what the node
    wrote (claim counts, coverage, confidence, error) — never transcript text;
  * a **generation per LLM call** (`TracedProvider`, wrapped around every
    provider `get_llm_provider()` returns): provider, model, latency, token
    counts, call_id, language, the prompt and the output.

**PII.** The pipeline only ever holds redacted text (docs/09-DECISIONS.md,
2026-10-06), so prompts are already redacted. As a backstop for any future
call site, *every string in every payload* goes through `redact()` before it
is handed to the Langfuse client — and the client is constructed with that as
its `mask` too. `tests/observability/test_tracing.py` submits known PII
through the real pipeline and asserts it reaches no trace payload.

Off unless `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are both set: every
function here is then a cheap no-op, so tests, the eval harness and the
Kaggle notebook run without a Langfuse server.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from sawti.config import get_settings
from sawti.llm.provider import LLMProvider, ResponseModelT
from sawti.llm.usage import capture_usage, total
from sawti.privacy.redaction import redact

logger = logging.getLogger(__name__)

#: The Langfuse object new observations attach to (a trace, or a node span).
_parent: ContextVar[Any] = ContextVar("sawti_trace_parent", default=None)
#: call_id / language of the run being traced, for generation metadata.
_run: ContextVar[dict[str, str] | None] = ContextVar("sawti_trace_run", default=None)


def scrub(value: Any) -> Any:
    """`redact()` every string inside `value` (dicts, lists, tuples), leaving other types alone."""
    if isinstance(value, str):
        return redact(value).redacted_text
    if isinstance(value, dict):
        return {key: scrub(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [scrub(item) for item in value]
    return value


def _mask(*, data: Any) -> Any:
    """The Langfuse SDK's `mask` hook: the same scrub, applied by the client itself."""
    return scrub(data)


@lru_cache
def get_client() -> Any | None:
    """The process's Langfuse client, or None when tracing is not configured.

    Created lazily, so a Celery worker builds its own after fork rather than
    inheriting a client whose background sender thread did not survive it.
    """
    settings = get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        mask=_mask,
    )


def flush() -> None:
    """Send everything queued. Call at the end of a task: workers may exit before the sender runs."""
    client = get_client()
    if client is not None:
        try:
            client.flush()
        except Exception:  # tracing must never fail the work it observes
            logger.exception("langfuse flush failed")


@contextlib.contextmanager
def call_trace(call_id: str, language: str, *, name: str) -> Iterator[None]:
    """Open the trace for one graph run of one call; nodes and LLM calls inside attach to it."""
    run_token = _run.set({"call_id": call_id, "language": language})
    client = get_client()
    parent_token = None
    if client is not None:
        try:
            trace = client.trace(
                name=name,
                session_id=call_id,
                metadata=scrub({"call_id": call_id, "language": language}),
                tags=[f"language:{language}"],
            )
            parent_token = _parent.set(trace)
        except Exception:
            logger.exception("langfuse trace failed for %s", call_id)
    try:
        yield
    finally:
        if parent_token is not None:
            _parent.reset(parent_token)
        _run.reset(run_token)


def _language(state: dict[str, Any]) -> str:
    language = state.get("language")
    return str(getattr(language, "value", language or ""))


def _node_summary(update: dict[str, Any]) -> dict[str, Any]:
    """Numbers only: what a node wrote, without any text from the call."""
    summary: dict[str, Any] = {}
    lists = (
        "commitments",
        "compliance_flags",
        "rubric_scores",
        "rejected_claims",
        "rejected_sentiment_points",
    )
    for key in lists:
        if key in update:
            summary[f"{key}_count"] = len(update[key])
    for key in ("grounding_coverage", "confidence", "requires_human_review", "review_status"):
        if key in update:
            summary[key] = update[key]
    if update.get("error"):
        summary["error"] = str(update["error"])[:500]
    return summary


NodeFn = Callable[[Any], Awaitable[dict[str, Any]]]


def traced_node(name: str, fn: NodeFn) -> NodeFn:
    """Wrap a graph node so each execution is a span under the current trace.

    `escalate` raises LangGraph's interrupt on purpose; that ends the span as
    "interrupted" and re-raises — it is control flow, not an error.
    """

    async def _node(state: Any) -> dict[str, Any]:
        parent = _parent.get()
        if parent is None:
            return await fn(state)
        meta = {"node": name, "call_id": str(state.get("call_id", "")), "language": _language(state)}
        span = None
        try:
            span = parent.span(name=f"node:{name}", metadata=scrub(meta), start_time=datetime.now(UTC))
        except Exception:
            logger.exception("langfuse span failed")
        token = _parent.set(span) if span is not None else None
        try:
            update = await fn(state)
        except BaseException as exc:
            if span is not None:
                from langgraph.errors import GraphInterrupt

                interrupted = isinstance(exc, GraphInterrupt)
                span.end(
                    level="DEFAULT" if interrupted else "ERROR",
                    status_message="interrupted for human review" if interrupted else repr(exc)[:500],
                    end_time=datetime.now(UTC),
                )
            raise
        finally:
            if token is not None:
                _parent.reset(token)
        if span is not None:
            span.end(output=scrub(_node_summary(update)), end_time=datetime.now(UTC))
        return update

    _node.__name__ = getattr(fn, "__name__", name)
    return _node


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


class TracedProvider(LLMProvider):
    """Wraps any provider: each call becomes a Langfuse generation (when tracing is on).

    Transparent otherwise — same results, same exceptions. `get_llm_provider()`
    returns every provider wrapped in this.
    """

    def __init__(self, inner: LLMProvider, *, provider_name: str) -> None:
        self._inner = inner
        self._provider_name = provider_name

    @property
    def inner(self) -> LLMProvider:
        """The wrapped provider."""
        return self._inner

    @property
    def model_name(self) -> str:
        return str(getattr(self._inner, "model_name", type(self._inner).__name__))

    @contextlib.asynccontextmanager
    async def _observe(self, name: str, prompt: str, system: str | None) -> AsyncIterator[dict[str, Any]]:
        """Time the call, capture its usage, and emit one generation with whatever happened."""
        result: dict[str, Any] = {}
        start = datetime.now(UTC)
        started = time.perf_counter()
        error: BaseException | None = None
        with capture_usage() as usages:
            try:
                yield result
            except BaseException as exc:
                error = exc
                raise
            finally:
                self._emit(name, prompt, system, result.get("output"), usages, start, started, error)

    def _emit(
        self,
        name: str,
        prompt: str,
        system: str | None,
        output: Any,
        usages: list[Any],
        start: datetime,
        started: float,
        error: BaseException | None,
    ) -> None:
        parent = _parent.get() or get_client()
        if parent is None:
            return
        run = _run.get() or {}
        used = total(usages)
        try:
            parent.generation(
                name=name,
                model=used.model or self.model_name,
                start_time=start,
                end_time=datetime.now(UTC),
                input=scrub(
                    [{"role": "system", "content": system or ""}, {"role": "user", "content": prompt}]
                ),
                output=scrub(_dump(output)),
                usage={"input": used.prompt_tokens, "output": used.completion_tokens, "unit": "TOKENS"}
                if used.prompt_tokens is not None or used.completion_tokens is not None
                else None,
                metadata=scrub(
                    {
                        "provider": self._provider_name,
                        "call_id": run.get("call_id", ""),
                        "language": run.get("language", ""),
                        "latency_s": round(time.perf_counter() - started, 3),
                    }
                ),
                level="ERROR" if error is not None else "DEFAULT",
                status_message=repr(error)[:500] if error is not None else None,
            )
        except Exception:
            logger.exception("langfuse generation failed")

    async def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        """See `LLMProvider.complete`."""
        async with self._observe("llm:complete", prompt, system) as result:
            result["output"] = await self._inner.complete(prompt, system=system, **kwargs)
        return str(result["output"])

    async def structured_complete(
        self,
        prompt: str,
        *,
        response_model: type[ResponseModelT],
        system: str | None = None,
        **kwargs: Any,
    ) -> ResponseModelT:
        """See `LLMProvider.structured_complete`."""
        async with self._observe(f"llm:{response_model.__name__}", prompt, system) as result:
            result["output"] = await self._inner.structured_complete(
                prompt, response_model=response_model, system=system, **kwargs
            )
        output: ResponseModelT = result["output"]
        return output
