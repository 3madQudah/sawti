"""Token usage reported by LLM providers, captured per call without shared state.

Phase 6.3. Langfuse needs token counts per LLM call, and so does the serving
benchmark. Providers call `record_usage()` after each request; whoever wants
the numbers wraps the call in `capture_usage()`. The capture lives in a
`ContextVar`, so concurrent calls (the benchmark runs 16 at once in one event
loop) each see only their own usage — asyncio gives every task its own copy of
the context.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class Usage:
    """One provider request's token counts, as the provider reported them (None = not reported)."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model: str | None = None
    #: Why generation stopped, as the provider said it ("stop", "length", "STOP", "MAX_TOKENS", ...).
    finish_reason: str | None = None


_capture: ContextVar[list[Usage] | None] = ContextVar("sawti_llm_usage", default=None)


def record_usage(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    *,
    model: str | None = None,
    finish_reason: str | None = None,
) -> None:
    """Report one request's usage to the active capture, if any. A no-op otherwise."""
    captured = _capture.get()
    if captured is not None:
        captured.append(
            Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=model,
                finish_reason=finish_reason,
            )
        )


@contextlib.contextmanager
def capture_usage() -> Iterator[list[Usage]]:
    """Collect every `record_usage()` made inside the block (a provider may make several, e.g. retries)."""
    captured: list[Usage] = []
    outer = _capture.get()
    token = _capture.set(captured)
    try:
        yield captured
    finally:
        _capture.reset(token)
        # Nested captures (the tracer's inside the benchmark's) must not
        # swallow usage: whatever the inner block saw, the outer one saw too.
        if outer is not None:
            outer.extend(captured)


def total(usages: list[Usage]) -> Usage:
    """Sum a capture into one `Usage`; a side is None only if no request reported it."""
    prompt = [u.prompt_tokens for u in usages if u.prompt_tokens is not None]
    completion = [u.completion_tokens for u in usages if u.completion_tokens is not None]
    models = [u.model for u in usages if u.model]
    return Usage(
        prompt_tokens=sum(prompt) if prompt else None,
        completion_tokens=sum(completion) if completion else None,
        model=models[-1] if models else None,
    )
