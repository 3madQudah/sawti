"""Abstract interface for LLM access — every model call in Sawti goes through this.

Phase 0: foundational abstraction; concrete providers are phase 1.
"""

from __future__ import annotations

import asyncio
import contextlib
from abc import ABC, abstractmethod
from collections.abc import Coroutine, Iterator
from contextvars import ContextVar
from typing import Any, TypeVar

from pydantic import BaseModel

from sawti.config import get_settings

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)
_ResultT = TypeVar("_ResultT")

# A provider call that never returns will stall an entire batch run: a 150-call
# generation job was observed hanging for minutes inside a single request with
# no client-side deadline. Callers that loop over many calls must bound each one
# so a hang degrades into an ordinary retryable error.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0


#: HTTP statuses a provider call may succeed on if simply tried again later:
#: request timeout, rate limit / quota (429), and server-side unavailability.
TRANSIENT_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


class TransientProviderError(RuntimeError):
    """A provider failure worth retrying later — raised by providers that know.

    Most SDKs signal this with their own exception types instead; see
    `is_transient_provider_error`, which recognizes those too.
    """


def is_transient_provider_error(exc: BaseException) -> bool:
    """Whether `exc` is a provider failure that a later retry may succeed on.

    The single classification point for phase 6.2's retry policy: transient
    errors are retried by the Celery task with backoff; everything else goes
    through the graph's `error` -> `escalate` path, so a human sees the call.

    Recognized as transient: `TransientProviderError`, `TimeoutError`, built-in
    connection errors, `httpx` transport errors (google-genai, openai and
    anthropic all sit on httpx), the `openai` client's own connection /
    timeout errors (the vLLM provider), and any exception carrying an HTTP status in
    `TRANSIENT_HTTP_STATUSES` as `.code` (google-genai `APIError`) or
    `.status_code` (openai / anthropic SDK errors).

    Note a daily-quota 429 is classified transient too; it will exhaust the
    retries and end `failed`, which is the honest outcome.
    """
    if isinstance(exc, TransientProviderError | TimeoutError | ConnectionError):
        return True
    try:
        import httpx

        if isinstance(exc, httpx.TransportError):
            return True
    except ImportError:  # pragma: no cover - httpx ships with every provider SDK here
        pass
    try:
        import openai  # the vLLM provider's client wraps httpx errors in its own types

        if isinstance(exc, openai.APIConnectionError):  # includes APITimeoutError
            return True
    except ImportError:  # pragma: no cover - openai is a base dependency
        pass
    for attribute in ("code", "status_code"):
        status = getattr(exc, attribute, None)
        if isinstance(status, int) and status in TRANSIENT_HTTP_STATUSES:
            return True
    return False


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Cloud (Anthropic) and self-hosted (vLLM) backends must be swappable by
    config (`SAWTI_LLM_PROVIDER`) alone — no call site should ever import a
    concrete provider directly. Use `get_llm_provider()` to obtain the
    configured instance.
    """

    @abstractmethod
    async def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        """Return a single text completion for `prompt`.

        Args:
            prompt: The user-turn content to send to the model.
            system: Optional system prompt.
            **kwargs: Provider-specific generation parameters (temperature, max_tokens, ...).

        Returns:
            The model's text completion.

        Raises:
            NotImplementedError: Until a concrete provider implements this.
        """
        # TODO(phase-1): implement per-provider completion call.
        raise NotImplementedError

    @abstractmethod
    async def structured_complete(
        self,
        prompt: str,
        *,
        response_model: type[ResponseModelT],
        system: str | None = None,
        **kwargs: Any,
    ) -> ResponseModelT:
        """Return a completion validated against `response_model`.

        Args:
            prompt: The user-turn content to send to the model.
            response_model: A Pydantic model class the response must validate against.
            system: Optional system prompt.
            **kwargs: Provider-specific generation parameters.

        Returns:
            An instance of `response_model`.

        Raises:
            NotImplementedError: Until a concrete provider implements this.
        """
        # TODO(phase-1): implement structured/tool-call based extraction into response_model.
        raise NotImplementedError


def run_with_timeout(
    coro: Coroutine[Any, Any, _ResultT],
    *,
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> _ResultT:
    """Run a provider coroutine to completion under a wall-clock deadline.

    Args:
        coro: The provider call to await.
        timeout: Seconds to allow before giving up.

    Returns:
        Whatever `coro` returned.

    Raises:
        TimeoutError: If `coro` did not finish within `timeout`. Callers with a
            retry loop should treat this like any other provider error.
    """

    async def _await() -> _ResultT:
        return await asyncio.wait_for(coro, timeout=timeout)

    return asyncio.run(_await())


#: A provider to use instead of the configured one, for the duration of
#: `provider_override()` — the benchmark's injection point (phase 6.3).
_override: ContextVar[LLMProvider | None] = ContextVar("sawti_provider_override", default=None)


@contextlib.contextmanager
def provider_override(provider: LLMProvider) -> Iterator[None]:
    """Make `get_llm_provider()` return `provider` inside the block, in this context only.

    A context variable rather than patching a module attribute, so concurrent
    graph runs in one event loop each see the provider their own task set,
    and nothing leaks once the block exits. The provider is returned as given
    — the caller decides whether to wrap it for tracing.
    """
    token = _override.set(provider)
    try:
        yield
    finally:
        _override.reset(token)


def get_llm_provider() -> LLMProvider:
    """Return the LLM provider configured via `sawti.config.Settings.llm_provider`.

    Since phase 6.3 the provider comes wrapped in
    `sawti.observability.tracing.TracedProvider`, so every LLM call made
    through this function is a Langfuse generation when tracing is configured
    (and a transparent pass-through when it is not).

    Returns:
        A concrete `LLMProvider` instance selected purely by config, traced.

    Raises:
        NotImplementedError: If `llm_provider` is set to a provider whose
            concrete class has not been implemented yet.
    """
    override = _override.get()
    if override is not None:
        return override
    from sawti.observability.tracing import TracedProvider

    return TracedProvider(_build_provider(), provider_name=get_settings().llm_provider)


def _build_provider() -> LLMProvider:
    """The configured concrete provider, unwrapped. See `get_llm_provider`."""
    settings = get_settings()
    if settings.llm_provider == "gemini":
        from sawti.llm.gemini_provider import GeminiProvider

        return GeminiProvider(api_key=settings.gemini_api_key or "", model=settings.gemini_model)
    if settings.llm_provider == "anthropic":
        from sawti.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key=settings.anthropic_api_key or "", model=settings.llm_model)
    if settings.llm_provider == "fake":
        from sawti.llm.fake_provider import FakeProvider

        return FakeProvider(
            latency_seconds=settings.fake_llm_latency_seconds,
            ungrounded_claims=settings.fake_llm_ungrounded_claims,
        )
    if settings.llm_provider == "vllm":
        from sawti.llm.vllm_provider import VLLMProvider

        return VLLMProvider(base_url=settings.vllm_base_url, model=settings.vllm_model or settings.llm_model)
    raise NotImplementedError(f"Unknown llm_provider: {settings.llm_provider!r}")
