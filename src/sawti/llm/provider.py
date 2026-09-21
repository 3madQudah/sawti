"""Abstract interface for LLM access — every model call in Sawti goes through this.

Phase 0: foundational abstraction; concrete providers are phase 1.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Coroutine
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


def get_llm_provider() -> LLMProvider:
    """Return the LLM provider configured via `sawti.config.Settings.llm_provider`.

    Returns:
        A concrete `LLMProvider` instance selected purely by config.

    Raises:
        NotImplementedError: If `llm_provider` is set to a provider whose
            concrete class has not been implemented yet.
    """
    settings = get_settings()
    if settings.llm_provider == "gemini":
        from sawti.llm.gemini_provider import GeminiProvider

        return GeminiProvider(api_key=settings.gemini_api_key or "", model=settings.gemini_model)
    if settings.llm_provider == "anthropic":
        from sawti.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key=settings.anthropic_api_key or "", model=settings.llm_model)
    if settings.llm_provider == "vllm":
        from sawti.llm.vllm_provider import VLLMProvider

        return VLLMProvider(base_url=settings.vllm_base_url, model=settings.vllm_model or settings.llm_model)
    raise NotImplementedError(f"Unknown llm_provider: {settings.llm_provider!r}")
