"""Abstract interface for LLM access — every model call in Sawti goes through this.

Phase 0: foundational abstraction; concrete providers are phase 1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


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


def get_llm_provider() -> LLMProvider:
    """Return the LLM provider configured via `sawti.config.Settings.llm_provider`.

    Returns:
        A concrete `LLMProvider` instance selected purely by config.

    Raises:
        NotImplementedError: Until provider construction is wired up.
    """
    # TODO(phase-1): dispatch on get_settings().llm_provider to construct
    # AnthropicProvider or VLLMProvider without callers needing to know which.
    raise NotImplementedError
