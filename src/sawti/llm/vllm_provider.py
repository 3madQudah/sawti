"""Self-hosted vLLM-backed implementation of `LLMProvider` (OpenAI-compatible endpoint).

Phase 1: self-hosted provider implementation.
"""

from __future__ import annotations

from typing import Any

from sawti.llm.provider import LLMProvider, ResponseModelT


class VLLMProvider(LLMProvider):
    """LLMProvider backed by a self-hosted vLLM OpenAI-compatible server."""

    def __init__(self, base_url: str, model: str) -> None:
        """Initialize the provider.

        Args:
            base_url: Base URL of the vLLM OpenAI-compatible server.
            model: Model id served by vLLM.
        """
        # TODO(phase-1): construct an openai.AsyncOpenAI client pointed at base_url.
        raise NotImplementedError

    async def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        """See `LLMProvider.complete`."""
        # TODO(phase-1): call self._client.chat.completions.create(...) and return the text content.
        raise NotImplementedError

    async def structured_complete(
        self,
        prompt: str,
        *,
        response_model: type[ResponseModelT],
        system: str | None = None,
        **kwargs: Any,
    ) -> ResponseModelT:
        """See `LLMProvider.structured_complete`."""
        # TODO(phase-1): use guided/structured decoding to fill response_model, then .model_validate().
        raise NotImplementedError
