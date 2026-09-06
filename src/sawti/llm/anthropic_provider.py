"""Anthropic-backed implementation of `LLMProvider`.

Phase 1: cloud provider implementation.
"""

from __future__ import annotations

from typing import Any

from sawti.llm.provider import LLMProvider, ResponseModelT


class AnthropicProvider(LLMProvider):
    """LLMProvider backed by the Anthropic Claude API."""

    def __init__(self, api_key: str, model: str) -> None:
        """Initialize the provider.

        Args:
            api_key: Anthropic API key.
            model: Model id to use for completions (e.g. "claude-sonnet-5").
        """
        # TODO(phase-1): construct an anthropic.AsyncAnthropic client and store api_key/model.
        raise NotImplementedError

    async def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        """See `LLMProvider.complete`."""
        # TODO(phase-1): call self._client.messages.create(...) and return the text content.
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
        # TODO(phase-1): use tool-use/structured output to fill response_model, then .model_validate().
        raise NotImplementedError
