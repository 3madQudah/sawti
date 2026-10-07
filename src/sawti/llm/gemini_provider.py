"""Gemini-backed implementation of `LLMProvider` (free-tier default for phase 1).

Phase 1: cloud provider implementation, used for synthetic data generation.
"""

from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai import types

from sawti.llm.provider import LLMProvider, ResponseModelT
from sawti.llm.usage import record_usage


class GeminiProvider(LLMProvider):
    """LLMProvider backed by the Google Gemini API (`google-genai` SDK)."""

    def __init__(self, api_key: str, model: str) -> None:
        """Initialize the provider.

        Args:
            api_key: Gemini API key.
            model: Model id to use for completions (e.g. "gemini-2.5-flash").
        """
        self._client = genai.Client(api_key=api_key)
        self._model = model

    @property
    def model_name(self) -> str:
        """The configured model id (an alias like `gemini-flash-lite-latest` resolves server-side)."""
        return self._model

    def _record(self, response: Any) -> None:
        """Report token usage, and the concrete model version the alias resolved to (phase 6.3)."""
        meta = getattr(response, "usage_metadata", None)
        record_usage(
            getattr(meta, "prompt_token_count", None),
            getattr(meta, "candidates_token_count", None),
            model=getattr(response, "model_version", None) or self._model,
        )

    async def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        """See `LLMProvider.complete`."""
        config = (
            types.GenerateContentConfig(system_instruction=system, **kwargs) if system or kwargs else None
        )
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        self._record(response)
        return response.text or ""

    async def structured_complete(
        self,
        prompt: str,
        *,
        response_model: type[ResponseModelT],
        system: str | None = None,
        **kwargs: Any,
    ) -> ResponseModelT:
        """See `LLMProvider.structured_complete`.

        Uses Gemini's JSON-schema structured output support: the Pydantic
        `response_model` is converted to a plain JSON-schema dict via
        `model_json_schema()` (rather than handed to the SDK as a class,
        which does not support the full range of Pydantic schemas — nested
        models, enums, `$defs`/`$ref`), and the returned JSON is validated
        against `response_model` afterward so the same validation errors a
        caller would get from constructing the model directly still apply.
        """
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=response_model.model_json_schema(),
            **kwargs,
        )
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        self._record(response)
        data = json.loads(response.text or "{}")
        return response_model.model_validate(data)
