"""Self-hosted vLLM-backed implementation of `LLMProvider` (OpenAI-compatible endpoint).

Phase 6.3. Talks to `vllm serve` over its OpenAI-compatible HTTP API with the
`openai` client — no vLLM import here, so this module (and the service) never
needs vLLM installed; only the serving host does (the Kaggle notebook pins it,
docs/09-DECISIONS.md 2026-10-07).

* **Structured output** uses guided decoding through the standard
  `response_format={"type": "json_schema", ...}` request field, against the
  same JSON schema Gemini receives (`response_model.model_json_schema()`).
  The decoder can only emit JSON matching the schema's *structure*; the
  response is still validated by `response_model`, whose field validators
  (offset arithmetic, chronological sentiment, ...) a grammar cannot enforce.
  That validation result is what the benchmark reports as schema-valid.
* **LoRA**: vLLM serves an adapter registered with `--lora-modules
  <name>=<path>` under `<name>`, so adapter on vs off is just which `model`
  this provider names — the adapter's name, or the base model's.
* **Qwen3 thinking** is off by default (`chat_template_kwargs`), matching the
  format the Phase 5 adapter was trained in (`<think></think>` empty).
"""

from __future__ import annotations

from typing import Any

import httpx
from openai import AsyncOpenAI

from sawti.llm.provider import LLMProvider, ResponseModelT
from sawti.llm.usage import record_usage

#: Qwen3's chat template thinks unless told not to; see the module docstring.
DEFAULT_CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}
DEFAULT_MAX_TOKENS = 4096
#: Generous: at concurrency 16 on two T4s a single long generation can take minutes.
DEFAULT_TIMEOUT_SECONDS = 900.0


class VLLMProvider(LLMProvider):
    """LLMProvider backed by a self-hosted vLLM OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        chat_template_kwargs: dict[str, Any] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the provider.

        Args:
            base_url: Base URL of the vLLM OpenAI-compatible server (ending in `/v1`).
            model: Model id served by vLLM — a LoRA adapter's `--lora-modules`
                name, or the base model's name for the adapter-off arm.
            max_tokens: Generation budget per request.
            temperature: 0.0 = greedy, the default for reproducible extraction.
            chat_template_kwargs: Passed to the server's chat template.
                Defaults to `DEFAULT_CHAT_TEMPLATE_KWARGS` (Qwen3 thinking off);
                pass `{}` to send nothing.
            timeout: Per-request timeout, seconds.
            http_client: Injected transport, for tests (mocked at the HTTP level).
        """
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._chat_template_kwargs = (
            DEFAULT_CHAT_TEMPLATE_KWARGS if chat_template_kwargs is None else chat_template_kwargs
        )
        # vLLM ignores the key unless started with --api-key; the client requires one.
        self._client = AsyncOpenAI(
            base_url=base_url, api_key="EMPTY", timeout=timeout, max_retries=0, http_client=http_client
        )

    @property
    def model_name(self) -> str:
        """The served model id this provider requests."""
        return self._model

    def _messages(self, prompt: str, system: str | None) -> list[dict[str, str]]:
        return ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]

    def _extra_body(self) -> dict[str, Any] | None:
        return {"chat_template_kwargs": self._chat_template_kwargs} if self._chat_template_kwargs else None

    async def _chat(self, prompt: str, system: str | None, **request: Any) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=self._messages(prompt, system),  # type: ignore[arg-type]
            max_tokens=request.pop("max_tokens", self._max_tokens),
            temperature=request.pop("temperature", self._temperature),
            extra_body=self._extra_body(),
            **request,
        )
        usage = response.usage
        record_usage(
            usage.prompt_tokens if usage else None,
            usage.completion_tokens if usage else None,
            model=response.model or self._model,
            finish_reason=response.choices[0].finish_reason if response.choices else None,
        )
        return response.choices[0].message.content or ""

    async def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        """See `LLMProvider.complete`."""
        return await self._chat(prompt, system, **kwargs)

    async def structured_complete(
        self,
        prompt: str,
        *,
        response_model: type[ResponseModelT],
        system: str | None = None,
        **kwargs: Any,
    ) -> ResponseModelT:
        """See `LLMProvider.structured_complete`. Guided decoding, then `response_model` validation.

        Raises:
            pydantic.ValidationError: The output does not validate — including
                truncated JSON when generation hit `max_tokens`.
        """
        content = await self._chat(
            prompt,
            system,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                },
            },
            **kwargs,
        )
        return response_model.model_validate_json(content)
