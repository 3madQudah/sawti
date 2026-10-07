"""Local, GPU-resident HF-model-backed `LLMProvider` — no cloud API call.

Phase 5: the held-out-calls accuracy comparison
(`scripts/eval_held_out_calls.py`) needs to run the exact same extraction
path (`sawti.eval.plain_extraction.run_plain_extraction`) against a locally
loaded Qwen3-8B — base and QLoRA-tuned — instead of a cloud provider. This
wraps an already-loaded `transformers` model (optionally PEFT-wrapped) as
an `LLMProvider`, so it can be dropped into that path via the exact
provider-injection point `tests/eval/test_plain_extraction.py` already
relies on (`monkeypatch.setattr("sawti.eval.plain_extraction.
get_llm_provider", lambda: provider)`) — not a new mechanism, the existing
one, just used outside a test for once.

CUDA-only in practice (an 8B model), but nothing in this module imports
`torch` at module load time — only inside `_generate`, so importing this
class does not require a GPU or even `transformers` to be installed.

Gemini's `structured_complete` gets guaranteed JSON via
`response_mime_type="application/json"`; a local model generating raw text
has no such guarantee, so this provider prompts for JSON explicitly and
retries (with sampling, since a failed *greedy* attempt would just fail
identically again) up to `max_retries` times before raising.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sawti.llm.provider import LLMProvider, ResponseModelT
from sawti.llm.usage import record_usage

logger = logging.getLogger(__name__)

_JSON_INSTRUCTION = (
    "Respond with ONLY a single JSON object matching this schema — no markdown "
    "code fences, no explanation before or after it:\n{schema}"
)

DEFAULT_MAX_NEW_TOKENS = 2048


class LocalHFProviderError(RuntimeError):
    """Raised when the wrapped model never produced a schema-valid response."""


def _extract_json(text: str) -> Any:
    """Best-effort JSON extraction from a model's raw text.

    Strips a leading/trailing markdown code fence if present, then takes the
    substring from the first `{` to the last `}` — local instruct models
    routinely wrap JSON in prose or fences despite being told not to.

    Raises:
        ValueError: No `{`/`}` pair found, or the extracted substring isn't
            valid JSON.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object found in model output: {text[:200]!r}")
    return json.loads(stripped[start : end + 1])


class LocalHFProvider(LLMProvider):
    """`LLMProvider` backed by an already-loaded local `transformers` model."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_retries: int = 3,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the provider.

        Args:
            model: A loaded `transformers` (or PEFT-wrapped) causal-LM model.
            tokenizer: Its matching tokenizer.
            max_new_tokens: Generation budget per call.
            max_retries: Extra `structured_complete` attempts (with sampling)
                after an initial greedy one that fails to parse/validate.
            chat_template_kwargs: Extra arguments for `apply_chat_template`.
                Phase 6.3 passes `{"enable_thinking": False}` for Qwen3, which
                matches the adapter's training format; the default (None)
                reproduces Phase 5 exactly, which ran in thinking mode — see
                docs/09-DECISIONS.md, 2026-10-07.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._max_new_tokens = max_new_tokens
        self._max_retries = max_retries
        self._chat_template_kwargs = dict(chat_template_kwargs or {})

    @property
    def model_name(self) -> str:
        """The loaded checkpoint's name, for traces and benchmark records."""
        config = getattr(self._model, "config", None)
        return str(getattr(config, "_name_or_path", None) or type(self._model).__name__)

    def _generate(self, messages: list[dict[str, str]], *, do_sample: bool, temperature: float = 0.7) -> str:
        import torch

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **self._chat_template_kwargs
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        generate_kwargs: dict[str, Any] = {"do_sample": do_sample}
        if do_sample:
            generate_kwargs["temperature"] = temperature
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens, **generate_kwargs
            )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        record_usage(int(inputs["input_ids"].shape[1]), int(len(generated)), model=self.model_name)
        decoded = self._tokenizer.decode(generated, skip_special_tokens=True)
        return str(decoded).strip()

    async def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        """See `LLMProvider.complete`."""
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        return self._generate(messages, do_sample=False)

    async def structured_complete(
        self,
        prompt: str,
        *,
        response_model: type[ResponseModelT],
        system: str | None = None,
        **kwargs: Any,
    ) -> ResponseModelT:
        """See `LLMProvider.structured_complete`.

        Raises:
            LocalHFProviderError: Every attempt (1 greedy + `max_retries`
                sampled) produced text that wasn't valid JSON matching
                `response_model`.
        """
        schema = json.dumps(response_model.model_json_schema())
        instruction = _JSON_INSTRUCTION.format(schema=schema)
        json_system = f"{system}\n\n{instruction}" if system else instruction
        messages = [{"role": "system", "content": json_system}, {"role": "user", "content": prompt}]

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            raw_text = self._generate(messages, do_sample=attempt > 0)
            try:
                data = _extract_json(raw_text)
                return response_model.model_validate(data)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LocalHFProvider: attempt %d/%d produced invalid output: %r",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )

        raise LocalHFProviderError(
            f"local model never produced output matching {response_model.__name__} after "
            f"{self._max_retries + 1} attempt(s): {last_error!r}"
        )
