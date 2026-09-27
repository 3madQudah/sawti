"""Tests for `sawti.llm.local_hf_provider`.

Phase 5. Nothing here needs CUDA or a real model — `_generate` is the only
method that touches `torch`, and it does so through a duck-typed
model/tokenizer pair (`torch` itself, CPU-only, is already a transitive
dependency of `sentence-transformers` in this project's base environment).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from sawti.llm.local_hf_provider import LocalHFProvider, LocalHFProviderError, _extract_json


class _Answer(BaseModel):
    value: int


# --- _extract_json -----------------------------------------------------------


def test_extract_json_parses_a_bare_object() -> None:
    assert _extract_json('{"value": 1}') == {"value": 1}


def test_extract_json_strips_a_markdown_fence() -> None:
    text = '```json\n{"value": 1}\n```'
    assert _extract_json(text) == {"value": 1}


def test_extract_json_strips_a_bare_fence_with_no_language_tag() -> None:
    text = '```\n{"value": 1}\n```'
    assert _extract_json(text) == {"value": 1}


def test_extract_json_finds_the_object_amid_surrounding_prose() -> None:
    text = 'Sure, here is the answer:\n{"value": 1}\nLet me know if you need anything else.'
    assert _extract_json(text) == {"value": 1}


def test_extract_json_raises_when_no_braces_present() -> None:
    with pytest.raises(ValueError, match="no JSON object found"):
        _extract_json("I don't know the answer.")


def test_extract_json_raises_on_malformed_json_between_braces() -> None:
    with pytest.raises(ValueError):
        _extract_json("{value: 1,}")


# --- LocalHFProvider.structured_complete, fake model/tokenizer ---------------


class _ScriptedTokenizer:
    """Duck-types the tokenizer calls the provider makes, no network needed."""

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return messages[-1]["content"]

    def __call__(self, text: str, return_tensors: str):
        import torch
        from transformers import BatchEncoding

        assert return_tensors == "pt"
        return BatchEncoding(data={"input_ids": torch.tensor([[1, 2, 3]])})

    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return self._next_response()

    def _next_response(self) -> str:  # overridden per-instance in tests below
        raise NotImplementedError


class _ScriptedModel:
    """Ignores its inputs; `generate` just returns a fixed-shape tensor.

    The actual decoded text comes from `_ScriptedTokenizer.decode`, which
    each test overrides — the model's own return value only needs to be
    shaped correctly for `output_ids[0][prompt_len:]` to slice cleanly.
    """

    device = "cpu"

    def generate(self, *, input_ids, max_new_tokens: int, do_sample: bool, **kwargs):
        import torch

        return torch.cat([input_ids, torch.zeros((1, 1), dtype=input_ids.dtype)], dim=1)


def _provider_with_responses(responses: list[str], **kwargs) -> LocalHFProvider:
    """A LocalHFProvider whose `_generate` calls return `responses` in order."""
    provider = LocalHFProvider(_ScriptedModel(), _ScriptedTokenizer(), **kwargs)
    remaining = list(responses)

    def fake_generate(messages, *, do_sample, temperature=0.7):
        return remaining.pop(0)

    provider._generate = fake_generate  # type: ignore[method-assign]
    return provider


async def test_structured_complete_succeeds_on_first_attempt() -> None:
    provider = _provider_with_responses(['{"value": 42}'])

    result = await provider.structured_complete("what is it?", response_model=_Answer)

    assert result == _Answer(value=42)


async def test_structured_complete_retries_after_invalid_json_then_succeeds() -> None:
    provider = _provider_with_responses(["not json at all", '{"value": 7}'], max_retries=3)

    result = await provider.structured_complete("what is it?", response_model=_Answer)

    assert result == _Answer(value=7)


async def test_structured_complete_raises_after_exhausting_retries() -> None:
    provider = _provider_with_responses(["nope", "still nope", "nope again"], max_retries=2)

    with pytest.raises(LocalHFProviderError, match="_Answer"):
        await provider.structured_complete("what is it?", response_model=_Answer)


async def test_structured_complete_prompts_for_json_matching_the_schema() -> None:
    provider = LocalHFProvider(_ScriptedModel(), _ScriptedTokenizer())
    seen_messages: list[list[dict[str, str]]] = []

    def fake_generate(messages, *, do_sample, temperature=0.7):
        seen_messages.append(messages)
        return '{"value": 1}'

    provider._generate = fake_generate  # type: ignore[method-assign]

    await provider.structured_complete("what is it?", response_model=_Answer, system="be terse")

    system_content = seen_messages[0][0]["content"]
    assert "be terse" in system_content
    assert "value" in system_content  # the schema itself is embedded


async def test_complete_returns_generated_text() -> None:
    provider = LocalHFProvider(_ScriptedModel(), _ScriptedTokenizer())
    provider._generate = lambda messages, *, do_sample, temperature=0.7: "the answer"  # type: ignore[method-assign]

    result = await provider.complete("what is it?")

    assert result == "the answer"
