"""Tests for `scripts/check_forgetting.py`.

Phase 5, step 4. `build_model_and_tokenizer_pair`/`run` need a real CUDA
`peft`/`bitsandbytes` stack (guarded with `pytest.importorskip`, skips
here). `generate_responses`'s tensor-slicing logic is tested for real
against a fake model/tokenizer pair — `torch` (CPU) and
`transformers.BatchEncoding` are already present in this project's base
environment (transitive deps of `sentence-transformers`), so that much
needs no GPU. `compare_responses` is pure Python and fully tested here.
"""

from __future__ import annotations

import pytest
from check_forgetting import GENERIC_PROMPTS, compare_responses, generate_responses


def test_generic_prompts_are_small_and_unrelated_to_call_analysis() -> None:
    """A handful, per the module's own scoping — not a benchmark-sized set."""
    assert 3 <= len(GENERIC_PROMPTS) <= 10
    call_analysis_terms = ("commitment", "compliance", "rubric", "transcript", "call_id")
    for prompt in GENERIC_PROMPTS:
        lowered = prompt.lower()
        assert not any(term in lowered for term in call_analysis_terms)


def test_compare_responses_flags_empty_tuned_response() -> None:
    rows = compare_responses(base={"q": "a full base answer"}, tuned={"q": ""})
    assert rows[0]["flagged"] is True


def test_compare_responses_flags_length_collapse() -> None:
    rows = compare_responses(base={"q": "x" * 100}, tuned={"q": "x" * 5})
    assert rows[0]["flagged"] is True


def test_compare_responses_does_not_flag_comparable_length() -> None:
    rows = compare_responses(base={"q": "x" * 100}, tuned={"q": "y" * 80})
    assert rows[0]["flagged"] is False


def test_compare_responses_flags_prompt_missing_from_tuned() -> None:
    rows = compare_responses(base={"q": "a full base answer"}, tuned={})
    assert rows[0]["flagged"] is True
    assert rows[0]["tuned_response"] == ""


def test_compare_responses_covers_every_base_prompt() -> None:
    base = {"q1": "answer one", "q2": "answer two"}
    tuned = {"q1": "reply one", "q2": "reply two"}
    rows = compare_responses(base, tuned)
    assert {row["prompt"] for row in rows} == {"q1", "q2"}


# --- generate_responses: real tensor-slicing logic, fake model/tokenizer ---


class _FakeGenerationModel:
    """Duck-types the `transformers` model interface `generate_responses` needs."""

    device = "cpu"

    def __init__(self, new_token_ids: list[int]) -> None:
        self._new_token_ids = new_token_ids

    def generate(self, *, input_ids, max_new_tokens: int, do_sample: bool):
        import torch

        assert do_sample is False
        assert max_new_tokens > 0
        new_tokens = torch.tensor([self._new_token_ids])
        return torch.cat([input_ids, new_tokens], dim=1)


class _FakeGenerationTokenizer:
    """Duck-types the tokenizer calls `generate_responses` makes, no network needed."""

    def __init__(self, prompt_token_ids: list[int], decoded_text: str) -> None:
        self._prompt_token_ids = prompt_token_ids
        self._decoded_text = decoded_text
        self.seen_prompts: list[str] = []

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return messages[0]["content"]

    def __call__(self, text: str, return_tensors: str):
        import torch
        from transformers import BatchEncoding

        assert return_tensors == "pt"
        self.seen_prompts.append(text)
        return BatchEncoding(data={"input_ids": torch.tensor([self._prompt_token_ids])})

    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return self._decoded_text


def test_generate_responses_strips_the_prompt_tokens_before_decoding() -> None:
    """The core correctness property: only newly-generated tokens get decoded, not the prompt."""
    model = _FakeGenerationModel(new_token_ids=[42, 43])
    tokenizer = _FakeGenerationTokenizer(prompt_token_ids=[1, 2, 3], decoded_text="the new tokens")

    responses = generate_responses(model, tokenizer, prompts=("hello there",))

    assert responses == {"hello there": "the new tokens"}
    assert tokenizer.seen_prompts == ["hello there"]


def test_generate_responses_covers_every_prompt() -> None:
    model = _FakeGenerationModel(new_token_ids=[7])
    tokenizer = _FakeGenerationTokenizer(prompt_token_ids=[1], decoded_text="ok")

    responses = generate_responses(model, tokenizer, prompts=("a", "b", "c"))

    assert set(responses) == {"a", "b", "c"}
    assert all(value == "ok" for value in responses.values())


# --- CUDA-only paths: run for real on Colab with the `finetune` extra, skip here ---


def test_build_model_and_tokenizer_pair_needs_peft() -> None:
    pytest.importorskip("peft")
    pytest.importorskip("bitsandbytes")
