"""Tests for `scripts/train_qlora.py`.

Phase 5, step 3. This module's heavy dependencies (`torch`+CUDA,
`bitsandbytes`, `peft`, `trl`, `datasets`) are CUDA-only and not installed on
this project's Mac dev environment — see the module's own docstring. Every
heavy import happens lazily inside the functions that need it, specifically
so this test file can exercise the pure-Python parts (JSONL loading, chat
message formatting, template rendering against a fake tokenizer) without
requiring a GPU. Tests that do need one of those libraries use
`pytest.importorskip` so they run for real wherever the `finetune` extra is
installed (Colab) and skip cleanly here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from train_qlora import (
    LORA_ALPHA,
    LORA_R,
    LORA_TARGET_MODULES,
    load_jsonl,
    render_text,
    to_chat_messages,
)


class _FakeTokenizer:
    """A minimal stand-in for `transformers.PreTrainedTokenizerBase.apply_chat_template`."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        return "\n".join(f"<{m['role']}>{m['content']}" for m in conversation)


def _record(**overrides: object) -> dict[str, object]:
    base = {
        "call_id": "call_0000_ar",
        "language": "ar",
        "error_location": "commitments",
        "topic": "commitments",
        "source": "reviewer_action",
        "reviewer_id": "five-batch-experiment",
        "transcript": "Agent: hello\nCustomer: hi",
        "instruction": "Correct the commitments.",
        "input": "[]",
        "output": '[{"description": "call back"}]',
        "note": None,
    }
    base.update(overrides)
    return base


def test_load_jsonl_reads_records_in_order(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    records = [_record(call_id=f"call_{i}") for i in range(3)]
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    loaded = load_jsonl(path)

    assert [r["call_id"] for r in loaded] == ["call_0", "call_1", "call_2"]


def test_load_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(f"{json.dumps(_record())}\n\n\n", encoding="utf-8")

    assert len(load_jsonl(path)) == 1


def test_to_chat_messages_puts_correction_in_the_assistant_turn() -> None:
    record = _record(output='[{"description": "corrected"}]')
    messages = to_chat_messages(record)

    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == record["output"]


def test_to_chat_messages_user_turn_includes_transcript_and_original_output() -> None:
    record = _record(transcript="Agent: verbatim transcript text", input='[{"description": "wrong"}]')
    messages = to_chat_messages(record)

    assert "verbatim transcript text" in messages[0]["content"]
    assert '"description": "wrong"' in messages[0]["content"]
    assert record["instruction"] in messages[0]["content"]


def test_render_text_calls_tokenizer_chat_template_untokenized() -> None:
    record = _record()
    text = render_text(_FakeTokenizer(), record)

    assert text.startswith("<user>")
    assert "<assistant>" in text
    assert record["output"] in text


def test_lora_hyperparameters_follow_qlora_paper_convention() -> None:
    """alpha = 2x rank is the documented QLoRA-paper convention this script uses, not tuned."""
    assert LORA_ALPHA == 2 * LORA_R
    # Every linear projection, not attention-only — see module docstring.
    assert set(LORA_TARGET_MODULES) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }


def test_build_datasets_needs_the_datasets_package(tmp_path: Path) -> None:
    """Runs for real on a CUDA host with the `finetune` extra installed; skips here."""
    pytest.importorskip("datasets")
    from train_qlora import build_datasets

    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    train_path.write_text(json.dumps(_record()), encoding="utf-8")
    val_path.write_text(json.dumps(_record(call_id="call_val")), encoding="utf-8")

    train_dataset, val_dataset = build_datasets(train_path, val_path, _FakeTokenizer())

    assert len(train_dataset) == 1
    assert len(val_dataset) == 1
    assert "text" in train_dataset.column_names


def test_build_lora_config_needs_peft() -> None:
    """Runs for real on a CUDA host with the `finetune` extra installed; skips here."""
    pytest.importorskip("peft")
    from train_qlora import build_lora_config

    config = build_lora_config()

    assert config.r == LORA_R
    assert config.lora_alpha == LORA_ALPHA
    assert config.task_type == "CAUSAL_LM"


def test_train_needs_cuda_finetune_stack() -> None:
    """The full training path is exercised on Colab, not here — see module docstring."""
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    pytest.importorskip("bitsandbytes")
