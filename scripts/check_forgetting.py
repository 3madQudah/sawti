"""Catastrophic-forgetting scaffold: base Qwen3-8B vs the QLoRA-tuned adapter.

Phase 5, step 4. **Scaffold only — never run in this session (no CUDA).**
Runs on Colab immediately after `scripts/train_qlora.py`, in the same
notebook session, so the adapter it compares against is already on disk and
the base checkpoint is already downloaded.

Purpose, deliberately narrow: this project's job here is to catch a *large*
regression in general instruction-following after fine-tuning on 18 narrow,
synthetic, call-analysis-correction examples — not to be a full eval suite,
and not to score response quality. A handful of generic prompts, run through
both models, compared for the shape a real forgetting regression takes
(empty output, or a drastic length collapse — degeneration into repetition
or refusal). Judging whether a *present*, reasonably-sized response is
actually good is left to whoever reads `data/finetune/forgetting_eval.json`.

Usage (Colab, after scripts/train_qlora.py has produced an adapter):
    uv run python scripts/check_forgetting.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from sawti.config import get_settings

logger = logging.getLogger("check_forgetting")

DEFAULT_ADAPTER_DIR = Path("data/finetune/qlora_adapter")
DEFAULT_OUTPUT_PATH = Path("data/finetune/forgetting_eval.json")
MAX_NEW_TOKENS = 200

# A handful of prompts with nothing to do with call analysis — general
# instruction-following and reasoning. Small and simple by design: this
# script's job is to catch a large regression, not to be a benchmark.
GENERIC_PROMPTS: tuple[str, ...] = (
    "What is the capital of France?",
    "Write a two-line haiku about the ocean.",
    "What is 17 multiplied by 23?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Translate 'good morning' into Spanish.",
    "List three primary colors.",
    "Explain what a for-loop does, in one sentence, to someone new to programming.",
)

# A tuned response under this fraction of the base response's length is
# flagged — the length-collapse shape a degenerated model produces. A
# judgment call, stated explicitly, same convention as
# `sawti.memory.diff.RUBRIC_SCORE_TOLERANCE`: not derived from data (there
# is no prior forgetting-eval run to derive it from), chosen to catch a
# large regression without flagging ordinary response-length variance.
LENGTH_COLLAPSE_RATIO = 0.2


def generate_responses(
    model: Any, tokenizer: Any, prompts: tuple[str, ...] = GENERIC_PROMPTS
) -> dict[str, str]:
    """Generate one greedy response per prompt from `model`, via its chat template.

    CUDA required for a real model, but the tensor-slicing logic here is
    exercised for real in `tests/scripts/test_check_forgetting.py` against a
    fake model/tokenizer pair — `torch` (CPU) is already a transitive
    dependency of `sentence-transformers` in this project's base
    environment, so that much needs no GPU.
    """
    import torch

    responses: dict[str, str] = {}
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        responses[prompt] = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return responses


def compare_responses(base: dict[str, str], tuned: dict[str, str]) -> list[dict[str, Any]]:
    """Pair up base/tuned responses per prompt and flag the regressions worth a human look."""
    rows: list[dict[str, Any]] = []
    for prompt, base_text in base.items():
        tuned_text = tuned.get(prompt, "")
        length_collapsed = bool(base_text) and len(tuned_text) < LENGTH_COLLAPSE_RATIO * len(base_text)
        flagged = not tuned_text.strip() or length_collapsed
        rows.append(
            {
                "prompt": prompt,
                "base_response": base_text,
                "tuned_response": tuned_text,
                "flagged": flagged,
            }
        )
    return rows


def build_model_and_tokenizer_pair(
    base_model: str, adapter_dir: Path
) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
    """Load `(base_model, tokenizer)` and `(base_model + LoRA adapter, tokenizer)`. CUDA required.

    Two separate base-model loads (one bare, one wrapped by `PeftModel`)
    rather than attaching/detaching one adapter in place — simpler, and the
    memory cost of loading an already-4-bit-quantized 8B model twice is
    small next to the download/quantization time either way (see
    `scripts/train_qlora.py`'s runtime estimate).
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=quant_config, device_map="auto"
    )
    tuned_base = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=quant_config, device_map="auto"
    )
    tuned = PeftModel.from_pretrained(tuned_base, str(adapter_dir))
    return (base, tokenizer), (tuned, tokenizer)


def run(
    *,
    base_model: str,
    adapter_dir: Path = DEFAULT_ADAPTER_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> list[dict[str, Any]]:
    """Run the full base-vs-tuned comparison and write it to `output_path`. CUDA required."""
    (base, tokenizer), (tuned, _tuned_tokenizer) = build_model_and_tokenizer_pair(base_model, adapter_dir)
    base_responses = generate_responses(base, tokenizer)
    tuned_responses = generate_responses(tuned, tokenizer)
    rows = compare_responses(base_responses, tuned_responses)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    flagged = [row for row in rows if row["flagged"]]
    if flagged:
        logger.warning("%d/%d prompts flagged for review — see %s", len(flagged), len(rows), output_path)
    else:
        logger.info("No prompts flagged (%d/%d checked) — see %s", len(rows), len(rows), output_path)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model", type=str, default=None, help="Overrides SAWTI_FINETUNE_BASE_MODEL."
    )
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    base_model = args.base_model or get_settings().finetune_base_model
    run(base_model=base_model, adapter_dir=args.adapter_dir, output_path=args.output_path)


if __name__ == "__main__":
    main()
