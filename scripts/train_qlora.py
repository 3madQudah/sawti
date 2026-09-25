"""QLoRA fine-tuning of the Phase 5 base model over the corrections dataset.

Phase 5, step 3. **CUDA-only — run on Colab or another CUDA host, never on
this project's Mac dev environment.** This machine has no CUDA (`.venv-linux`
is aarch64, `docker-compose.yml` has no GPU service; see
docs/09-DECISIONS.md), and `bitsandbytes`'s 4-bit quantization kernels need a
real GPU. Install the extra first: `uv pip install -e ".[finetune]"`.

What this trains on: `data/finetune/train.jsonl` / `val.jsonl`, built by
`scripts/build_finetune_dataset.py`. **Every example there is synthetic** —
manufactured by diffing an agent output against an LLM-generated reference
label, never a real QA reviewer's judgment (see that script's docstring and
docs/09-DECISIONS.md, 2026-09-25). This run validates the QLoRA training
*mechanism* end-to-end on 18 training examples; it is not evidence of
real-world learning capacity, and the loss curve it produces should be read
that way.

Hyperparameters (`LORA_*`, `BNB_*`, training args below) are the QLoRA
paper's (Dettmers et al. 2023) documented defaults, not tuned against this
dataset — 18 examples is far too small to tune hyperparameters against
without simply overfitting the tuning itself. See docs/09-DECISIONS.md.

Runtime/tier estimate (not measured — this script has never run, since it
needs CUDA this session doesn't have): a T4 (16 GB VRAM, Colab's free tier)
should be sufficient — QLoRA on an 8B model needs roughly 5-6 GB of VRAM for
the 4-bit weights plus LoRA adapter/optimizer state, comfortably under a
T4's 16 GB. Expect ~10-15 minutes wall-clock, almost all of it downloading
and quantizing the ~16 GB bf16 base checkpoint from the Hub; actual training
on 18 examples for 3 epochs (~14 optimizer steps at this script's batch
size / gradient accumulation) is well under a minute of GPU compute. An A100
would only meaningfully help if the dataset or epoch count grows a lot from
here. Sanity-check the first training step's loss before walking away from a
longer run — `notebooks/qlora_train.ipynb` runs this script directly, no
separate smoke test.

Usage (Colab / any CUDA host):
    uv pip install -e ".[finetune]"
    uv run python scripts/train_qlora.py

Output: a LoRA adapter (not merged into the base model) at
`data/finetune/qlora_adapter/`, plus a per-step loss CSV at
`data/finetune/train_log.csv` (`step,epoch,train_loss,eval_loss`).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from sawti.config import get_settings

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger("train_qlora")

DEFAULT_TRAIN_PATH = Path("data/finetune/train.jsonl")
DEFAULT_VAL_PATH = Path("data/finetune/val.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/finetune/qlora_adapter")
DEFAULT_LOG_PATH = Path("data/finetune/train_log.csv")

# --- LoRA hyperparameters — QLoRA-paper defaults, not tuned. See module docstring. ---
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Every linear projection, attention and MLP alike — the QLoRA paper's own
# finding that adapting attention-only underperforms adapting everything.
LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
LORA_BIAS = "none"

# --- 4-bit NF4 quantization — the QLoRA paper's recommended configuration. ---
BNB_QUANT_TYPE = "nf4"
BNB_USE_DOUBLE_QUANT = True
BNB_COMPUTE_DTYPE = "bfloat16"

# --- Training hyperparameters ---
# A dataset this small needs several passes to produce any loss signal worth
# reading; 3 epochs is a common small-SFT-set default, not tuned against
# this run's own loss curve (which does not exist yet — see module docstring).
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.03
# Covers the longest observed training example (~4.9K chars => well under
# 2K tokens for a BPE-style tokenizer) with headroom.
MAX_SEQ_LENGTH = 3072


class ChatTemplateTokenizer(Protocol):
    """The one method this module needs from a tokenizer — real or a test fake.

    Deliberately loose (`**kwargs`, broad return type): a real
    `transformers.PreTrainedTokenizerBase.apply_chat_template` takes many
    more optional parameters than this module ever passes, and this
    Protocol only needs to describe the call shape `render_text()` actually
    uses.
    """

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = ...,
        add_generation_prompt: bool = ...,
        **kwargs: Any,
    ) -> Any: ...


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a `scripts/build_finetune_dataset.py`-shaped JSONL file into records."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def to_chat_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Turn one dataset record into a chat-format SFT pair.

    The user turn carries the instruction, transcript, and the agent's
    original (wrong) field output; the assistant turn is the reviewer's
    corrected output — the "self-correction" framing
    `scripts/build_finetune_dataset.py`'s own docstring describes: given a
    transcript and its own mistake, produce the fix.
    """
    user_content = (
        f"{record['instruction']}\n\n"
        f"Transcript:\n{record['transcript']}\n\n"
        f"Agent's output (`{record['topic']}`):\n{record['input']}"
    )
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": record["output"]},
    ]


def render_text(tokenizer: ChatTemplateTokenizer, record: dict[str, Any]) -> str:
    """Render one record through `tokenizer`'s chat template, as plain text (not tokenized)."""
    messages = to_chat_messages(record)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def _text_dataset(records: list[dict[str, Any]], tokenizer: ChatTemplateTokenizer) -> Dataset:
    from datasets import Dataset

    return Dataset.from_dict({"text": [render_text(tokenizer, record) for record in records]})


def build_datasets(
    train_path: Path, val_path: Path, tokenizer: ChatTemplateTokenizer
) -> tuple[Dataset, Dataset]:
    """Render `train_path`/`val_path` into `datasets.Dataset`s with a single `"text"` column."""
    return (
        _text_dataset(load_jsonl(train_path), tokenizer),
        _text_dataset(load_jsonl(val_path), tokenizer),
    )


def build_model_and_tokenizer(base_model: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load `base_model` 4-bit NF4-quantized, plus its tokenizer. CUDA required."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=BNB_QUANT_TYPE,
        bnb_4bit_use_double_quant=BNB_USE_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=getattr(torch, BNB_COMPUTE_DTYPE),
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        # Qwen's tokenizers commonly ship without a pad token; the eos token
        # is the standard stand-in for causal-LM SFT padding.
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=quant_config, device_map="auto"
    )
    return model, tokenizer


def build_lora_config() -> Any:
    """The `peft.LoraConfig` this run trains — see the `LORA_*` constants above."""
    from peft import LoraConfig

    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(LORA_TARGET_MODULES),
        bias=LORA_BIAS,
        task_type="CAUSAL_LM",
    )


def _csv_logger_callback(log_path: Path) -> Any:
    """A `transformers.TrainerCallback` that appends every log event to `log_path` as CSV.

    Plain CSV, not Langfuse: Langfuse traces LLM *calls*, and nothing in
    this codebase has wired it into anything training-loop-shaped — building
    that integration for one small run would be new observability infra for
    a problem `csv.writer` already solves. See docs/09-DECISIONS.md.
    """
    from transformers import TrainerCallback

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    writer.writerow(["step", "epoch", "train_loss", "eval_loss"])
    handle.flush()

    class CsvLoggerCallback(TrainerCallback):
        def on_log(
            self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **_: Any
        ) -> None:
            if not logs:
                return
            writer.writerow([state.global_step, logs.get("epoch"), logs.get("loss"), logs.get("eval_loss")])
            handle.flush()

    return CsvLoggerCallback()


def train(
    *,
    base_model: str,
    train_path: Path = DEFAULT_TRAIN_PATH,
    val_path: Path = DEFAULT_VAL_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    log_path: Path = DEFAULT_LOG_PATH,
    num_train_epochs: int = NUM_TRAIN_EPOCHS,
) -> Path:
    """Run the full QLoRA fine-tune and save the resulting adapter to `output_dir`.

    CUDA required — every heavy import (`torch`, `transformers`, `peft`,
    `trl`) happens inside this function (and the functions it calls) so that
    importing this module on a machine without the `finetune` extra
    installed — this project's own Mac dev environment included — does not
    fail on import, only on an actual call to `train()`/`main()`.
    """
    from peft import prepare_model_for_kbit_training
    from transformers import TrainingArguments
    from trl import SFTTrainer

    model, tokenizer = build_model_and_tokenizer(base_model)
    model = prepare_model_for_kbit_training(model)
    lora_config = build_lora_config()
    train_dataset, val_dataset = build_datasets(train_path, val_path, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1,
        bf16=True,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        callbacks=[_csv_logger_callback(log_path)],
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model", type=str, default=None, help="Overrides SAWTI_FINETUNE_BASE_MODEL."
    )
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--num-train-epochs", type=int, default=NUM_TRAIN_EPOCHS)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    base_model = args.base_model or get_settings().finetune_base_model
    logger.info("Training %s on %s, evaluating on %s", base_model, args.train_path, args.val_path)

    output_dir = train(
        base_model=base_model,
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        log_path=args.log_path,
        num_train_epochs=args.num_train_epochs,
    )
    logger.info("LoRA adapter saved to %s (loss log: %s)", output_dir, args.log_path)


if __name__ == "__main__":
    main()
