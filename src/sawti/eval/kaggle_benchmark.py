"""Driver the Kaggle notebook calls: run one benchmark arm, or assemble the results file.

Phase 6.3. Lives in the repo (and so in the `sawti-repo` Kaggle bundle) rather
than in notebook cells, so it is tested locally and the notebook stays a thin
sequence of commands. Every arm goes through `sawti.eval.serving_benchmark`,
i.e. the service's own graph:

    python -m sawti.eval.kaggle_benchmark vllm --arm vllm_adapter_c16 --model sawti-qlora --concurrency 16
    python -m sawti.eval.kaggle_benchmark hf   --arm hf_adapter_c1 --adapter <adapter dir> --limit 3
    python -m sawti.eval.kaggle_benchmark assemble --out /kaggle/working/sawti_kaggle_results.json

`--limit N` takes the first N calls of the interleaved ar/en/mixed order, so a
small subset stays balanced across languages. `--deadline-epoch` stops new
calls from starting once the notebook's GPU-time budget is spent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sawti.eval.serving_benchmark import CallInput, load_call_inputs, load_checkpoint, run_arm
from sawti.llm.provider import LLMProvider

DEFAULT_CHECKPOINT = Path("/kaggle/working/sawti_runs.jsonl")
DEFAULT_CALLS_DIR = Path("/kaggle/input/sawti-comparison-calls")
DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"
MAX_TOKENS = 4096


def load_calls(calls_dir: Path, limit: int | None = None) -> list[CallInput]:
    """The comparison calls from the uploaded bundle (redacted transcripts), interleaved by language."""
    call_set = json.loads((calls_dir / "call_set.json").read_text(encoding="utf-8"))
    calls = load_call_inputs({k: v for k, v in call_set.items() if k != "_meta"}, calls_dir / "transcripts")
    return calls[:limit] if limit else calls


def _run(provider: LLMProvider, args: argparse.Namespace, note: dict[str, str]) -> int:
    calls = load_calls(args.calls_dir, args.limit)
    written = asyncio.run(
        run_arm(
            provider,
            calls,
            arm=args.arm,
            checkpoint=args.checkpoint,
            concurrency=args.concurrency,
            note=note,
            deadline_epoch=args.deadline_epoch,
            retry_wait_s=10.0,  # a local server: a transient error is a restart, not a quota
        )
    )
    records, _ = load_checkpoint(args.checkpoint)
    done = {r.call_id for r in records if r.arm == args.arm}
    print(f"{args.arm}: +{len(written)} this run, {len(done)}/{len(calls)} recorded")
    return 0


def build_vllm_provider(args: argparse.Namespace) -> LLMProvider:
    from sawti.llm.vllm_provider import VLLMProvider

    return VLLMProvider(args.base_url, args.model, max_tokens=MAX_TOKENS)


def build_hf_provider(args: argparse.Namespace) -> LLMProvider:  # pragma: no cover - needs a GPU
    """Qwen3-8B in fp16 across both T4s (`device_map="auto"`), optionally with the LoRA adapter.

    fp16, not Phase 5's 4-bit: the same precision as the vLLM arms, so the
    difference measured is the serving engine. Thinking off, like vLLM.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sawti.llm.local_hf_provider import LocalHFProvider

    tokenizer: Any = AutoTokenizer.from_pretrained(args.base_model)  # type: ignore[no-untyped-call]
    model: Any = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16, device_map="auto"
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    return LocalHFProvider(
        model, tokenizer, max_new_tokens=MAX_TOKENS, chat_template_kwargs={"enable_thinking": False}
    )


def _gpu_info() -> list[str]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def assemble(checkpoint: Path, out: Path, extra_meta: list[Path]) -> dict[str, Any]:
    """One results file: every record and segment, plus the environment it ran in."""
    records, segments = load_checkpoint(checkpoint)
    meta: dict[str, Any] = {
        "assembled_at_epoch": time.time(),
        "hardware": "free Kaggle T4x2",
        "gpus": _gpu_info(),
        "python": platform.python_version(),
    }
    for path in extra_meta:
        if path.exists():
            meta[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "meta": meta,
        "records": [r.model_dump(mode="json") for r in records],
        "segments": [s.model_dump(mode="json") for s in segments],
    }
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "records": len(records),
        "arms": sorted({r.arm for r in records}),
        "per_arm": {arm: sum(r.arm == arm for r in records) for arm in sorted({r.arm for r in records})},
    }


def main(
    argv: list[str] | None = None,
    *,
    providers: dict[str, Callable[[argparse.Namespace], LLMProvider]] | None = None,
) -> int:
    providers = providers or {"vllm": build_vllm_provider, "hf": build_hf_provider}
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("vllm", "hf"):
        p = sub.add_parser(name)
        p.add_argument("--arm", required=True)
        p.add_argument("--concurrency", type=int, default=1)
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--calls-dir", type=Path, default=DEFAULT_CALLS_DIR)
        p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
        p.add_argument("--deadline-epoch", type=float, default=None)
        if name == "vllm":
            p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
            p.add_argument("--model", required=True, help="the LoRA's served name, or the base model's")
        else:
            p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
            p.add_argument("--adapter", type=Path, default=None)
    p = sub.add_parser("assemble")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--meta", type=Path, nargs="*", default=[])
    args = parser.parse_args(argv)

    if args.command == "assemble":
        print(json.dumps(assemble(args.checkpoint, args.out, args.meta), indent=2))
        return 0
    provider = providers[args.command](args)
    engine = "vllm" if args.command == "vllm" else "hf-transformers"
    note = {
        "engine": engine,
        "hardware": "free Kaggle T4x2",
        "model": str(getattr(args, "model", None) or args.base_model),
    }
    if args.command == "hf":
        note["adapter"] = "on" if args.adapter else "off"
    return _run(provider, args, note)


if __name__ == "__main__":
    raise SystemExit(main())
