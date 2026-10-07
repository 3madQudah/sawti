"""Build the phase 6.3 cloud vs self-hosted comparison table, per language category.

Inputs: the Gemini checkpoint (`scripts/run_cloud_arm.py`) and the results
file downloaded from the Kaggle notebook. Every arm is scored by one function,
`sawti.eval.serving_benchmark.score_records` (the project's metrics,
unchanged), against `data/ground_truth`.

Columns, per language (never blended): accuracy, rubric agreement, grounding
precision, unsupported claim rate, schema-valid rate, latency p50/p95 with n,
throughput, cost per 1000 calls, transcript bytes sent to a third party.

**Cost — priced as if paid; the actual spend of every run here is $0.**

* Gemini: the paid standard-tier token price of the model the calls actually
  resolved to (`gemini-3.5-flash-lite`: $0.30 / 1M input, $2.50 / 1M output;
  ai.google.dev/gemini-api/docs/pricing, last updated 2026-10-07), times each
  language's mean prompt and output tokens.
* Self-hosted: the hourly price of an equivalent 2×T4 rental × the GPU time
  per 1000 calls at the best measured concurrency. GPU time is allocated to a
  language by its output tokens (the arm's wall time per output token × that
  language's mean output tokens), because throughput is measured on the
  mixed-language workload, not per language. Price: Google Cloud T4 at $0.42
  per GPU-hour all-in, i.e. $0.845/h for two (getdeploying.com/gpus/nvidia-t4,
  updated 2026-10-07 — a price aggregator; Google's own pricing page is the
  primary source and should be re-checked). Sensitivity: the cheapest listed
  2×T4 rental, $1.11/h (same source, same date).

**Egress** — UTF-8 bytes of transcript text sent to a third party per call,
measured in the request. Gemini: sent to Google per call. Kaggle arms: the
transcripts were uploaded to Kaggle (Google) once, as the bundle — so this
benchmark *does* send them to a third party; the requests themselves only go
to localhost. The zero-egress property belongs to running the same container
on your own hardware, which this run did not do. All transcripts are
synthetic and redacted.

Usage:
    uv run python scripts/assemble_serving_comparison.py --kaggle sawti_kaggle_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sawti.data.ground_truth import load_ground_truth
from sawti.eval.serving_benchmark import CallRecord, Segment, load_checkpoint, score_records, throughput

GEMINI_CHECKPOINT = Path("data/serving_benchmark/gemini.jsonl")
GROUND_TRUTH_DIR = Path("data/ground_truth")
SYNTHETIC_DIR = Path("data/synthetic")
OUT_JSON = Path("data/serving_benchmark/comparison.json")

#: USD per 1M tokens, paid standard tier, by the model id the API reported.
GEMINI_PRICES = {"gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50}}
GEMINI_PRICE_SOURCE = "ai.google.dev/gemini-api/docs/pricing (last updated 2026-10-07), standard paid tier"
T4X2_HOURLY_USD = 0.845
T4X2_HOURLY_USD_SENSITIVITY = 1.11
T4X2_PRICE_SOURCE = (
    "getdeploying.com/gpus/nvidia-t4 (updated 2026-10-07): Google Cloud T4 $0.42/GPU-h all-in "
    "-> $0.845/h for two; cheapest listed 2xT4 rental $1.11/h (sensitivity)"
)
HARDWARE = {
    "gemini": "Google-hosted (Gemini API, free tier, 1 in flight)",
    "kaggle": "free Kaggle T4x2 (2x NVIDIA T4 16 GB, sm_75, fp16, TP=2)",
}
ARM_ORDER = ("gemini", "vllm_adapter_c1", "vllm_adapter_c16", "vllm_base_c16", "hf_adapter_c1")
ARM_LABEL = {
    "gemini": "Cloud — Gemini (gemini-flash-lite-latest)",
    "vllm_adapter_c1": "Self-hosted — vLLM, QLoRA adapter, concurrency 1",
    "vllm_adapter_c16": "Self-hosted — vLLM, QLoRA adapter, concurrency 16",
    "vllm_base_c16": "Self-hosted — vLLM, base Qwen3-8B, concurrency 16",
    "hf_adapter_c1": "Self-hosted — HF transformers fp16, adapter, concurrency 1",
}


def load_kaggle(path: Path) -> tuple[list[CallRecord], list[Segment], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        [CallRecord.model_validate(r) for r in payload["records"]],
        [Segment.model_validate(s) for s in payload["segments"]],
        payload.get("meta", {}),
    )


def gemini_cost_per_1000(row: dict[str, Any], model: str | None) -> float | None:
    price = GEMINI_PRICES.get(model or "")
    if price is None or row["prompt_tokens_mean"] is None or row["completion_tokens_mean"] is None:
        return None
    per_call = (
        row["prompt_tokens_mean"] * price["input"] / 1e6
        + row["completion_tokens_mean"] * price["output"] / 1e6
    )
    return round(per_call * 1000, 4)


def gpu_cost_per_1000(
    row: dict[str, Any], arm_tp: dict[str, Any], arm_tokens_out: int, hourly: float
) -> float | None:
    """Hourly price × GPU-hours per 1000 calls, GPU time allocated to this language by output tokens."""
    if not arm_tp["seconds"] or not arm_tokens_out or row["completion_tokens_mean"] is None:
        return None
    seconds_per_token = arm_tp["seconds"] / arm_tokens_out
    seconds_per_call = seconds_per_token * row["completion_tokens_mean"]
    return round(hourly / 3600 * seconds_per_call * 1000, 4)


def build(kaggle_path: Path | None, gemini_path: Path = GEMINI_CHECKPOINT) -> dict[str, Any]:
    records: list[CallRecord] = []
    segments: list[Segment] = []
    meta: dict[str, Any] = {}
    if gemini_path.exists():
        gemini_records, gemini_segments = load_checkpoint(gemini_path)
        records += gemini_records
        segments += gemini_segments
    if kaggle_path is not None:
        k_records, k_segments, meta = load_kaggle(kaggle_path)
        records += k_records
        segments += k_segments
    truth = load_ground_truth(GROUND_TRUTH_DIR)

    arms: dict[str, Any] = {}
    for arm in [a for a in ARM_ORDER if any(r.arm == a for r in records)]:
        rows = [r for r in records if r.arm == arm]
        scored = score_records(rows, truth, transcript_dir=SYNTHETIC_DIR)
        tp = throughput(records, segments, arm=arm)
        tokens_out = sum(r.completion_tokens or 0 for r in rows)
        models = sorted({r.model for r in rows if r.model})
        for row in scored.values():
            if arm == "gemini":
                row["cost_per_1000_usd"] = gemini_cost_per_1000(row, models[-1] if models else None)
                row["third_party_transcript_bytes_per_call"] = row["transcript_bytes_sent_mean"]
            else:
                row["cost_per_1000_usd"] = gpu_cost_per_1000(row, tp, tokens_out, T4X2_HOURLY_USD)
                row["cost_per_1000_usd_sensitivity"] = gpu_cost_per_1000(
                    row, tp, tokens_out, T4X2_HOURLY_USD_SENSITIVITY
                )
                # Requests went to localhost; the transcript reached Kaggle once, as the uploaded bundle.
                row["third_party_transcript_bytes_per_call"] = row["transcript_bytes_sent_mean"]
        arms[arm] = {
            "label": ARM_LABEL[arm],
            "hardware": HARDWARE["gemini" if arm == "gemini" else "kaggle"],
            "models": models,
            "throughput": tp,
            "per_language": scored,
        }
    return {
        "arms": arms,
        "kaggle_meta": meta,
        "pricing": {"gemini": GEMINI_PRICE_SOURCE, "t4x2": T4X2_PRICE_SOURCE, "actual_spend_usd": 0},
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def to_markdown(result: dict[str, Any]) -> str:
    lines = []
    for language in ("ar", "en", "mixed"):
        lines += [
            f"### {language}",
            "",
            "| Arm | Hardware | n | Accuracy | Rubric agr. | Grounding prec. | Unsupported | Schema-valid "
            "| Latency p50 / p95 (s) | Throughput (calls/min) | Cost / 1000 (USD, as if paid) "
            "| Transcript bytes to a third party / call |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for arm in result["arms"].values():
            row = arm["per_language"].get(language)
            if row is None:
                continue
            lines.append(
                f"| {arm['label']} | {arm['hardware']} | {row['n']} | {_fmt(row['accuracy'])} "
                f"| {_fmt(row['rubric_agreement'])} | {_fmt(row['grounding_precision'])} "
                f"| {_fmt(row['unsupported_claim_rate'])} | {_fmt(row['schema_valid_rate'])} "
                f"| {_fmt(row['latency_p50_s'], 1)} / {_fmt(row['latency_p95_s'], 1)} "
                f"| {_fmt(arm['throughput']['calls_per_min'], 2)} | {_fmt(row.get('cost_per_1000_usd'), 2)} "
                f"| {_fmt(row['third_party_transcript_bytes_per_call'], 0)} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--kaggle", type=Path, default=None)
    parser.add_argument("--gemini", type=Path, default=GEMINI_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args()
    result = build(args.kaggle, args.gemini)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
