"""Held-out-calls accuracy comparison: base Qwen3-8B vs. QLoRA-tuned.

Phase 5 — the part of the original scope that never got built (see
docs/09-DECISIONS.md, 2026-09-27). **CUDA-only**, run on Colab alongside
`scripts/train_qlora.py` / `scripts/check_forgetting.py`.

What this measures: the 15 calls in `data/finetune/held_out_call_ids.json`
were never trained or validated on. Both the base and QLoRA-tuned model run
the *same* extraction path this project already uses for Phase 1's
baseline — `sawti.eval.plain_extraction.run_plain_extraction`, reused via
the exact provider-injection point `tests/eval/test_plain_extraction.py`
already relies on, not reinvented — over each held-out call's transcript.
Both prediction sets are scored against `data/ground_truth` with the exact
same `sawti.eval.metrics` functions Phases 1/2/4 use: `accuracy_by_category`,
`rubric_agreement_by_category`, `grounding_precision_by_category`,
`unsupported_claim_rate_by_category`. The delta (tuned − base) is the
number this script exists to produce.

Read the result carefully: every ground-truth label and every training
example remains synthetic (see `scripts/build_finetune_dataset.py`'s
docstring and `docs/09-DECISIONS.md`). The LoRA adapter was trained only on
narrow field-correction pairs (transcript + wrong field → corrected field —
see `scripts/train_qlora.py`'s docstring), never on the "produce a full
CallAnalysis from a transcript" prompt this script uses. A positive delta
here is a small-sample (15-call) signal that the fine-tuning transfers to
the broader task it was never directly trained on, not proof of a
production-ready improvement.

Usage (Colab, after scripts/train_qlora.py has produced an adapter):
    uv run python scripts/eval_held_out_calls.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch

from sawti.config import get_settings
from sawti.data.ground_truth import load_ground_truth
from sawti.eval.metrics import (
    accuracy_by_category,
    grounding_precision_by_category,
    rubric_agreement_by_category,
    unsupported_claim_rate_by_category,
)
from sawti.eval.plain_extraction import run_plain_extraction
from sawti.schemas import CallAnalysis, Language

logger = logging.getLogger("eval_held_out_calls")

DEFAULT_HELD_OUT_PATH = Path("data/finetune/held_out_call_ids.json")
DEFAULT_GROUND_TRUTH_DIR = Path("data/ground_truth")
DEFAULT_SYNTHETIC_DIR = Path("data/synthetic")
DEFAULT_ADAPTER_DIR = Path("data/finetune/qlora_adapter")
DEFAULT_OUTPUT_PATH = Path("data/finetune/held_out_eval.json")

_METRIC_ORDER = ("Accuracy", "Rubric agreement", "Grounding precision", "Unsupported claim rate")
_LANGUAGE_ORDER = (Language.AR, Language.EN, Language.MIXED)

Metrics = dict[str, dict[Language, float]]


def load_held_out_call_ids(path: Path) -> dict[Language, list[str]]:
    """Every held-out call_id in `path`, grouped by language (the `_meta` key is skipped)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[Language, list[str]] = {}
    for key, call_ids in payload.items():
        if key == "_meta":
            continue
        result[Language(key)] = list(call_ids)
    return result


def _extract_all(
    call_ids: list[str],
    *,
    synthetic_dir: Path,
    provider: Any,
) -> list[CallAnalysis]:
    """Run `run_plain_extraction` over every one of `call_ids`, with `provider` injected.

    `provider` replaces `get_llm_provider()`'s configured result for the
    duration of this call only — the same injection point
    `tests/eval/test_plain_extraction.py` uses, applied here to a real
    provider instead of a test stub.

    A missing transcript or a failed extraction is logged and the call is
    skipped, matching `sawti.eval.runner.run_eval`'s own convention: excluded
    from every metric, never silently zero-filled. `run_plain_extraction`'s
    own retry budget is 0 here — `LocalHFProvider.structured_complete`
    already retries internally with sampling; retrying again on top would
    just re-run that same internal retry loop from scratch.
    """
    predictions: list[CallAnalysis] = []
    with patch("sawti.eval.plain_extraction.get_llm_provider", lambda: provider):
        for index, call_id in enumerate(call_ids, start=1):
            transcript_path = synthetic_dir / f"{call_id}.txt"
            if not transcript_path.is_file():
                logger.error("[%d/%d] %s: transcript missing, skipping", index, len(call_ids), call_id)
                continue
            try:
                predictions.append(run_plain_extraction(transcript_path, max_retries=0))
            except Exception as exc:
                logger.error("[%d/%d] %s: extraction failed: %r", index, len(call_ids), call_id, exc)
                continue
            logger.info("[%d/%d] %s: extracted", index, len(call_ids), call_id)
    return predictions


def _metrics_for(
    predictions: list[CallAnalysis], reference: list[CallAnalysis], *, synthetic_dir: Path
) -> Metrics:
    return {
        "Accuracy": accuracy_by_category(predictions, reference),
        "Rubric agreement": rubric_agreement_by_category(predictions, reference),
        "Grounding precision": grounding_precision_by_category(predictions, transcript_dir=synthetic_dir),
        "Unsupported claim rate": unsupported_claim_rate_by_category(
            predictions, transcript_dir=synthetic_dir
        ),
    }


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "—"


def render_comparison(base_metrics: Metrics, tuned_metrics: Metrics, *, counts: dict[Language, int]) -> str:
    """A base/tuned/delta table per metric, for every language present in `counts`."""
    languages = [language for language in _LANGUAGE_ORDER if counts.get(language, 0) > 0]
    lines = [
        "N calls scored: " + ", ".join(f"{language.value}=`{counts[language]}`" for language in languages),
        "",
    ]
    for metric_name in _METRIC_ORDER:
        lines.append(f"**{metric_name}**")
        lines.append("| Language | Base | Tuned | Δ (tuned − base) |")
        lines.append("| --- | --- | --- | --- |")
        base_scores = base_metrics.get(metric_name, {})
        tuned_scores = tuned_metrics.get(metric_name, {})
        for language in languages:
            base_value = base_scores.get(language)
            tuned_value = tuned_scores.get(language)
            delta = (
                tuned_value - base_value if base_value is not None and tuned_value is not None else None
            )
            delta_str = f"{delta:+.3f}" if delta is not None else "—"
            lines.append(f"| {language.value} | {_fmt(base_value)} | {_fmt(tuned_value)} | {delta_str} |")
        lines.append("")
    return "\n".join(lines)


def run(
    *,
    base_model: str,
    adapter_dir: Path = DEFAULT_ADAPTER_DIR,
    held_out_path: Path = DEFAULT_HELD_OUT_PATH,
    ground_truth_dir: Path = DEFAULT_GROUND_TRUTH_DIR,
    synthetic_dir: Path = DEFAULT_SYNTHETIC_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Run the full base-vs-tuned held-out comparison and write it to `output_path`.

    CUDA required — `build_model_and_tokenizer_pair` (from
    `scripts/check_forgetting.py`, reused rather than duplicated a third
    time) loads two real copies of `base_model`, one bare and one wrapped by
    the LoRA adapter at `adapter_dir`.

    Raises:
        ValueError: A held-out call_id has no matching record in
            `ground_truth_dir` — the corpus and the held-out file have
            drifted apart.
    """
    from check_forgetting import build_model_and_tokenizer_pair

    from sawti.llm.local_hf_provider import LocalHFProvider

    held_out_by_language = load_held_out_call_ids(held_out_path)
    all_call_ids = [call_id for ids in held_out_by_language.values() for call_id in ids]

    ground_truth_all = load_ground_truth(ground_truth_dir)
    call_id_set = set(all_call_ids)
    reference = [record for record in ground_truth_all if record.call_id in call_id_set]
    missing = call_id_set - {record.call_id for record in reference}
    if missing:
        message = f"held-out call_ids missing from {ground_truth_dir}: {sorted(missing)}"
        raise ValueError(message)

    (base_model_obj, base_tokenizer), (tuned_model_obj, tuned_tokenizer) = build_model_and_tokenizer_pair(
        base_model, adapter_dir
    )
    base_provider = LocalHFProvider(base_model_obj, base_tokenizer)
    tuned_provider = LocalHFProvider(tuned_model_obj, tuned_tokenizer)

    base_predictions = _extract_all(all_call_ids, synthetic_dir=synthetic_dir, provider=base_provider)
    tuned_predictions = _extract_all(all_call_ids, synthetic_dir=synthetic_dir, provider=tuned_provider)

    scored_ids = {p.call_id for p in base_predictions} & {p.call_id for p in tuned_predictions}
    counts: dict[Language, int] = {}
    for record in reference:
        if record.call_id in scored_ids:
            counts[record.language] = counts.get(record.language, 0) + 1

    base_metrics = _metrics_for(base_predictions, reference, synthetic_dir=synthetic_dir)
    tuned_metrics = _metrics_for(tuned_predictions, reference, synthetic_dir=synthetic_dir)
    comparison = render_comparison(base_metrics, tuned_metrics, counts=counts)

    def _serialize(metrics: Metrics) -> dict[str, dict[str, float]]:
        return {
            name: {language.value: value for language, value in scores.items()}
            for name, scores in metrics.items()
        }

    result = {
        "base_metrics": _serialize(base_metrics),
        "tuned_metrics": _serialize(tuned_metrics),
        "counts": {language.value: count for language, count in counts.items()},
        "comparison_table": comparison,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(comparison)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model", type=str, default=None, help="Overrides SAWTI_FINETUNE_BASE_MODEL."
    )
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--held-out-path", type=Path, default=DEFAULT_HELD_OUT_PATH)
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GROUND_TRUTH_DIR)
    parser.add_argument("--synthetic-dir", type=Path, default=DEFAULT_SYNTHETIC_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    base_model = args.base_model or get_settings().finetune_base_model

    run(
        base_model=base_model,
        adapter_dir=args.adapter_dir,
        held_out_path=args.held_out_path,
        ground_truth_dir=args.ground_truth_dir,
        synthetic_dir=args.synthetic_dir,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
