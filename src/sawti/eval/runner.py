"""Evaluation run orchestration: load reference labels, run the baseline, score per category, write results.

Phase 1: eval runner.

The baseline extraction path lives in `sawti.eval.plain_extraction` and is
deliberately kept at arm's length from the reference-label prompt — see that
module's docstring for why.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from sawti.config import get_settings
from sawti.data.ground_truth import load_ground_truth
from sawti.eval.metrics import (
    DEFAULT_SYNTHETIC_DIR,
    accuracy_by_category,
    grounding_precision_by_category,
    rubric_agreement_by_category,
    unsupported_claim_rate_by_category,
)
from sawti.eval.plain_extraction import run_plain_extraction
from sawti.schemas import CallAnalysis, Language

logger = logging.getLogger(__name__)

# Shown at the top of every results section this runner writes. The numbers
# below it are meaningless without it, so it is not optional and not a
# footnote — see `_render_results`.
REFERENCE_LABEL_CAVEAT = (
    "Reference labels for this run are LLM-generated, not human-reviewed — "
    "see docs/09-DECISIONS.md."
)

_METRIC_ORDER = (
    "Accuracy",
    "Rubric agreement",
    "Grounding precision",
    # Lower is better — the only metric in this table where that is true.
    "Unsupported claim rate",
)

_LANGUAGE_ORDER = (Language.AR, Language.EN, Language.MIXED)


def _format_score(scores: dict[Language, float], language: Language) -> str:
    """Render one cell of the results table, or an em dash when absent."""
    value = scores.get(language)
    return f"{value:.3f}" if value is not None else "—"


def _render_results(
    results: dict[str, dict[Language, float]],
    *,
    counts: dict[Language, int],
    failures: list[tuple[str, str]],
    experiment_name: str | None = None,
) -> str:
    """Render a dated results section for `eval_results.md`.

    The LLM-generated-reference caveat is emitted first, before the table,
    and is not conditional on anything.

    Args:
        results: Metric name to per-language scores.
        counts: Calls scored per language.
        failures: `(call_id, reason)` for calls that produced no `CallAnalysis`.
        experiment_name: Heading label. Defaults to the phase 1 baseline's, so
            the existing caller renders exactly as before.
    """
    settings = get_settings()
    label = experiment_name or "phase 1 baseline (plain-prompt extraction, no memory)"
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d")
    total_calls = sum(counts.values())

    lines = [
        f"## Round: {generated_at} — {label}",
        "",
        f"> **{REFERENCE_LABEL_CAVEAT}** These figures measure agreement between two "
        "LLM-driven processes, not accuracy against human judgment. Grounding "
        "precision is the exception: it is checked against the transcripts "
        "themselves and needs no reference labels.",
        "",
        "- Ground truth: `data/ground_truth` (LLM-generated reference labels, `human_reviewed: false`)",
        f"- Agent config / model: `{settings.llm_provider}` / `{settings.gemini_model}`",
        f"- Extraction: `{label}`",
        "- Memory: `off`",
        "- N calls scored: "
        + ", ".join(f"{language.value}=`{counts.get(language, 0)}`" for language in _LANGUAGE_ORDER)
        + f" (total `{total_calls}`)",
    ]
    if failures:
        lines.append(
            f"- Extraction failures: `{len(failures)}` call(s) produced no valid "
            "`CallAnalysis` and are excluded from every metric above "
            f"({', '.join(call_id for call_id, _ in failures[:10])}"
            + (", …" if len(failures) > 10 else "")
            + ")"
        )

    lines += [
        "",
        "| Metric | ar | en | mixed |",
        "| --- | --- | --- | --- |",
    ]
    for metric_name in _METRIC_ORDER:
        scores = results.get(metric_name, {})
        cells = " | ".join(_format_score(scores, language) for language in _LANGUAGE_ORDER)
        lines.append(f"| {metric_name} | {cells} |")

    lines.append("")
    return "\n".join(lines)


# An extractor turns one transcript file into one validated `CallAnalysis`.
# `run_plain_extraction` (the phase 1 baseline) and
# `sawti.eval.experiments.grounded_graph.extract_via_graph` (the phase 2 agent)
# both satisfy it, which is what makes the two rounds comparable: same reference
# set, same metrics, same harness — only this callable differs.
Extractor = Callable[[Path], CallAnalysis]


def _default_extractor(path: Path, *, max_retries: int, initial_delay: float) -> CallAnalysis:
    """Phase 1 baseline extractor, kept as the default so existing callers are unchanged."""
    return run_plain_extraction(path, max_retries=max_retries, initial_delay=initial_delay)


def run_eval(
    ground_truth_path: Path,
    *,
    output_path: Path,
    synthetic_dir: Path = DEFAULT_SYNTHETIC_DIR,
    sleep_seconds: float = 4.0,
    max_retries: int = 4,
    extractor: Extractor | None = None,
    experiment_name: str | None = None,
) -> dict[str, dict[Language, float]]:
    """Run the full evaluation pipeline and append results to `output_path`.

    Loads the reference labels, runs `run_plain_extraction` over each
    corresponding transcript, scores all three per-category metrics, and
    appends a dated section to `output_path` — led by the caveat that the
    reference labels are LLM-generated.

    A call whose extraction fails (provider exhausted, or output that does
    not satisfy `CallAnalysis`) is excluded from the metrics and reported by
    count in the written section, so a shrinking denominator can never pass
    for a rising score.

    Args:
        ground_truth_path: Directory of reference-label records.
        output_path: Where to append the results table (e.g. `eval_results.md`).
        synthetic_dir: Directory of `<call_id>.txt` transcripts.
        sleep_seconds: Delay between LLM calls, to respect free-tier rate limits.
        max_retries: Provider-error retries per call (not content retries).
        extractor: How to turn a transcript into a `CallAnalysis`. Defaults to the
            phase 1 plain-prompt baseline, so existing callers keep their behaviour.
        experiment_name: Label written into the results section heading.

    Returns:
        A mapping from metric name to its per-`Language` category scores.

    Raises:
        ValueError: If `ground_truth_path` holds no usable records.
    """
    reference = load_ground_truth(ground_truth_path)
    if not reference:
        raise ValueError(f"{ground_truth_path} contains no reference-label records")
    logger.info("loaded %d reference label record(s)", len(reference))

    predictions: list[CallAnalysis] = []
    failures: list[tuple[str, str]] = []

    for index, record in enumerate(reference, start=1):
        transcript_path = synthetic_dir / f"{record.call_id}.txt"
        if not transcript_path.is_file():
            failures.append((record.call_id, "transcript file missing"))
            logger.error("[%d/%d] %s: transcript missing", index, len(reference), record.call_id)
            continue
        try:
            predictions.append(
                extractor(transcript_path)
                if extractor is not None
                else _default_extractor(
                    transcript_path, max_retries=max_retries, initial_delay=sleep_seconds
                )
            )
            logger.info("[%d/%d] extracted %s", index, len(reference), record.call_id)
        except Exception as exc:
            failures.append((record.call_id, repr(exc)))
            logger.error("[%d/%d] %s FAILED: %r", index, len(reference), record.call_id, exc)
        if index < len(reference):
            time.sleep(sleep_seconds)

    scored_ids = {prediction.call_id for prediction in predictions}
    counts: dict[Language, int] = {}
    for record in reference:
        if record.call_id in scored_ids:
            counts[record.language] = counts.get(record.language, 0) + 1

    results: dict[str, dict[Language, float]] = {
        "Accuracy": accuracy_by_category(predictions, reference),
        "Rubric agreement": rubric_agreement_by_category(predictions, reference),
        "Grounding precision": grounding_precision_by_category(
            predictions, transcript_dir=synthetic_dir
        ),
        "Unsupported claim rate": unsupported_claim_rate_by_category(
            predictions, transcript_dir=synthetic_dir
        ),
    }

    section = _render_results(
        results, counts=counts, failures=failures, experiment_name=experiment_name
    )
    existing = output_path.read_text(encoding="utf-8") if output_path.is_file() else ""
    separator = "" if existing.endswith("\n\n") or not existing else "\n"
    output_path.write_text(existing + separator + section, encoding="utf-8")
    logger.info("appended results section to %s", output_path)

    return results
