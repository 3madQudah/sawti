"""Evaluation metrics — always computed and reported per language category.

Phase 1: eval foundation. Never add a function here that returns a single
blended float across languages; every metric returns a per-category mapping.
Structurally enforced by the return type: `dict[Language, float]`, not `float`.

A note on what these numbers mean right now: the records these metrics are
scored against are LLM-generated reference labels, not human-reviewed ground
truth (see `sawti.data.ground_truth` and `docs/09-DECISIONS.md`). Agreement
against them is agreement between two LLM-driven processes.
`grounding_precision_by_category` is the exception — it checks predictions
against the transcript itself, so it needs no reference labels and is not
affected by that limitation.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from sawti.quotes import is_verbatim
from sawti.schemas import CallAnalysis, Language, Quote

DEFAULT_SYNTHETIC_DIR = Path("data/synthetic")

# Sentiment movement smaller than this counts as "flat" rather than a
# direction. Sentiment scores run -1..1, so 0.1 is a 5% swing.
_SENTIMENT_FLAT_EPSILON = 0.1

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    """Lowercase word tokens of `text`, for rough cross-language text overlap."""
    return {token.lower() for token in _TOKEN_RE.findall(text)}


def _text_overlap(left: str, right: str) -> float:
    """Jaccard token overlap of two strings, in 0.0-1.0.

    Deliberately crude: it is comparing two free-text descriptions of the
    same event ("agent will cancel the line today" vs "cancel account
    immediately"), where exact match is the wrong bar and full semantic
    similarity would mean another model call inside the metric.
    """
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _pair_by_call_id(
    predictions: list[CallAnalysis], ground_truth: list[CallAnalysis]
) -> list[tuple[CallAnalysis, CallAnalysis]]:
    """Pair predictions with reference records on `call_id`.

    Records present on only one side are dropped: a metric cannot say
    anything about a call it only half has. Ordering follows `ground_truth`.
    """
    by_call_id = {prediction.call_id: prediction for prediction in predictions}
    return [
        (by_call_id[reference.call_id], reference)
        for reference in ground_truth
        if reference.call_id in by_call_id
    ]


def _mean(values: list[float]) -> float:
    """Arithmetic mean, or 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


def _commitment_agreement(prediction: CallAnalysis, reference: CallAnalysis) -> float:
    """Score commitment extraction: half count agreement, half description match.

    Both empty scores 1.0 (correctly finding nothing is correct). One empty
    and the other not scores 0.0. Otherwise: the count component is
    `1 - |n_pred - n_ref| / max(n_pred, n_ref)`, and the description
    component is the mean, over reference commitments, of the best token
    overlap against any predicted commitment — a greedy best-match rather
    than an assignment problem, because commitment lists are short.
    """
    predicted, expected = prediction.commitments, reference.commitments
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0

    count_score = 1.0 - abs(len(predicted) - len(expected)) / max(len(predicted), len(expected))
    description_score = _mean(
        [
            max(_text_overlap(item.description, candidate.description) for candidate in predicted)
            for item in expected
        ]
    )
    return (count_score + description_score) / 2


def _sentiment_direction(analysis: CallAnalysis) -> str:
    """Classify the call's overall sentiment movement as up / down / flat / unknown.

    "unknown" means fewer than two points, i.e. no movement was recorded at
    all — distinct from "flat", which is a measured lack of movement.
    """
    points = analysis.sentiment_trajectory.points
    if len(points) < 2:
        return "unknown"
    delta = points[-1].score - points[0].score
    if abs(delta) < _SENTIMENT_FLAT_EPSILON:
        return "flat"
    return "up" if delta > 0 else "down"


def accuracy_by_category(
    predictions: list[CallAnalysis], ground_truth: list[CallAnalysis]
) -> dict[Language, float]:
    """Compute accuracy, grouped by `Language` category.

    Scoring rule (a judgment call, so stated explicitly). Predictions and
    references are paired on `call_id`; each pair scores the unweighted mean
    of four field-level components, and each language's accuracy is the mean
    of its pairs' scores:

      1. **summary presence** — 1.0 when both sides have a non-empty summary.
         Note this component is near-trivially satisfied, because
         `CallAnalysis.summary` has `min_length=1`; it can only fail when a
         record was built outside the schema. It is kept because "did the
         system produce a summary at all" is the intended field-level check,
         but it should not be read as evidence the summaries *agree*.
      2. **commitments** — see `_commitment_agreement`: half count agreement,
         half fuzzy description overlap.
      3. **compliance flags presence** — 1.0 when the two sides agree on
         *whether* the call contains any violation. Deliberately binary:
         rule_id vocabularies are model-invented and not comparable, so
         matching individual flags would measure naming, not detection.
      4. **sentiment trend direction** — 1.0 when both sides' first-to-last
         sentiment movement falls in the same class (up / down / flat /
         unknown); see `_sentiment_direction`.

    Args:
        predictions: Model-produced analyses.
        ground_truth: Corresponding reference analyses (currently LLM-generated,
            not human-reviewed — see module docstring).

    Returns:
        A mapping from each `Language` present in the data to its accuracy.
        Never a single blended float across all categories.
    """
    per_language: dict[Language, list[float]] = defaultdict(list)

    for prediction, reference in _pair_by_call_id(predictions, ground_truth):
        summary_score = 1.0 if prediction.summary.strip() and reference.summary.strip() else 0.0
        commitment_score = _commitment_agreement(prediction, reference)
        compliance_score = (
            1.0 if bool(prediction.compliance_flags) == bool(reference.compliance_flags) else 0.0
        )
        sentiment_score = (
            1.0 if _sentiment_direction(prediction) == _sentiment_direction(reference) else 0.0
        )
        per_language[reference.language].append(
            _mean([summary_score, commitment_score, compliance_score, sentiment_score])
        )

    return {language: _mean(scores) for language, scores in per_language.items()}


def rubric_agreement_by_category(
    predictions: list[CallAnalysis], ground_truth: list[CallAnalysis]
) -> dict[Language, float]:
    """Compute rubric-score agreement with the reference labels, grouped by `Language`.

    Scoring rule: for each paired call, take every criterion scored on both
    sides, compute the mean absolute difference of the two 0.0-1.0 scores,
    and report `1 - MAE` so that higher is better and the number stays on the
    same 0.0-1.0 scale as the scores themselves. A call whose criteria do not
    overlap at all scores 0.0 rather than being dropped — a prediction that
    scored nothing comparable is a failure, not a missing observation. Each
    language's agreement is the mean over its calls.

    Args:
        predictions: Model-produced analyses.
        ground_truth: Corresponding reference analyses.

    Returns:
        A mapping from each `Language` present in the data to its agreement score.
    """
    per_language: dict[Language, list[float]] = defaultdict(list)

    for prediction, reference in _pair_by_call_id(predictions, ground_truth):
        predicted_scores = {score.criterion: score.score for score in prediction.rubric_scores}
        differences = [
            abs(predicted_scores[score.criterion] - score.score)
            for score in reference.rubric_scores
            if score.criterion in predicted_scores
        ]
        per_language[reference.language].append(1.0 - _mean(differences) if differences else 0.0)

    return {language: _mean(scores) for language, scores in per_language.items()}


def _evidence_quotes(analysis: CallAnalysis) -> list[Quote]:
    """Every quote a `CallAnalysis` stakes a claim on, in one flat list."""
    quotes: list[Quote] = [item.evidence for item in analysis.commitments]
    quotes.extend(flag.evidence for flag in analysis.compliance_flags)
    quotes.extend(score.evidence for score in analysis.rubric_scores)
    quotes.extend(point.quote for point in analysis.sentiment_trajectory.points)
    return quotes


def grounding_precision_by_category(
    predictions: list[CallAnalysis],
    *,
    transcript_dir: Path = DEFAULT_SYNTHETIC_DIR,
) -> dict[Language, float]:
    """Compute the fraction of claims whose evidence verifiably quotes the transcript, per category.

    This metric needs no reference labels at all — it checks the prediction
    against the source transcript directly, via the same
    `sawti.quotes` matcher used to author and locate quotes everywhere else.
    That makes it the one number here that is not an LLM-vs-LLM comparison.

    Scoring rule: pooled (micro-averaged) over every evidence quote of every
    call in a language — commitments, compliance flags, rubric scores, and
    sentiment points alike — so a call with many claims weighs more than a
    call with few. A quote counts as grounded only if its exact text appears
    in the transcript; the recorded offsets are not trusted either, since a
    model-supplied offset can be arithmetically valid and still point at the
    wrong span.

    A prediction whose transcript file is missing is skipped with no claims
    counted, so a missing file cannot silently look like perfect grounding.

    Args:
        predictions: Model-produced analyses to check.
        transcript_dir: Directory holding `<call_id>.txt` transcripts.

    Returns:
        A mapping from each `Language` present in the data to its grounding precision.
    """
    grounded: dict[Language, int] = defaultdict(int)
    total: dict[Language, int] = defaultdict(int)

    transcript_cache: dict[str, str | None] = {}

    for prediction in predictions:
        if prediction.call_id not in transcript_cache:
            path = transcript_dir / f"{prediction.call_id}.txt"
            transcript_cache[prediction.call_id] = (
                path.read_text(encoding="utf-8") if path.is_file() else None
            )
        transcript = transcript_cache[prediction.call_id]
        if transcript is None:
            continue

        for quote in _evidence_quotes(prediction):
            total[prediction.language] += 1
            if is_verbatim(transcript, quote.text):
                grounded[prediction.language] += 1

    return {
        language: (grounded[language] / count if count else 0.0) for language, count in total.items()
    }
