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

from sawti.agent.nodes.ground import is_grounded
from sawti.quotes import is_verbatim
from sawti.schemas import CallAnalysis, Claim, Language, Quote

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


def unsupported_claim_rate_by_category(
    predictions: list[CallAnalysis],
    *,
    transcript_dir: Path = DEFAULT_SYNTHETIC_DIR,
) -> dict[Language, float]:
    """Fraction of emitted claims whose evidence does not hold up, per category. Lower is better.

    This is the phase 2 headline metric, and the one number that says whether
    the grounding gate did anything. Like `grounding_precision_by_category` it
    needs no reference labels — it checks predictions against the transcripts
    themselves — so it is not an LLM-vs-LLM comparison.

    Scoring rule: pooled over every `Claim` in every call of a language —
    commitments, compliance flags, and rubric scores. Sentiment points are
    excluded because `SentimentPoint` is not a `Claim`; it is an interpretation
    of the call rather than an assertion about it, and the project rule this
    metric tracks is about claims.

    A claim counts as unsupported when its quote text does not appear verbatim
    in the transcript. This deliberately reuses `sawti.agent.nodes.ground`'s own
    predicate rather than restating it, so the metric measures exactly what the
    graph's gate enforces and the two can never drift apart.

    It differs from `grounding_precision_by_category` in *what it counts*, not in
    how it decides: that metric pools every evidence quote including sentiment
    points and reports the fraction that verify; this one pools `Claim`s only and
    reports the fraction that do not. On claim-only input the two are
    complements.

    What a drop in this number does and does not mean: it means fewer
    unsupported claims reach anything downstream. It does **not** by itself mean
    the model hallucinates less — a pipeline that rejects bad claims and one
    that never produces them both score 0.0 here. Read it alongside the
    grounding-gate counts recorded in `eval_results.md`.

    A prediction whose transcript file is missing is skipped with no claims
    counted, so a missing file cannot look like a perfect score.

    Args:
        predictions: Analyses as the pipeline emits them.
        transcript_dir: Directory holding `<call_id>.txt` transcripts.

    Returns:
        A mapping from each `Language` present in the data to its unsupported
        claim rate, in [0.0, 1.0]. Never a single blended float.
    """
    unsupported: dict[Language, int] = defaultdict(int)
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

        claims: list[Claim] = [*prediction.commitments, *prediction.compliance_flags]
        claims.extend(prediction.rubric_scores)
        for claim in claims:
            total[prediction.language] += 1
            if not is_grounded(transcript, claim):
                unsupported[prediction.language] += 1

    return {
        language: (unsupported[language] / count if count else 0.0)
        for language, count in total.items()
    }


# ---------------------------------------------------------------------------
# Phase 3: ASR metrics
# ---------------------------------------------------------------------------

#: Arabic diacritics (harakat, shadda, sukun) and the superscript alef.
_ARABIC_DIACRITICS = re.compile(r"[ً-ٰٟ]")
#: Tatweel / kashida, a purely decorative letter-stretching character.
_TATWEEL = re.compile(r"ـ")
#: Apostrophes are *deleted*, not spaced: replacing them would split "let's"
#: into two tokens, so an ASR "lets" would score two errors against a
#: one-token reference for what is at most one. Applies to the Latin half of
#: code-switched text, where contractions are common.
_APOSTROPHE = re.compile(
    r"(?<=[A-Za-z])[\u0027\u2018\u2019\u02BC](?=[A-Za-z])"
)
#: Punctuation to drop, Arabic and Latin, plus the common typographic quotes.
#: Replaced by a space, since these genuinely separate words.
_PUNCTUATION = re.compile(r"[.,!?;:\-–—_'\"“”‘’()\[\]{}/\\«»…،؛؟٪]")
_WHITESPACE = re.compile(r"\s+")

#: Orthographic folds applied before scoring Arabic WER.
_ARABIC_FOLDS = str.maketrans(
    {
        "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",  # alef variants
        "ى": "ي",  # alef maqsura -> ya
        "ة": "ه",  # ta marbuta -> ha
        "ؤ": "و", "ئ": "ي",  # hamza carriers
        "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
        "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    }
)


def normalize_for_wer(text: str) -> str:
    """Apply the project's orthographic normalization before WER scoring.

    Arabic has several ways to write the same word, and Whisper picks between
    them differently than the reference does. Scoring raw text charges the
    recognizer for orthography it never got wrong: an unnormalized Arabic WER
    is inflated by hamza seating, ta-marbuta and absent diacritics, none of
    which change what was said.

    The rule, applied in order:

    1. Strip diacritics (harakat, shadda, sukun, superscript alef) and tatweel.
    2. Fold alef variants (أ إ آ ٱ) to bare alef.
    3. Fold alef maqsura (ى) to ya, ta marbuta (ة) to ha.
    4. Fold hamza carriers (ؤ → و, ئ → ي).
    5. Map Arabic-Indic digits to ASCII.
    6. Lowercase, which matters for the Latin half of code-switched text.
    7. Strip punctuation and collapse whitespace.

    This is the standard convention in Arabic ASR evaluation, so the numbers it
    produces are comparable to published figures. The unnormalized WER is
    reported alongside it in `eval_results.md` so nothing is hidden — see
    `docs/09-DECISIONS.md`.

    Args:
        text: Raw reference or hypothesis text.

    Returns:
        The normalized text, space-separated.
    """
    text = _ARABIC_DIACRITICS.sub("", text)
    text = _TATWEEL.sub("", text)
    text = text.translate(_ARABIC_FOLDS)
    text = text.lower()
    text = _APOSTROPHE.sub("", text)
    text = _PUNCTUATION.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Levenshtein distance between two token sequences.

    Iterative two-row implementation: the corpus is small, but a full matrix
    over long calls is needless memory.
    """
    if not reference:
        return len(hypothesis)
    previous = list(range(len(reference) + 1))
    for hypothesis_index, hypothesis_token in enumerate(hypothesis, start=1):
        current = [hypothesis_index]
        for reference_index, reference_token in enumerate(reference, start=1):
            current.append(
                previous[reference_index - 1]
                if reference_token == hypothesis_token
                else 1 + min(previous[reference_index - 1], previous[reference_index], current[-1])
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str, *, normalize: bool = True) -> float:
    """Word error rate of `hypothesis` against `reference`.

    Args:
        reference: The true transcript text.
        hypothesis: The ASR-produced text.
        normalize: Apply `normalize_for_wer` to both sides first. `False`
            gives the raw, unnormalized rate.

    Returns:
        (substitutions + insertions + deletions) / reference word count.
        Can exceed 1.0 when the hypothesis is longer than the reference —
        deliberately not clipped, since a runaway hallucination should show
        as worse than total loss rather than saturating at 1.0.
        An empty reference scores 0.0 against an empty hypothesis, else 1.0.
    """
    if normalize:
        reference, hypothesis = normalize_for_wer(reference), normalize_for_wer(hypothesis)

    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    if not reference_tokens:
        return 0.0 if not hypothesis_tokens else 1.0

    return _edit_distance(reference_tokens, hypothesis_tokens) / len(reference_tokens)


def wer_by_category(
    pairs: list[tuple[Language, str, str]], *, normalize: bool = True
) -> dict[Language, float]:
    """Compute word error rate, grouped by `Language`.

    Aggregated per category the way WER is conventionally aggregated — total
    errors over total reference words, not the mean of per-call rates — so one
    short call cannot swing the category.

    Args:
        pairs: `(language, reference_text, hypothesis_text)` per call.
        normalize: Apply `normalize_for_wer` before scoring.

    Returns:
        A mapping from each `Language` present in the data to its WER. Never a
        single blended float: `ar`, `en` and `mixed` ASR difficulty differ far
        too much for an average across them to mean anything.
    """
    errors: dict[Language, int] = defaultdict(int)
    reference_words: dict[Language, int] = defaultdict(int)

    for language, reference, hypothesis in pairs:
        if normalize:
            reference, hypothesis = normalize_for_wer(reference), normalize_for_wer(hypothesis)
        reference_tokens, hypothesis_tokens = reference.split(), hypothesis.split()
        errors[language] += _edit_distance(reference_tokens, hypothesis_tokens)
        reference_words[language] += len(reference_tokens)

    return {
        language: (errors[language] / count if count else 0.0)
        for language, count in reference_words.items()
    }


#: Resolution at which the diarization timelines are compared, in seconds.
_DIARIZATION_FRAME_SEC = 0.01


def _speaker_at(segments: list[tuple[float, float, str]], time_sec: float) -> str | None:
    """Speaker covering `time_sec`, or `None` where no segment covers it."""
    for start, end, speaker in segments:
        if start <= time_sec < end:
            return speaker
    return None


def speaker_attribution_accuracy(
    reference: list[tuple[float, float, str]],
    hypothesis: list[tuple[float, float, str]],
    *,
    frame_sec: float = _DIARIZATION_FRAME_SEC,
) -> float:
    """Fraction of reference speech time attributed to the right speaker.

    Compares the two timelines frame by frame over the reference's speech
    regions. Frames where the reference has no speaker — the silence between
    turns — are excluded: crediting a diarizer for correctly labelling silence
    would inflate the score toward whatever fraction of the call is quiet.

    This is deliberately not DER. DER charges false alarm, missed detection and
    confusion separately against total speech time; this is the simpler
    question "of the speech we know the speaker of, how much did we get right",
    which is what a sanity check on a two-speaker synthetic call needs. Both
    timelines must already use the same label vocabulary — see
    `sawti.asr.diarization.assign_roles_by_first_speaker`.

    Args:
        reference: True `(start_sec, end_sec, speaker)` segments.
        hypothesis: Predicted `(start_sec, end_sec, speaker)` segments.
        frame_sec: Comparison resolution.

    Returns:
        Accuracy in [0.0, 1.0]. Returns 0.0 when the reference has no speech.
    """
    if not reference:
        return 0.0

    total = 0
    correct = 0
    end_sec = max(end for _, end, _ in reference)
    frames = int(end_sec / frame_sec)

    for frame in range(frames):
        time_sec = frame * frame_sec
        true_speaker = _speaker_at(reference, time_sec)
        if true_speaker is None:
            continue
        total += 1
        if _speaker_at(hypothesis, time_sec) == true_speaker:
            correct += 1

    return correct / total if total else 0.0


def speaker_attribution_accuracy_by_category(
    entries: list[tuple[Language, list[tuple[float, float, str]], list[tuple[float, float, str]]]],
) -> dict[Language, float]:
    """Compute speaker attribution accuracy, grouped by `Language`.

    Args:
        entries: `(language, reference_segments, hypothesis_segments)` per call.

    Returns:
        A mapping from each `Language` present to its mean attribution
        accuracy. Never a single blended float.
    """
    per_language: dict[Language, list[float]] = defaultdict(list)
    for language, reference, hypothesis in entries:
        per_language[language].append(speaker_attribution_accuracy(reference, hypothesis))

    return {language: _mean(scores) for language, scores in per_language.items()}
