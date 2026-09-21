"""The phase 1 baseline: plain-prompt extraction, with no coaching of any kind.

Phase 1: baseline extraction path.

This module exists to be *separate* from
`sawti.data.ground_truth.generate_reference_labels`, and the separation is
the point. The reference labels this baseline is scored against are
themselves LLM-generated (see `docs/09-DECISIONS.md`), so the single
methodological safeguard available is that the two paths do not share a
prompt: the reference path gets a long, careful, domain-specific system
prompt with explicit instructions about verbatim evidence and rubric
calibration, while this path gets a short generic instruction and nothing
else.

Rules for this module — do not "improve" it:
  * The prompt stays minimal and generic. No worked examples, no rubric
    guidance, no compliance taxonomy, no verbatim-quote coaching.
  * No retry on bad content. Whatever the model produces is the baseline.
    The only retries here are on transport/provider failures (rate limits,
    5xx), which are not the model's answer.
  * Nothing corrects the model's output. Quotes that turn out not to be in
    the transcript are left exactly as returned — detecting them is
    `sawti.eval.metrics.grounding_precision_by_category`'s job, and silently
    fixing them here would erase the thing being measured.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from pydantic import BaseModel, Field

from sawti.config import get_settings
from sawti.data.ground_truth import RUBRIC_CRITERIA, infer_call_id_and_language
from sawti.llm.provider import get_llm_provider, run_with_timeout
from sawti.quotes import UNKNOWN_SPEAKER, find_quote_matches
from sawti.schemas import (
    CallAnalysis,
    Commitment,
    ComplianceFlag,
    Quote,
    RubricScore,
    SentimentPoint,
    SentimentTrajectory,
    Severity,
)

logger = logging.getLogger(__name__)


class PlainCommitment(BaseModel):
    """A commitment, as returned by the plain prompt."""

    quote: str
    promised_by: str
    description: str


class PlainComplianceFlag(BaseModel):
    """A compliance flag, as returned by the plain prompt."""

    quote: str
    rule_id: str
    severity: Severity
    description: str


class PlainRubricScore(BaseModel):
    """A rubric score, as returned by the plain prompt."""

    quote: str
    criterion: str
    score: float = Field(..., ge=0.0, le=1.0)
    justification: str


class PlainSentimentPoint(BaseModel):
    """A sentiment sample, as returned by the plain prompt."""

    quote: str
    speaker: str
    score: float = Field(..., ge=-1.0, le=1.0)
    timestamp_sec: float = Field(..., ge=0.0)


class PlainAnalysis(BaseModel):
    """The plain prompt's raw response shape.

    Like the reference path, this asks for quote text rather than character
    offsets — but for a different reason. There, it is quality control. Here,
    it is simply that a model-invented `start_char` almost never satisfies
    `Quote`'s offset invariants, so asking for offsets would fail the run at
    parse time and measure nothing at all. The quote *text* is untouched
    model output either way, which is what grounding precision scores.
    """

    summary: str
    commitments: list[PlainCommitment]
    compliance_flags: list[PlainComplianceFlag]
    rubric_scores: list[PlainRubricScore]
    sentiment_points: list[PlainSentimentPoint]
    confidence: float = Field(..., ge=0.0, le=1.0)


_PLAIN_SYSTEM_PROMPT = "You analyze contact center call transcripts."

_PLAIN_PROMPT_TEMPLATE = (
    "Extract call_id, language, summary, commitments, compliance_flags, "
    "sentiment_trajectory, and score these 7 criteria: {criteria}.\n"
    "For each commitment, compliance flag, rubric score, and sentiment point, "
    "include the quote from the transcript it is based on.\n\n"
    "{transcript}"
)


def _quote_as_returned(transcript: str, quote_text: str) -> Quote:
    """Wrap the model's quote text in a `Quote` without correcting it.

    If the text is in the transcript, the real offsets and speaker are
    recorded. If it is not — the model paraphrased, translated, or invented
    it — the text is preserved exactly as returned and given placeholder
    offsets (`0..len(text)`) purely so the record can be constructed;
    `Quote` forbids `end_char <= start_char` and requires the span length to
    match the text, so some value is structurally required here.

    Those placeholders are not a claim that the quote is grounded, and
    nothing downstream treats them as one: grounding precision re-checks
    every quote's text against the transcript and ignores the offsets
    entirely. An empty quote is given a single-space text, the minimum
    `Quote` accepts, and will score as ungrounded.
    """
    if not quote_text:
        return Quote(text=" ", speaker=UNKNOWN_SPEAKER, start_char=0, end_char=1)
    matches = find_quote_matches(transcript, quote_text)
    if matches:
        match = matches[0]
        return Quote(
            text=quote_text,
            speaker=match.speaker or UNKNOWN_SPEAKER,
            start_char=match.start_char,
            end_char=match.end_char,
        )
    return Quote(text=quote_text, speaker=UNKNOWN_SPEAKER, start_char=0, end_char=len(quote_text))


def _to_call_analysis(raw: PlainAnalysis, transcript: str, transcript_path: Path) -> CallAnalysis:
    """Assemble the model's raw answer into a `CallAnalysis`, content untouched.

    Raises:
        ValidationError: If the model's answer does not satisfy
            `CallAnalysis` — e.g. sentiment points that are not in
            chronological order. That is a real baseline failure and is
            reported as one, not repaired.
    """
    call_id, language = infer_call_id_and_language(transcript_path)
    threshold = get_settings().confidence_threshold

    return CallAnalysis(
        call_id=call_id,
        language=language,
        summary=raw.summary,
        commitments=[
            Commitment(
                evidence=_quote_as_returned(transcript, item.quote),
                promised_by=item.promised_by or UNKNOWN_SPEAKER,
                description=item.description,
            )
            for item in raw.commitments
        ],
        compliance_flags=[
            ComplianceFlag(
                evidence=_quote_as_returned(transcript, flag.quote),
                rule_id=flag.rule_id,
                severity=flag.severity,
                description=flag.description,
            )
            for flag in raw.compliance_flags
        ],
        rubric_scores=[
            RubricScore(
                evidence=_quote_as_returned(transcript, score.quote),
                criterion=score.criterion,
                score=score.score,
                justification=score.justification,
            )
            for score in raw.rubric_scores
        ],
        sentiment_trajectory=SentimentTrajectory(
            points=[
                SentimentPoint(
                    quote=_quote_as_returned(transcript, point.quote),
                    speaker=point.speaker or UNKNOWN_SPEAKER,
                    score=point.score,
                    timestamp_sec=point.timestamp_sec,
                )
                for point in raw.sentiment_points
            ]
        ),
        confidence=raw.confidence,
        requires_human_review=raw.confidence < threshold,
    )


def run_plain_extraction(
    transcript_path: Path,
    *,
    max_retries: int = 4,
    initial_delay: float = 4.0,
) -> CallAnalysis:
    """Run the baseline plain-prompt extraction over one transcript.

    The prompt is a single generic instruction naming the fields and the 7
    criteria, with no further guidance. Whatever comes back is the baseline.

    `max_retries` covers provider/transport failures only — free-tier APIs
    rate-limit aggressively and a 429 is not the model's answer. A response
    that parses but is wrong, ungrounded, or missing criteria is returned or
    raised as-is; it is never re-prompted.

    Args:
        transcript_path: Path to a `call_<idx>_<lang>.txt` transcript.
        max_retries: Retries on provider errors, with exponential backoff.
        initial_delay: Seconds before the first retry; doubles each time.

    Returns:
        The model's analysis as a validated `CallAnalysis`.

    Raises:
        ValidationError: If the model's answer does not satisfy `CallAnalysis`.
        Exception: The provider's own error, if every attempt failed.
    """
    transcript = transcript_path.read_text(encoding="utf-8")
    prompt = _PLAIN_PROMPT_TEMPLATE.format(
        criteria=", ".join(RUBRIC_CRITERIA),
        transcript=transcript,
    )
    provider = get_llm_provider()

    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            raw = run_with_timeout(
                provider.structured_complete(
                    prompt,
                    response_model=PlainAnalysis,
                    system=_PLAIN_SYSTEM_PROMPT,
                )
            )
        except Exception as exc:
            if attempt == max_retries:
                raise
            logger.warning(
                "plain extraction provider error for %s (attempt %d/%d): %r; retrying in %.1fs",
                transcript_path.name,
                attempt + 1,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)
            delay *= 2
            continue
        # Parsed. Whatever it says is the baseline — no content retry.
        return _to_call_analysis(raw, transcript, transcript_path)

    raise AssertionError("unreachable")  # pragma: no cover
