"""Load, validate, and generate the reference labels used for evaluation.

Phase 1: ground truth loading, required by eval.runner from phase 1 onward.

Reference records live as one `*.json` file per call in `data/ground_truth/`,
each shaped like `sawti.schemas.CallAnalysis`.

IMPORTANT — how these labels are produced, and what that costs:

The original plan was a human review pass: blank authoring templates from
`scripts/generate_ground_truth_templates.py`, filled in by hand with
`scripts/find_quote.py` for offsets. That did not happen. The labels
currently in `data/ground_truth/` are produced by `generate_reference_labels`
below — an LLM, prompted carefully, with its quotes verified programmatically
against the transcript but its *content* verified by nobody.

They are therefore "reference labels", not ground truth. Any metric computed
against them measures agreement between two LLM-driven processes, not
accuracy against human judgment. Every generated file carries a
`_provenance` block saying so, and `docs/09-DECISIONS.md` records the
decision. Do not quietly re-describe these as human-reviewed.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from sawti.config import get_settings
from sawti.llm.provider import get_llm_provider, run_with_timeout
from sawti.quotes import UNKNOWN_SPEAKER, find_quote_matches
from sawti.schemas import (
    CallAnalysis,
    Commitment,
    ComplianceFlag,
    Language,
    Quote,
    RubricScore,
    SentimentPoint,
    SentimentTrajectory,
    Severity,
)

logger = logging.getLogger(__name__)

# The 7 QA rubric criteria every `CallAnalysis.rubric_scores` must cover.
# Fixed and agreed with the user — do not add, remove, or rename any of these.
RUBRIC_CRITERIA: tuple[str, ...] = (
    "professionalism",
    "empathy",
    "resolution_effectiveness",
    "communication_clarity",
    "call_closure",
    "accuracy_of_information",
    "procedure_adherence",
)

# Matches how `sawti.data.generate_calls.generate_batch` names synthetic
# transcripts (call_<idx>_<lang>.txt) — ground-truth call_ids mirror that
# stem, so the language category can be cross-checked from the id alone.
_CALL_ID_LANGUAGE_RE = re.compile(r"_(ar|en|mixed)$")

# Full synthetic-transcript filename: call_<idx>_<lang>.txt. The call_id is
# the filename stem. Canonical here; `scripts/generate_ground_truth_templates.py`
# imports `infer_call_id_and_language` from this module rather than redefining it.
_FILENAME_RE = re.compile(r"^(?P<call_id>call_\d+_(?P<lang>ar|en|mixed))\.txt$")


def _language_from_call_id(call_id: str) -> Language | None:
    """Infer the `Language` encoded in a call_id's `_ar`/`_en`/`_mixed` suffix.

    Returns None if `call_id` does not end with a recognized language suffix
    (e.g. a hand-picked call_id that doesn't follow the synthetic naming
    convention) — such records are skipped by the cross-check, not rejected.
    """
    match = _CALL_ID_LANGUAGE_RE.search(call_id)
    return Language(match.group(1)) if match else None


def infer_call_id_and_language(transcript_path: Path) -> tuple[str, Language]:
    """Infer (call_id, language) from a synthetic transcript's filename.

    Args:
        transcript_path: Path to a `call_<idx>_<ar|en|mixed>.txt` transcript.

    Returns:
        The filename stem as `call_id`, and the `Language` encoded in its
        `_ar`/`_en`/`_mixed` suffix.

    Raises:
        ValueError: If the filename doesn't match the expected pattern.
    """
    match = _FILENAME_RE.match(transcript_path.name)
    if not match:
        raise ValueError(
            f"{transcript_path.name!r} does not match expected pattern 'call_<idx>_<ar|en|mixed>.txt'"
        )
    return match.group("call_id"), Language(match.group("lang"))


def load_ground_truth(path: Path) -> list[CallAnalysis]:
    """Load and validate every `*.json` ground-truth record in `path`.

    Args:
        path: Directory containing ground-truth `*.json` files.

    Returns:
        Validated `CallAnalysis` instances, one per file, sorted by filename.

    Raises:
        ValueError: If any file is not valid JSON, or fails `CallAnalysis`
            validation (e.g. an authoring template that hasn't been filled
            in yet) — the error names the offending file and the cause.
    """
    records: list[CallAnalysis] = []
    for file_path in sorted(path.glob("*.json")):
        raw = file_path.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{file_path}: not valid JSON ({exc})") from exc
        try:
            records.append(CallAnalysis.model_validate(data))
        except ValidationError as exc:
            raise ValueError(f"{file_path}: failed CallAnalysis validation:\n{exc}") from exc
    return records


def validate_ground_truth(records: list[CallAnalysis]) -> None:
    """Validate a set of ground-truth records for internal consistency.

    Checks, across all records:
      * No two records share the same `call_id`.
      * Every record has a `RubricScore` for each of the 7 `RUBRIC_CRITERIA`.
      * Every record's `language` agrees with the language encoded in its own
        `call_id` (the `_ar`/`_en`/`_mixed` filename suffix), where present.

    Args:
        records: Ground-truth `CallAnalysis` records to validate.

    Raises:
        ValueError: Listing every problem found, not just the first.
    """
    problems: list[str] = []

    call_id_counts: dict[str, int] = {}
    for record in records:
        call_id_counts[record.call_id] = call_id_counts.get(record.call_id, 0) + 1
    for call_id, count in sorted(call_id_counts.items()):
        if count > 1:
            problems.append(f"duplicate call_id {call_id!r} appears {count} times")

    for record in records:
        present_criteria = {score.criterion for score in record.rubric_scores}
        missing = [c for c in RUBRIC_CRITERIA if c not in present_criteria]
        if missing:
            problems.append(f"call_id {record.call_id!r} is missing rubric criteria: {missing}")

        expected_language = _language_from_call_id(record.call_id)
        if expected_language is not None and record.language != expected_language:
            problems.append(
                f"call_id {record.call_id!r} encodes language {expected_language.value!r} "
                f"but record.language is {record.language.value!r}"
            )

    if problems:
        raise ValueError("ground truth validation failed:\n" + "\n".join(f"  - {p}" for p in problems))


# --------------------------------------------------------------------------
# Reference-label generation (LLM, not human review — see module docstring)
# --------------------------------------------------------------------------
#
# The model is asked for evidence as verbatim quote TEXT ONLY. It is never
# asked for, and never trusted with, character offsets: those are computed
# here from the transcript via `sawti.quotes.find_quote_matches`. A quote the
# model returns that does not appear verbatim in the transcript is a
# retryable failure, not something to paper over.


class DraftCommitment(BaseModel):
    """A commitment as the model reports it: evidence is quote text, not a `Quote`."""

    evidence_quote: str = Field(
        ..., description="Verbatim text copied exactly from the transcript. No offsets."
    )
    promised_by: str = Field(..., description="Speaker who made the promise: 'Agent' or 'Customer'.")
    description: str = Field(..., description="What was promised, in normalized form.")
    deadline: str | None = Field(
        default=None,
        description=(
            "ISO 8601 datetime if the call stated an explicit absolute date/time, "
            "otherwise null. Never guess a date from a relative phrase."
        ),
    )


class DraftComplianceFlag(BaseModel):
    """A compliance violation as the model reports it: evidence is quote text."""

    evidence_quote: str = Field(
        ..., description="Verbatim text copied exactly from the transcript. No offsets."
    )
    rule_id: str = Field(..., description="Short stable identifier for the violated rule.")
    severity: Severity
    description: str = Field(..., description="What was violated and why it matters.")


class DraftRubricScore(BaseModel):
    """One rubric criterion score as the model reports it: evidence is quote text."""

    evidence_quote: str = Field(
        ..., description="Verbatim text copied exactly from the transcript. No offsets."
    )
    criterion: str = Field(..., description="Exactly one of the requested criterion names.")
    score: float = Field(..., ge=0.0, le=1.0)
    justification: str = Field(..., description="Why this score, referring to what happened in the call.")


class DraftSentimentPoint(BaseModel):
    """One sentiment sample as the model reports it: the anchor is quote text."""

    evidence_quote: str = Field(
        ..., description="Verbatim text copied exactly from the transcript. No offsets."
    )
    speaker: str = Field(..., description="'Agent' or 'Customer'.")
    score: float = Field(..., ge=-1.0, le=1.0, description="Polarity, -1 negative to 1 positive.")
    timestamp_sec: float = Field(
        ..., ge=0.0, description="Approximate offset in seconds from call start, assuming ~8s per turn."
    )


class DraftAnalysis(BaseModel):
    """A full call analysis as the model reports it, before quotes are located.

    Deliberately omits `call_id` and `language`: both are derived from the
    transcript filename, so the model is never given the chance to disagree
    with the dataset about which call it just read.

    Every field is required, including the lists. A field with a default is
    optional in the generated JSON Schema, and Gemini simply omits optional
    fields — `rubric_scores` came back missing on every call until it was
    made required. An empty list must be emitted explicitly.
    """

    summary: str = Field(..., description="A few sentences: what the customer wanted and what happened.")
    commitments: list[DraftCommitment] = Field(
        ..., description="Every promise made in the call. Emit an empty list if there are none."
    )
    compliance_flags: list[DraftComplianceFlag] = Field(
        ..., description="Every compliance violation. Emit an empty list if there are none."
    )
    rubric_scores: list[DraftRubricScore] = Field(
        ..., description="One entry for EVERY requested criterion. Never omit this field."
    )
    sentiment_points: list[DraftSentimentPoint] = Field(
        ..., description="3-6 samples of the customer's sentiment, in chronological order."
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Your confidence in this whole analysis.")


_REFERENCE_SYSTEM_PROMPT = (
    "You are a senior QA analyst at a bilingual (Arabic/English) contact center in "
    "Jordan. You are producing the reference analysis for one recorded call — the "
    "record other people's work will be measured against. Be thorough, specific, "
    "and conservative: report what the transcript actually shows, never what a "
    "typical call of this kind usually contains.\n"
    "\n"
    "THE ONE RULE THAT MATTERS MOST — EVIDENCE MUST BE VERBATIM:\n"
    "Every claim and every score you make must be supported by an `evidence_quote` "
    "that is an EXACT, CHARACTER-FOR-CHARACTER COPY of a span of text from the "
    "transcript below. Copy and paste it; do not retype it, translate it, "
    "summarize it, normalize its spelling or punctuation, fix its typos, or "
    "change its diacritics. For Arabic, reproduce the exact Arabic script as "
    "written. Do NOT include the 'Agent: ' or 'Customer: ' line prefix in the "
    "quote. Keep each quote short enough to be a single clean span — one clause "
    "or one sentence — but long enough to be unambiguous.\n"
    "Do NOT report character offsets or positions of any kind. Those are computed "
    "separately from your quote text; your only job is to copy the text exactly.\n"
    "\n"
    "WHAT TO ANALYZE:\n"
    "1. summary — what the customer wanted, what the agent did, how it ended.\n"
    "2. commitments — every promise made by either party: a callback, a refund, a "
    "cancellation, an escalation, a fix within some time. One entry per promise. "
    "If the call contains no promises, return an empty list; do not invent one.\n"
    "3. compliance_flags — every point where the agent broke a contact center "
    "rule: failing to identify themselves or the company, failing to verify the "
    "customer's identity before discussing account details, being rude or "
    "dismissive, interrupting, disclosing information improperly, promising "
    "something they cannot deliver, ending the call without confirming "
    "resolution, ignoring an explicit request. Choose a short snake_case rule_id "
    "and a severity. If the agent did nothing wrong, return an empty list.\n"
    "4. sentiment_points — 3 to 6 samples tracking how the CUSTOMER's sentiment "
    "moves across the call, in the order they occur, with increasing "
    "timestamp_sec (estimate ~8 seconds per conversational turn).\n"
    "5. rubric_scores — score EVERY ONE of the criteria listed in the user "
    "message, using exactly the criterion names given, one entry each, no extras "
    "and none missing. Use the full 0.0-1.0 range; a merely adequate call is "
    "around 0.6-0.7, not 0.9.\n"
    "\n"
    "Finally, report your own `confidence` in the analysis as a whole (0.0-1.0). "
    "Lower it when the call is ambiguous, very short, or hard to read."
)


def _build_reference_prompt(
    transcript: str,
    criteria: Sequence[str],
    *,
    feedback: str | None = None,
) -> str:
    """Build the user-turn prompt for one reference-label generation attempt.

    Args:
        transcript: Full transcript text to analyze.
        criteria: Rubric criterion names the model must score, all of them.
        feedback: On a retry, what went wrong last time — currently always a
            verbatim-quote failure. Placed at the top so it is read first.

    Returns:
        The prompt string.
    """
    parts: list[str] = []
    if feedback:
        parts.append(
            "YOUR PREVIOUS ANSWER WAS REJECTED. Read this before answering again:\n"
            f"{feedback}\n"
            "Every evidence_quote must be copied character-for-character from the "
            "transcript below. Pick shorter, simpler spans this time — a few words "
            "you can copy exactly — rather than long ones you might alter.\n"
        )
    parts.append(
        "Analyze the following contact center call transcript and return the "
        "structured reference analysis.\n\n"
        f"Score exactly these {len(criteria)} rubric criteria, using these exact "
        f"names: {', '.join(criteria)}.\n\n"
        "--- TRANSCRIPT BEGINS ---\n"
        f"{transcript}\n"
        "--- TRANSCRIPT ENDS ---"
    )
    return "\n".join(parts)


class QuoteNotVerbatimError(ValueError):
    """Raised when a model-proposed quote does not appear verbatim in the transcript.

    Carries the offending quote texts so the retry prompt can name them.
    """

    def __init__(self: QuoteNotVerbatimError, quotes: Sequence[str]) -> None:
        self.quotes = list(quotes)
        preview = "; ".join(repr(q[:60]) for q in self.quotes[:5])
        super().__init__(f"{len(self.quotes)} quote(s) not found verbatim in the transcript: {preview}")


def locate_quote(transcript: str, quote_text: str) -> Quote:
    """Turn model-proposed quote text into a real `Quote` with computed offsets.

    The offsets come from the transcript, never from the model. When the quote
    occurs more than once the first occurrence is used — the alternative is
    failing an otherwise-good label over an ambiguity that does not change
    what the quote says.

    Args:
        transcript: Full transcript text the quote must come from.
        quote_text: Verbatim text the model claims to have copied.

    Returns:
        A validated `Quote` with real `start_char`/`end_char` and the speaker
        parsed from the containing line.

    Raises:
        QuoteNotVerbatimError: If `quote_text` is empty or does not appear
            verbatim in `transcript`.
    """
    if not quote_text.strip():
        raise QuoteNotVerbatimError([quote_text])
    matches = find_quote_matches(transcript, quote_text)
    if not matches:
        raise QuoteNotVerbatimError([quote_text])
    match = matches[0]
    return Quote(
        text=quote_text,
        speaker=match.speaker or UNKNOWN_SPEAKER,
        start_char=match.start_char,
        end_char=match.end_char,
    )


def _parse_deadline(raw: str | None) -> datetime | None:
    """Parse a model-supplied ISO 8601 deadline, tolerating junk by returning None.

    A malformed deadline is not worth failing (and re-billing) a whole call
    over — `Commitment.deadline` is optional, and the commitment's `evidence`
    and `description` carry the substance.
    """
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        logger.debug("discarding unparseable deadline %r", raw)
        return None


def _assemble_analysis(
    draft: DraftAnalysis,
    transcript: str,
    *,
    call_id: str,
    language: Language,
    criteria: Sequence[str],
) -> CallAnalysis:
    """Locate every draft quote and assemble a validated `CallAnalysis`.

    Args:
        draft: The model's analysis, with evidence as quote text.
        transcript: Full transcript text, the sole source of quote offsets.
        call_id: Call id derived from the transcript filename.
        language: Language category derived from the transcript filename.
        criteria: Rubric criteria that must all be present exactly once.

    Returns:
        A validated `CallAnalysis`.

    Raises:
        QuoteNotVerbatimError: If any evidence quote is not verbatim. Every
            bad quote is collected first, so one retry can fix all of them.
        ValueError: If the rubric criteria do not match `criteria` exactly.
        ValidationError: If the assembled record fails `CallAnalysis` validation.
    """
    unlocatable: list[str] = []

    def _try_locate(quote_text: str) -> Quote | None:
        try:
            return locate_quote(transcript, quote_text)
        except QuoteNotVerbatimError:
            unlocatable.append(quote_text)
            return None

    commitments: list[Commitment] = []
    for item in draft.commitments:
        evidence = _try_locate(item.evidence_quote)
        if evidence is not None:
            commitments.append(
                Commitment(
                    evidence=evidence,
                    promised_by=item.promised_by or evidence.speaker,
                    description=item.description,
                    deadline=_parse_deadline(item.deadline),
                )
            )

    compliance_flags: list[ComplianceFlag] = []
    for flag in draft.compliance_flags:
        evidence = _try_locate(flag.evidence_quote)
        if evidence is not None:
            compliance_flags.append(
                ComplianceFlag(
                    evidence=evidence,
                    rule_id=flag.rule_id,
                    severity=flag.severity,
                    description=flag.description,
                )
            )

    rubric_scores: list[RubricScore] = []
    for score in draft.rubric_scores:
        evidence = _try_locate(score.evidence_quote)
        if evidence is not None:
            rubric_scores.append(
                RubricScore(
                    evidence=evidence,
                    criterion=score.criterion,
                    score=score.score,
                    justification=score.justification,
                )
            )

    points: list[SentimentPoint] = []
    for point in draft.sentiment_points:
        quote = _try_locate(point.evidence_quote)
        if quote is not None:
            points.append(
                SentimentPoint(
                    quote=quote,
                    speaker=point.speaker or quote.speaker,
                    score=point.score,
                    timestamp_sec=point.timestamp_sec,
                )
            )
    # `SentimentTrajectory` requires chronological order. Sorting reorders the
    # model's points but changes none of them; the alternative is discarding a
    # whole analysis over the order a list happened to come back in.
    points.sort(key=lambda p: p.timestamp_sec)

    if unlocatable:
        raise QuoteNotVerbatimError(unlocatable)

    # Enforced here rather than left to `validate_ground_truth`: a record
    # missing criteria is unusable for rubric agreement, and at this point a
    # retry is still cheap.
    present = [score.criterion for score in rubric_scores]
    missing = [c for c in criteria if c not in present]
    extra = [c for c in present if c not in criteria]
    if missing or extra:
        raise ValueError(f"rubric criteria mismatch: missing={missing}, unexpected={extra}")

    threshold = get_settings().confidence_threshold
    return CallAnalysis(
        call_id=call_id,
        language=language,
        summary=draft.summary,
        commitments=commitments,
        compliance_flags=compliance_flags,
        rubric_scores=rubric_scores,
        sentiment_trajectory=SentimentTrajectory(points=points),
        confidence=draft.confidence,
        requires_human_review=draft.confidence < threshold,
    )


def generate_reference_labels(
    transcript_path: Path,
    *,
    criteria: Sequence[str] = RUBRIC_CRITERIA,
    max_retries: int = 4,
    initial_delay: float = 4.0,
) -> CallAnalysis:
    """Generate the LLM reference analysis for one transcript.

    This replaces the human review pass that was originally planned. The
    result is NOT human-verified ground truth — see the module docstring and
    `docs/09-DECISIONS.md`. What *is* verified is grounding: every quote in
    the returned record provably appears in the transcript at the offsets
    recorded, because those offsets are computed here rather than reported by
    the model.

    Retries with exponential backoff (the same pattern as
    `sawti.data.generate_calls._generate_with_retry`) on:
      * provider errors — free-tier APIs rate-limit aggressively;
      * a quote the model did not copy verbatim, which is re-prompted with
        an explicit explanation of what it got wrong;
      * a rubric criteria mismatch, or a record that fails `CallAnalysis`
        validation.

    Args:
        transcript_path: Path to a `call_<idx>_<ar|en|mixed>.txt` transcript.
        criteria: Rubric criteria to score; defaults to all 7 `RUBRIC_CRITERIA`.
        max_retries: Retries after the first attempt before giving up.
        initial_delay: Seconds to wait before the first retry; doubles each time.

    Returns:
        A validated `CallAnalysis` whose every quote is verbatim and located.

    Raises:
        ValueError: If `transcript_path` is empty or misnamed, or if the model
            never produced a fully locatable, valid analysis within
            `max_retries`.
    """
    call_id, language = infer_call_id_and_language(transcript_path)
    transcript = transcript_path.read_text(encoding="utf-8")
    if not transcript.strip():
        raise ValueError(f"{transcript_path} is empty — nothing to analyze")

    provider = get_llm_provider()
    feedback: str | None = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        is_last_attempt = attempt == max_retries
        prompt = _build_reference_prompt(transcript, criteria, feedback=feedback)
        try:
            draft = run_with_timeout(
                provider.structured_complete(
                    prompt,
                    response_model=DraftAnalysis,
                    system=_REFERENCE_SYSTEM_PROMPT,
                    temperature=0.2,
                )
            )
            return _assemble_analysis(
                draft,
                transcript,
                call_id=call_id,
                language=language,
                criteria=criteria,
            )
        except QuoteNotVerbatimError as exc:
            reason = str(exc)
            feedback = (
                "These evidence_quote values do NOT appear anywhere in the "
                "transcript — you paraphrased, translated, or retyped them instead "
                "of copying them:\n"
                + "\n".join(f"  - {quote!r}" for quote in exc.quotes)
            )
        except (ValueError, ValidationError) as exc:
            reason = str(exc)
            feedback = f"Your previous answer was structurally invalid: {reason}"
        except Exception as exc:  # provider/transport failure — not the model's fault
            reason = f"provider error: {exc!r}"
            feedback = None

        if is_last_attempt:
            raise ValueError(
                f"generate_reference_labels failed for {transcript_path.name} after "
                f"{max_retries} retries: {reason}"
            )
        logger.warning(
            "reference label attempt %d/%d failed for %s (%s); retrying in %.1fs",
            attempt + 1,
            max_retries,
            transcript_path.name,
            reason,
            delay,
        )
        time.sleep(delay)
        delay *= 2

    raise AssertionError("unreachable")  # pragma: no cover
