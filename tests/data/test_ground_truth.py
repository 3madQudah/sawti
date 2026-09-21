"""Tests for `sawti.data.ground_truth`.

Phase 1: mirrors `src/sawti/data/ground_truth.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sawti.config import get_settings
from sawti.data.ground_truth import (
    RUBRIC_CRITERIA,
    DraftAnalysis,
    DraftCommitment,
    DraftRubricScore,
    DraftSentimentPoint,
    QuoteNotVerbatimError,
    generate_reference_labels,
    load_ground_truth,
    locate_quote,
    validate_ground_truth,
)
from sawti.schemas import CallAnalysis, Language, Quote, RubricScore


def _evidence(text: str = "I understand your frustration.") -> Quote:
    return Quote(text=text, speaker="Agent", start_char=0, end_char=len(text))


def _rubric_scores() -> list[RubricScore]:
    return [
        RubricScore(evidence=_evidence(), criterion=criterion, score=0.8, justification="meets expectations")
        for criterion in RUBRIC_CRITERIA
    ]


def _call_analysis(call_id: str = "call_0000_ar", language: Language = Language.AR) -> CallAnalysis:
    return CallAnalysis(
        call_id=call_id,
        language=language,
        summary="Customer requested to close their account; agent processed it.",
        rubric_scores=_rubric_scores(),
        confidence=0.9,
        requires_human_review=False,
    )


def _write(directory: Path, name: str, analysis: CallAnalysis) -> Path:
    file_path = directory / name
    file_path.write_text(analysis.model_dump_json(), encoding="utf-8")
    return file_path


def test_load_ground_truth_returns_validated_call_analyses(tmp_path: Path) -> None:
    """load_ground_truth() returns one validated CallAnalysis per *.json file."""
    _write(tmp_path, "call_0000_ar.json", _call_analysis("call_0000_ar", Language.AR))
    _write(tmp_path, "call_0001_en.json", _call_analysis("call_0001_en", Language.EN))

    records = load_ground_truth(tmp_path)

    assert len(records) == 2
    assert all(isinstance(record, CallAnalysis) for record in records)
    assert {record.call_id for record in records} == {"call_0000_ar", "call_0001_en"}


def test_load_ground_truth_raises_on_malformed_record(tmp_path: Path) -> None:
    """load_ground_truth() raises, naming the file, when a record fails CallAnalysis validation."""
    bad_path = tmp_path / "call_0000_ar.json"
    bad_path.write_text(json.dumps({"call_id": "call_0000_ar", "language": "ar"}), encoding="utf-8")

    with pytest.raises(ValueError, match="call_0000_ar.json"):
        load_ground_truth(tmp_path)


def test_load_ground_truth_raises_on_invalid_json(tmp_path: Path) -> None:
    """load_ground_truth() raises, naming the file, when a file isn't valid JSON at all."""
    bad_path = tmp_path / "call_0000_ar.json"
    bad_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="call_0000_ar.json"):
        load_ground_truth(tmp_path)


def test_validate_ground_truth_accepts_a_valid_set() -> None:
    """validate_ground_truth() raises nothing for a consistent set of records."""
    records = [_call_analysis("call_0000_ar", Language.AR), _call_analysis("call_0001_en", Language.EN)]

    validate_ground_truth(records)  # should not raise


def test_validate_ground_truth_rejects_duplicate_call_ids() -> None:
    """validate_ground_truth() raises when two records share the same call_id."""
    records = [_call_analysis("call_0000_ar", Language.AR), _call_analysis("call_0000_ar", Language.AR)]

    with pytest.raises(ValueError, match="duplicate call_id"):
        validate_ground_truth(records)


def test_validate_ground_truth_rejects_missing_rubric_criterion() -> None:
    """validate_ground_truth() raises when a record is missing a required rubric criterion."""
    analysis = _call_analysis("call_0000_ar", Language.AR)
    incomplete = analysis.model_copy(update={"rubric_scores": analysis.rubric_scores[:-1]})

    with pytest.raises(ValueError, match="missing rubric criteria"):
        validate_ground_truth([incomplete])


def test_validate_ground_truth_rejects_language_mismatch_with_call_id() -> None:
    """validate_ground_truth() raises when record.language disagrees with its call_id suffix."""
    mismatched = _call_analysis("call_0000_ar", Language.EN)

    with pytest.raises(ValueError, match="encodes language"):
        validate_ground_truth([mismatched])


def test_validate_ground_truth_reports_all_problems_at_once() -> None:
    """validate_ground_truth() lists every problem found, not just the first."""
    same_id_ar = _call_analysis("call_0000_ar", Language.AR)
    same_id_en = _call_analysis("call_0000_ar", Language.EN)  # duplicate id + language mismatch

    with pytest.raises(ValueError) as exc_info:
        validate_ground_truth([same_id_ar, same_id_en])

    message = str(exc_info.value)
    assert "duplicate call_id" in message
    assert "encodes language" in message


# --------------------------------------------------------------------------
# generate_reference_labels — LLM-generated labels with programmatic grounding
# --------------------------------------------------------------------------

TRANSCRIPT = (
    "Agent: Hello, you have reached support, my name is Rami.\n"
    "Customer: I was charged twice this month and I want a refund.\n"
    "Agent: I will process the refund today and send you a confirmation.\n"
    "Customer: Thank you, that helps.\n"
)


def _draft(
    *,
    summary: str = "Customer was double charged; agent promised a refund today.",
    commitment_quote: str = "I will process the refund today",
    rubric_quote: str = "my name is Rami",
    sentiment_quote: str = "Thank you, that helps.",
    criteria: tuple[str, ...] = RUBRIC_CRITERIA,
    confidence: float = 0.9,
) -> DraftAnalysis:
    """A DraftAnalysis whose quotes are all verbatim in TRANSCRIPT by default."""
    return DraftAnalysis(
        summary=summary,
        commitments=[
            DraftCommitment(
                evidence_quote=commitment_quote,
                promised_by="Agent",
                description="Process the refund today.",
                deadline=None,
            )
        ],
        compliance_flags=[],
        rubric_scores=[
            DraftRubricScore(
                evidence_quote=rubric_quote,
                criterion=criterion,
                score=0.7,
                justification="adequate",
            )
            for criterion in criteria
        ],
        sentiment_points=[
            DraftSentimentPoint(
                evidence_quote=sentiment_quote, speaker="Customer", score=0.5, timestamp_sec=24.0
            )
        ],
        confidence=confidence,
    )


def _transcript_file(tmp_path: Path, name: str = "call_0000_ar.txt") -> Path:
    path = tmp_path / name
    path.write_text(TRANSCRIPT, encoding="utf-8")
    return path


class _ScriptedProvider:
    """An LLMProvider stub that returns a queued draft (or raises) per call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts: list[str] = []

    async def complete(self, prompt, *, system=None, **kwargs):  # pragma: no cover - unused
        raise AssertionError("generate_reference_labels must use structured_complete")

    async def structured_complete(self, prompt, *, response_model, system=None, **kwargs):
        self.prompts.append(prompt)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def _no_sleep(monkeypatch):
    """Make retry backoff instant so tests don't actually wait."""
    monkeypatch.setattr("sawti.data.ground_truth.time.sleep", lambda _seconds: None)


def _patch_provider(monkeypatch, provider) -> None:
    monkeypatch.setattr("sawti.data.ground_truth.get_llm_provider", lambda: provider)


def test_generate_reference_labels_locates_quotes_and_validates(tmp_path, monkeypatch, _no_sleep):
    """A clean draft yields a CallAnalysis whose offsets are computed from the transcript."""
    provider = _ScriptedProvider([_draft()])
    _patch_provider(monkeypatch, provider)

    analysis = generate_reference_labels(_transcript_file(tmp_path))

    assert analysis.call_id == "call_0000_ar"
    assert analysis.language is Language.AR
    # Offsets came from the transcript, not the model: they slice back exactly.
    evidence = analysis.commitments[0].evidence
    assert TRANSCRIPT[evidence.start_char : evidence.end_char] == evidence.text
    assert evidence.speaker == "Agent"
    assert {score.criterion for score in analysis.rubric_scores} == set(RUBRIC_CRITERIA)
    assert len(provider.prompts) == 1


def test_generate_reference_labels_retries_when_a_quote_is_not_verbatim(
    tmp_path, monkeypatch, _no_sleep
):
    """A paraphrased quote is retried, and the retry prompt names the offending text."""
    bad = _draft(commitment_quote="I'll refund you right away")  # not in the transcript
    provider = _ScriptedProvider([bad, _draft()])
    _patch_provider(monkeypatch, provider)

    analysis = generate_reference_labels(_transcript_file(tmp_path))

    assert len(provider.prompts) == 2
    retry_prompt = provider.prompts[1]
    assert "YOUR PREVIOUS ANSWER WAS REJECTED" in retry_prompt
    assert "I'll refund you right away" in retry_prompt
    evidence = analysis.commitments[0].evidence
    assert TRANSCRIPT[evidence.start_char : evidence.end_char] == evidence.text


def test_generate_reference_labels_raises_when_retries_are_exhausted(
    tmp_path, monkeypatch, _no_sleep
):
    """A model that never quotes verbatim fails loudly rather than returning junk."""
    bad = _draft(commitment_quote="a quote that is simply not there")
    provider = _ScriptedProvider([bad] * 3)
    _patch_provider(monkeypatch, provider)

    with pytest.raises(ValueError, match="after 2 retries"):
        generate_reference_labels(_transcript_file(tmp_path), max_retries=2)

    assert len(provider.prompts) == 3


def test_generate_reference_labels_retries_on_missing_rubric_criteria(
    tmp_path, monkeypatch, _no_sleep
):
    """A draft missing rubric criteria is retried — such a record is unusable."""
    incomplete = _draft(criteria=RUBRIC_CRITERIA[:3])
    provider = _ScriptedProvider([incomplete, _draft()])
    _patch_provider(monkeypatch, provider)

    analysis = generate_reference_labels(_transcript_file(tmp_path))

    assert len(provider.prompts) == 2
    assert len(analysis.rubric_scores) == len(RUBRIC_CRITERIA)


def test_generate_reference_labels_retries_on_provider_error(tmp_path, monkeypatch, _no_sleep):
    """A transport failure is retried without blaming the model's content."""
    provider = _ScriptedProvider([RuntimeError("429 rate limited"), _draft()])
    _patch_provider(monkeypatch, provider)

    analysis = generate_reference_labels(_transcript_file(tmp_path))

    assert len(provider.prompts) == 2
    assert "YOUR PREVIOUS ANSWER WAS REJECTED" not in provider.prompts[1]
    assert analysis.summary


def test_generate_reference_labels_sorts_sentiment_points_chronologically(
    tmp_path, monkeypatch, _no_sleep
):
    """Out-of-order sentiment points are ordered, not rejected — content is unchanged."""
    draft = _draft()
    draft.sentiment_points = [
        DraftSentimentPoint(
            evidence_quote="Thank you, that helps.", speaker="Customer", score=0.5, timestamp_sec=24.0
        ),
        DraftSentimentPoint(
            evidence_quote="I was charged twice this month",
            speaker="Customer",
            score=-0.6,
            timestamp_sec=8.0,
        ),
    ]
    _patch_provider(monkeypatch, _ScriptedProvider([draft]))

    analysis = generate_reference_labels(_transcript_file(tmp_path))

    points = analysis.sentiment_trajectory.points
    assert [point.timestamp_sec for point in points] == [8.0, 24.0]
    assert [point.score for point in points] == [-0.6, 0.5]


def test_generate_reference_labels_forces_human_review_below_threshold(
    tmp_path, monkeypatch, _no_sleep
):
    """A low-confidence analysis is routed to human review, per the schema invariant."""
    threshold = get_settings().confidence_threshold
    _patch_provider(monkeypatch, _ScriptedProvider([_draft(confidence=threshold - 0.2)]))

    analysis = generate_reference_labels(_transcript_file(tmp_path))

    assert analysis.requires_human_review is True


def test_generate_reference_labels_rejects_a_misnamed_transcript(tmp_path, monkeypatch):
    """A file that doesn't follow the call_<idx>_<lang>.txt convention is rejected."""
    path = tmp_path / "not_a_call.txt"
    path.write_text(TRANSCRIPT, encoding="utf-8")

    with pytest.raises(ValueError, match="does not match expected pattern"):
        generate_reference_labels(path)


def test_locate_quote_computes_offsets_and_speaker():
    """locate_quote() derives offsets from the transcript rather than trusting input."""
    quote = locate_quote(TRANSCRIPT, "I want a refund")

    assert TRANSCRIPT[quote.start_char : quote.end_char] == "I want a refund"
    assert quote.speaker == "Customer"


def test_locate_quote_raises_for_a_non_verbatim_quote():
    """locate_quote() refuses a paraphrase instead of fuzzy-matching it."""
    with pytest.raises(QuoteNotVerbatimError):
        locate_quote(TRANSCRIPT, "I would like my money back")
