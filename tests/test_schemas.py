"""Tests for `sawti.schemas`.

Phase 0: mirrors `src/sawti/schemas.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from sawti.schemas import (
    CallAnalysis,
    Commitment,
    Correction,
    Language,
    Quote,
    SentimentPoint,
    SentimentTrajectory,
)


def _quote(text: str = "I will call you back tomorrow.", start: int = 0) -> Quote:
    return Quote(text=text, speaker="agent", start_char=start, end_char=start + len(text))


def test_quote_requires_offset_span_to_match_text_length() -> None:
    """Quote rejects start/end offsets whose span doesn't match len(text)."""
    with pytest.raises(ValidationError):
        Quote(text="hello", speaker="agent", start_char=0, end_char=10)


def test_quote_requires_end_after_start() -> None:
    """Quote rejects end_char <= start_char."""
    with pytest.raises(ValidationError):
        Quote(text="hello", speaker="agent", start_char=10, end_char=5)


def test_claim_without_quote_is_rejected() -> None:
    """A Claim subclass cannot be constructed without an `evidence` quote."""
    with pytest.raises(ValidationError):
        Commitment(promised_by="agent", description="will follow up")  # type: ignore[call-arg]


def test_claim_with_blank_evidence_text_is_rejected() -> None:
    """A Claim subclass rejects evidence whose quote text is empty/whitespace-only."""
    with pytest.raises(ValidationError):
        Commitment(
            evidence=Quote(text=" ", speaker="agent", start_char=0, end_char=1),
            promised_by="agent",
            description="will follow up",
        )


def test_sentiment_trajectory_rejects_out_of_order_points() -> None:
    """SentimentTrajectory rejects points that are not in non-decreasing timestamp order."""
    with pytest.raises(ValidationError):
        SentimentTrajectory(
            points=[
                SentimentPoint(quote=_quote(), speaker="agent", score=0.1, timestamp_sec=10.0),
                SentimentPoint(quote=_quote(), speaker="agent", score=0.2, timestamp_sec=5.0),
            ]
        )


def test_call_analysis_rejects_low_confidence_without_human_review() -> None:
    """CallAnalysis rejects confidence below threshold when requires_human_review is False."""
    with pytest.raises(ValidationError):
        CallAnalysis(
            call_id="call-1",
            language=Language.EN,
            summary="short call",
            confidence=0.1,
            requires_human_review=False,
        )


def test_call_analysis_accepts_low_confidence_with_human_review() -> None:
    """CallAnalysis accepts confidence below threshold when requires_human_review is True."""
    analysis = CallAnalysis(
        call_id="call-1",
        language=Language.EN,
        summary="short call",
        confidence=0.1,
        requires_human_review=True,
    )
    assert analysis.requires_human_review is True


def test_correction_requires_reviewer_id_and_timestamp() -> None:
    """Correction cannot be constructed without both reviewer_id and timestamp."""
    original = CallAnalysis(
        call_id="call-1",
        language=Language.AR,
        summary="short call",
        confidence=0.95,
        requires_human_review=False,
    )
    with pytest.raises(ValidationError):
        Correction(
            call_id="call-1",
            original=original,
            corrected=original,
            error_location="rubric_scores[0]",
        )  # type: ignore[call-arg]


def test_correction_accepts_valid_reviewer_id_and_timestamp() -> None:
    """Correction validates successfully when reviewer_id and timestamp are provided."""
    original = CallAnalysis(
        call_id="call-1",
        language=Language.MIXED,
        summary="short call",
        confidence=0.95,
        requires_human_review=False,
    )
    correction = Correction(
        call_id="call-1",
        original=original,
        corrected=original,
        error_location="rubric_scores[0]",
        reviewer_id="reviewer-42",
        timestamp=datetime.now(UTC),
    )
    assert correction.reviewer_id == "reviewer-42"
