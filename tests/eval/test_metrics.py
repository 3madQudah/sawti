"""Tests for `sawti.eval.metrics`.

Phase 1: mirrors `src/sawti/eval/metrics.py`. Constructed `CallAnalysis`
fixtures with hand-computed expected scores — every metric must stay
per-language and must never collapse into a single blended number.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sawti.data.ground_truth import RUBRIC_CRITERIA
from sawti.eval.metrics import (
    accuracy_by_category,
    grounding_precision_by_category,
    rubric_agreement_by_category,
)
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

TRANSCRIPT = (
    "Agent: Hello, you have reached support, my name is Rami.\n"
    "Customer: I was charged twice this month and I want a refund.\n"
    "Agent: I will process the refund today and send you a confirmation.\n"
    "Customer: Thank you, that helps.\n"
)

GROUNDED_QUOTE = "I will process the refund today"
UNGROUNDED_QUOTE = "I'll sort that out for you immediately"


def _quote(text: str = GROUNDED_QUOTE, speaker: str = "Agent") -> Quote:
    """A structurally valid Quote; offsets are not what the metrics check."""
    return Quote(text=text, speaker=speaker, start_char=0, end_char=len(text))


def _rubric(scores: dict[str, float] | None = None, quote_text: str = GROUNDED_QUOTE):
    values = scores or dict.fromkeys(RUBRIC_CRITERIA, 0.8)
    return [
        RubricScore(
            evidence=_quote(quote_text),
            criterion=criterion,
            score=score,
            justification="because",
        )
        for criterion, score in values.items()
    ]


def _analysis(
    call_id: str = "call_0000_ar",
    language: Language = Language.AR,
    *,
    summary: str = "Customer was double charged; agent promised a refund.",
    commitment_descriptions: tuple[str, ...] = ("process the refund today",),
    has_compliance_flag: bool = False,
    rubric: dict[str, float] | None = None,
    sentiment: tuple[float, ...] = (-0.6, 0.5),
    quote_text: str = GROUNDED_QUOTE,
) -> CallAnalysis:
    return CallAnalysis(
        call_id=call_id,
        language=language,
        summary=summary,
        commitments=[
            Commitment(evidence=_quote(quote_text), promised_by="Agent", description=description)
            for description in commitment_descriptions
        ],
        compliance_flags=(
            [
                ComplianceFlag(
                    evidence=_quote(quote_text),
                    rule_id="no_identity_verification",
                    severity=Severity.MEDIUM,
                    description="did not verify identity",
                )
            ]
            if has_compliance_flag
            else []
        ),
        rubric_scores=_rubric(rubric, quote_text),
        sentiment_trajectory=SentimentTrajectory(
            points=[
                SentimentPoint(
                    quote=_quote(quote_text, "Customer"),
                    speaker="Customer",
                    score=score,
                    timestamp_sec=float(index * 8),
                )
                for index, score in enumerate(sentiment)
            ]
        ),
        confidence=0.9,
        requires_human_review=False,
    )


@pytest.fixture
def transcript_dir(tmp_path: Path) -> Path:
    """A transcript directory holding one transcript per call_id used here."""
    for call_id in ("call_0000_ar", "call_0001_en", "call_0002_mixed"):
        (tmp_path / f"{call_id}.txt").write_text(TRANSCRIPT, encoding="utf-8")
    return tmp_path


# --- accuracy ------------------------------------------------------------


def test_accuracy_is_one_when_prediction_matches_reference():
    """Identical prediction and reference score a perfect 1.0 in their category."""
    reference = [_analysis()]

    assert accuracy_by_category([_analysis()], reference) == {Language.AR: 1.0}


def test_accuracy_groups_by_language_and_never_blends():
    """Each language gets its own score; a perfect en does not lift a broken ar."""
    reference = [
        _analysis("call_0000_ar", Language.AR),
        _analysis("call_0001_en", Language.EN),
    ]
    predictions = [
        # ar: wrong on commitments (none found), compliance, and sentiment direction
        _analysis(
            "call_0000_ar",
            Language.AR,
            commitment_descriptions=(),
            has_compliance_flag=True,
            sentiment=(0.5, -0.6),
        ),
        _analysis("call_0001_en", Language.EN),
    ]

    scores = accuracy_by_category(predictions, reference)

    assert set(scores) == {Language.AR, Language.EN}
    assert scores[Language.EN] == 1.0
    # ar: summary 1.0, commitments 0.0, compliance 0.0, sentiment 0.0 -> mean 0.25
    assert scores[Language.AR] == pytest.approx(0.25)


def test_accuracy_rewards_agreeing_that_a_call_has_no_commitments():
    """Both sides finding no commitments is correct, not an empty-vs-empty miss."""
    reference = [_analysis(commitment_descriptions=())]
    predictions = [_analysis(commitment_descriptions=())]

    assert accuracy_by_category(predictions, reference) == {Language.AR: 1.0}


def test_accuracy_scores_fuzzy_commitment_descriptions_partially():
    """A commitment described in different words scores between 0 and 1, not 0."""
    reference = [_analysis(commitment_descriptions=("process the refund today",))]
    predictions = [_analysis(commitment_descriptions=("refund the customer today",))]

    score = accuracy_by_category(predictions, reference)[Language.AR]

    # summary 1.0 + compliance 1.0 + sentiment 1.0 + partial commitment, over 4
    assert 0.75 < score < 1.0


def test_accuracy_ignores_calls_missing_from_either_side():
    """A prediction with no matching reference (or vice versa) is not scored."""
    reference = [_analysis("call_0000_ar", Language.AR)]
    predictions = [_analysis("call_0009_en", Language.EN)]

    assert accuracy_by_category(predictions, reference) == {}


# --- rubric agreement ----------------------------------------------------


def test_rubric_agreement_is_one_for_identical_scores():
    """Identical rubric scores give agreement 1.0 (MAE of 0)."""
    reference = [_analysis()]

    assert rubric_agreement_by_category([_analysis()], reference) == {Language.AR: 1.0}


def test_rubric_agreement_is_one_minus_mean_absolute_error():
    """Agreement is exactly 1 - MAE over the criteria scored on both sides."""
    reference = [_analysis(rubric=dict.fromkeys(RUBRIC_CRITERIA, 0.8))]
    # every criterion off by 0.2 -> MAE 0.2 -> agreement 0.8
    predictions = [_analysis(rubric=dict.fromkeys(RUBRIC_CRITERIA, 0.6))]

    scores = rubric_agreement_by_category(predictions, reference)

    assert scores[Language.AR] == pytest.approx(0.8)


def test_rubric_agreement_scores_zero_when_no_criteria_overlap():
    """A prediction that scored nothing comparable is a failure, not a skip."""
    reference = [_analysis()]
    predictions = [_analysis(rubric={"some_other_criterion": 0.8})]

    assert rubric_agreement_by_category(predictions, reference) == {Language.AR: 0.0}


def test_rubric_agreement_reports_each_language_separately():
    """Rubric agreement is per category, like every other metric here."""
    reference = [
        _analysis("call_0000_ar", Language.AR, rubric=dict.fromkeys(RUBRIC_CRITERIA, 0.8)),
        _analysis("call_0002_mixed", Language.MIXED, rubric=dict.fromkeys(RUBRIC_CRITERIA, 0.8)),
    ]
    predictions = [
        _analysis("call_0000_ar", Language.AR, rubric=dict.fromkeys(RUBRIC_CRITERIA, 0.8)),
        _analysis("call_0002_mixed", Language.MIXED, rubric=dict.fromkeys(RUBRIC_CRITERIA, 0.3)),
    ]

    scores = rubric_agreement_by_category(predictions, reference)

    assert scores[Language.AR] == pytest.approx(1.0)
    assert scores[Language.MIXED] == pytest.approx(0.5)


# --- grounding precision -------------------------------------------------


def test_grounding_precision_is_one_when_every_quote_is_verbatim(transcript_dir):
    """Every claim quoting the transcript exactly scores a precision of 1.0."""
    predictions = [_analysis(quote_text=GROUNDED_QUOTE)]

    scores = grounding_precision_by_category(predictions, transcript_dir=transcript_dir)

    assert scores == {Language.AR: 1.0}


def test_grounding_precision_is_zero_for_invented_quotes(transcript_dir):
    """A quote that is not in the transcript is not grounded, whatever its offsets say."""
    predictions = [_analysis(quote_text=UNGROUNDED_QUOTE)]

    scores = grounding_precision_by_category(predictions, transcript_dir=transcript_dir)

    assert scores == {Language.AR: 0.0}


def test_grounding_precision_pools_claims_within_a_language(transcript_dir):
    """Precision is micro-averaged over claims, so claim-heavy calls weigh more."""
    grounded = _analysis("call_0000_ar", Language.AR, quote_text=GROUNDED_QUOTE)
    ungrounded = _analysis("call_0002_mixed", Language.MIXED, quote_text=UNGROUNDED_QUOTE)

    scores = grounding_precision_by_category([grounded, ungrounded], transcript_dir=transcript_dir)

    assert scores[Language.AR] == 1.0
    assert scores[Language.MIXED] == 0.0


def test_grounding_precision_needs_no_ground_truth(transcript_dir):
    """The metric takes predictions only — it is checked against the transcript."""
    scores = grounding_precision_by_category(
        [_analysis(quote_text=GROUNDED_QUOTE)], transcript_dir=transcript_dir
    )

    assert scores == {Language.AR: 1.0}


def test_grounding_precision_skips_calls_whose_transcript_is_missing(tmp_path):
    """A missing transcript contributes no claims, rather than scoring as perfect."""
    predictions = [_analysis("call_0000_ar", Language.AR)]

    assert grounding_precision_by_category(predictions, transcript_dir=tmp_path) == {}


def test_grounding_precision_counts_partially_grounded_calls(transcript_dir):
    """A call mixing real and invented quotes lands strictly between 0 and 1."""
    mixed = _analysis("call_0000_ar", Language.AR, quote_text=GROUNDED_QUOTE)
    mixed.compliance_flags = [
        ComplianceFlag(
            evidence=_quote(UNGROUNDED_QUOTE),
            rule_id="made_up",
            severity=Severity.LOW,
            description="invented",
        )
    ]

    score = grounding_precision_by_category([mixed], transcript_dir=transcript_dir)[Language.AR]

    assert 0.0 < score < 1.0
