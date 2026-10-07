"""Tests for `sawti.api.views` and `sawti.api.records`: the pure mapping functions.

Phase 6.2.
"""

from __future__ import annotations

import pytest

from sawti.api.records import NO_ANALYSIS_SUMMARY, analysis_from_state, escalation_reason
from sawti.api.views import split_turns, turn_language
from sawti.data.transcript_parser import UnattributableLineError
from sawti.schemas import Language


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("النت عندي مقطوع", Language.AR),
        ("My internet is down", Language.EN),
        ("الـ internet عندي slow", Language.MIXED),
        ("رقمي هو [PHONE] شكراً", Language.AR),  # a placeholder is not English
        ("12345 ...", Language.EN),
    ],
)
def test_turn_language_is_by_script(text: str, expected: Language) -> None:
    assert turn_language(text) == expected


def test_split_turns_offsets_survive_blank_lines_and_padding() -> None:
    text = "Agent:   Hello there.  \n\nCustomer: مرحبا\r\nAgent: Bye\n"
    turns = split_turns(text, "c")
    assert [(t.index, t.speaker, t.text) for t in turns] == [
        (0, "Agent", "Hello there."),
        (1, "Customer", "مرحبا"),
        (2, "Agent", "Bye"),
    ]
    for turn in turns:
        assert text[turn.start_char : turn.end_char] == turn.text


def test_split_turns_refuses_to_guess_a_speaker() -> None:
    with pytest.raises(UnattributableLineError):
        split_turns("Agent: hi\nwho said this?\n", "c")


def test_escalation_reason_prefers_the_error() -> None:
    assert escalation_reason({"error": "extract: boom"}, threshold=0.7) == "Analysis error: extract: boom"


def test_escalation_reason_is_none_for_an_auto_pass() -> None:
    assert escalation_reason({"requires_human_review": False}, threshold=0.7) is None


def test_errored_state_still_yields_a_reviewable_analysis() -> None:
    analysis = analysis_from_state({"error": "extract: boom"}, call_id="c", language=Language.AR)
    assert analysis.summary == NO_ANALYSIS_SUMMARY
    assert analysis.confidence == 0.0 and analysis.requires_human_review


# --- plan_corrections: what induction will see -------------------------------

from datetime import UTC, datetime  # noqa: E402

from sawti.api.records import InvalidVerdictError, plan_corrections  # noqa: E402
from sawti.api.schemas import ReviewSubmission  # noqa: E402
from sawti.memory.induction import _describe_field  # noqa: E402
from sawti.schemas import CallAnalysis, Commitment, ComplianceFlag, Quote, RubricScore  # noqa: E402


def _quote(text: str = "I will call you") -> Quote:
    return Quote(text=text, speaker="Agent", start_char=7, end_char=7 + len(text))


def _analysis() -> CallAnalysis:
    return CallAnalysis(
        call_id="c",
        language=Language.EN,
        summary="s",
        commitments=[
            Commitment(
                evidence=_quote(),
                promised_by="Agent",
                description="Call back.",
                deadline=datetime(2026, 10, 8, tzinfo=UTC),
            )
        ],
        compliance_flags=[
            ComplianceFlag(
                evidence=_quote(), rule_id="no_greeting", severity="low", description="No greeting."
            )
        ],
        rubric_scores=[RubricScore(evidence=_quote(), criterion="empathy", score=0.8, justification="Warm.")],
        confidence=0.5,
        requires_human_review=True,
    )


@pytest.mark.parametrize(
    ("claim_id", "verdict", "location"),
    [
        ("commitments[0]", {"verdict": "reject"}, "commitments"),
        ("commitments[0]", {"verdict": "correct", "changes": {"description": "Refund."}}, "commitments[0]"),
        ("commitments[0]", {"verdict": "correct", "changes": {"deadline": None}}, "commitments[0]"),
        ("compliance_flags[0]", {"verdict": "reject"}, "compliance_flags"),
        (
            "compliance_flags[0]",
            {"verdict": "correct", "changes": {"severity": "high"}},
            "compliance_flags[0]",
        ),
        ("rubric_scores[0]", {"verdict": "reject"}, "rubric_scores[empathy]"),
        ("rubric_scores[0]", {"verdict": "correct", "changes": {"score": 0.3}}, "rubric_scores[empathy]"),
        (
            "rubric_scores[0]",
            {"verdict": "correct", "changes": {"justification": "Curt."}},
            "rubric_scores[empathy].justification",
        ),
    ],
)
def test_every_non_confirm_verdict_gives_induction_a_visible_difference(
    claim_id: str, verdict: dict[str, object], location: str
) -> None:
    """corrected != original, and the field induction renders differs between them."""
    original = _analysis()
    submission = ReviewSubmission.model_validate({"verdicts": [{"claim_id": claim_id, **verdict}]})
    [planned] = plan_corrections(submission, original)
    assert planned.error_location == location
    assert planned.corrected != original
    assert _describe_field(planned.corrected, location) != _describe_field(original, location)


def test_a_change_that_changes_nothing_is_rejected() -> None:
    submission = ReviewSubmission.model_validate(
        {"verdicts": [{"claim_id": "rubric_scores[0]", "verdict": "correct", "changes": {"score": 0.8}}]}
    )
    with pytest.raises(InvalidVerdictError, match="leave the claim as it was"):
        plan_corrections(submission, _analysis())


def test_confirm_plans_no_correction() -> None:
    submission = ReviewSubmission.model_validate(
        {"verdicts": [{"claim_id": "commitments[0]", "verdict": "confirm"}]}
    )
    assert plan_corrections(submission, _analysis()) == []
