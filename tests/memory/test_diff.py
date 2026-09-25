"""Tests for `sawti.memory.diff`.

Phase 4: mirrors `src/sawti/memory/diff.py`. Moved here from
`scripts/generate_synthetic_corrections.py` (Stage 2) when the function
itself moved, to be shared with `sawti.eval.experiments.five_batch` (Stage 8).
"""

from __future__ import annotations

from datetime import UTC, datetime

from sawti.memory.diff import first_difference, topic_for
from sawti.schemas import CallAnalysis, Commitment, ComplianceFlag, Language, Quote, RubricScore


def _quote(text: str) -> Quote:
    return Quote(text=text, speaker="Agent", start_char=0, end_char=len(text))


def _commitment(*, deadline: datetime | None) -> Commitment:
    return Commitment(
        evidence=_quote("we will call back"), promised_by="Agent", description="callback", deadline=deadline
    )


def _flag(rule_id: str) -> ComplianceFlag:
    return ComplianceFlag(evidence=_quote("hello"), rule_id=rule_id, severity="low", description="x")


def _score(criterion: str, value: float) -> RubricScore:
    return RubricScore(evidence=_quote("ok"), criterion=criterion, score=value, justification="x")


def _analysis(
    call_id: str,
    *,
    language: Language = Language.EN,
    commitments: list[Commitment] | None = None,
    compliance_flags: list[ComplianceFlag] | None = None,
    rubric_scores: list[RubricScore] | None = None,
) -> CallAnalysis:
    return CallAnalysis(
        call_id=call_id,
        language=language,
        summary="placeholder summary",
        commitments=commitments or [],
        compliance_flags=compliance_flags or [],
        rubric_scores=rubric_scores or [],
        confidence=0.9,
        requires_human_review=False,
    )


def test_no_difference_returns_none() -> None:
    agent = _analysis("call_0000_en", rubric_scores=[_score("professionalism", 0.8)])
    truth = _analysis("call_0000_en", rubric_scores=[_score("professionalism", 0.8)])
    assert first_difference(agent, truth) is None


def test_commitment_count_mismatch() -> None:
    agent = _analysis("call_0000_en", commitments=[])
    truth = _analysis("call_0000_en", commitments=[_commitment(deadline=None)])
    assert first_difference(agent, truth) == "commitments"


def test_commitment_deadline_presence_mismatch() -> None:
    agent = _analysis("call_0000_en", commitments=[_commitment(deadline=None)])
    truth = _analysis("call_0000_en", commitments=[_commitment(deadline=datetime.now(UTC))])
    assert first_difference(agent, truth) == "commitments[0].deadline"


def test_compliance_flag_set_mismatch() -> None:
    agent = _analysis("call_0000_en", compliance_flags=[_flag("rude_agent")])
    truth = _analysis("call_0000_en", compliance_flags=[_flag("no_id_check")])
    assert first_difference(agent, truth) == "compliance_flags"


def test_rubric_score_beyond_tolerance() -> None:
    agent = _analysis("call_0000_en", rubric_scores=[_score("empathy", 0.9)])
    truth = _analysis("call_0000_en", rubric_scores=[_score("empathy", 0.2)])
    assert first_difference(agent, truth) == "rubric_scores[empathy]"


def test_rubric_score_within_tolerance_is_ignored() -> None:
    agent = _analysis("call_0000_en", rubric_scores=[_score("empathy", 0.8)])
    truth = _analysis("call_0000_en", rubric_scores=[_score("empathy", 0.65)])
    assert first_difference(agent, truth) is None


def test_free_text_wording_alone_is_not_a_difference() -> None:
    """summary/description wording is deliberately excluded — see the docstring."""
    agent = _analysis("call_0000_en", commitments=[_commitment(deadline=None)])
    truth = _analysis("call_0000_en", commitments=[_commitment(deadline=None)])
    truth.commitments[0].description = "a completely different description"
    assert first_difference(agent, truth) is None


def test_topic_for_strips_list_index() -> None:
    assert topic_for("commitments[0].deadline") == "commitments.deadline"


def test_topic_for_strips_bracketed_criterion() -> None:
    assert topic_for("rubric_scores[empathy]") == "rubric_scores"


def test_topic_for_groups_different_indices_together() -> None:
    assert topic_for("commitments[0].deadline") == topic_for("commitments[3].deadline")


def test_topic_for_leaves_unbracketed_location_unchanged() -> None:
    assert topic_for("compliance_flags") == "compliance_flags"
