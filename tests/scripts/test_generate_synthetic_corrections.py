"""Tests for `scripts/generate_synthetic_corrections.py`.

Phase 4: mirrors the synthetic-corrections generator script.

The end-to-end test needs a real Postgres (via `capture_correction()`'s
persistence path) — run `docker compose up -d` first. See
`docs/09-DECISIONS.md`. `_first_difference` itself needs no database.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from generate_synthetic_corrections import _first_difference, generate_synthetic_corrections

from sawti.db.models import Base, Call, CallAnalysisRecord, ReviewerAction
from sawti.db.session import get_engine, get_session
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


class TestFirstDifference:
    """`_first_difference` is pure and deliberately narrow — see its docstring."""

    def test_no_difference_returns_none(self) -> None:
        agent = _analysis("call_0000_en", rubric_scores=[_score("professionalism", 0.8)])
        truth = _analysis("call_0000_en", rubric_scores=[_score("professionalism", 0.8)])
        assert _first_difference(agent, truth) is None

    def test_commitment_count_mismatch(self) -> None:
        agent = _analysis("call_0000_en", commitments=[])
        truth = _analysis("call_0000_en", commitments=[_commitment(deadline=None)])
        assert _first_difference(agent, truth) == "commitments"

    def test_commitment_deadline_presence_mismatch(self) -> None:
        agent = _analysis("call_0000_en", commitments=[_commitment(deadline=None)])
        truth = _analysis("call_0000_en", commitments=[_commitment(deadline=datetime.now(UTC))])
        assert _first_difference(agent, truth) == "commitments[0].deadline"

    def test_compliance_flag_set_mismatch(self) -> None:
        agent = _analysis("call_0000_en", compliance_flags=[_flag("rude_agent")])
        truth = _analysis("call_0000_en", compliance_flags=[_flag("no_id_check")])
        assert _first_difference(agent, truth) == "compliance_flags"

    def test_rubric_score_beyond_tolerance(self) -> None:
        agent = _analysis("call_0000_en", rubric_scores=[_score("empathy", 0.9)])
        truth = _analysis("call_0000_en", rubric_scores=[_score("empathy", 0.2)])
        assert _first_difference(agent, truth) == "rubric_scores[empathy]"

    def test_rubric_score_within_tolerance_is_ignored(self) -> None:
        agent = _analysis("call_0000_en", rubric_scores=[_score("empathy", 0.8)])
        truth = _analysis("call_0000_en", rubric_scores=[_score("empathy", 0.65)])
        assert _first_difference(agent, truth) is None

    def test_free_text_wording_alone_is_not_a_difference(self) -> None:
        """summary/description wording is deliberately excluded — see the docstring."""
        agent = _analysis("call_0000_en", commitments=[_commitment(deadline=None)])
        truth = _analysis("call_0000_en", commitments=[_commitment(deadline=None)])
        truth.commitments[0].description = "a completely different description"
        assert _first_difference(agent, truth) is None


@pytest.fixture(scope="module", autouse=True)
def _schema() -> None:
    Base.metadata.create_all(get_engine())


def test_generate_synthetic_corrections_captures_only_disagreements(tmp_path: Path) -> None:
    """Only the call where agent and ground truth disagree produces a Correction."""
    ground_truth_dir = tmp_path / "ground_truth"
    synthetic_dir = tmp_path / "synthetic"
    ground_truth_dir.mkdir()
    synthetic_dir.mkdir()

    truth_matching = _analysis("call_match_en", rubric_scores=[_score("empathy", 0.8)])
    truth_diff = _analysis("call_diff_en", commitments=[_commitment(deadline=datetime.now(UTC))])
    agent_matching = truth_matching
    agent_diff = _analysis("call_diff_en", commitments=[_commitment(deadline=None)])

    for record in (truth_matching, truth_diff):
        (ground_truth_dir / f"{record.call_id}.json").write_text(record.model_dump_json(), encoding="utf-8")
        (synthetic_dir / f"{record.call_id}.txt").write_text("transcript text", encoding="utf-8")

    outputs = {"call_match_en": agent_matching, "call_diff_en": agent_diff}

    def fake_extractor(path: Path) -> CallAnalysis:
        return outputs[path.stem]

    corrections, counts = generate_synthetic_corrections(
        ground_truth_dir,
        synthetic_dir=synthetic_dir,
        sleep_seconds=0.0,
        extractor=fake_extractor,
    )

    assert counts.compared[Language.EN] == 2
    assert counts.matched[Language.EN] == 1
    assert counts.corrected[Language.EN] == 1
    assert counts.failed[Language.EN] == 0

    assert len(corrections) == 1
    [correction] = corrections
    assert correction.call_id == "call_diff_en"
    assert correction.reviewer_id == "synthetic-day1"
    assert correction.error_location == "commitments[0].deadline"
    assert correction.note is not None
    assert "not a real QA review" in correction.note

    # Cleanup: the FK-satisfying rows this run persisted for call_diff_en.
    with get_session() as session:
        record = session.get(CallAnalysisRecord, correction.original.id)
        call_row_id = record.call_id if record is not None else None
        for action in session.query(ReviewerAction).filter_by(call_analysis_id=correction.original.id):
            session.delete(action)
        if record is not None:
            session.delete(record)
        if call_row_id is not None:
            call_row = session.get(Call, call_row_id)
            if call_row is not None:
                session.delete(call_row)


def test_generate_synthetic_corrections_respects_limit_per_language(tmp_path: Path) -> None:
    """limit_per_language caps how many calls are even sent to the extractor, per language."""
    ground_truth_dir = tmp_path / "ground_truth"
    synthetic_dir = tmp_path / "synthetic"
    ground_truth_dir.mkdir()
    synthetic_dir.mkdir()

    records = [_analysis(f"call_{i:04d}_en", rubric_scores=[_score("empathy", 0.8)]) for i in range(3)]
    for record in records:
        (ground_truth_dir / f"{record.call_id}.json").write_text(record.model_dump_json(), encoding="utf-8")
        (synthetic_dir / f"{record.call_id}.txt").write_text("transcript text", encoding="utf-8")

    seen_call_ids: list[str] = []

    def fake_extractor(path: Path) -> CallAnalysis:
        seen_call_ids.append(path.stem)
        return next(record for record in records if record.call_id == path.stem)

    corrections, counts = generate_synthetic_corrections(
        ground_truth_dir,
        synthetic_dir=synthetic_dir,
        limit_per_language=1,
        sleep_seconds=0.0,
        extractor=fake_extractor,
    )

    assert len(seen_call_ids) == 1
    assert counts.compared[Language.EN] == 1
    assert corrections == []
