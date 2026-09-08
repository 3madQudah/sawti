"""Tests for `sawti.data.ground_truth`.

Phase 1: mirrors `src/sawti/data/ground_truth.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sawti.data.ground_truth import RUBRIC_CRITERIA, load_ground_truth, validate_ground_truth
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
