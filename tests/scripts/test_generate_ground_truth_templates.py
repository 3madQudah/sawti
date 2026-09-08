"""Tests for `scripts/generate_ground_truth_templates.py`.

Phase 1 item 3: ground-truth authoring template generation. Never invents
call content — these tests assert every generated value is a placeholder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from generate_ground_truth_templates import (
    build_template,
    generate_templates,
    infer_call_id_and_language,
)

from sawti.data.ground_truth import RUBRIC_CRITERIA
from sawti.schemas import Language


def test_infer_call_id_and_language_parses_filename() -> None:
    """infer_call_id_and_language() reads call_id and language from the filename."""
    call_id, language = infer_call_id_and_language(Path("data/synthetic/call_0007_mixed.txt"))

    assert call_id == "call_0007_mixed"
    assert language is Language.MIXED


def test_infer_call_id_and_language_rejects_unexpected_filename() -> None:
    """infer_call_id_and_language() raises on a filename that doesn't match the expected pattern."""
    with pytest.raises(ValueError, match="does not match"):
        infer_call_id_and_language(Path("data/synthetic/not_a_call.txt"))


def test_build_template_prefills_only_criterion_names() -> None:
    """build_template() leaves every value empty/null except the 7 rubric criterion names."""
    template = build_template("call_0000_ar", Language.AR)

    assert template["call_id"] == "call_0000_ar"
    assert template["language"] == "ar"
    assert template["summary"] == ""
    assert template["commitments"] == []
    assert template["compliance_flags"] == []
    assert template["confidence"] is None
    assert template["requires_human_review"] is None
    assert template["sentiment_trajectory"] == {"points": []}
    assert "_instructions" in template

    criteria = [entry["criterion"] for entry in template["rubric_scores"]]
    assert criteria == list(RUBRIC_CRITERIA)
    for entry in template["rubric_scores"]:
        assert entry["score"] is None
        assert entry["justification"] == ""
        assert entry["evidence"] == {"text": "", "speaker": "", "start_char": None, "end_char": None}


def test_generate_templates_writes_one_json_per_transcript(tmp_path: Path) -> None:
    """generate_templates() writes one ground-truth template per synthetic transcript."""
    synthetic_dir = tmp_path / "synthetic"
    synthetic_dir.mkdir()
    (synthetic_dir / "call_0000_ar.txt").write_text("Agent: hi.\n", encoding="utf-8")
    (synthetic_dir / "call_0001_en.txt").write_text("Agent: hi.\n", encoding="utf-8")
    output_dir = tmp_path / "ground_truth"

    written = generate_templates(synthetic_dir, output_dir, overwrite=False)

    assert len(written) == 2
    assert sorted(p.name for p in output_dir.glob("*.json")) == ["call_0000_ar.json", "call_0001_en.json"]
    data = json.loads((output_dir / "call_0000_ar.json").read_text(encoding="utf-8"))
    assert data["call_id"] == "call_0000_ar"
    assert data["language"] == "ar"


def test_generate_templates_does_not_overwrite_existing_by_default(tmp_path: Path) -> None:
    """generate_templates() skips files that already exist unless overwrite=True."""
    synthetic_dir = tmp_path / "synthetic"
    synthetic_dir.mkdir()
    (synthetic_dir / "call_0000_ar.txt").write_text("Agent: hi.\n", encoding="utf-8")
    output_dir = tmp_path / "ground_truth"
    output_dir.mkdir()
    existing = output_dir / "call_0000_ar.json"
    existing.write_text('{"summary": "already reviewed"}', encoding="utf-8")

    written = generate_templates(synthetic_dir, output_dir, overwrite=False)

    assert written == []
    assert existing.read_text(encoding="utf-8") == '{"summary": "already reviewed"}'


def test_generate_templates_overwrites_when_requested(tmp_path: Path) -> None:
    """generate_templates() replaces existing template files when overwrite=True."""
    synthetic_dir = tmp_path / "synthetic"
    synthetic_dir.mkdir()
    (synthetic_dir / "call_0000_ar.txt").write_text("Agent: hi.\n", encoding="utf-8")
    output_dir = tmp_path / "ground_truth"
    output_dir.mkdir()
    (output_dir / "call_0000_ar.json").write_text('{"stale": true}', encoding="utf-8")

    written = generate_templates(synthetic_dir, output_dir, overwrite=True)

    assert len(written) == 1
    data = json.loads((output_dir / "call_0000_ar.json").read_text(encoding="utf-8"))
    assert "stale" not in data
