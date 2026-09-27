"""Tests for `scripts/eval_held_out_calls.py`.

Phase 5 — the held-out-calls base-vs-tuned accuracy comparison
(docs/09-DECISIONS.md, 2026-09-27). `run()`'s model-loading step needs
CUDA and is not exercised here; everything else — held-out-id loading,
the extraction loop (via the same `get_llm_provider` injection point
`tests/eval/test_plain_extraction.py` already relies on), and the
comparison-table rendering — runs for real, no GPU needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval_held_out_calls import (
    _extract_all,
    _metrics_for,
    load_held_out_call_ids,
    render_comparison,
)

from sawti.data.ground_truth import RUBRIC_CRITERIA
from sawti.schemas import CallAnalysis, Language


def test_load_held_out_call_ids_groups_by_language_and_skips_meta(tmp_path: Path) -> None:
    path = tmp_path / "held_out_call_ids.json"
    path.write_text(
        json.dumps(
            {
                "ar": ["call_0000_ar", "call_0001_ar"],
                "en": ["call_0002_en"],
                "_meta": {"ar": {"seed": 1}},
            }
        ),
        encoding="utf-8",
    )

    result = load_held_out_call_ids(path)

    assert result == {
        Language.AR: ["call_0000_ar", "call_0001_ar"],
        Language.EN: ["call_0002_en"],
    }


# --- _extract_all: real run_plain_extraction, fake provider, no CUDA -------


def _response(*, quote_text: str) -> dict:
    return {
        "summary": "Refund processed.",
        "commitments": [{"quote": quote_text, "promised_by": "Agent", "description": "refund"}],
        "compliance_flags": [],
        "rubric_scores": [
            {"quote": quote_text, "criterion": criterion, "score": 0.7, "justification": "ok"}
            for criterion in RUBRIC_CRITERIA
        ],
        "sentiment_points": [
            {"quote": quote_text, "speaker": "Customer", "score": 0.5, "timestamp_sec": 0.0}
        ],
        "confidence": 0.8,
    }


class _StubProvider:
    """Mirrors `tests/eval/test_plain_extraction.py`'s stub — one response per call."""

    def __init__(self, responses: dict[str, dict | Exception]) -> None:
        self._responses = responses

    async def complete(self, prompt, *, system=None, **kwargs):  # pragma: no cover - unused
        raise AssertionError("run_plain_extraction must use structured_complete")

    async def structured_complete(self, prompt, *, response_model, system=None, **kwargs):
        # The transcript is appended after the fixed instruction text, so a
        # substring match on the transcript identifies which call this is.
        for call_id, response in self._responses.items():
            if call_id in prompt:
                if isinstance(response, Exception):
                    raise response
                return response_model.model_validate(response)
        raise AssertionError(f"no stubbed response matches prompt: {prompt[:200]!r}")


def _write_transcript(synthetic_dir: Path, call_id: str, quote: str) -> None:
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    (synthetic_dir / f"{call_id}.txt").write_text(
        f"Agent: {quote}\nCustomer: thanks [{call_id}]", encoding="utf-8"
    )


def test_extract_all_returns_one_call_analysis_per_successful_call(tmp_path: Path) -> None:
    synthetic_dir = tmp_path / "synthetic"
    _write_transcript(synthetic_dir, "call_0000_ar", "we will refund you today")
    _write_transcript(synthetic_dir, "call_0001_ar", "we will call back")

    provider = _StubProvider(
        {
            "call_0000_ar": _response(quote_text="we will refund you today"),
            "call_0001_ar": _response(quote_text="we will call back"),
        }
    )

    predictions = _extract_all(
        ["call_0000_ar", "call_0001_ar"], synthetic_dir=synthetic_dir, provider=provider
    )

    assert {p.call_id for p in predictions} == {"call_0000_ar", "call_0001_ar"}


def test_extract_all_skips_a_missing_transcript(tmp_path: Path) -> None:
    synthetic_dir = tmp_path / "synthetic"
    _write_transcript(synthetic_dir, "call_0000_ar", "we will refund you today")
    provider = _StubProvider({"call_0000_ar": _response(quote_text="we will refund you today")})

    predictions = _extract_all(
        ["call_0000_ar", "call_missing_ar"], synthetic_dir=synthetic_dir, provider=provider
    )

    assert [p.call_id for p in predictions] == ["call_0000_ar"]


def test_extract_all_skips_a_call_whose_extraction_raises(tmp_path: Path) -> None:
    synthetic_dir = tmp_path / "synthetic"
    _write_transcript(synthetic_dir, "call_0000_ar", "we will refund you today")
    _write_transcript(synthetic_dir, "call_0001_ar", "we will call back")
    provider = _StubProvider(
        {
            "call_0000_ar": _response(quote_text="we will refund you today"),
            "call_0001_ar": RuntimeError("model exploded"),
        }
    )

    predictions = _extract_all(
        ["call_0000_ar", "call_0001_ar"], synthetic_dir=synthetic_dir, provider=provider
    )

    assert [p.call_id for p in predictions] == ["call_0000_ar"]


# --- render_comparison --------------------------------------------------------


def _quote(text: str) -> dict:
    return {"text": text, "speaker": "Agent", "start_char": 0, "end_char": len(text)}


def _minimal_analysis(call_id: str, *, language: Language = Language.AR) -> CallAnalysis:
    return CallAnalysis(
        call_id=call_id,
        language=language,
        summary="placeholder",
        commitments=[],
        compliance_flags=[],
        rubric_scores=[],
        confidence=0.9,
        requires_human_review=False,
    )


def test_render_comparison_only_shows_languages_with_scored_calls() -> None:
    base_metrics = {"Accuracy": {Language.AR: 0.5}}
    tuned_metrics = {"Accuracy": {Language.AR: 0.7}}

    table = render_comparison(base_metrics, tuned_metrics, counts={Language.AR: 15})
    rows = [line for line in table.splitlines() if line.startswith("| ")]

    assert any(row.startswith("| ar |") for row in rows)
    assert not any(row.startswith("| en |") for row in rows)
    assert not any(row.startswith("| mixed |") for row in rows)


def test_render_comparison_computes_the_delta() -> None:
    base_metrics = {"Accuracy": {Language.AR: 0.500}}
    tuned_metrics = {"Accuracy": {Language.AR: 0.700}}

    table = render_comparison(base_metrics, tuned_metrics, counts={Language.AR: 15})

    assert "+0.200" in table


def test_render_comparison_handles_a_missing_score_gracefully() -> None:
    base_metrics = {"Accuracy": {}}
    tuned_metrics = {"Accuracy": {Language.AR: 0.7}}

    table = render_comparison(base_metrics, tuned_metrics, counts={Language.AR: 15})

    assert "—" in table


def test_metrics_for_reuses_the_shared_accuracy_methodology(tmp_path: Path) -> None:
    """Not a new metric implementation — the same `sawti.eval.metrics` functions Phase 1/2/4 use."""
    synthetic_dir = tmp_path / "synthetic"
    synthetic_dir.mkdir()
    (synthetic_dir / "call_0000_ar.txt").write_text("Agent: hello\nCustomer: hi", encoding="utf-8")

    prediction = _minimal_analysis("call_0000_ar")
    reference = _minimal_analysis("call_0000_ar")

    metrics = _metrics_for([prediction], [reference], synthetic_dir=synthetic_dir)

    assert set(metrics) == {
        "Accuracy",
        "Rubric agreement",
        "Grounding precision",
        "Unsupported claim rate",
    }
    assert metrics["Accuracy"][Language.AR] == pytest.approx(1.0)


# --- run(): CUDA-only paths need real peft/bitsandbytes; skip cleanly here --


def test_run_needs_the_finetune_cuda_stack() -> None:
    pytest.importorskip("peft")
    pytest.importorskip("bitsandbytes")


def test_run_raises_on_a_held_out_id_missing_from_ground_truth(tmp_path: Path) -> None:
    """The corpus/held-out-file mismatch guard needs no model — runs for real."""
    from eval_held_out_calls import run

    held_out_path = tmp_path / "held_out_call_ids.json"
    held_out_path.write_text(json.dumps({"ar": ["call_nowhere_ar"]}), encoding="utf-8")

    ground_truth_dir = tmp_path / "ground_truth"
    ground_truth_dir.mkdir()
    truth = CallAnalysis(
        call_id="call_0000_ar",
        language=Language.AR,
        summary="s",
        confidence=0.9,
        requires_human_review=False,
    )
    (ground_truth_dir / "call_0000_ar.json").write_text(
        json.dumps({**truth.model_dump(mode="json"), "_provenance": {"human_reviewed": False}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="call_nowhere_ar"):
        run(
            base_model="unused",
            adapter_dir=tmp_path / "adapter",
            held_out_path=held_out_path,
            ground_truth_dir=ground_truth_dir,
            synthetic_dir=tmp_path / "synthetic",
            output_path=tmp_path / "out.json",
        )
