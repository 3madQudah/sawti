"""Tests for `sawti.eval.runner`.

Phase 1: mirrors `src/sawti/eval/runner.py`. The LLM provider is mocked end
to end — these tests must never make a network call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sawti.data.ground_truth import RUBRIC_CRITERIA
from sawti.eval.runner import REFERENCE_LABEL_CAVEAT, run_eval
from sawti.schemas import (
    CallAnalysis,
    Commitment,
    Language,
    Quote,
    RubricScore,
    SentimentPoint,
    SentimentTrajectory,
)

TRANSCRIPT = (
    "Agent: Hello, you have reached support, my name is Rami.\n"
    "Customer: I was charged twice this month and I want a refund.\n"
    "Agent: I will process the refund today and send you a confirmation.\n"
    "Customer: Thank you, that helps.\n"
)

QUOTE_TEXT = "I will process the refund today"


def _quote(text: str = QUOTE_TEXT) -> Quote:
    return Quote(text=text, speaker="Agent", start_char=0, end_char=len(text))


def _reference(call_id: str, language: Language) -> CallAnalysis:
    return CallAnalysis(
        call_id=call_id,
        language=language,
        summary="Customer was double charged; agent promised a refund.",
        commitments=[
            Commitment(evidence=_quote(), promised_by="Agent", description="process the refund today")
        ],
        compliance_flags=[],
        rubric_scores=[
            RubricScore(evidence=_quote(), criterion=criterion, score=0.8, justification="ok")
            for criterion in RUBRIC_CRITERIA
        ],
        sentiment_trajectory=SentimentTrajectory(
            points=[
                SentimentPoint(quote=_quote(), speaker="Customer", score=-0.6, timestamp_sec=0.0),
                SentimentPoint(quote=_quote(), speaker="Customer", score=0.5, timestamp_sec=16.0),
            ]
        ),
        confidence=0.9,
        requires_human_review=False,
    )


class _StubProvider:
    """Returns a fixed plain-extraction response for every call."""

    def __init__(self, *, quote_text: str = QUOTE_TEXT, rubric_score: float = 0.8) -> None:
        self.quote_text = quote_text
        self.rubric_score = rubric_score
        self.calls = 0

    async def complete(self, prompt, *, system=None, **kwargs):  # pragma: no cover - unused
        raise AssertionError("the baseline must use structured_complete")

    async def structured_complete(self, prompt, *, response_model, system=None, **kwargs):
        self.calls += 1
        return response_model.model_validate(
            {
                "summary": "Double charge; refund promised.",
                "commitments": [
                    {
                        "quote": self.quote_text,
                        "promised_by": "Agent",
                        "description": "process the refund today",
                    }
                ],
                "compliance_flags": [],
                "rubric_scores": [
                    {
                        "quote": self.quote_text,
                        "criterion": criterion,
                        "score": self.rubric_score,
                        "justification": "ok",
                    }
                    for criterion in RUBRIC_CRITERIA
                ],
                "sentiment_points": [
                    {
                        "quote": self.quote_text,
                        "speaker": "Customer",
                        "score": -0.6,
                        "timestamp_sec": 0.0,
                    },
                    {
                        "quote": self.quote_text,
                        "speaker": "Customer",
                        "score": 0.5,
                        "timestamp_sec": 16.0,
                    },
                ],
                "confidence": 0.8,
            }
        )


@pytest.fixture
def dataset(tmp_path: Path):
    """A two-call dataset: reference labels, transcripts, and an output path."""
    ground_truth_dir = tmp_path / "ground_truth"
    synthetic_dir = tmp_path / "synthetic"
    ground_truth_dir.mkdir()
    synthetic_dir.mkdir()

    for call_id, language in (("call_0000_ar", Language.AR), ("call_0001_en", Language.EN)):
        record = _reference(call_id, language)
        payload = {"_provenance": {"method": "llm_generated", "human_reviewed": False}}
        payload.update(json.loads(record.model_dump_json()))
        (ground_truth_dir / f"{call_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        (synthetic_dir / f"{call_id}.txt").write_text(TRANSCRIPT, encoding="utf-8")

    return ground_truth_dir, synthetic_dir, tmp_path / "eval_results.md"


@pytest.fixture
def _stub_llm(monkeypatch):
    """Install a stub provider and remove all sleeping."""
    provider = _StubProvider()
    monkeypatch.setattr("sawti.eval.plain_extraction.get_llm_provider", lambda: provider)
    monkeypatch.setattr("sawti.eval.runner.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("sawti.eval.plain_extraction.time.sleep", lambda _seconds: None)
    return provider


def test_run_eval_returns_all_metrics_per_language(dataset, _stub_llm):
    """run_eval() returns every metric keyed by name, each mapping Language to a float."""
    ground_truth_dir, synthetic_dir, output_path = dataset

    results = run_eval(ground_truth_dir, output_path=output_path, synthetic_dir=synthetic_dir)

    assert set(results) == {
        "Accuracy",
        "Rubric agreement",
        "Grounding precision",
        "Unsupported claim rate",
    }
    for scores in results.values():
        assert set(scores) == {Language.AR, Language.EN}
        assert all(isinstance(value, float) for value in scores.values())
    assert _stub_llm.calls == 2


def test_run_eval_scores_a_matching_baseline_perfectly(dataset, _stub_llm):
    """A baseline that agrees with the reference labels scores 1.0 everywhere."""
    ground_truth_dir, synthetic_dir, output_path = dataset

    results = run_eval(ground_truth_dir, output_path=output_path, synthetic_dir=synthetic_dir)

    assert results["Accuracy"][Language.AR] == pytest.approx(1.0)
    assert results["Rubric agreement"][Language.AR] == pytest.approx(1.0)
    assert results["Grounding precision"][Language.AR] == pytest.approx(1.0)


def test_run_eval_writes_the_llm_reference_caveat_above_the_table(dataset, _stub_llm):
    """The LLM-generated-reference warning is written, and written before the table."""
    ground_truth_dir, synthetic_dir, output_path = dataset

    run_eval(ground_truth_dir, output_path=output_path, synthetic_dir=synthetic_dir)
    written = output_path.read_text(encoding="utf-8")

    assert REFERENCE_LABEL_CAVEAT in written
    assert "docs/09-DECISIONS.md" in written
    assert written.index(REFERENCE_LABEL_CAVEAT) < written.index("| Metric | ar | en | mixed |")


def test_run_eval_writes_a_per_language_table(dataset, _stub_llm):
    """Results are written as a per-language table, never as one blended number."""
    ground_truth_dir, synthetic_dir, output_path = dataset

    run_eval(ground_truth_dir, output_path=output_path, synthetic_dir=synthetic_dir)
    written = output_path.read_text(encoding="utf-8")

    assert "| Metric | ar | en | mixed |" in written
    for metric_name in ("Accuracy", "Rubric agreement", "Grounding precision"):
        assert f"| {metric_name} |" in written


def test_run_eval_appends_rather_than_overwriting(dataset, _stub_llm):
    """A second round is appended; the earlier round's history is preserved."""
    ground_truth_dir, synthetic_dir, output_path = dataset
    output_path.write_text("# Eval Results\n\nexisting content\n", encoding="utf-8")

    run_eval(ground_truth_dir, output_path=output_path, synthetic_dir=synthetic_dir)
    run_eval(ground_truth_dir, output_path=output_path, synthetic_dir=synthetic_dir)
    written = output_path.read_text(encoding="utf-8")

    assert "existing content" in written
    assert written.count("| Metric | ar | en | mixed |") == 2
    assert written.count(REFERENCE_LABEL_CAVEAT) == 2


def test_run_eval_reports_extraction_failures_instead_of_hiding_them(dataset, monkeypatch):
    """A call whose extraction fails is excluded from metrics and named in the output."""
    ground_truth_dir, synthetic_dir, output_path = dataset

    def _explode(transcript_path, **kwargs):
        if transcript_path.stem == "call_0001_en":
            raise RuntimeError("provider exhausted")
        return _reference(transcript_path.stem, Language.AR)

    monkeypatch.setattr("sawti.eval.runner.run_plain_extraction", _explode)
    monkeypatch.setattr("sawti.eval.runner.time.sleep", lambda _seconds: None)

    results = run_eval(ground_truth_dir, output_path=output_path, synthetic_dir=synthetic_dir)
    written = output_path.read_text(encoding="utf-8")

    assert Language.EN not in results["Accuracy"]
    assert "Extraction failures: `1`" in written
    assert "call_0001_en" in written


def test_run_eval_rejects_an_empty_reference_set(tmp_path, _stub_llm):
    """An empty ground-truth directory fails loudly rather than scoring nothing as perfect."""
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(ValueError, match="no reference-label records"):
        run_eval(empty, output_path=tmp_path / "out.md", synthetic_dir=tmp_path)


def test_run_eval_uses_a_supplied_extractor_instead_of_the_baseline(dataset, _stub_llm):
    """run_eval() runs whatever extractor it is given — this is what makes phase 2 comparable.

    The phase 1 baseline and the phase 2 graph differ only by this callable;
    reference set, metrics, and reporting stay identical.
    """
    ground_truth_dir, synthetic_dir, output_path = dataset
    seen: list[str] = []

    def _fake_extractor(path):
        seen.append(path.name)
        # Language is irrelevant here; what matters is that run_eval used *this*
        # callable and produced real predictions from it.
        return _reference(path.stem, Language.AR)

    run_eval(
        ground_truth_dir,
        output_path=output_path,
        synthetic_dir=synthetic_dir,
        extractor=_fake_extractor,
        experiment_name="phase 2 test",
    )

    assert len(seen) == 2
    # The baseline extractor must not have been called at all.
    assert _stub_llm.calls == 0
    written = output_path.read_text(encoding="utf-8")
    assert "phase 2 test" in written
    # And the run really scored something — no swallowed extractor error.
    assert "Extraction failures" not in written
