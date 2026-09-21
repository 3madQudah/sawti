"""Tests for `sawti.eval.experiments.baseline`.

Phase 1: mirrors `src/sawti/eval/experiments/baseline.py`.
"""

from __future__ import annotations

from pathlib import Path

from sawti.eval.experiments.baseline import (
    DEFAULT_GROUND_TRUTH_DIR,
    DEFAULT_OUTPUT_PATH,
    run_baseline,
)
from sawti.schemas import Language


def test_run_baseline_defaults_to_the_project_paths():
    """The no-argument defaults point at data/ground_truth and eval_results.md."""
    assert DEFAULT_GROUND_TRUTH_DIR == Path("data/ground_truth")
    assert DEFAULT_OUTPUT_PATH == Path("eval_results.md")


def test_run_baseline_disables_memory_retrieval(monkeypatch, tmp_path):
    """run_baseline() runs the eval pipeline with memory retrieval switched off.

    The baseline path is `sawti.eval.plain_extraction`, which calls the
    provider directly and consults no rule store — so "memory off" is a
    property of which extraction path is used, and this pins that choice.
    """
    import sawti.eval.plain_extraction as plain_extraction

    captured: dict[str, object] = {}

    def _fake_run_eval(ground_truth_path, *, output_path, synthetic_dir):
        captured["ground_truth_path"] = ground_truth_path
        captured["output_path"] = output_path
        captured["synthetic_dir"] = synthetic_dir
        return {"Accuracy": {Language.AR: 1.0}}

    monkeypatch.setattr("sawti.eval.experiments.baseline.run_eval", _fake_run_eval)

    results = run_baseline(tmp_path / "gt", output_path=tmp_path / "out.md")

    assert captured["ground_truth_path"] == tmp_path / "gt"
    assert captured["output_path"] == tmp_path / "out.md"
    assert results == {"Accuracy": {Language.AR: 1.0}}
    # No memory module is imported or consulted by the baseline extraction path.
    assert not hasattr(plain_extraction, "retrieve_rules")


def test_run_baseline_writes_per_category_results(monkeypatch, tmp_path):
    """run_baseline() writes results broken down per language category."""
    monkeypatch.setattr(
        "sawti.eval.experiments.baseline.run_eval",
        lambda gt, *, output_path, synthetic_dir: (
            output_path.write_text(
                "| Metric | ar | en | mixed |\n| --- | --- | --- | --- |\n"
                "| Accuracy | 0.800 | 0.900 | 0.700 |\n",
                encoding="utf-8",
            ),
            {"Accuracy": {Language.AR: 0.8, Language.EN: 0.9, Language.MIXED: 0.7}},
        )[1],
    )

    output_path = tmp_path / "out.md"
    results = run_baseline(tmp_path / "gt", output_path=output_path)

    assert set(results["Accuracy"]) == {Language.AR, Language.EN, Language.MIXED}
    assert "| Metric | ar | en | mixed |" in output_path.read_text(encoding="utf-8")
