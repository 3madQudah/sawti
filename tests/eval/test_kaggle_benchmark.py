"""Tests for `sawti.eval.kaggle_benchmark`, the Kaggle notebook's driver (phase 6.3).

Run against the real bundle layout (built by `scripts/build_kaggle_bundles.py`)
with the fake provider standing in for vLLM / HF, so the exact commands the
notebook issues are exercised here, minus the GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from sawti.eval import kaggle_benchmark as kb
from sawti.llm.fake_provider import FakeProvider

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_kaggle_bundles


@pytest.fixture(scope="module")
def calls_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("kaggle")
    build_kaggle_bundles.build_calls(out, "someone")
    return out / "sawti-comparison-calls"


def test_limit_keeps_the_subset_balanced_across_languages(calls_dir: Path) -> None:
    subset = kb.load_calls(calls_dir, limit=15)
    assert [c.language.value for c in subset].count("ar") == 5
    assert {c.language.value for c in subset} == {"ar", "en", "mixed"}
    assert len(kb.load_calls(calls_dir)) == 45


def test_the_notebooks_commands_run_arms_and_assemble_one_results_file(
    calls_dir: Path, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "runs.jsonl"
    fake = {"vllm": lambda args: FakeProvider(), "hf": lambda args: FakeProvider()}
    common = ["--calls-dir", str(calls_dir), "--checkpoint", str(checkpoint)]
    assert (
        kb.main(
            ["vllm", "--arm", "vllm_adapter_c1", "--model", "sawti-qlora", "--limit", "6", *common],
            providers=fake,
        )
        == 0
    )
    assert (
        kb.main(
            ["vllm", "--arm", "vllm_adapter_c16", "--model", "sawti-qlora", "--concurrency", "16", *common],
            providers=fake,
        )
        == 0
    )
    assert (
        kb.main(["hf", "--arm", "hf_adapter_c1", "--adapter", "/x", "--limit", "3", *common], providers=fake)
        == 0
    )

    feasibility = tmp_path / "feasibility.json"
    feasibility.write_text(json.dumps({"vllm_version": "0.18.1", "ok": True}))
    out = tmp_path / "results.json"
    summary = kb.main(
        ["assemble", "--checkpoint", str(checkpoint), "--out", str(out), "--meta", str(feasibility)]
    )
    assert summary == 0
    results = json.loads(out.read_text())
    assert {r["arm"] for r in results["records"]} == {"vllm_adapter_c1", "vllm_adapter_c16", "hf_adapter_c1"}
    assert sum(r["arm"] == "vllm_adapter_c16" for r in results["records"]) == 45
    assert results["meta"]["feasibility"]["vllm_version"] == "0.18.1"
    assert results["meta"]["hardware"] == "free Kaggle T4x2"
    notes = {s["arm"]: s["note"] for s in results["segments"]}
    assert notes["hf_adapter_c1"] == {
        "engine": "hf-transformers",
        "hardware": "free Kaggle T4x2",
        "model": "Qwen/Qwen3-8B",
        "adapter": "on",
    }
    assert all(r["prediction"] for r in results["records"])  # re-scorable later


def test_rerunning_an_arm_only_runs_what_is_missing(calls_dir: Path, tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs.jsonl"
    fake = {"vllm": lambda args: FakeProvider(), "hf": lambda args: FakeProvider()}
    args = [
        "vllm",
        "--arm",
        "a",
        "--model",
        "m",
        "--calls-dir",
        str(calls_dir),
        "--checkpoint",
        str(checkpoint),
    ]
    kb.main([*args, "--limit", "3"], providers=fake)
    kb.main(args, providers=fake)
    from sawti.eval.serving_benchmark import load_checkpoint

    records, _ = load_checkpoint(checkpoint)
    assert len(records) == 45 and len({r.call_id for r in records}) == 45
