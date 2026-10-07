"""Tests for `scripts/assemble_serving_comparison.py` (phase 6.3)."""

from __future__ import annotations

import json
from pathlib import Path

import assemble_serving_comparison as assemble
import build_kaggle_bundles
import pytest

from sawti.eval import kaggle_benchmark as kb
from sawti.eval.serving_benchmark import load_checkpoint
from sawti.llm.fake_provider import FakeProvider


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A Kaggle results file and a Gemini checkpoint, from the real drivers with the fake provider."""
    root = tmp_path_factory.mktemp("cmp")
    build_kaggle_bundles.build_calls(root, "u")
    calls = root / "sawti-comparison-calls"
    fake = {"vllm": lambda args: FakeProvider(), "hf": lambda args: FakeProvider()}
    ck = root / "runs.jsonl"
    for arm, extra in (
        ("vllm_adapter_c16", ["--concurrency", "16"]),
        ("vllm_base_c16", ["--concurrency", "16"]),
    ):
        kb.main(
            [
                "vllm",
                "--arm",
                arm,
                "--model",
                "m",
                "--calls-dir",
                str(calls),
                "--checkpoint",
                str(ck),
                *extra,
            ],
            providers=fake,
        )
    kaggle = root / "results.json"
    kb.main(["assemble", "--checkpoint", str(ck), "--out", str(kaggle)])
    gemini = root / "gemini.jsonl"
    kb.main(
        ["vllm", "--arm", "gemini", "--model", "m", "--calls-dir", str(calls), "--checkpoint", str(gemini)],
        providers=fake,
    )
    # Pretend these came from the real API: Gemini reports its resolved model.
    lines = [json.loads(line) for line in gemini.read_text().splitlines()]
    for line in lines:
        if line["type"] == "record":
            line["model"] = "gemini-3.5-flash-lite"
    gemini.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return kaggle, gemini


def test_every_arm_is_scored_per_language_with_hardware_and_cost(results: tuple[Path, Path]) -> None:
    kaggle, gemini = results
    out = assemble.build(kaggle, gemini)
    assert list(out["arms"]) == ["gemini", "vllm_adapter_c16", "vllm_base_c16"]
    for arm in out["arms"].values():
        assert set(arm["per_language"]) == {"ar", "en", "mixed"}
        assert arm["hardware"]
        for row in arm["per_language"].values():
            assert row["n"] == 15 and row["cost_per_1000_usd"] is not None
            assert row["third_party_transcript_bytes_per_call"] > 0
    assert out["arms"]["vllm_adapter_c16"]["hardware"].startswith("free Kaggle T4x2")
    assert out["pricing"]["actual_spend_usd"] == 0


def test_gemini_cost_uses_the_paid_token_price_of_the_resolved_model() -> None:
    row = {"prompt_tokens_mean": 1000.0, "completion_tokens_mean": 2000.0}
    # 1000 * 0.30/1M + 2000 * 2.50/1M = 0.0053 per call -> 5.3 per 1000
    assert assemble.gemini_cost_per_1000(row, "gemini-3.5-flash-lite") == pytest.approx(5.3)
    assert assemble.gemini_cost_per_1000(row, "some-unpriced-model") is None


def test_gpu_cost_allocates_wall_time_by_output_tokens() -> None:
    # 3600 s / 360000 output tokens = 0.01 s/token; a 2000-token call = 20 s; 1000 calls = 5.556 h
    cost = assemble.gpu_cost_per_1000({"completion_tokens_mean": 2000.0}, {"seconds": 3600.0}, 360_000, 0.845)
    assert cost == pytest.approx(0.845 * 20000 / 3600, rel=1e-3)


def test_markdown_has_one_table_per_language_and_never_a_blended_row(results: tuple[Path, Path]) -> None:
    text = assemble.to_markdown(assemble.build(*results))
    assert [line for line in text.splitlines() if line.startswith("### ")] == [
        "### ar",
        "### en",
        "### mixed",
    ]
    assert "overall" not in text.lower() and "all languages" not in text.lower()
    assert load_checkpoint(results[1])[0]
