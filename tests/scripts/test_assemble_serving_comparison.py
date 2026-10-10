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
    headings = [line for line in text.splitlines() if line.startswith("### ")]
    assert headings[:3] == ["### ar", "### en", "### mixed"]
    assert "### Failures" in headings
    assert "overall" not in text.lower() and "all languages" not in text.lower()
    assert load_checkpoint(results[1])[0]


def test_a_sleep_inflated_segment_falls_back_to_summed_latency_and_says_so() -> None:
    from datetime import UTC, datetime, timedelta

    from sawti.eval.serving_benchmark import CallRecord, Segment
    from sawti.schemas import Language

    start = datetime(2026, 10, 7, tzinfo=UTC)
    rows = [
        CallRecord(
            arm="g",
            concurrency=1,
            call_id=f"c{i}",
            language=Language.EN,
            started_at=start,
            latency_s=60.0,
            outcome="ok",
            completion_tokens=600,
        )
        for i in range(3)
    ]
    asleep = [
        Segment(
            arm="g", concurrency=1, started_at=start, finished_at=start + timedelta(days=2), calls_completed=3
        )
    ]
    tp = assemble.arm_throughput(rows, asleep, "g")
    assert tp["calls_per_min"] == 1.0 and "host sleep" in tp["method"]
    awake = [
        Segment(
            arm="g",
            concurrency=1,
            started_at=start,
            finished_at=start + timedelta(seconds=185),
            calls_completed=3,
        )
    ]
    assert assemble.arm_throughput(rows, awake, "g")["method"].startswith("segment wall time")


def test_failures_report_inferred_length_and_the_base_model_on_the_same_call() -> None:
    from datetime import UTC, datetime

    from sawti.eval.serving_benchmark import CallRecord
    from sawti.schemas import Language

    now = datetime.now(UTC)
    bad = CallRecord(
        arm="vllm_adapter_c16",
        concurrency=16,
        call_id="call_0100_ar",
        language=Language.AR,
        started_at=now,
        latency_s=1.0,
        outcome="schema_invalid",
        completion_tokens=4096,
        error="Invalid JSON: EOF while parsing an object at line 27181 column 0",
    )
    base = CallRecord(
        arm="vllm_base_c16",
        concurrency=16,
        call_id="call_0100_ar",
        language=Language.AR,
        started_at=now,
        latency_s=1.0,
        outcome="ok",
        completion_tokens=3051,
    )
    recorded = bad.model_copy(update={"call_id": "x", "finish_reasons": ["length"]})
    [first, second] = assemble.failure_details([bad, base, recorded])
    assert first["finish_reason"].startswith("length (inferred") and first["json_broke_off_at_line"] == 27181
    assert first["base_model_on_same_call"] == "ok (3051 tokens)"
    assert second["finish_reason"] == "length" and second["base_model_on_same_call"] == "not run"


def test_a_call_stretched_by_host_sleep_is_left_out_of_latency_and_throughput_and_named() -> None:
    from datetime import UTC, datetime, timedelta

    from sawti.eval.serving_benchmark import CallRecord, Segment
    from sawti.schemas import Language

    start = datetime(2026, 10, 7, tzinfo=UTC)
    rows = [
        CallRecord(
            arm="g",
            concurrency=1,
            call_id=f"c{i}",
            language=Language.EN,
            started_at=start,
            latency_s=30.0,
            outcome="ok",
            completion_tokens=300,
        )
        for i in range(4)
    ]
    rows.append(rows[0].model_copy(update={"call_id": "slept", "latency_s": 9783.0}))
    seg = [
        Segment(
            arm="g", concurrency=1, started_at=start, finished_at=start + timedelta(days=2), calls_completed=5
        )
    ]
    tp = assemble.arm_throughput(rows, seg, "g")
    assert tp["excluded_as_host_sleep"] == ["slept"]
    assert tp["calls"] == 4 and tp["calls_per_min"] == 2.0
    assert "slept" in tp["method"]
