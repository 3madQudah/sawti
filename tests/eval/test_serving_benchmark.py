"""Tests for `sawti.eval.serving_benchmark` (phase 6.3).

The fake provider stands in for every real one; what is under test is the
measuring, the classification, the checkpoint/resume behavior and that the
service's own pipeline and metrics are what run.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from sawti.data.ground_truth import load_ground_truth
from sawti.eval import serving_benchmark as sb
from sawti.llm.fake_provider import FakeProvider
from sawti.llm.provider import LLMProvider, TransientProviderError
from sawti.privacy.redaction import redact
from sawti.schemas import Language

SYNTHETIC = Path("data/synthetic")
GROUND_TRUTH = Path("data/ground_truth")


def _call(call_id: str) -> sb.CallInput:
    raw = (SYNTHETIC / f"{call_id}.txt").read_text(encoding="utf-8")
    return sb.CallInput(
        call_id=call_id,
        language=Language(call_id.rsplit("_", 1)[1]),
        redacted_transcript=redact(raw).redacted_text,
    )


class _Raising(LLMProvider):
    """Raises the queued exceptions in order, then behaves like the fake provider."""

    def __init__(self, *errors: BaseException) -> None:
        self.errors = list(errors)
        self.calls = 0

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError

    async def structured_complete(self, prompt: str, *, response_model: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return await FakeProvider().structured_complete(prompt, response_model=response_model, **kwargs)


def _validation_error() -> ValidationError:
    class _M(BaseModel):
        x: int

    try:
        _M.model_validate({"x": "not a number"})
    except ValidationError as exc:
        return exc
    raise AssertionError


async def test_a_good_call_records_prediction_tokens_bytes_and_latency() -> None:
    call = _call("call_0006_en")
    record = await sb.run_call(FakeProvider(), call, arm="fake")
    assert record.outcome == "ok" and record.schema_valid
    assert record.prediction is not None and record.prediction["call_id"] == "call_0006_en"
    assert record.prompt_tokens and record.completion_tokens and record.model == "fake"
    assert record.transcript_bytes_sent == len(call.redacted_transcript.encode("utf-8"))
    assert record.request_bytes_sent > record.transcript_bytes_sent  # plus the system prompt
    assert record.latency_s >= 0 and record.llm_requests == 1


async def test_schema_invalid_and_other_failures_are_told_apart_and_carry_no_prediction() -> None:
    invalid = await sb.run_call(_Raising(_validation_error()), _call("call_0006_en"), arm="x")
    broken = await sb.run_call(_Raising(RuntimeError("boom")), _call("call_0006_en"), arm="x")
    assert (invalid.outcome, invalid.prediction) == ("schema_invalid", None)
    assert (broken.outcome, broken.prediction) == ("error", None)
    assert "boom" in (broken.error or "")


async def test_run_arm_checkpoints_and_resumes_without_redoing_calls(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run.jsonl"
    calls = [_call("call_0000_ar"), _call("call_0006_en"), _call("call_0004_mixed")]
    provider = _Raising()
    first = await sb.run_arm(provider, calls[:2], arm="a", checkpoint=checkpoint, concurrency=2)
    second = await sb.run_arm(provider, calls, arm="a", checkpoint=checkpoint, concurrency=2)
    assert [r.call_id for r in second] == ["call_0004_mixed"]
    assert provider.calls == 3
    records, segments = sb.load_checkpoint(checkpoint)
    assert sorted(r.call_id for r in records) == sorted(c.call_id for c in calls)
    assert len(first) == 2 and len(segments) == 2
    assert sb.throughput(records, segments, arm="a")["calls"] == 3


async def test_a_torn_last_line_is_ignored_on_resume(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run.jsonl"
    await sb.run_arm(_Raising(), [_call("call_0006_en")], arm="a", checkpoint=checkpoint)
    with checkpoint.open("a") as handle:
        handle.write('{"type": "record", "arm": "a", "call_')  # killed mid-write
    records, _ = sb.load_checkpoint(checkpoint)
    assert [r.call_id for r in records] == ["call_0006_en"]


async def test_transient_errors_back_off_and_retry(tmp_path: Path) -> None:
    provider = _Raising(TransientProviderError("429"))
    [record] = await sb.run_arm(
        provider, [_call("call_0006_en")], arm="a", checkpoint=tmp_path / "r.jsonl", retry_wait_s=0
    )
    assert record.outcome == "ok" and provider.calls == 2


async def test_the_daily_quota_stops_cleanly_and_keeps_what_finished(tmp_path: Path) -> None:
    checkpoint = tmp_path / "r.jsonl"
    quota = TransientProviderError("429 GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    provider = _Raising()
    await sb.run_arm(provider, [_call("call_0006_en")], arm="a", checkpoint=checkpoint)
    provider.errors = [quota]
    with pytest.raises(sb.QuotaExhaustedError):
        await sb.run_arm(
            provider, [_call("call_0006_en"), _call("call_0000_ar")], arm="a", checkpoint=checkpoint
        )
    records, _ = sb.load_checkpoint(checkpoint)
    assert [r.call_id for r in records] == ["call_0006_en"]


async def test_retrieved_rules_reach_the_prompt_and_are_recorded() -> None:
    seen: dict[str, Any] = {}

    class _Spy(FakeProvider):
        async def structured_complete(self, prompt: str, *, response_model: Any, **kwargs: Any) -> Any:
            seen["system"] = kwargs.get("system")
            return await super().structured_complete(prompt, response_model=response_model, **kwargs)

    record = await sb.run_call(
        _Spy(),
        _call("call_0006_en"),
        arm="m",
        rules_for=lambda call: (["Always quote callback times."], ["r-1"]),
    )
    assert record.retrieved_rule_ids == ["r-1"]
    assert "Always quote callback times." in seen["system"]


async def test_concurrent_calls_keep_their_own_tokens(tmp_path: Path) -> None:
    calls = [_call(c) for c in ("call_0000_ar", "call_0006_en", "call_0004_mixed", "call_0001_ar")]
    records = await sb.run_arm(
        FakeProvider(latency_seconds=0.01), calls, arm="c", checkpoint=tmp_path / "c.jsonl", concurrency=16
    )
    solo = {c.call_id: (await sb.run_call(FakeProvider(), c, arm="s")).prompt_tokens for c in calls}
    assert {r.call_id: r.prompt_tokens for r in records} == solo


async def test_scoring_reports_each_language_with_the_project_metrics(tmp_path: Path) -> None:
    calls = [_call("call_0000_ar"), _call("call_0006_en"), _call("call_0004_mixed")]
    records = [await sb.run_call(FakeProvider(), c, arm="f") for c in calls]
    records.append(await sb.run_call(_Raising(_validation_error()), _call("call_0001_ar"), arm="f"))
    scores = sb.score_records(records, load_ground_truth(GROUND_TRUTH), transcript_dir=SYNTHETIC)
    assert set(scores) == {"ar", "en", "mixed"}
    assert (scores["ar"]["n"], scores["ar"]["schema_valid"], scores["ar"]["schema_valid_rate"]) == (2, 1, 0.5)
    for language in scores.values():
        for key in ("accuracy", "rubric_agreement", "grounding_precision", "unsupported_claim_rate"):
            assert key in language
        assert language["latency_p50_s"] is not None


def test_daily_quota_and_schema_failures_are_recognized() -> None:
    assert sb.is_daily_quota(RuntimeError("limit GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
    assert not sb.is_daily_quota(RuntimeError("429 per minute"))
    assert sb.is_schema_failure(_validation_error())
    assert not sb.is_schema_failure(RuntimeError("x"))


def test_run_arm_is_a_no_op_when_everything_is_done(tmp_path: Path) -> None:
    checkpoint = tmp_path / "r.jsonl"
    asyncio.run(sb.run_arm(FakeProvider(), [_call("call_0006_en")], arm="a", checkpoint=checkpoint))
    assert (
        asyncio.run(sb.run_arm(FakeProvider(), [_call("call_0006_en")], arm="a", checkpoint=checkpoint)) == []
    )


def test_load_call_inputs_interleaves_languages_and_redacts(tmp_path: Path) -> None:
    (tmp_path / "a_ar.txt").write_text("Agent: رقمي 0791234567\n", encoding="utf-8")
    (tmp_path / "b_ar.txt").write_text("Agent: hi\n", encoding="utf-8")
    (tmp_path / "c_en.txt").write_text("Agent: [PHONE] already redacted\n", encoding="utf-8")
    inputs = sb.load_call_inputs({"ar": ["a_ar", "b_ar"], "en": ["c_en"]}, tmp_path)
    assert [(i.call_id, i.language.value) for i in inputs] == [("a_ar", "ar"), ("c_en", "en"), ("b_ar", "ar")]
    assert "0791234567" not in inputs[0].redacted_transcript
    assert inputs[1].redacted_transcript == "Agent: [PHONE] already redacted\n"  # idempotent


async def test_nothing_new_starts_after_the_deadline(tmp_path: Path) -> None:
    import time

    records = await sb.run_arm(
        FakeProvider(),
        [_call("call_0006_en")],
        arm="a",
        checkpoint=tmp_path / "r.jsonl",
        deadline_epoch=time.time() - 1,
    )
    assert records == []
    assert sb.load_checkpoint(tmp_path / "r.jsonl")[0] == []
