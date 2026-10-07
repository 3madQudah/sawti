"""Tests for the phase 6.3 Gemini runners: `run_cloud_arm.py` and `run_memory_comparison.py`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import run_cloud_arm
import run_memory_comparison as memory

from sawti.config import Settings
from sawti.eval.serving_benchmark import CallInput
from sawti.schemas import Language

RULES = [
    {"id": "r-callback", "text": "callback times", "source_call_ids": ["call_0001_ar"]},
    {"id": "r-refund", "text": "refund deadlines", "source_call_ids": ["call_0002_ar", "call_0003_mixed"]},
    {"id": "r-greeting", "text": "greeting name", "source_call_ids": []},
]


def test_leave_one_out_never_returns_a_rule_learned_from_the_same_call() -> None:
    assert [r["id"] for r in memory.leave_one_out(RULES, "call_0001_ar")] == ["r-refund", "r-greeting"]
    assert [r["id"] for r in memory.leave_one_out(RULES, "call_0003_mixed")] == ["r-callback", "r-greeting"]
    assert [r["id"] for r in memory.leave_one_out(RULES, "call_0099_en", top_k=2)] == [
        "r-callback",
        "r-refund",
    ]


def test_retrieval_ranks_by_similarity_then_applies_leave_one_out() -> None:
    def embed(text: str) -> np.ndarray[Any, Any]:
        return (
            np.array([text.count("refund"), text.count("callback"), text.count("greeting")], dtype=float)
            + 1e-9
        )

    rules_for = memory.make_rules_for(RULES, embed=embed)
    call = CallInput(
        call_id="call_0002_ar", language=Language.AR, redacted_transcript="refund refund callback"
    )
    texts, ids = rules_for(call)
    assert ids[0] == "r-callback"  # r-refund is the best match but came from this very call
    assert "r-refund" not in ids and texts[0] == "callback times"


def test_provenance_fingerprints_the_key_and_never_contains_it(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "AIza-not-a-real-key-123456"
    settings = Settings(_env_file=None, GEMINI_API_KEY=secret)  # type: ignore[call-arg]
    monkeypatch.setattr(run_cloud_arm, "get_settings", lambda: settings)
    note = run_cloud_arm.provenance("gemini")
    assert secret not in str(note)
    assert len(note["api_key_sha256_12"]) == 12 and note["configured_model"] == settings.gemini_model


async def test_run_executes_both_arms_and_both_repeats(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    """Exercises `run()` end to end (a missing parameter in it once slipped past the helper tests)."""
    import json

    from sawti.eval.serving_benchmark import load_checkpoint
    from sawti.llm.fake_provider import FakeProvider

    snapshot = tmp_path / "rules.json"
    snapshot.write_text(json.dumps({"rules": RULES}))
    monkeypatch.setattr(memory, "SNAPSHOT", snapshot)
    monkeypatch.setattr(
        memory,
        "all_calls",
        lambda: memory.load_call_inputs(
            {"ar": ["call_0001_ar"], "en": ["call_0006_en"]}, memory.SYNTHETIC_DIR
        ),
    )
    real = memory.make_rules_for
    monkeypatch.setattr(
        memory, "make_rules_for", lambda rules: real(rules, embed=lambda t: np.array([len(t), 1.0]))
    )
    checkpoint = tmp_path / "runs.jsonl"
    await memory.run(
        FakeProvider(),
        repeats=2,
        min_interval=0.0,
        note={"provider": "fake"},
        checkpoint=checkpoint,
        concurrency=2,
    )

    records, segments = load_checkpoint(checkpoint)
    assert sorted({(r.arm, r.repeat) for r in records}) == [
        ("memory_off", 0),
        ("memory_off", 1),
        ("memory_on_35_loo", 0),
        ("memory_on_35_loo", 1),
    ]
    on = [r for r in records if r.arm == "memory_on_35_loo"]
    assert all("r-callback" not in r.retrieved_rule_ids for r in on if r.call_id == "call_0001_ar")
    assert all(not r.retrieved_rule_ids for r in records if r.arm == "memory_off")
    assert {s.note["provider"] for s in segments} == {"fake"}
