"""Tests for `sawti.eval.experiments.five_batch`.

Phase 4: mirrors `src/sawti/eval/experiments/five_batch.py`.

Integration test against a real Postgres instance — the memory-on arm's
correction capture goes through `sawti.memory.capture.capture_correction()`.
Run `docker compose up -d` first; see `docs/09-DECISIONS.md`. The LLM
provider and the embedder are both faked throughout — no real model calls,
and no assertion here about whether memory actually helps (that is an
empirical question for a real run, not something a fake-extractor test
should assert).

IMPORTANT — every test passes its own unique `reviewer_id`/`agent_external_id`
to `run_five_batch()`, never the module's production defaults
(`REVIEWER_ID`/`_PLACEHOLDER_AGENT_EXTERNAL_ID`). A real experiment run and
this test suite share a database; a cleanup fixture that matched on the
production identity once deleted a real, concurrently-running experiment's
`Agent`/`Call` rows out from under it (see `docs/09-DECISIONS.md`). Using a
fresh identity per test makes that class of bug structurally impossible
here, not just something to remember to avoid.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sawti.db.models import Agent, Call, CallAnalysisRecord, ReviewerAction
from sawti.db.session import get_session
from sawti.eval.experiments.five_batch import (
    BatchScores,
    FiveBatchResult,
    _Checkpoint,
    _record_rule_outcomes,
    _render_report,
    _save_checkpoint,
    run_five_batch,
)
from sawti.memory.capture import capture_correction, get_or_create_placeholder_agent, persist_analysis_record
from sawti.memory.rule_schema import MemoryRule
from sawti.memory.store import MemoryStore
from sawti.schemas import CallAnalysis, Commitment, Language, Quote


def _quote(text: str) -> Quote:
    return Quote(text=text, speaker="Agent", start_char=0, end_char=len(text))


def _ground_truth(call_id: str, language: Language) -> CallAnalysis:
    """Ground truth: a commitment WITH a deadline."""
    return CallAnalysis(
        call_id=call_id,
        language=language,
        summary="Customer requested a refund.",
        commitments=[
            Commitment(
                evidence=_quote("we will refund you by Friday"),
                promised_by="Agent",
                description="Refund",
                deadline=datetime.now(UTC),
            )
        ],
        confidence=0.9,
        requires_human_review=False,
    )


def _agent_output(call_id: str, language: Language) -> CallAnalysis:
    """Deliberately differs from ground truth (no deadline), so first_difference always fires."""
    return CallAnalysis(
        call_id=call_id,
        language=language,
        summary="Customer requested a refund.",
        commitments=[
            Commitment(
                evidence=_quote("we will refund you"),
                promised_by="Agent",
                description="Refund",
                deadline=None,
            )
        ],
        confidence=0.9,
        requires_human_review=False,
    )


_VOCAB = ["refund", "deadline", "unrelated"]


def _fake_embed(text: str) -> np.ndarray:
    words = set(text.lower().split())
    return np.array([1.0 if word in words else 0.0 for word in _VOCAB])


class _UniversalFakeProvider:
    """Handles the induction, conflict, and consolidation structured_complete calls with canned responses."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        raise AssertionError("unexpected complete() call")

    async def structured_complete(
        self, prompt: str, *, response_model: Any, system: str | None = None, **kwargs: Any
    ) -> Any:
        self.prompts.append(prompt)
        field_names = set(response_model.model_fields)
        if "contradicts" in field_names:
            return response_model(contradicts=False, reasoning="fake: never conflicts")
        if "rule_text" in field_names:
            return response_model(rule_text="Flag commitments without a stated deadline as incomplete.")
        raise AssertionError(f"unexpected response_model: {response_model}")


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch: pytest.MonkeyPatch) -> _UniversalFakeProvider:
    """Every sawti.memory module that calls an LLM gets the same fake provider."""
    provider = _UniversalFakeProvider()
    monkeypatch.setattr("sawti.memory.induction.get_llm_provider", lambda: provider)
    monkeypatch.setattr("sawti.memory.conflict.get_llm_provider", lambda: provider)
    monkeypatch.setattr("sawti.memory.consolidate.get_llm_provider", lambda: provider)
    return provider


@pytest.fixture
def _ground_truth_dir(tmp_path: Path) -> tuple[Path, Path]:
    """A small ground-truth/synthetic pair: 2 calls per language, 6 total."""
    ground_truth_dir = tmp_path / "ground_truth"
    synthetic_dir = tmp_path / "synthetic"
    ground_truth_dir.mkdir()
    synthetic_dir.mkdir()

    index = 0
    for language in (Language.AR, Language.EN, Language.MIXED):
        for _ in range(2):
            call_id = f"call_{index:04d}_{language.value}"
            record = _ground_truth(call_id, language)
            (ground_truth_dir / f"{call_id}.json").write_text(record.model_dump_json(), encoding="utf-8")
            (synthetic_dir / f"{call_id}.txt").write_text(
                "Agent: we will refund you by Friday.\n", encoding="utf-8"
            )
            index += 1
    return ground_truth_dir, synthetic_dir


def _fake_extractor_factory() -> tuple[Any, list[tuple[str, list[str] | None]]]:
    """A fake extractor recording every (call_id, retrieved_rules) it was called with."""
    calls: list[tuple[str, list[str] | None]] = []

    def fake_extractor(path: Path, retrieved_rules: list[str] | None) -> CallAnalysis:
        call_id = path.stem
        language = Language(call_id.rsplit("_", 1)[-1])
        calls.append((call_id, retrieved_rules))
        return _agent_output(call_id, language)

    return fake_extractor, calls


@pytest.fixture
def _test_identity() -> Any:
    """A unique reviewer_id/agent external_id for this test alone, and cleanup scoped to only it.

    Never the module's production `REVIEWER_ID`/`_PLACEHOLDER_AGENT_EXTERNAL_ID`
    — see the module docstring for why that distinction is load-bearing, not
    cosmetic.
    """
    identity = f"test-five-batch-{uuid.uuid4()}"
    yield identity
    with get_session() as session:
        for action in session.query(ReviewerAction).filter_by(reviewer_id=identity):
            session.delete(action)
        agent = session.query(Agent).filter_by(external_id=identity).one_or_none()
        if agent is not None:
            for call in session.query(Call).filter_by(agent_id=agent.id):
                for record in session.query(CallAnalysisRecord).filter_by(call_id=call.id):
                    session.delete(record)
                session.delete(call)
            session.delete(agent)


def test_run_five_batch_compares_memory_on_vs_off_per_batch(
    _ground_truth_dir: tuple[Path, Path], _test_identity: str
) -> None:
    """run_five_batch() runs each of the five batches with memory both enabled and disabled."""
    ground_truth_dir, synthetic_dir = _ground_truth_dir
    fake_extractor, calls = _fake_extractor_factory()
    store = MemoryStore(embed=_fake_embed)

    result = run_five_batch(
        ground_truth_dir,
        output_path=ground_truth_dir.parent / "results.md",
        synthetic_dir=synthetic_dir,
        sleep_seconds=0.0,
        extractor=fake_extractor,
        store=store,
        embed=_fake_embed,
        reviewer_id=_test_identity,
        agent_external_id=_test_identity,
    )

    assert result.n_batches == 5
    assert len(result.memory) == 5
    assert len(result.control) == 5

    # retrieve_fn is only wired for the memory arm — its calls carry a
    # (possibly empty) list; the control arm's carry None.
    memory_calls = [call for call in calls if call[1] is not None]
    control_calls = [call for call in calls if call[1] is None]
    assert len(memory_calls) == 6
    assert len(control_calls) == 6
    assert sum(result.corrections_captured) == 6


def test_run_five_batch_reports_deltas_per_language_category(
    _ground_truth_dir: tuple[Path, Path], _test_identity: str
) -> None:
    """run_five_batch() never collapses the memory-vs-no-memory delta across languages."""
    ground_truth_dir, synthetic_dir = _ground_truth_dir
    fake_extractor, _calls = _fake_extractor_factory()
    store = MemoryStore(embed=_fake_embed)

    result = run_five_batch(
        ground_truth_dir,
        output_path=ground_truth_dir.parent / "results.md",
        synthetic_dir=synthetic_dir,
        sleep_seconds=0.0,
        extractor=fake_extractor,
        store=store,
        embed=_fake_embed,
        reviewer_id=_test_identity,
        agent_external_id=_test_identity,
    )

    for batch_scores in (*result.memory, *result.control):
        assert isinstance(batch_scores.accuracy, dict)

    report = _render_report(result)
    for language in (Language.AR, Language.EN, Language.MIXED):
        assert f"### {language.value} — Accuracy per batch" in report
    assert "overall" not in report.lower()


def test_run_five_batch_retries_transient_extraction_failures(
    _ground_truth_dir: tuple[Path, Path], _test_identity: str
) -> None:
    """A transient extraction failure is retried rather than immediately excluded.

    This is the fix for a real issue hit running the actual experiment:
    Gemini's free tier returned 503s at a ~40% rate under load, which
    without a retry would have permanently dropped a large, non-random
    slice of the 150-call ground truth from the result.
    """
    ground_truth_dir, synthetic_dir = _ground_truth_dir
    attempt_counts: dict[str, int] = {}

    def flaky_extractor(path: Path, retrieved_rules: list[str] | None) -> CallAnalysis:
        call_id = path.stem
        language = Language(call_id.rsplit("_", 1)[-1])
        attempt_counts[call_id] = attempt_counts.get(call_id, 0) + 1
        if attempt_counts[call_id] < 2:
            raise RuntimeError("transient provider error")
        return _agent_output(call_id, language)

    store = MemoryStore(embed=_fake_embed)

    result = run_five_batch(
        ground_truth_dir,
        output_path=ground_truth_dir.parent / "results.md",
        synthetic_dir=synthetic_dir,
        sleep_seconds=0.0,
        max_retries=2,
        retry_initial_delay=0.0,
        extractor=flaky_extractor,
        store=store,
        embed=_fake_embed,
        reviewer_id=_test_identity,
        agent_external_id=_test_identity,
    )

    # Every call needed at least one retry (fails once per call_id, the first
    # time it's seen; attempt_counts is shared across both arms, so a
    # call_id's count keeps climbing once it has already succeeded once).
    assert attempt_counts
    assert all(count >= 2 for count in attempt_counts.values())
    # Nothing was permanently lost to the transient failure.
    assert sum(result.corrections_captured) == 6


def test_run_five_batch_retries_transient_induction_failures(
    _ground_truth_dir: tuple[Path, Path], _test_identity: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient failure inside induce_rule/insert_rule is retried, not left to crash the run.

    Regression test for a real incident: a hard 429 RESOURCE_EXHAUSTED
    inside an unretried induce_rule() call crashed the whole process,
    even though extraction itself was already retried. See
    docs/09-DECISIONS.md. Overrides the autouse `_patch_llm` fixture's
    provider with a flaky one, for this test only.
    """
    ground_truth_dir, synthetic_dir = _ground_truth_dir
    fake_extractor, _calls = _fake_extractor_factory()
    call_counts: dict[str, int] = {}

    class _FlakyInductionProvider(_UniversalFakeProvider):
        async def structured_complete(
            self, prompt: str, *, response_model: Any, system: str | None = None, **kwargs: Any
        ) -> Any:
            field_names = set(response_model.model_fields)
            kind = "contradicts" if "contradicts" in field_names else "rule_text"
            call_counts[kind] = call_counts.get(kind, 0) + 1
            if call_counts[kind] == 1:
                raise RuntimeError("429 RESOURCE_EXHAUSTED (simulated)")
            return await super().structured_complete(prompt, response_model=response_model, **kwargs)

    provider = _FlakyInductionProvider()
    monkeypatch.setattr("sawti.memory.induction.get_llm_provider", lambda: provider)
    monkeypatch.setattr("sawti.memory.conflict.get_llm_provider", lambda: provider)
    monkeypatch.setattr("sawti.memory.consolidate.get_llm_provider", lambda: provider)

    store = MemoryStore(embed=_fake_embed)
    result = run_five_batch(
        ground_truth_dir,
        output_path=ground_truth_dir.parent / "results.md",
        synthetic_dir=synthetic_dir,
        sleep_seconds=0.0,
        max_retries=2,
        retry_initial_delay=0.0,
        extractor=fake_extractor,
        store=store,
        embed=_fake_embed,
        reviewer_id=_test_identity,
        agent_external_id=_test_identity,
    )

    # The run completed at all despite the injected transient failure, and
    # nothing was permanently lost to it.
    assert sum(result.corrections_captured) == 6


def test_run_five_batch_resumes_from_checkpoint(
    _ground_truth_dir: tuple[Path, Path], _test_identity: str, tmp_path: Path
) -> None:
    """A pre-existing checkpoint is resumed from: already-completed batches are not re-run.

    Regression test for a real incident: a run lost 3 already-completed
    batches to a provider's daily quota exhausted mid-batch-4, with no way
    to resume short of starting over. See docs/09-DECISIONS.md. Rather than
    simulating an actual mid-run crash, this hand-writes a checkpoint
    representing "batch 1 of 2 already done" and confirms `run_five_batch()`
    picks up at batch 2 instead of re-extracting batch 1.
    """
    ground_truth_dir, synthetic_dir = _ground_truth_dir
    checkpoint_path = tmp_path / "checkpoint.json"

    restored_batch_scores = BatchScores(
        n_calls={Language.AR: 1, Language.EN: 1, Language.MIXED: 1},
        accuracy={Language.AR: 1.0, Language.EN: 1.0, Language.MIXED: 1.0},
        rubric_agreement={},
        grounding_precision={Language.AR: 1.0, Language.EN: 1.0, Language.MIXED: 1.0},
        unsupported_claim_rate={Language.AR: 0.0, Language.EN: 0.0, Language.MIXED: 0.0},
    )
    pre_existing_rule = MemoryRule(rule_text="a pre-existing rule", created_at=datetime.now(UTC))
    _save_checkpoint(
        checkpoint_path,
        _Checkpoint(
            n_batches=2,
            completed_batches=1,
            batch_sizes=[{Language.AR: 1, Language.EN: 1, Language.MIXED: 1}] * 2,
            memory_scores=[restored_batch_scores],
            control_scores=[restored_batch_scores],
            corrections_captured=[0],
            active_rules_after_batch=[1],
            memory_rules=[pre_existing_rule],
        ),
    )

    fake_extractor, calls = _fake_extractor_factory()

    result = run_five_batch(
        ground_truth_dir,
        output_path=tmp_path / "results.md",
        synthetic_dir=synthetic_dir,
        n_batches=2,
        sleep_seconds=0.0,
        extractor=fake_extractor,
        embed=_fake_embed,
        reviewer_id=_test_identity,
        agent_external_id=_test_identity,
        checkpoint_path=checkpoint_path,
    )

    # 2 calls per language x 2 batches of 1 each; only batch 2 (3 calls x 2
    # arms = 6) should ever reach the extractor — batch 1 was restored, not
    # re-run.
    assert len(calls) == 6
    assert len(result.memory) == 2
    assert result.memory[0] == restored_batch_scores
    assert len(result.control) == 2
    assert result.control[0] == restored_batch_scores
    # A fully successful run removes its own checkpoint.
    assert not checkpoint_path.is_file()


def test_run_five_batch_raises_on_checkpoint_batch_count_mismatch(
    _ground_truth_dir: tuple[Path, Path], _test_identity: str, tmp_path: Path
) -> None:
    """A checkpoint written for a different n_batches is rejected, not silently misapplied."""
    ground_truth_dir, synthetic_dir = _ground_truth_dir
    checkpoint_path = tmp_path / "checkpoint.json"
    _save_checkpoint(
        checkpoint_path,
        _Checkpoint(
            n_batches=5,
            completed_batches=1,
            batch_sizes=[],
            memory_scores=[],
            control_scores=[],
            corrections_captured=[],
            active_rules_after_batch=[],
            memory_rules=[],
        ),
    )
    fake_extractor, _calls = _fake_extractor_factory()

    with pytest.raises(ValueError, match="n_batches"):
        run_five_batch(
            ground_truth_dir,
            output_path=tmp_path / "results.md",
            synthetic_dir=synthetic_dir,
            n_batches=2,
            sleep_seconds=0.0,
            extractor=fake_extractor,
            embed=_fake_embed,
            reviewer_id=_test_identity,
            agent_external_id=_test_identity,
            checkpoint_path=checkpoint_path,
        )


def _batch_scores(accuracy: float) -> BatchScores:
    all_languages = (Language.AR, Language.EN, Language.MIXED)
    return BatchScores(
        n_calls=dict.fromkeys(all_languages, 1),
        accuracy=dict.fromkeys(all_languages, accuracy),
        rubric_agreement=dict.fromkeys(all_languages, 1.0),
        grounding_precision=dict.fromkeys(all_languages, 1.0),
        unsupported_claim_rate=dict.fromkeys(all_languages, 0.0),
    )


def test_render_report_judges_the_memory_control_delta_not_memorys_own_trajectory() -> None:
    """A real bug, fixed: memory's own accuracy can fall while its advantage over

    control still grows every batch (the control fell further) — that is a win
    for memory, not a failure, because the control was never flat to begin
    with. The old logic checked whether memory's raw score rose and mislabeled
    exactly this case as "did not rise". See docs/09-DECISIONS.md.
    """
    result = FiveBatchResult(
        n_batches=2,
        batch_sizes=[{Language.AR: 1}, {Language.AR: 1}],
        memory=[_batch_scores(0.9), _batch_scores(0.8)],  # memory's own score falls
        control=[_batch_scores(0.85), _batch_scores(0.5)],  # control falls further
        corrections_captured=[0, 0],
        active_rules_after_batch=[0, 0],
    )

    report = _render_report(result)

    # Δ rises: 0.9-0.85=+0.05 -> 0.8-0.5=+0.3, and beats control both batches.
    assert "Δ (memory − control) trend: rising" in report
    assert "beats control every batch" in report
    assert "Memory's advantage over control grows every batch, in every language category" in report
    assert "does not grow consistently" not in report


def test_record_rule_outcomes_judges_each_rule_by_its_own_topic(_test_identity: str) -> None:
    """Two rules retrieved for the same call get independent verdicts, not one shared per-call verdict.

    This is the exact distinction the wiring exists to make: a call's
    first_difference() finds at most one field wrong, but multiple rules on
    different topics may have been retrieved for it. Only the rule whose
    own topic matches the diff should be marked a failure.
    """
    truth = _ground_truth("call_topic_test", Language.EN)
    prediction = _agent_output("call_topic_test", Language.EN)  # differs on commitments[0].deadline

    with get_session() as session:
        agent_id = get_or_create_placeholder_agent(
            session, external_id=_test_identity, name="topic-test placeholder agent"
        )
        persist_analysis_record(session, prediction, agent_id=agent_id, transcript="irrelevant transcript")

    on_topic_correction = capture_correction(
        prediction, truth, error_location="commitments[0].deadline", reviewer_id=_test_identity
    )
    off_topic_correction = capture_correction(
        prediction, truth, error_location="compliance_flags", reviewer_id=_test_identity
    )

    on_topic_rule = MemoryRule(
        rule_text="a rule about commitment deadlines",
        source_correction_ids=[on_topic_correction.id],
        created_at=datetime.now(UTC),
    )
    off_topic_rule = MemoryRule(
        rule_text="a rule about compliance flags",
        source_correction_ids=[off_topic_correction.id],
        created_at=datetime.now(UTC),
    )
    store = MemoryStore(embed=_fake_embed)
    store.add(on_topic_rule)
    store.add(off_topic_rule)

    _record_rule_outcomes(
        predictions=[prediction],
        batch=[truth],
        retrieved_by_call={prediction.call_id: [on_topic_rule, off_topic_rule]},
        store=store,
    )

    updated_on_topic = next(rule for rule in store.all_active_rules() if rule.id == on_topic_rule.id)
    updated_off_topic = next(rule for rule in store.all_active_rules() if rule.id == off_topic_rule.id)
    assert updated_on_topic.failure_count == 1
    assert updated_on_topic.success_count == 0
    assert updated_off_topic.success_count == 1
    assert updated_off_topic.failure_count == 0


def test_run_five_batch_retires_an_underperforming_rule(
    _ground_truth_dir: tuple[Path, Path], _test_identity: str
) -> None:
    """maybe_retire() actually fires at the end of a batch for a rule that keeps failing."""
    ground_truth_dir, synthetic_dir = _ground_truth_dir

    seed_truth = _ground_truth("call_seed", Language.AR)
    seed_prediction = _agent_output("call_seed", Language.AR)  # always differs on commitments[0].deadline
    with get_session() as session:
        agent_id = get_or_create_placeholder_agent(
            session, external_id=_test_identity, name="retirement-test placeholder agent"
        )
        persist_analysis_record(session, seed_prediction, agent_id=agent_id, transcript="seed transcript")
    seed_correction = capture_correction(
        seed_prediction, seed_truth, error_location="commitments[0].deadline", reviewer_id=_test_identity
    )
    # 9 applications already, all failures — one more failure this batch
    # crosses maybe_retire()'s default min_applications=10.
    struggling_rule = MemoryRule(
        rule_text="a rule about commitment deadlines that keeps failing",
        source_correction_ids=[seed_correction.id],
        created_at=datetime.now(UTC),
        success_count=0,
        failure_count=9,
    )
    store = MemoryStore(embed=_fake_embed)
    store.add(struggling_rule)

    fake_extractor, _calls = _fake_extractor_factory()

    run_five_batch(
        ground_truth_dir,
        output_path=ground_truth_dir.parent / "results.md",
        synthetic_dir=synthetic_dir,
        n_batches=1,
        sleep_seconds=0.0,
        extractor=fake_extractor,
        store=store,
        embed=_fake_embed,
        reviewer_id=_test_identity,
        agent_external_id=_test_identity,
    )

    active_ids = {rule.id for rule in store.all_active_rules()}
    assert struggling_rule.id not in active_ids
