"""Tests for `scripts/build_finetune_dataset.py`.

Phase 5, step 2. The Postgres-backed tests need a running instance (`docker
compose up -d`) — same requirement as `tests/scripts/
test_generate_synthetic_corrections.py`, which this file's fixtures mirror.
Every fixture call_id is namespaced under `finetune_test_` so it can never
collide with the real project corpus (`data/ground_truth`) or its Postgres
rows — `build_dataset()` scopes Postgres corrections to the ground-truth
corpus it was handed, so an isolated call_id namespace is what keeps these
tests from reading (or polluting) real data.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from build_finetune_dataset import (
    DatasetExample,
    _build_examples,
    _extract_field,
    _relevant_claims,
    _to_record,
    build_dataset,
    ensure_held_out_call_ids,
    select_held_out_call_ids,
    stratified_split,
)

from sawti.db.models import Base, Call, CallAnalysisRecord, ReviewerAction
from sawti.db.session import get_engine, get_session
from sawti.memory.capture import (
    capture_correction,
    get_or_create_placeholder_agent,
    persist_analysis_record,
)
from sawti.schemas import (
    CallAnalysis,
    Commitment,
    ComplianceFlag,
    Correction,
    Language,
    Quote,
    RubricScore,
)


def _quote(text: str) -> Quote:
    return Quote(text=text, speaker="Agent", start_char=0, end_char=len(text))


def _commitment(*, text: str = "we will call back", deadline: datetime | None = None) -> Commitment:
    return Commitment(
        evidence=_quote(text), promised_by="Agent", description="callback", deadline=deadline
    )


def _flag(*, text: str = "we recorded this call", rule_id: str = "recording_disclosure") -> ComplianceFlag:
    return ComplianceFlag(evidence=_quote(text), rule_id=rule_id, severity="low", description="x")


def _score(*, text: str = "ok", criterion: str = "empathy", value: float = 0.8) -> RubricScore:
    return RubricScore(evidence=_quote(text), criterion=criterion, score=value, justification="x")


def _analysis(
    call_id: str,
    *,
    language: Language = Language.AR,
    commitments: list[Commitment] | None = None,
    compliance_flags: list[ComplianceFlag] | None = None,
    rubric_scores: list[RubricScore] | None = None,
) -> CallAnalysis:
    return CallAnalysis(
        call_id=call_id,
        language=language,
        summary="placeholder summary",
        commitments=commitments or [],
        compliance_flags=compliance_flags or [],
        rubric_scores=rubric_scores or [],
        confidence=0.9,
        requires_human_review=False,
    )


def _correction(
    call_id: str,
    *,
    error_location: str,
    original: CallAnalysis,
    corrected: CallAnalysis,
    reviewer_id: str = "x",
) -> Correction:
    return Correction(
        call_id=call_id,
        original=original,
        corrected=corrected,
        error_location=error_location,
        reviewer_id=reviewer_id,
        timestamp=datetime.now(UTC),
    )


# --- select_held_out_call_ids / ensure_held_out_call_ids -------------------


def test_select_held_out_call_ids_is_deterministic_and_sized() -> None:
    call_ids = [f"call_{i:04d}_ar" for i in range(60)]
    first = select_held_out_call_ids(call_ids, fraction=0.25, seed=42)
    second = select_held_out_call_ids(call_ids, fraction=0.25, seed=42)
    assert first == second
    assert len(first) == 15
    assert set(first) <= set(call_ids)


def test_select_held_out_call_ids_different_seed_differs() -> None:
    call_ids = [f"call_{i:04d}_ar" for i in range(60)]
    a = select_held_out_call_ids(call_ids, fraction=0.25, seed=1)
    b = select_held_out_call_ids(call_ids, fraction=0.25, seed=2)
    assert a != b


def test_ensure_held_out_call_ids_writes_file_then_reuses_it(tmp_path: Path) -> None:
    call_ids = [f"call_{i:04d}_ar" for i in range(20)]
    first = ensure_held_out_call_ids(
        call_ids, language=Language.AR, output_dir=tmp_path, fraction=0.25, seed=1
    )
    assert (tmp_path / "held_out_call_ids.json").is_file()
    assert len(first) == 5

    # A different seed is ignored on the second call — the file already answers.
    second = ensure_held_out_call_ids(
        call_ids, language=Language.AR, output_dir=tmp_path, fraction=0.25, seed=999
    )
    assert second == first


def test_ensure_held_out_call_ids_rejects_unknown_ids(tmp_path: Path) -> None:
    call_ids = [f"call_{i:04d}_ar" for i in range(20)]
    ensure_held_out_call_ids(call_ids, language=Language.AR, output_dir=tmp_path, fraction=0.25, seed=1)

    with pytest.raises(ValueError, match="not in the current corpus"):
        ensure_held_out_call_ids(
            ["call_9999_ar"], language=Language.AR, output_dir=tmp_path, fraction=0.25, seed=1
        )


def test_ensure_held_out_call_ids_is_independent_per_language(tmp_path: Path) -> None:
    ar_ids = [f"call_{i:04d}_ar" for i in range(20)]
    en_ids = [f"call_{i:04d}_en" for i in range(20)]
    ar_held_out = ensure_held_out_call_ids(
        ar_ids, language=Language.AR, output_dir=tmp_path, fraction=0.25, seed=1
    )
    en_held_out = ensure_held_out_call_ids(
        en_ids, language=Language.EN, output_dir=tmp_path, fraction=0.25, seed=1
    )
    assert set(ar_held_out).isdisjoint(en_held_out)


# --- _extract_field / _relevant_claims --------------------------------------


def test_extract_field_whole_commitments_list() -> None:
    analysis = _analysis("c1", commitments=[_commitment(text="a"), _commitment(text="b")])
    field = _extract_field(analysis, "commitments")
    assert len(field) == 2
    assert field[0]["description"] == "callback"


def test_extract_field_commitment_deadline_by_index() -> None:
    commitments = [_commitment(text="a"), _commitment(text="b", deadline=datetime.now(UTC))]
    analysis = _analysis("c1", commitments=commitments)
    field = _extract_field(analysis, "commitments[1].deadline")
    assert field["deadline"] is not None


def test_extract_field_commitment_index_out_of_range_is_none() -> None:
    analysis = _analysis("c1", commitments=[_commitment(text="a")])
    assert _extract_field(analysis, "commitments[5].deadline") is None


def test_extract_field_compliance_flags() -> None:
    analysis = _analysis("c1", compliance_flags=[_flag(rule_id="r1")])
    field = _extract_field(analysis, "compliance_flags")
    assert field[0]["rule_id"] == "r1"


def test_extract_field_rubric_score_by_criterion() -> None:
    scores = [_score(criterion="empathy", value=0.5), _score(criterion="call_closure")]
    analysis = _analysis("c1", rubric_scores=scores)
    field = _extract_field(analysis, "rubric_scores[empathy]")
    assert field["score"] == 0.5


def test_extract_field_rubric_score_missing_criterion_is_none() -> None:
    analysis = _analysis("c1", rubric_scores=[_score(criterion="empathy")])
    assert _extract_field(analysis, "rubric_scores[call_closure]") is None


def test_extract_field_unrecognized_shape_raises() -> None:
    analysis = _analysis("c1")
    with pytest.raises(ValueError, match="Unrecognized error_location"):
        _extract_field(analysis, "summary")


def test_relevant_claims_matches_extract_field_shapes() -> None:
    analysis = _analysis("c1", commitments=[_commitment(text="a")])
    assert _relevant_claims(analysis, "commitments") == analysis.commitments
    assert _relevant_claims(analysis, "commitments[0].deadline") == [analysis.commitments[0]]
    assert _relevant_claims(analysis, "commitments[7].deadline") == []


# --- _build_examples: grounding, held-out exclusion, dedupe ----------------


def test_build_examples_drops_ungrounded_corrected_claims() -> None:
    transcript = "Agent: we will call back tomorrow\nCustomer: ok"
    original = _analysis("call_a_ar", commitments=[])
    corrected = _analysis("call_a_ar", commitments=[_commitment(text="never in the transcript")])
    correction = _correction(
        "call_a_ar", error_location="commitments", original=original, corrected=corrected
    )

    examples, dropped = _build_examples([(correction, transcript, "reviewer_action")], held_out=set())

    assert examples == []
    assert len(dropped) == 1
    assert "grounding" in dropped[0]["reason"]


def test_build_examples_keeps_grounded_claim() -> None:
    transcript = "Agent: we will call back tomorrow\nCustomer: ok"
    original = _analysis("call_a_ar", commitments=[])
    corrected = _analysis("call_a_ar", commitments=[_commitment(text="we will call back tomorrow")])
    correction = _correction(
        "call_a_ar", error_location="commitments", original=original, corrected=corrected
    )

    examples, dropped = _build_examples([(correction, transcript, "reviewer_action")], held_out=set())

    assert dropped == []
    assert len(examples) == 1
    assert examples[0].topic == "commitments"
    assert examples[0].source == "reviewer_action"


def test_build_examples_excludes_held_out_calls() -> None:
    transcript = "Agent: we will call back tomorrow\nCustomer: ok"
    original = _analysis("call_a_ar", commitments=[])
    corrected = _analysis("call_a_ar", commitments=[_commitment(text="we will call back tomorrow")])
    correction = _correction(
        "call_a_ar", error_location="commitments", original=original, corrected=corrected
    )

    examples, dropped = _build_examples(
        [(correction, transcript, "reviewer_action")], held_out={"call_a_ar"}
    )

    assert examples == []
    assert dropped == []  # held-out calls are excluded silently, not reported as a data problem


def test_build_examples_dedupes_identical_call_location_target() -> None:
    transcript = "Agent: we will call back tomorrow\nCustomer: ok"
    original = _analysis("call_a_ar", commitments=[])
    corrected = _analysis("call_a_ar", commitments=[_commitment(text="we will call back tomorrow")])
    correction_1 = _correction(
        "call_a_ar", error_location="commitments", original=original, corrected=corrected
    )
    correction_2 = _correction(
        "call_a_ar", error_location="commitments", original=original, corrected=corrected
    )

    examples, _dropped = _build_examples(
        [(correction_1, transcript, "reviewer_action"), (correction_2, transcript, "phase1_2_diff")],
        held_out=set(),
    )

    assert len(examples) == 1


def test_build_examples_rubric_missing_criterion_is_dropped() -> None:
    transcript = "Agent: ok\nCustomer: fine"
    original = _analysis("call_a_ar", rubric_scores=[_score(criterion="empathy")])
    corrected = _analysis("call_a_ar", rubric_scores=[])  # criterion vanished — nothing to target
    correction = _correction(
        "call_a_ar", error_location="rubric_scores[empathy]", original=original, corrected=corrected
    )

    examples, dropped = _build_examples([(correction, transcript, "reviewer_action")], held_out=set())

    assert examples == []
    assert dropped[0]["reason"] == "corrected field not resolvable"


# --- stratified_split --------------------------------------------------------


def _example(call_id: str, topic: str) -> DatasetExample:
    return DatasetExample(
        call_id=call_id,
        language=Language.AR,
        error_location=topic,
        topic=topic,
        source="reviewer_action",
        reviewer_id="x",
        transcript="t",
        original_field={},
        corrected_field={"x": 1},
        note=None,
    )


def test_stratified_split_keeps_rare_topic_entirely_in_train() -> None:
    examples = [_example("call_0", "rare_topic")]
    train, val = stratified_split(examples, val_fraction=0.15, seed=1)
    assert train == examples
    assert val == []


def test_stratified_split_proportions_a_larger_group() -> None:
    examples = [_example(f"call_{i}", "commitments") for i in range(20)]
    train, val = stratified_split(examples, val_fraction=0.15, seed=1)
    assert len(val) == 3  # round(20 * 0.15)
    assert len(train) == 17
    assert {e.call_id for e in train}.isdisjoint({e.call_id for e in val})


def test_stratified_split_every_topic_represented_in_train() -> None:
    examples = [_example(f"call_{i}", "commitments") for i in range(10)] + [
        _example(f"other_{i}", "compliance_flags") for i in range(10)
    ]
    train, _val = stratified_split(examples, val_fraction=0.15, seed=1)
    train_topics = {e.topic for e in train}
    assert train_topics == {"commitments", "compliance_flags"}


def test_stratified_split_is_deterministic() -> None:
    examples = [_example(f"call_{i}", "commitments") for i in range(20)]
    train_a, val_a = stratified_split(examples, val_fraction=0.15, seed=7)
    train_b, val_b = stratified_split(examples, val_fraction=0.15, seed=7)
    assert [e.call_id for e in train_a] == [e.call_id for e in train_b]
    assert [e.call_id for e in val_a] == [e.call_id for e in val_b]


# --- _to_record --------------------------------------------------------------


def test_to_record_is_json_conformant() -> None:
    example = _example("call_0", "commitments")
    record = _to_record(example)
    required_keys = (
        "call_id",
        "language",
        "error_location",
        "topic",
        "source",
        "transcript",
        "instruction",
        "input",
        "output",
    )
    for key in required_keys:
        assert key in record
    # input/output must themselves be JSON text, ready for a chat-template renderer.
    json.loads(record["input"])
    json.loads(record["output"])


# --- build_dataset: end-to-end with real Postgres, fake extractor ----------


@pytest.fixture(scope="module", autouse=True)
def _schema() -> None:
    Base.metadata.create_all(get_engine())


def _seed_reviewer_action(
    call_id: str, *, language: Language, error_location: str, transcript: str
) -> None:
    """Persist one Postgres ReviewerAction the way the real pipeline would, for one call."""
    original = _analysis(call_id, language=language, commitments=[])
    corrected = _analysis(
        call_id, language=language, commitments=[_commitment(text="we will fix this today")]
    )

    with get_session() as session:
        agent_id = get_or_create_placeholder_agent(
            session, external_id="finetune-test-agent", name="finetune test placeholder agent"
        )
    with get_session() as session:
        persist_analysis_record(session, original, agent_id=agent_id, transcript=transcript)
    capture_correction(original, corrected, error_location=error_location, reviewer_id="test-reviewer")


def _cleanup_call(call_id_prefix: str) -> None:
    with get_session() as session:
        for record in session.query(CallAnalysisRecord).all():
            if record.payload.get("call_id", "").startswith(call_id_prefix):
                for action in session.query(ReviewerAction).filter_by(call_analysis_id=record.id):
                    session.delete(action)
                call_row = session.get(Call, record.call_id)
                session.delete(record)
                if call_row is not None:
                    session.delete(call_row)


def test_build_dataset_default_is_ar_only_and_respects_held_out(tmp_path: Path) -> None:
    prefix = "finetune_test_default"
    ground_truth_dir = tmp_path / "ground_truth"
    synthetic_dir = tmp_path / "synthetic"
    output_dir = tmp_path / "finetune"
    ground_truth_dir.mkdir()
    synthetic_dir.mkdir()

    call_ids = [f"{prefix}_{i:02d}_ar" for i in range(8)]
    transcript_text = "Agent: we will fix this today\nCustomer: thank you"

    for call_id in call_ids:
        commitments = [_commitment(text="we will fix this today")]
        truth = _analysis(call_id, language=Language.AR, commitments=commitments)
        (ground_truth_dir / f"{call_id}.json").write_text(truth.model_dump_json(), encoding="utf-8")
        (synthetic_dir / f"{call_id}.txt").write_text(transcript_text, encoding="utf-8")

    def fake_extractor(path: Path) -> CallAnalysis:
        # Agent proposes nothing — every eligible call disagrees with ground truth.
        return _analysis(path.stem, language=Language.AR, commitments=[])

    try:
        manifest = build_dataset(
            languages=[Language.AR],
            ground_truth_dir=ground_truth_dir,
            synthetic_dir=synthetic_dir,
            output_dir=output_dir,
            extractor=fake_extractor,
            sleep_seconds=0.0,
            diff_cache_dir=None,
            held_out_fraction=0.25,
            held_out_seed=1,
            val_fraction=0.15,
            split_seed=1,
        )

        assert manifest["languages_included"] == ["ar"]
        assert manifest["per_language"]["ar"]["corpus_size"] == 8
        assert manifest["per_language"]["ar"]["held_out_count"] == 2
        assert manifest["per_language"]["ar"]["eligible_count"] == 6

        held_out_payload = json.loads((output_dir / "held_out_call_ids.json").read_text())
        held_out_ids = set(held_out_payload["ar"])
        train_val_ids = set()
        for filename in ("train.jsonl", "val.jsonl"):
            for line in (output_dir / filename).read_text(encoding="utf-8").splitlines():
                train_val_ids.add(json.loads(line)["call_id"])

        assert held_out_ids.isdisjoint(train_val_ids)
        assert train_val_ids <= set(call_ids) - held_out_ids
        per_language = manifest["per_language"]["ar"]
        assert manifest["train_count"] + manifest["val_count"] == per_language["total_examples"]
    finally:
        _cleanup_call(prefix)


def test_build_dataset_uses_existing_postgres_corrections(tmp_path: Path) -> None:
    prefix = "finetune_test_pg"
    ground_truth_dir = tmp_path / "ground_truth"
    synthetic_dir = tmp_path / "synthetic"
    output_dir = tmp_path / "finetune"
    ground_truth_dir.mkdir()
    synthetic_dir.mkdir()

    call_ids = [f"{prefix}_{i:02d}_ar" for i in range(4)]
    transcript_text = "Agent: we will fix this today\nCustomer: thank you"
    for call_id in call_ids:
        commitments = [_commitment(text="we will fix this today")]
        truth = _analysis(call_id, language=Language.AR, commitments=commitments)
        (ground_truth_dir / f"{call_id}.json").write_text(truth.model_dump_json(), encoding="utf-8")
        (synthetic_dir / f"{call_id}.txt").write_text(transcript_text, encoding="utf-8")

    seeded_call_id = call_ids[0]
    _seed_reviewer_action(
        seeded_call_id, language=Language.AR, error_location="commitments", transcript=transcript_text
    )

    def fake_extractor(path: Path) -> CallAnalysis:
        # Agent matches ground truth everywhere — no diff-augmented pairs, so
        # only the Postgres-sourced example should reach the output.
        commitments = [_commitment(text="we will fix this today")]
        return _analysis(path.stem, language=Language.AR, commitments=commitments)

    try:
        manifest = build_dataset(
            languages=[Language.AR],
            ground_truth_dir=ground_truth_dir,
            synthetic_dir=synthetic_dir,
            output_dir=output_dir,
            extractor=fake_extractor,
            sleep_seconds=0.0,
            diff_cache_dir=None,
            held_out_fraction=0.0,
            held_out_seed=1,
            val_fraction=0.15,
            split_seed=1,
        )

        per_language = manifest["per_language"]["ar"]
        assert per_language["examples_from_reviewer_action"] == 1
        assert per_language["examples_from_phase1_2_diff"] == 0
    finally:
        _cleanup_call(prefix)


def test_build_dataset_en_and_mixed_are_opt_in_only(tmp_path: Path) -> None:
    prefix = "finetune_test_optin"
    ground_truth_dir = tmp_path / "ground_truth"
    synthetic_dir = tmp_path / "synthetic"
    output_dir = tmp_path / "finetune"
    ground_truth_dir.mkdir()
    synthetic_dir.mkdir()

    ar_id = f"{prefix}_ar"
    truth = _analysis(ar_id, language=Language.AR)
    (ground_truth_dir / f"{ar_id}.json").write_text(truth.model_dump_json(), encoding="utf-8")
    (synthetic_dir / f"{ar_id}.txt").write_text("Agent: hello\nCustomer: hi", encoding="utf-8")

    def fake_extractor(path: Path) -> CallAnalysis:
        return _analysis(path.stem, language=Language.AR)

    try:
        manifest = build_dataset(
            languages=[Language.AR],
            ground_truth_dir=ground_truth_dir,
            synthetic_dir=synthetic_dir,
            output_dir=output_dir,
            extractor=fake_extractor,
            sleep_seconds=0.0,
            diff_cache_dir=None,
            held_out_fraction=0.0,
            held_out_seed=1,
        )
        assert manifest["languages_included"] == ["ar"]
        assert "en" not in manifest["per_language"]
        assert "mixed" not in manifest["per_language"]
    finally:
        _cleanup_call(prefix)
