"""Build the Phase 5 QLoRA fine-tuning dataset from QA corrections.

Phase 5, step 2: turns `Correction`/`ReviewerAction` records — plus a fresh
Phase 1/2 ground-truth-vs-agent-output diff pass over held-in calls — into
instruction-tuning pairs.

STAND-IN, NOT SIGNAL. See `scripts/generate_synthetic_corrections.py`'s own
docstring: every `Correction` this repository has, from either source this
script reads, is manufactured by diffing an agent output against an
LLM-generated reference label — never a real QA reviewer's judgment. This
dataset validates the QLoRA training mechanism end-to-end. It is not evidence
of real-world learning capacity. See `docs/09-DECISIONS.md`, 2026-09-25.

Why two sources, not one:

  1. Existing `ReviewerAction` rows (reviewer_id `five-batch-experiment` or
     `synthetic-day1`) — captured previously, persisted in Postgres.
  2. A fresh diff pass this script runs itself, over every held-in call that
     doesn't already have a Postgres correction: `sawti.memory.diff.
     first_difference` against the same Phase 2 grounded agent
     (`sawti.eval.experiments.grounded_graph.extract_via_graph`) used to
     produce (1) and `scripts/generate_synthetic_corrections.py`. This is
     the same mechanism, run wider, because Postgres alone left `ar` below a
     usable sample size (see the 2026-09-25 dataset-scoping decision).

Why calls are held out before anything else: Phase 5 step 4 needs to compare
the base model against the QLoRA-tuned one on calls neither ever trained on.
`ensure_held_out_call_ids()` fixes that set — 25% of each included language's
corpus, chosen deterministically — and writes it to
`data/finetune/held_out_call_ids.json` before a single training pair is
built. Every later step in this script treats that file as authoritative
(loads it back rather than recomputing), and `build_dataset()` hard-fails if
a held-out call_id ever ends up in train.jsonl or val.jsonl.

Usage:
    uv run python scripts/build_finetune_dataset.py
    uv run python scripts/build_finetune_dataset.py --include-en --include-mixed
    uv run python scripts/build_finetune_dataset.py --skip-diff-augmentation
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from re import compile as re_compile
from typing import Any

from sawti.agent.nodes.ground import is_grounded
from sawti.data.ground_truth import load_ground_truth
from sawti.db.models import Call as CallRow
from sawti.db.models import CallAnalysisRecord, ReviewerAction
from sawti.db.session import get_session
from sawti.eval.experiments.grounded_graph import extract_via_graph
from sawti.memory.diff import first_difference, topic_for
from sawti.schemas import CallAnalysis, Claim, Correction, Language

logger = logging.getLogger("build_finetune_dataset")

DEFAULT_GROUND_TRUTH_DIR = Path("data/ground_truth")
DEFAULT_SYNTHETIC_DIR = Path("data/synthetic")
DEFAULT_OUTPUT_DIR = Path("data/finetune")

# 60 ar / 30 en / 60 mixed calls in the corpus (data/ground_truth). A quarter
# of each included language is reserved for the Phase 5 step-4 before/after
# comparison and never trains — see the module docstring.
HELD_OUT_FRACTION = 0.25
# Fixed, not "whatever the clock says" — a held-out set that could change
# between runs defeats the point of writing it to disk. Dated, like every
# other seed choice in this project's decisions log.
HELD_OUT_SEED = 20260925

VAL_FRACTION = 0.15
SPLIT_SEED = 20260925

# Never a real reviewer id — this script's own diff pass is exactly as
# synthetic as `scripts/generate_synthetic_corrections.py`'s "synthetic-day1",
# just run over a different (wider, held-in-only) slice of calls. Kept
# distinct from that script's id so a downstream reader can tell the two
# generation runs apart.
DIFF_AUGMENTATION_REVIEWER_ID = "phase1-2-diff-augmentation"
DIFF_AUGMENTATION_NOTE = (
    "auto-generated from a fresh Phase 1/2 ground-truth-vs-agent-output diff, not a real QA review"
)

# Quoted from eval_results.md's 2026-09-25 five-batch finding, caveat 1.
# Printed whenever --include-en/--include-mixed is used, so opting in to a
# category the done-when criterion did NOT validate is never silent.
EN_MIXED_CAVEAT = (
    "CAVEAT (eval_results.md, 2026-09-25 five-batch finding): only `ar` met the "
    "done-when bar every batch; `en`'s memory advantage was real but partial "
    "(3 of 5 batches showed none), and `mixed` showed no consistent effect. "
    "Including this language's corrections in the fine-tuning set is NOT "
    "validated by that result."
)

DATASET_CAVEAT = (
    "Every example in this dataset is synthetic: sourced from auto-generated "
    "Correction/ReviewerAction rows (reviewer_id 'five-batch-experiment' or "
    "'synthetic-day1') or from this script's own fresh ground-truth-vs-agent-"
    "output diffs (reviewer_id 'phase1-2-diff-augmentation'). None is a real "
    "human QA judgment. See docs/09-DECISIONS.md, 2026-09-25."
)

Extractor = Callable[[Path], CallAnalysis]

_COMMITMENT_DEADLINE_RE = re_compile(r"^commitments\[(\d+)\]\.deadline$")
_RUBRIC_RE = re_compile(r"^rubric_scores\[([^\]]+)\]$")

INSTRUCTION_BY_TOPIC: dict[str, str] = {
    "commitments": (
        "The transcript below is a customer-service call. The agent's analysis "
        "pipeline extracted the commitments (promises made during the call) shown "
        "in `input`. Review them against the transcript and correct any errors — "
        "commitments that were missed, invented, or wrongly counted."
    ),
    "commitments.deadline": (
        "The transcript below is a customer-service call. The agent's analysis "
        "pipeline extracted the commitment shown in `input`. Review whether it was "
        "given a deadline, and correct the `deadline` field if the agent got it wrong."
    ),
    "compliance_flags": (
        "The transcript below is a customer-service call. The agent's analysis "
        "pipeline extracted the compliance flags shown in `input`. Review them "
        "against the transcript and correct any errors — flags that were missed or "
        "wrongly raised."
    ),
    "rubric_scores": (
        "The transcript below is a customer-service call. The agent's analysis "
        "pipeline scored the call on a QA rubric criterion, shown in `input`. "
        "Review the score against the transcript and correct it if it is off."
    ),
}


def _relevant_claims(analysis: CallAnalysis, error_location: str) -> list[Claim]:
    """The `corrected` claims a grounding check must run against, for one `error_location`.

    Raises:
        ValueError: `error_location` isn't one of the four shapes
            `sawti.memory.diff.first_difference` produces — the only shapes
            this dataset's sources can contain.
    """
    if error_location == "commitments":
        return list(analysis.commitments)
    if error_location == "compliance_flags":
        return list(analysis.compliance_flags)
    match = _COMMITMENT_DEADLINE_RE.match(error_location)
    if match:
        index = int(match.group(1))
        return [analysis.commitments[index]] if index < len(analysis.commitments) else []
    match = _RUBRIC_RE.match(error_location)
    if match:
        return [score for score in analysis.rubric_scores if score.criterion == match.group(1)]
    raise ValueError(f"Unrecognized error_location shape: {error_location!r}")


def _extract_field(analysis: CallAnalysis, error_location: str) -> Any:
    """The JSON-able value at `error_location` within `analysis` — the input/target payload."""
    if error_location == "commitments":
        return [c.model_dump(mode="json") for c in analysis.commitments]
    if error_location == "compliance_flags":
        return [f.model_dump(mode="json") for f in analysis.compliance_flags]
    match = _COMMITMENT_DEADLINE_RE.match(error_location)
    if match:
        index = int(match.group(1))
        if index >= len(analysis.commitments):
            return None
        return analysis.commitments[index].model_dump(mode="json")
    match = _RUBRIC_RE.match(error_location)
    if match:
        for score in analysis.rubric_scores:
            if score.criterion == match.group(1):
                return score.model_dump(mode="json")
        return None
    raise ValueError(f"Unrecognized error_location shape: {error_location!r}")


def select_held_out_call_ids(call_ids: list[str], *, fraction: float, seed: int) -> list[str]:
    """Deterministically pick `fraction` of `call_ids`, sorted for a stable result."""
    ordered = sorted(call_ids)
    count = round(len(ordered) * fraction)
    return sorted(Random(seed).sample(ordered, count))


def ensure_held_out_call_ids(
    call_ids: list[str],
    *,
    language: Language,
    output_dir: Path,
    fraction: float,
    seed: int,
) -> list[str]:
    """Return this language's held-out call_ids, computing and persisting them on first use.

    A second call (this run or a future one) with the same `output_dir` reads
    the file back rather than recomputing — the whole point of writing it
    immediately is that the held-out set can never silently drift.

    Raises:
        ValueError: The file already names a held-out id that isn't in
            `call_ids` — a corpus/file mismatch, not something to paper over.
    """
    path = output_dir / "held_out_call_ids.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    existing = payload.get(language.value)
    if existing is not None:
        unknown = sorted(set(existing) - set(call_ids))
        if unknown:
            raise ValueError(
                f"held_out_call_ids.json's {language.value!r} entries include ids not in "
                f"the current corpus: {unknown}"
            )
        return sorted(existing)

    held_out = select_held_out_call_ids(call_ids, fraction=fraction, seed=seed)
    payload[language.value] = held_out
    meta = payload.setdefault("_meta", {})
    meta[language.value] = {
        "fraction": fraction,
        "seed": seed,
        "corpus_size": len(call_ids),
        "held_out_count": len(held_out),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return held_out


@dataclass
class DatasetExample:
    """One field-level correction, resolved into a training pair."""

    call_id: str
    language: Language
    error_location: str
    topic: str
    source: str
    reviewer_id: str
    transcript: str
    original_field: Any
    corrected_field: Any
    note: str | None


def _pg_corrections_for_language(language: Language) -> list[tuple[Correction, str]]:
    """Every persisted `ReviewerAction` for `language`, as `(Correction, transcript_text)`."""
    with get_session() as session:
        rows = (
            session.query(ReviewerAction, CallRow.transcript)
            .join(CallAnalysisRecord, ReviewerAction.call_analysis_id == CallAnalysisRecord.id)
            .join(CallRow, CallAnalysisRecord.call_id == CallRow.id)
            .filter(CallRow.language == language.value)
            .all()
        )
        # Resolved inside the session: `action.payload`/`transcript` are lazy
        # column reads that would raise DetachedInstanceError once the
        # session closes on `with` exit.
        return [(Correction.model_validate(action.payload), transcript or "") for action, transcript in rows]


def _diff_augmented_pairs(
    call_ids: list[str],
    *,
    ground_truth_dir: Path,
    synthetic_dir: Path,
    extractor: Extractor,
    cache_dir: Path | None,
    sleep_seconds: float,
) -> tuple[list[Correction], dict[str, str]]:
    """Run `extractor` over each of `call_ids` and diff its output against ground truth.

    Never persisted to Postgres — these are dataset-only, ephemeral to this
    script's output. `cache_dir`, when given, memoizes each call's agent
    output by call_id so a re-run (or a fix applied after a partial run)
    doesn't re-spend LLM quota on calls already extracted; see
    `docs/09-DECISIONS.md`'s 2026-09-25 entry on this project's Gemini
    free-tier rate/daily limits.

    Returns:
        Captured `Correction`s (one per call where the agent disagreed with
        ground truth) and a `call_id -> transcript text` map covering every
        call in `call_ids` that had a readable transcript file.
    """
    ground_truth_by_id = {record.call_id: record for record in load_ground_truth(ground_truth_dir)}
    corrections: list[Correction] = []
    transcripts: dict[str, str] = {}

    for index, call_id in enumerate(call_ids, start=1):
        truth = ground_truth_by_id.get(call_id)
        if truth is None:
            logger.error("[%d/%d] %s: no ground-truth record, skipping", index, len(call_ids), call_id)
            continue
        transcript_path = synthetic_dir / f"{call_id}.txt"
        if not transcript_path.is_file():
            logger.error("[%d/%d] %s: transcript missing, skipping", index, len(call_ids), call_id)
            continue
        transcript = transcript_path.read_text(encoding="utf-8")
        transcripts[call_id] = transcript

        cache_path = cache_dir / f"{call_id}.json" if cache_dir is not None else None
        if cache_path is not None and cache_path.is_file():
            agent_output = CallAnalysis.model_validate_json(cache_path.read_text(encoding="utf-8"))
            logger.info("[%d/%d] %s: agent output from cache", index, len(call_ids), call_id)
        else:
            try:
                agent_output = extractor(transcript_path)
            except Exception as exc:
                logger.error("[%d/%d] %s: agent extraction failed: %r", index, len(call_ids), call_id, exc)
                continue
            if cache_path is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
                cache_path.write_text(agent_output.model_dump_json(), encoding="utf-8")
            if index < len(call_ids):
                time.sleep(sleep_seconds)

        diff_path = first_difference(agent_output, truth)
        if diff_path is None:
            logger.info("[%d/%d] %s: agent matches ground truth", index, len(call_ids), call_id)
            continue

        corrections.append(
            Correction(
                call_id=call_id,
                original=agent_output,
                corrected=truth,
                error_location=diff_path,
                reviewer_id=DIFF_AUGMENTATION_REVIEWER_ID,
                note=DIFF_AUGMENTATION_NOTE,
                timestamp=datetime.now(UTC),
            )
        )
        logger.info("[%d/%d] %s: diff at %s", index, len(call_ids), call_id, diff_path)

    return corrections, transcripts


def _build_examples(
    sourced_corrections: list[tuple[Correction, str, str]],
    held_out: set[str],
) -> tuple[list[DatasetExample], list[dict[str, Any]]]:
    """Turn `(Correction, transcript, source)` triples into grounded, deduplicated examples.

    Drops (and logs, in the returned list — never silently) any correction
    whose call is held out, has no transcript, targets a field shape this
    script doesn't recognize, whose corrected claims fail the grounding
    check (`sawti.agent.nodes.ground.is_grounded`), or whose target field
    can't be resolved at all.
    """
    examples: list[DatasetExample] = []
    dropped: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for correction, transcript, source in sourced_corrections:
        if correction.call_id in held_out:
            continue
        if not transcript:
            dropped.append(
                {
                    "call_id": correction.call_id,
                    "error_location": correction.error_location,
                    "reason": "no transcript text available",
                }
            )
            continue

        try:
            claims = _relevant_claims(correction.corrected, correction.error_location)
        except ValueError as exc:
            dropped.append(
                {
                    "call_id": correction.call_id,
                    "error_location": correction.error_location,
                    "reason": str(exc),
                }
            )
            continue

        ungrounded = [claim for claim in claims if not is_grounded(transcript, claim)]
        if ungrounded:
            dropped.append(
                {
                    "call_id": correction.call_id,
                    "error_location": correction.error_location,
                    "reason": f"{len(ungrounded)} corrected claim(s) failed the grounding check",
                }
            )
            continue

        corrected_field = _extract_field(correction.corrected, correction.error_location)
        if corrected_field is None:
            dropped.append(
                {
                    "call_id": correction.call_id,
                    "error_location": correction.error_location,
                    "reason": "corrected field not resolvable",
                }
            )
            continue
        original_field = _extract_field(correction.original, correction.error_location)

        corrected_json = json.dumps(corrected_field, sort_keys=True)
        dedupe_key = (correction.call_id, correction.error_location, corrected_json)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        examples.append(
            DatasetExample(
                call_id=correction.call_id,
                language=correction.corrected.language,
                error_location=correction.error_location,
                topic=topic_for(correction.error_location),
                source=source,
                reviewer_id=correction.reviewer_id,
                transcript=transcript,
                original_field=original_field,
                corrected_field=corrected_field,
                note=correction.note,
            )
        )

    return examples, dropped


def stratified_split(
    examples: list[DatasetExample], *, val_fraction: float, seed: int
) -> tuple[list[DatasetExample], list[DatasetExample]]:
    """Split `examples` 85/15 (by default) within each topic, not across the whole set.

    A topic with fewer than 2 examples goes entirely to train — one example
    can't be both a rare type's only training signal and its only
    validation signal, and it must not become val-only.
    """
    by_topic: dict[str, list[DatasetExample]] = defaultdict(list)
    for example in examples:
        by_topic[example.topic].append(example)

    rng = Random(seed)
    train: list[DatasetExample] = []
    val: list[DatasetExample] = []
    for topic in sorted(by_topic):
        group = sorted(by_topic[topic], key=lambda e: (e.call_id, e.error_location))
        rng.shuffle(group)
        if len(group) < 2:
            train.extend(group)
            continue
        n_val = max(1, round(len(group) * val_fraction))
        n_val = min(n_val, len(group) - 1)
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return train, val


def _to_record(example: DatasetExample) -> dict[str, Any]:
    default_instruction = f"Correct the '{example.topic}' field of this call analysis."
    instruction = INSTRUCTION_BY_TOPIC.get(example.topic, default_instruction)
    return {
        "call_id": example.call_id,
        "language": example.language.value,
        "error_location": example.error_location,
        "topic": example.topic,
        "source": example.source,
        "reviewer_id": example.reviewer_id,
        "transcript": example.transcript,
        "instruction": instruction,
        "input": json.dumps(example.original_field, ensure_ascii=False, sort_keys=True),
        "output": json.dumps(example.corrected_field, ensure_ascii=False, sort_keys=True),
        "note": example.note,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def build_dataset(
    *,
    languages: list[Language],
    ground_truth_dir: Path = DEFAULT_GROUND_TRUTH_DIR,
    synthetic_dir: Path = DEFAULT_SYNTHETIC_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    extractor: Extractor | None = None,
    sleep_seconds: float = 5.0,
    diff_cache_dir: Path | None = None,
    skip_diff_augmentation: bool = False,
    held_out_fraction: float = HELD_OUT_FRACTION,
    held_out_seed: int = HELD_OUT_SEED,
    val_fraction: float = VAL_FRACTION,
    split_seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    """Build `train.jsonl`/`val.jsonl`/`MANIFEST.json` in `output_dir` for `languages`.

    Raises:
        RuntimeError: A held-out call_id (any included language) ended up in
            train or val — hard failure, never a warning.
    """
    run_extractor = extractor or extract_via_graph
    output_dir.mkdir(parents=True, exist_ok=True)

    all_examples: list[DatasetExample] = []
    all_dropped: list[dict[str, Any]] = []
    held_out_by_language: dict[str, list[str]] = {}
    manifest_languages: dict[str, Any] = {}

    for language in languages:
        ground_truth = load_ground_truth(ground_truth_dir)
        call_ids = sorted(record.call_id for record in ground_truth if record.language == language)
        held_out = ensure_held_out_call_ids(
            call_ids,
            language=language,
            output_dir=output_dir,
            fraction=held_out_fraction,
            seed=held_out_seed,
        )
        held_out_by_language[language.value] = held_out
        held_out_set = set(held_out)
        call_id_set = set(call_ids)
        eligible = [call_id for call_id in call_ids if call_id not in held_out_set]

        # Scoped to this run's corpus: a Postgres correction for a call_id
        # outside `ground_truth_dir` (a stale row from a prior corpus
        # revision, or another test's fixture data) must never enter this
        # dataset — see docs/09-DECISIONS.md, 2026-09-25.
        pg_pairs = [
            (correction, transcript, "reviewer_action")
            for correction, transcript in _pg_corrections_for_language(language)
            if correction.call_id in call_id_set
        ]

        if skip_diff_augmentation:
            diff_corrections: list[Correction] = []
            diff_transcripts: dict[str, str] = {}
        else:
            diff_corrections, diff_transcripts = _diff_augmented_pairs(
                eligible,
                ground_truth_dir=ground_truth_dir,
                synthetic_dir=synthetic_dir,
                extractor=run_extractor,
                cache_dir=diff_cache_dir,
                sleep_seconds=sleep_seconds,
            )
        diff_pairs = [
            (correction, diff_transcripts[correction.call_id], "phase1_2_diff")
            for correction in diff_corrections
        ]

        examples, dropped = _build_examples(pg_pairs + diff_pairs, held_out_set)
        all_examples.extend(examples)
        all_dropped.extend(dropped)

        manifest_languages[language.value] = {
            "corpus_size": len(call_ids),
            "held_out_count": len(held_out),
            "eligible_count": len(eligible),
            "examples_from_reviewer_action": sum(1 for e in examples if e.source == "reviewer_action"),
            "examples_from_phase1_2_diff": sum(1 for e in examples if e.source == "phase1_2_diff"),
            "total_examples": len(examples),
        }

    train, val = stratified_split(all_examples, val_fraction=val_fraction, seed=split_seed)

    held_out_flat = {call_id for ids in held_out_by_language.values() for call_id in ids}
    leaked = held_out_flat & ({e.call_id for e in train} | {e.call_id for e in val})
    if leaked:
        message = f"Held-out call_ids leaked into train/val, refusing to write output: {sorted(leaked)}"
        raise RuntimeError(message)

    _write_jsonl(output_dir / "train.jsonl", [_to_record(e) for e in train])
    _write_jsonl(output_dir / "val.jsonl", [_to_record(e) for e in val])

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "languages_included": [language.value for language in languages],
        "held_out_fraction": held_out_fraction,
        "held_out_seed": held_out_seed,
        "val_fraction": val_fraction,
        "split_seed": split_seed,
        "diff_augmentation_skipped": skip_diff_augmentation,
        "held_out_call_ids_file": "held_out_call_ids.json",
        "per_language": manifest_languages,
        "train_count": len(train),
        "val_count": len(val),
        "dropped": all_dropped,
        "caveat": DATASET_CAVEAT,
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (output_dir / "MANIFEST.json").write_text(manifest_text, encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GROUND_TRUTH_DIR)
    parser.add_argument("--synthetic-dir", type=Path, default=DEFAULT_SYNTHETIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-en", action="store_true", help="Opt in to en corrections. Prints a caveat."
    )
    parser.add_argument(
        "--include-mixed", action="store_true", help="Opt in to mixed corrections. Prints a caveat."
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=5.0,
        help="Pacing between live LLM calls (15/min free-tier ceiling).",
    )
    parser.add_argument(
        "--skip-diff-augmentation",
        action="store_true",
        help="Use only existing Postgres corrections, no live LLM calls.",
    )
    parser.add_argument(
        "--diff-cache-dir",
        type=Path,
        default=None,
        help="Cache agent outputs here by call_id (default: <output-dir>/_diff_cache).",
    )
    parser.add_argument("--held-out-fraction", type=float, default=HELD_OUT_FRACTION)
    parser.add_argument("--held-out-seed", type=int, default=HELD_OUT_SEED)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    languages = [Language.AR]
    if args.include_en:
        logger.warning(EN_MIXED_CAVEAT)
        languages.append(Language.EN)
    if args.include_mixed:
        logger.warning(EN_MIXED_CAVEAT)
        languages.append(Language.MIXED)

    diff_cache_dir = args.diff_cache_dir or (args.output_dir / "_diff_cache")

    manifest = build_dataset(
        languages=languages,
        ground_truth_dir=args.ground_truth_dir,
        synthetic_dir=args.synthetic_dir,
        output_dir=args.output_dir,
        sleep_seconds=args.sleep_seconds,
        diff_cache_dir=diff_cache_dir,
        skip_diff_augmentation=args.skip_diff_augmentation,
        held_out_fraction=args.held_out_fraction,
        held_out_seed=args.held_out_seed,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
    )
    print(json.dumps(manifest["per_language"], indent=2, sort_keys=True))
    dropped_count = len(manifest["dropped"])
    print(f"train: {manifest['train_count']}, val: {manifest['val_count']}, dropped: {dropped_count}")


if __name__ == "__main__":
    main()
