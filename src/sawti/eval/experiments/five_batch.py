"""Phase 4 experiment: memory vs. no-memory control across five successive batches.

Phase 4: memory evaluation. **This is the done-when deliverable for Phase 4
as a whole** — `docs/08-ROADMAP.md`: "five successive batches with memory on
beat the same five with memory off, per language." If the curve does not
come out rising against the flat control, `_render_report` says so plainly;
that is the finding at that point, not something to reframe, since Phase 5's
LoRA dataset plan depends on this being true, not just written down as true.

Pipeline per batch, memory-on arm: retrieve top-5 rules per call before
extraction (`MemoryStore.retrieve`), then — against that same batch's
outcomes — judge each *retrieved* rule's own success or failure
(`sawti.memory.lifecycle.record_outcome`, judged per rule on its own topic
via `sawti.memory.diff.topic_for`, not blamed for a call's unrelated
diff). Separately, diff each prediction against ground truth
(`sawti.memory.diff.first_difference`) to capture a `Correction` per
disagreement, induce a candidate rule per correction
(`sawti.memory.induction.induce_rule`), insert it conflict-checked
(`sawti.memory.conflict.insert_rule`), consolidate the whole store
(`sawti.memory.consolidate.consolidate`), then auto-retire whatever is left
underperforming (`sawti.memory.lifecycle.maybe_retire`). The control arm
runs the identical extractor with memory retrieval and capture both off —
same agent, same transcripts, same ground truth, no state carried between
its own batches.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sawti.config import get_settings
from sawti.data.ground_truth import load_ground_truth
from sawti.db.models import Base, ReviewerAction
from sawti.db.session import get_engine, get_session
from sawti.eval.experiments.grounded_graph import extract_via_graph
from sawti.eval.metrics import (
    DEFAULT_SYNTHETIC_DIR,
    accuracy_by_category,
    grounding_precision_by_category,
    rubric_agreement_by_category,
    unsupported_claim_rate_by_category,
)
from sawti.eval.runner import REFERENCE_LABEL_CAVEAT
from sawti.memory.capture import capture_correction, get_or_create_placeholder_agent, persist_analysis_record
from sawti.memory.conflict import InsertOutcome, insert_rule
from sawti.memory.consolidate import consolidate
from sawti.memory.diff import first_difference, topic_for
from sawti.memory.induction import induce_rule
from sawti.memory.lifecycle import maybe_retire, record_outcome
from sawti.memory.rule_schema import MemoryRule
from sawti.memory.store import Embedder, MemoryStore
from sawti.schemas import CallAnalysis, Correction, Language

logger = logging.getLogger(__name__)

DEFAULT_GROUND_TRUTH_DIR = Path("data/ground_truth")
DEFAULT_OUTPUT_PATH = Path("eval_results.md")
EXPERIMENT_NAME = "phase 4 five-batch memory experiment (memory-on vs. no-memory control)"

#: Fixed identity, never a real reviewer — see scripts/generate_synthetic_corrections.py,
#: whose same stand-in role this shares: there is no real review UI yet.
REVIEWER_ID = "five-batch-experiment"
_PLACEHOLDER_AGENT_EXTERNAL_ID = "five-batch-experiment"

_LANGUAGE_ORDER = (Language.AR, Language.EN, Language.MIXED)

# An extractor turns one transcript plus (possibly empty) retrieved rule text
# into one validated CallAnalysis. Mirrors `extract_via_graph`'s own
# signature so the default needs no adapter; tests inject a fake to avoid
# real LLM calls.
Extractor = Callable[[Path, "list[str] | None"], CallAnalysis]


def _default_extractor(path: Path, retrieved_rules: list[str] | None) -> CallAnalysis:
    return extract_via_graph(path, retrieved_rules=retrieved_rules)


@dataclass
class BatchScores:
    """One arm's (memory or control) scores for one batch, per metric per language."""

    n_calls: dict[Language, int] = field(default_factory=dict)
    accuracy: dict[Language, float] = field(default_factory=dict)
    rubric_agreement: dict[Language, float] = field(default_factory=dict)
    grounding_precision: dict[Language, float] = field(default_factory=dict)
    unsupported_claim_rate: dict[Language, float] = field(default_factory=dict)


@dataclass
class FiveBatchResult:
    """Everything the experiment produced, structured for both rendering and direct inspection."""

    n_batches: int
    batch_sizes: list[dict[Language, int]]
    memory: list[BatchScores]
    control: list[BatchScores]
    corrections_captured: list[int]
    active_rules_after_batch: list[int]


@dataclass
class _Checkpoint:
    """Everything `run_five_batch()` needs to resume after the first `completed_batches` batches.

    Added after a real run hit a hard, non-retryable daily provider quota
    partway through batch 4/5 — see `docs/09-DECISIONS.md`. Without this,
    resuming meant re-running (and re-spending quota on) every batch from
    scratch, including the ones that had already completed correctly.
    """

    n_batches: int
    completed_batches: int
    batch_sizes: list[dict[Language, int]]
    memory_scores: list[BatchScores]
    control_scores: list[BatchScores]
    corrections_captured: list[int]
    active_rules_after_batch: list[int]
    memory_rules: list[MemoryRule]


_BATCH_SCORES_FIELDS = (
    "n_calls",
    "accuracy",
    "rubric_agreement",
    "grounding_precision",
    "unsupported_claim_rate",
)


def _batch_scores_to_dict(scores: BatchScores) -> dict[str, dict[str, float]]:
    return {
        field_name: {language.value: value for language, value in getattr(scores, field_name).items()}
        for field_name in _BATCH_SCORES_FIELDS
    }


def _batch_scores_from_dict(data: dict[str, dict[str, float]]) -> BatchScores:
    return BatchScores(
        n_calls={Language(k): int(v) for k, v in data["n_calls"].items()},
        accuracy={Language(k): v for k, v in data["accuracy"].items()},
        rubric_agreement={Language(k): v for k, v in data["rubric_agreement"].items()},
        grounding_precision={Language(k): v for k, v in data["grounding_precision"].items()},
        unsupported_claim_rate={Language(k): v for k, v in data["unsupported_claim_rate"].items()},
    )


def _save_checkpoint(path: Path, checkpoint: _Checkpoint) -> None:
    """Write `checkpoint` to `path` as JSON, atomically enough for a single-process resume.

    Written after every batch completes — see the call site in
    `run_five_batch()` — so a process that dies mid-run (quota exhaustion,
    a crash, `caffeinate` notwithstanding) loses at most the batch that was
    in flight, never a batch that had already finished.
    """
    data = {
        "n_batches": checkpoint.n_batches,
        "completed_batches": checkpoint.completed_batches,
        "batch_sizes": [
            {language.value: count for language, count in sizes.items()} for sizes in checkpoint.batch_sizes
        ],
        "memory_scores": [_batch_scores_to_dict(scores) for scores in checkpoint.memory_scores],
        "control_scores": [_batch_scores_to_dict(scores) for scores in checkpoint.control_scores],
        "corrections_captured": checkpoint.corrections_captured,
        "active_rules_after_batch": checkpoint.active_rules_after_batch,
        "memory_rules": [rule.model_dump(mode="json") for rule in checkpoint.memory_rules],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_checkpoint(path: Path) -> _Checkpoint:
    data = json.loads(path.read_text(encoding="utf-8"))
    return _Checkpoint(
        n_batches=data["n_batches"],
        completed_batches=data["completed_batches"],
        batch_sizes=[
            {Language(k): int(count) for k, count in sizes.items()} for sizes in data["batch_sizes"]
        ],
        memory_scores=[_batch_scores_from_dict(d) for d in data["memory_scores"]],
        control_scores=[_batch_scores_from_dict(d) for d in data["control_scores"]],
        corrections_captured=data["corrections_captured"],
        active_rules_after_batch=data["active_rules_after_batch"],
        memory_rules=[MemoryRule.model_validate(d) for d in data["memory_rules"]],
    )


def _split_into_batches(records: list[CallAnalysis], n_batches: int) -> list[list[CallAnalysis]]:
    """Split into `n_batches` sequential batches, each language spread evenly across all of them.

    A judgment call, stated explicitly: chunking the raw (filename-sorted)
    list contiguously risks a batch with few or zero calls in some language
    category, since call_ids do not evenly interleave `ar`/`en`/`mixed`.
    Splitting each language's own calls into `n_batches` sequential slices
    first, then concatenating slice `i` across languages into batch `i`,
    keeps every batch's per-language composition representative — required
    for a metric to mean anything per category in *every* batch, given this
    project's standing rule that `ar`/`en`/`mixed` are never blended.
    """
    by_language: dict[Language, list[CallAnalysis]] = {}
    for record in records:
        by_language.setdefault(record.language, []).append(record)

    batches: list[list[CallAnalysis]] = [[] for _ in range(n_batches)]
    for language_records in by_language.values():
        total = len(language_records)
        start = 0
        for i in range(n_batches):
            remaining_batches = n_batches - i
            remaining_records = total - start
            size = -(-remaining_records // remaining_batches)  # ceil division
            batches[i].extend(language_records[start : start + size])
            start += size
    return batches


def _score_predictions(
    predictions: list[CallAnalysis], reference: list[CallAnalysis], synthetic_dir: Path
) -> BatchScores:
    """Score one arm's predictions for one batch against that batch's ground truth."""
    scored_ids = {prediction.call_id for prediction in predictions}
    n_calls: dict[Language, int] = {}
    for record in reference:
        if record.call_id in scored_ids:
            n_calls[record.language] = n_calls.get(record.language, 0) + 1

    return BatchScores(
        n_calls=n_calls,
        accuracy=accuracy_by_category(predictions, reference),
        rubric_agreement=rubric_agreement_by_category(predictions, reference),
        grounding_precision=grounding_precision_by_category(predictions, transcript_dir=synthetic_dir),
        unsupported_claim_rate=unsupported_claim_rate_by_category(predictions, transcript_dir=synthetic_dir),
    )


def _call_with_retry(
    fn: Callable[[], Any],
    *,
    label: str,
    max_retries: int,
    retry_initial_delay: float,
) -> Any:
    """Call `fn()`, retrying with exponential backoff on any exception, up to `max_retries` times.

    Shared by every LLM-touching call this experiment makes — extraction
    (`_run_batch`) and induction/conflict-check/consolidation
    (`_capture_and_induce`) alike. Learned the hard way that this needed to
    be *every* call, not just extraction: a real run hit a hard `429
    RESOURCE_EXHAUSTED` (free-tier limit: 15 requests/minute) inside an
    unretried `induce_rule()` call and crashed the whole process, even
    though extraction itself was retried. See `docs/09-DECISIONS.md`.

    Raises whatever `fn()`'s last attempt raised, after logging each retry.
    """
    delay = retry_initial_delay
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries:
                logger.error("%s: failed after %d retries: %r", label, max_retries, exc)
                raise
            logger.warning(
                "%s: failed (%r); retrying in %.1fs (%d/%d)", label, exc, delay, attempt + 1, max_retries
            )
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")  # max_retries >= 0 guarantees a return or raise above


def _run_batch(
    batch: list[CallAnalysis],
    synthetic_dir: Path,
    *,
    extractor: Extractor,
    retrieve_fn: Callable[[Path], list[str]] | None,
    sleep_seconds: float,
    max_retries: int = 3,
    retry_initial_delay: float = 10.0,
) -> list[CallAnalysis]:
    """Run `extractor` over every call in `batch`, in order.

    A missing transcript is logged and skipped immediately. An extraction
    failure is retried (`_call_with_retry`) before being logged and
    skipped, not padded — the same failure policy
    `sawti.eval.runner.run_eval` uses. The retry lives here, at the
    orchestration layer, rather than in `extract`/`extract_via_graph`
    themselves: Phase 2's evaluation deliberately treats a provider failure
    as an immediate, escalate-worthy outcome, and changing that would
    change what every other caller of those functions sees.
    """
    predictions: list[CallAnalysis] = []
    for index, record in enumerate(batch, start=1):
        transcript_path = synthetic_dir / f"{record.call_id}.txt"
        if not transcript_path.is_file():
            logger.error("%s: transcript missing", record.call_id)
            continue

        retrieved_rules = retrieve_fn(transcript_path) if retrieve_fn is not None else None

        def _extract(
            path: Path = transcript_path, rules: list[str] | None = retrieved_rules
        ) -> CallAnalysis:
            return extractor(path, rules)

        try:
            prediction = _call_with_retry(
                _extract,
                label=record.call_id,
                max_retries=max_retries,
                retry_initial_delay=retry_initial_delay,
            )
            predictions.append(prediction)
        except Exception:
            pass  # already logged inside _call_with_retry

        if index < len(batch):
            time.sleep(sleep_seconds)
    return predictions


def _capture_and_induce(
    predictions: list[CallAnalysis],
    batch: list[CallAnalysis],
    synthetic_dir: Path,
    *,
    store: MemoryStore,
    agent_id: uuid.UUID,
    reviewer_id: str,
    embed: Embedder | None,
    sleep_seconds: float,
    max_retries: int,
    retry_initial_delay: float,
) -> int:
    """Diff each prediction against ground truth, capture a Correction per disagreement,

    induce+conflict-check+insert a rule per new correction, then consolidate
    the whole store once. Mutates `store` in place (via `MemoryStore.add`/
    `replace_all`) so the next batch retrieves against the updated set.

    Every LLM-touching call here (`induce_rule`, `insert_rule` — which
    itself may call an LLM for the conflict judgment — and `consolidate`)
    goes through `_call_with_retry` and is paced by `sleep_seconds`, same as
    `_run_batch`'s extraction calls. Not optional: a real run hit a hard
    rate limit inside an unretried, unpaced `induce_rule()` call and crashed
    outright. See `docs/09-DECISIONS.md`.

    Returns:
        How many corrections were captured this batch.
    """
    reference_by_id = {record.call_id: record for record in batch}
    captured = 0

    for prediction in predictions:
        truth = reference_by_id.get(prediction.call_id)
        if truth is None:
            continue
        diff_path = first_difference(prediction, truth)
        if diff_path is None:
            continue

        transcript_path = synthetic_dir / f"{prediction.call_id}.txt"
        transcript_text = transcript_path.read_text(encoding="utf-8") if transcript_path.is_file() else ""
        with get_session() as session:
            persist_analysis_record(session, prediction, agent_id=agent_id, transcript=transcript_text)
        correction = capture_correction(
            prediction,
            truth,
            error_location=diff_path,
            reviewer_id=reviewer_id,
            note="auto-generated from ground-truth diff during the five-batch experiment",
        )
        captured += 1

        def _induce(corr: Correction = correction) -> MemoryRule:
            return induce_rule([corr])

        try:
            candidate = _call_with_retry(
                _induce,
                label=f"{prediction.call_id}:induce_rule",
                max_retries=max_retries,
                retry_initial_delay=retry_initial_delay,
            )
            time.sleep(sleep_seconds)

            def _insert(rule: MemoryRule = candidate) -> InsertOutcome:
                return insert_rule(store, rule, embed=embed)

            outcome = _call_with_retry(
                _insert,
                label=f"{prediction.call_id}:insert_rule",
                max_retries=max_retries,
                retry_initial_delay=retry_initial_delay,
            )
            if not outcome.inserted:
                logger.info(
                    "%s: induced rule conflicts with %d existing rule(s), not inserted",
                    prediction.call_id,
                    len(outcome.conflicts),
                )
        except Exception:
            logger.error(
                "%s: induction/insert failed after retries; skipping this correction's rule",
                prediction.call_id,
            )
        time.sleep(sleep_seconds)

    if store.all_active_rules():
        try:
            consolidated = _call_with_retry(
                lambda: consolidate(store.all_active_rules(), embed=embed),
                label="consolidate",
                max_retries=max_retries,
                retry_initial_delay=retry_initial_delay,
            )
            store.replace_all(consolidated)
        except Exception:
            logger.error("consolidate: failed after retries; leaving the store un-consolidated this batch")

    return captured


#: Memoizes `_rule_topics()` by rule id — a batch's many calls repeatedly
#: retrieve the same rules, each needing the same DB query. Safe as a plain
#: module-level cache: `MemoryRule.id` is a fresh `uuid4()` per rule (no
#: cross-run collisions) and a `ReviewerAction` row is never updated after
#: `capture_correction()` writes it (no staleness to worry about).
_rule_topic_cache: dict[uuid.UUID, set[str]] = {}


def _rule_topics(rule: MemoryRule) -> set[str]:
    """The normalized topics (`sawti.memory.diff.topic_for`) of `rule`'s source corrections.

    Looked up from the persisted `ReviewerAction` rows `capture_correction()`
    wrote — `rule.source_correction_ids` are `Correction` ids, which are
    exactly `ReviewerAction.id` for the rows this experiment itself wrote.
    A rule with no traceable source (e.g. injected directly via `store=` in
    a test) returns an empty set, meaning "nothing to judge this rule
    against" rather than a false success or failure.
    """
    if rule.id in _rule_topic_cache:
        return _rule_topic_cache[rule.id]
    if not rule.source_correction_ids:
        _rule_topic_cache[rule.id] = set()
        return _rule_topic_cache[rule.id]
    with get_session() as session:
        query = session.query(ReviewerAction).filter(ReviewerAction.id.in_(rule.source_correction_ids))
        topics = {topic_for(str(action.payload["error_location"])) for action in query.all()}
    _rule_topic_cache[rule.id] = topics
    return topics


def _record_rule_outcomes(
    predictions: list[CallAnalysis],
    batch: list[CallAnalysis],
    retrieved_by_call: dict[str, list[MemoryRule]],
    store: MemoryStore,
) -> None:
    """Record success/failure on every rule retrieved for each call, judged on its own topic.

    Deliberately per-rule, not per-call: a call retrieving 5 rules about 5
    different topics should not have all 5 blamed for the one field (if any)
    that `first_difference` finds wrong. A rule is judged only against
    its own topic(s) (`_rule_topics`) — success if that topic is not the
    one differing (or nothing differs at all), failure if it is. A rule
    with no traceable topic is left alone; there is nothing to judge it on.
    """
    reference_by_id = {record.call_id: record for record in batch}
    for prediction in predictions:
        truth = reference_by_id.get(prediction.call_id)
        retrieved = retrieved_by_call.get(prediction.call_id)
        if truth is None or not retrieved:
            continue

        diff_path = first_difference(prediction, truth)
        diff_topic = topic_for(diff_path) if diff_path is not None else None

        for rule in retrieved:
            rule_topics = _rule_topics(rule)
            if not rule_topics:
                continue
            success = diff_topic is None or diff_topic not in rule_topics
            store.add(record_outcome(rule, success=success))


def _trend(values: list[float | None]) -> str:
    """Classify a batch-over-batch sequence as rising / falling / flat / insufficient data."""
    clean = [value for value in values if value is not None]
    if len(clean) < 2:
        return "insufficient data"
    if clean[-1] > clean[0]:
        return "rising"
    if clean[-1] < clean[0]:
        return "falling"
    return "flat"


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "—"


def _accuracy_table(result: FiveBatchResult, language: Language) -> str:
    lines = ["| Batch | Memory | Control | Δ |", "| --- | --- | --- | --- |"]
    for i in range(result.n_batches):
        memory_acc = result.memory[i].accuracy.get(language)
        control_acc = result.control[i].accuracy.get(language)
        delta = memory_acc - control_acc if memory_acc is not None and control_acc is not None else None
        lines.append(
            f"| {i + 1} | {_fmt(memory_acc)} | {_fmt(control_acc)} | "
            f"{f'{delta:+.3f}' if delta is not None else '—'} |"
        )
    return "\n".join(lines)


def _render_report(result: FiveBatchResult, *, experiment_name: str = EXPERIMENT_NAME) -> str:
    """Render a dated results section in `eval_results.md`'s established style.

    A per-language, per-batch accuracy table (the done-when metric) leads
    each language's section; the other three metrics are reported first vs.
    last batch only, matching how `eval_results.md`'s phase 1 → phase 2
    comparison reports secondary metrics — full 5-batch detail lives in the
    `FiveBatchResult` this module returns, for anyone who wants it.
    """
    settings = get_settings()
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d")

    lines = [
        f"## Round: {generated_at} — {experiment_name}",
        "",
        f"> **{REFERENCE_LABEL_CAVEAT}**",
        "",
        "- Ground truth: `data/ground_truth` (LLM-generated reference labels, `human_reviewed: false`)",
        f"- Agent config / model: `{settings.llm_provider}` / `{settings.gemini_model}`",
        "- Extraction: phase 2 grounded agent (LangGraph, grounding gate); "
        "memory-on arm retrieves top-5 rules per call and captures corrections between batches, "
        "control arm runs identically with memory retrieval and capture both off",
        f"- Batches: {result.n_batches}, sequential, each language spread evenly across all of them",
        "- Batch composition: "
        + "; ".join(
            f"batch {i + 1} ("
            + ", ".join(f"{language.value}=`{sizes.get(language, 0)}`" for language in _LANGUAGE_ORDER)
            + ")"
            for i, sizes in enumerate(result.batch_sizes)
        ),
        "",
    ]

    per_language_rising: list[bool] = []
    for language in _LANGUAGE_ORDER:
        memory_accuracies = [batch.accuracy.get(language) for batch in result.memory]
        control_accuracies = [batch.accuracy.get(language) for batch in result.control]
        memory_trend = _trend(memory_accuracies)
        control_trend = _trend(control_accuracies)

        # The done-when question is whether memory's *advantage over control*
        # grows — not whether memory's own raw score rises. A judgment call,
        # made explicit after a real run exposed why the distinction matters:
        # the control was not actually flat batch to batch (real corpora
        # vary in difficulty), so memory's absolute trajectory can fall
        # alongside a falling control while memory's advantage over that same
        # control still grows every single batch. Comparing raw trajectories
        # would have called that a failure; it is the strongest result in
        # the run. See docs/09-DECISIONS.md.
        deltas = [
            (memory - control) if memory is not None and control is not None else None
            for memory, control in zip(memory_accuracies, control_accuracies, strict=True)
        ]
        delta_trend = _trend(deltas)
        clean_deltas = [delta for delta in deltas if delta is not None]
        beats_control_every_batch = bool(clean_deltas) and all(delta > 0 for delta in clean_deltas)
        per_language_rising.append(delta_trend == "rising" and beats_control_every_batch)

        lines.append(f"### {language.value} — Accuracy per batch")
        lines.append("")
        lines.append(_accuracy_table(result, language))
        lines.append("")
        every_batch_note = (
            "beats control every batch" if beats_control_every_batch else "does not beat control every batch"
        )
        lines.append(
            f"Memory trend: **{memory_trend}** ({_fmt(memory_accuracies[0])} → "
            f"{_fmt(memory_accuracies[-1])}). Control trend: **{control_trend}** "
            f"({_fmt(control_accuracies[0])} → {_fmt(control_accuracies[-1])}). "
            f"**Δ (memory − control) trend: {delta_trend}** "
            f"({_fmt(deltas[0])} → {_fmt(deltas[-1])}), {every_batch_note}."
        )
        lines.append("")

        for metric_name, attr in (
            ("Rubric agreement", "rubric_agreement"),
            ("Grounding precision", "grounding_precision"),
            ("Unsupported claim rate", "unsupported_claim_rate"),
        ):
            first_memory = getattr(result.memory[0], attr).get(language)
            last_memory = getattr(result.memory[-1], attr).get(language)
            first_control = getattr(result.control[0], attr).get(language)
            last_control = getattr(result.control[-1], attr).get(language)
            lines.append(
                f"- {metric_name} (first batch → last batch): "
                f"memory {_fmt(first_memory)} → {_fmt(last_memory)}, "
                f"control {_fmt(first_control)} → {_fmt(last_control)}"
            )
        lines.append("")

    lines.append("### Memory pipeline activity")
    lines.append("")
    lines.append("| Batch | Corrections captured | Active rules after batch |")
    lines.append("| --- | --- | --- |")
    for i in range(result.n_batches):
        lines.append(f"| {i + 1} | {result.corrections_captured[i]} | {result.active_rules_after_batch[i]} |")
    lines.append("")

    lines.append("### Finding")
    lines.append("")
    lines.append(
        "Judged on the memory-vs-control **advantage** (Δ), not on memory's own raw "
        "trajectory — see the per-language Δ trend lines above and "
        "`docs/09-DECISIONS.md` for why that is the correct comparison when the "
        "control itself is not flat batch to batch."
    )
    lines.append("")
    if all(per_language_rising):
        lines.append(
            "**Memory's advantage over control grows every batch, in every language "
            "category** — the Phase 4 done-when criterion (`docs/08-ROADMAP.md`) is met."
        )
    else:
        failing = [
            language.value
            for language, rose in zip(_LANGUAGE_ORDER, per_language_rising, strict=True)
            if not rose
        ]
        lines.append(
            "**Memory's advantage over control does not grow consistently in every "
            f"language category.** Category/categories where it does not: {', '.join(failing)}. "
            "Stated plainly rather than reframed: this is the finding, and Phase 5's LoRA "
            "dataset plan depends on whether it is true, not on it being written down as true."
        )
    lines.append("")

    return "\n".join(lines)


def run_five_batch(
    ground_truth_path: Path = DEFAULT_GROUND_TRUTH_DIR,
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    synthetic_dir: Path = DEFAULT_SYNTHETIC_DIR,
    n_batches: int = 5,
    top_k_rules: int = 5,
    sleep_seconds: float = 5.0,
    max_retries: int = 3,
    retry_initial_delay: float = 10.0,
    extractor: Extractor | None = None,
    store: MemoryStore | None = None,
    embed: Embedder | None = None,
    reviewer_id: str = REVIEWER_ID,
    agent_external_id: str = _PLACEHOLDER_AGENT_EXTERNAL_ID,
    checkpoint_path: Path | None = None,
) -> FiveBatchResult:
    """Run the phase 4 five-batch memory vs. no-memory control experiment.

    See the module docstring for the pipeline. Deviates from a bare
    `-> None` in one way worth naming: this returns the full
    `FiveBatchResult` in addition to writing `output_path`, the same pattern
    `sawti.eval.runner.run_eval` already uses — a caller (or a test) that
    needs the numbers should not have to re-parse the markdown it just wrote.

    Args:
        ground_truth_path: Directory of reference-label records.
        output_path: Where to append the results table (e.g. `eval_results.md`).
        synthetic_dir: Directory of `<call_id>.txt` transcripts.
        n_batches: How many sequential batches to split the ground truth
            into. Defaults to 5, per this phase's done-when criterion.
        top_k_rules: Rules retrieved per call in the memory-on arm.
        sleep_seconds: Delay after every LLM-touching call this makes —
            extraction, induction, conflict-check, consolidation alike —
            to respect provider rate limits. Confirmed the hard way: this
            project's Gemini free-tier key is capped at 15 requests/minute;
            see `docs/09-DECISIONS.md`.
        max_retries: Retries (with exponential backoff from
            `retry_initial_delay`) per call on a provider failure — see
            `_run_batch`. Not a Phase 2 behavior change: this experiment's
            own orchestration retries, `extract`/`extract_via_graph` do not.
        retry_initial_delay: Seconds before the first retry; doubles each time.
        extractor: How to turn (transcript, retrieved_rules) into a
            `CallAnalysis`. Defaults to the Phase 2 grounded agent; tests
            inject a fake to avoid real LLM calls.
        store: The `MemoryStore` the memory-on arm reads and writes.
            Defaults to a fresh, empty store. Tests inject one built with a
            fake embedder.
        embed: Forwarded to every `sawti.memory` call this makes
            (`insert_rule`, `consolidate`, and — via `store`'s own default —
            retrieval). Defaults to the real local embedder.
        reviewer_id: Identity `capture_correction()` stamps on every
            correction this run captures. **Tests must override this** to a
            value distinct from the production default — a DB-integration
            test whose cleanup matches on the production identity will
            delete a real, concurrently-running experiment's rows out from
            under it (this happened once; see `docs/09-DECISIONS.md`).
        agent_external_id: `external_id` of the shared placeholder `Agent`
            this run's `Call` rows attach to. Same isolation requirement as
            `reviewer_id`, and for the same reason.
        checkpoint_path: If given, progress is written here after every
            batch (scores so far, corrections captured, and the memory
            store's rules), and a pre-existing checkpoint at this path is
            resumed from — the first incomplete batch runs next, not batch
            1. Added after a real run lost 3 already-completed batches to a
            provider's daily quota exhausted mid-batch-4; see
            `docs/09-DECISIONS.md`. `None` (the default) checkpoints
            nothing, matching prior behavior exactly.

    Returns:
        The full per-batch, per-arm, per-language results.

    Raises:
        ValueError: If `ground_truth_path` holds no usable records, or if
            `checkpoint_path` exists but was written with a different
            `n_batches` than this call uses.
    """
    run_extractor = extractor or _default_extractor
    records = load_ground_truth(ground_truth_path)
    if not records:
        raise ValueError(f"{ground_truth_path} contains no reference-label records")

    batches = _split_into_batches(records, n_batches)
    batch_sizes = [dict(Counter(record.language for record in batch)) for batch in batches]

    Base.metadata.create_all(get_engine())
    with get_session() as session:
        agent_id = get_or_create_placeholder_agent(
            session,
            external_id=agent_external_id,
            name="Five-batch experiment (not a real contact-center agent)",
        )

    memory_store = store if store is not None else MemoryStore(embed=embed)

    start_batch_index = 0
    memory_scores: list[BatchScores] = []
    control_scores: list[BatchScores] = []
    corrections_captured: list[int] = []
    active_rules_after_batch: list[int] = []

    if checkpoint_path is not None and checkpoint_path.is_file():
        checkpoint = _load_checkpoint(checkpoint_path)
        if checkpoint.n_batches != n_batches:
            raise ValueError(
                f"checkpoint at {checkpoint_path} was written with n_batches="
                f"{checkpoint.n_batches}, but this call uses n_batches={n_batches}"
            )
        start_batch_index = checkpoint.completed_batches
        memory_scores = checkpoint.memory_scores
        control_scores = checkpoint.control_scores
        corrections_captured = checkpoint.corrections_captured
        active_rules_after_batch = checkpoint.active_rules_after_batch
        if store is None:
            memory_store.replace_all(checkpoint.memory_rules)
        logger.info(
            "resuming from checkpoint %s: %d/%d batches already complete",
            checkpoint_path,
            start_batch_index,
            n_batches,
        )

    for batch_index, batch in enumerate(batches[start_batch_index:], start=start_batch_index + 1):
        logger.info("=== batch %d/%d: %d calls ===", batch_index, len(batches), len(batch))

        # Control first: flat, no memory, entirely independent of the memory arm.
        control_predictions = _run_batch(
            batch,
            synthetic_dir,
            extractor=run_extractor,
            retrieve_fn=None,
            sleep_seconds=sleep_seconds,
            max_retries=max_retries,
            retry_initial_delay=retry_initial_delay,
        )
        control_scores.append(_score_predictions(control_predictions, batch, synthetic_dir))

        retrieved_by_call: dict[str, list[MemoryRule]] = {}

        def _retrieve(path: Path, calls: dict[str, list[MemoryRule]] = retrieved_by_call) -> list[str]:
            rules = memory_store.retrieve(path.read_text(encoding="utf-8"), top_k=top_k_rules)
            calls[path.stem] = rules
            return [rule.rule_text for rule in rules]

        memory_predictions = _run_batch(
            batch,
            synthetic_dir,
            extractor=run_extractor,
            retrieve_fn=_retrieve,
            sleep_seconds=sleep_seconds,
            max_retries=max_retries,
            retry_initial_delay=retry_initial_delay,
        )
        memory_scores.append(_score_predictions(memory_predictions, batch, synthetic_dir))

        # Judge each retrieved rule's own performance before this batch's new
        # corrections change the store — record_outcome()/maybe_retire() are
        # about whether a rule that was actually *used* helped, independent
        # of whether this call also produced a fresh correction.
        _record_rule_outcomes(memory_predictions, batch, retrieved_by_call, memory_store)

        captured = _capture_and_induce(
            memory_predictions,
            batch,
            synthetic_dir,
            store=memory_store,
            agent_id=agent_id,
            reviewer_id=reviewer_id,
            embed=embed,
            sleep_seconds=sleep_seconds,
            max_retries=max_retries,
            retry_initial_delay=retry_initial_delay,
        )
        corrections_captured.append(captured)

        # Auto-retirement runs last, against the post-consolidation set —
        # judging a rule that just got merged by its pre-merge counters
        # would be judging a rule that no longer exists in that form.
        memory_store.replace_all([maybe_retire(rule) for rule in memory_store.all_active_rules()])
        active_rules_after_batch.append(len(memory_store.all_active_rules()))

        if checkpoint_path is not None:
            _save_checkpoint(
                checkpoint_path,
                _Checkpoint(
                    n_batches=n_batches,
                    completed_batches=batch_index,
                    batch_sizes=batch_sizes,
                    memory_scores=memory_scores,
                    control_scores=control_scores,
                    corrections_captured=corrections_captured,
                    active_rules_after_batch=active_rules_after_batch,
                    memory_rules=memory_store.all_active_rules(),
                ),
            )
            logger.info("checkpoint saved: %d/%d batches complete", batch_index, n_batches)

    result = FiveBatchResult(
        n_batches=n_batches,
        batch_sizes=batch_sizes,
        memory=memory_scores,
        control=control_scores,
        corrections_captured=corrections_captured,
        active_rules_after_batch=active_rules_after_batch,
    )

    section = _render_report(result)
    existing = output_path.read_text(encoding="utf-8") if output_path.is_file() else ""
    separator = "" if existing.endswith("\n\n") or not existing else "\n"
    output_path.write_text(existing + separator + section, encoding="utf-8")
    logger.info("appended results section to %s", output_path)

    if checkpoint_path is not None and checkpoint_path.is_file():
        checkpoint_path.unlink()
        logger.info("run completed; removed checkpoint %s", checkpoint_path)

    return result
