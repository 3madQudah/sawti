"""Generate synthetic reviewer corrections from Phase 2 agent vs. ground-truth diffs.

Phase 4: stand-in for real review data, to unblock rule induction (Stage 3)
onward before a real review UI exists.

STAND-IN, NOT SIGNAL. There is no review UI yet — `src/sawti/api/routes/reviews.py`
is a Phase 5 stub, needing `interrupt()`/checkpointer resume wiring that does
not exist. Every `Correction` this script produces is manufactured by
diffing the Phase 2 grounded agent's own output against the LLM-generated
reference labels in `data/ground_truth/` — not a real QA reviewer's
judgment about a real error. It is a data-shape fixture for exercising
Stages 3-8 of the memory pipeline, and must never be read as evidence about
what a human reviewer would actually flag. See `docs/09-DECISIONS.md`.

Persistence note: `capture_correction()` requires an existing
`CallAnalysisRecord` row for the agent's output (its `id` is
`ReviewerAction.call_analysis_id`'s foreign-key target), and nothing in this
codebase persists one yet — that is still-unbuilt Phase 5 service-layer
work. This script backfills that step itself: one shared placeholder `Agent`
("synthetic-corrections-script"), plus a `Call` and `CallAnalysisRecord` per
processed call, holding the actual agent output that was produced. This is
not fabricated content — it is the real analysis the grounded agent
produced — only the persistence step is synthetic, standing in for what
Phase 5's pipeline will eventually do for real runs.

Usage:
    uv run python scripts/generate_synthetic_corrections.py
    uv run python scripts/generate_synthetic_corrections.py --limit-per-language 5
"""

from __future__ import annotations

import argparse
import logging
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from sawti.data.ground_truth import RUBRIC_CRITERIA, load_ground_truth
from sawti.db.models import Agent, Base, Call, CallAnalysisRecord
from sawti.db.session import get_engine, get_session
from sawti.eval.experiments.grounded_graph import extract_via_graph
from sawti.memory.capture import capture_correction
from sawti.schemas import CallAnalysis, Correction, Language

logger = logging.getLogger("generate_synthetic_corrections")

DEFAULT_GROUND_TRUTH_DIR = Path("data/ground_truth")
DEFAULT_SYNTHETIC_DIR = Path("data/synthetic")

#: Fixed identity for every reviewer_id this script writes — never a real
#: QA reviewer's id, so a downstream reader can always tell synthetic
#: corrections apart from real ones by this field alone.
REVIEWER_ID = "synthetic-day1"

_SYNTHETIC_AGENT_EXTERNAL_ID = "synthetic-corrections-script"

# Rubric scores are 0.0-1.0. A judgment call, stated explicitly (project
# convention — see sawti.eval.metrics): a gap this size is treated as a real
# QA disagreement worth a correction, not the ordinary variance of two
# independently-written scores that both land in the same rough range.
_RUBRIC_SCORE_TOLERANCE = 0.25

# An extractor turns one transcript file into one validated `CallAnalysis`.
# Mirrors `sawti.eval.runner.Extractor` so this script and the eval harness
# can be tested the same way: inject a fake extractor, no LLM call needed.
Extractor = Callable[[Path], CallAnalysis]


@dataclass
class CorrectionCounts:
    """Per-language tally of what this run did, so the report never blends categories."""

    compared: Counter[Language] = field(default_factory=Counter)
    corrected: Counter[Language] = field(default_factory=Counter)
    matched: Counter[Language] = field(default_factory=Counter)
    failed: Counter[Language] = field(default_factory=Counter)


def _first_difference(agent: CallAnalysis, truth: CallAnalysis) -> str | None:
    """Find the first QA-relevant field where `agent` and `truth` disagree.

    Deliberately narrow, and a judgment call stated explicitly: `summary`,
    commitment/flag descriptions, and rubric justifications are free text
    that two independently-authored analyses almost never match word for
    word even when they agree in substance. Diffing them directly would
    make every call "differ" on wording, which is not a QA-actionable
    correction and would flood induction with noise instead of the general
    principle it is supposed to find. Only structurally meaningful content
    is compared, in a fixed order:

      1. commitments — count, then (index-paired) whether a deadline was
         given at all.
      2. compliance_flags — the set of `rule_id`s raised.
      3. rubric_scores — each of the seven canonical criteria, when both
         sides scored it, differing by more than `_RUBRIC_SCORE_TOLERANCE`.

    Args:
        agent: The Phase 2 grounded agent's own output.
        truth: The corresponding ground-truth reference label.

    Returns:
        A field path in `Correction.error_location`'s style (e.g.
        `"commitments[0].deadline"`), or `None` if nothing comparable
        differs.
    """
    if len(agent.commitments) != len(truth.commitments):
        return "commitments"
    for index, (predicted, expected) in enumerate(zip(agent.commitments, truth.commitments, strict=True)):
        if bool(predicted.deadline) != bool(expected.deadline):
            return f"commitments[{index}].deadline"

    agent_flag_ids = {flag.rule_id for flag in agent.compliance_flags}
    truth_flag_ids = {flag.rule_id for flag in truth.compliance_flags}
    if agent_flag_ids != truth_flag_ids:
        return "compliance_flags"

    agent_scores = {score.criterion: score.score for score in agent.rubric_scores}
    truth_scores = {score.criterion: score.score for score in truth.rubric_scores}
    for criterion in RUBRIC_CRITERIA:
        if criterion not in agent_scores or criterion not in truth_scores:
            continue
        if abs(agent_scores[criterion] - truth_scores[criterion]) > _RUBRIC_SCORE_TOLERANCE:
            return f"rubric_scores[{criterion}]"

    return None


def _get_or_create_synthetic_agent(session: Session) -> uuid.UUID:
    """Return the id of the shared placeholder `Agent` this script's `Call` rows attach to.

    Every `CallAnalysisRecord` needs a `Call`, which needs an `Agent` —
    there is no other way to model "output of a synthetic batch run" in the
    current schema. One `Agent`, clearly labeled via `external_id`, is
    reused across every call this script processes rather than minted fresh
    per call.
    """
    existing = session.query(Agent).filter_by(external_id=_SYNTHETIC_AGENT_EXTERNAL_ID).one_or_none()
    if existing is not None:
        return existing.id
    agent = Agent(
        name="Synthetic corrections script (not a real contact-center agent)",
        external_id=_SYNTHETIC_AGENT_EXTERNAL_ID,
        created_at=datetime.now(UTC),
    )
    session.add(agent)
    # `Agent.id`'s `default=uuid.uuid4` is a SQLAlchemy flush-time default, not
    # a construction-time one — `agent.id` is None until flushed.
    session.flush()
    return agent.id


def _persist_analysis_record(
    session: Session, analysis: CallAnalysis, *, agent_id: uuid.UUID, transcript: str
) -> None:
    """Persist a `Call` + `CallAnalysisRecord` for `analysis`, satisfying `ReviewerAction`'s FK.

    `CallAnalysisRecord.id` is set to `analysis.id` — the same id
    `capture_correction()` will use as `ReviewerAction.call_analysis_id` —
    so the two rows line up by construction.
    """
    now = datetime.now(UTC)
    call_row_id = uuid.uuid4()
    session.add(
        Call(
            id=call_row_id,
            agent_id=agent_id,
            transcript=transcript,
            language=analysis.language.value,
            occurred_at=now,
        )
    )
    session.add(
        CallAnalysisRecord(
            id=analysis.id,
            call_id=call_row_id,
            payload=analysis.model_dump(mode="json"),
            confidence=analysis.confidence,
            requires_human_review=analysis.requires_human_review,
            created_at=now,
        )
    )


def _select_records(
    ground_truth: list[CallAnalysis], *, limit_per_language: int | None
) -> list[CallAnalysis]:
    """Apply `limit_per_language`, keeping each language's own share rather than the first N overall."""
    if limit_per_language is None:
        return ground_truth
    seen: Counter[Language] = Counter()
    selected: list[CallAnalysis] = []
    for record in ground_truth:
        if seen[record.language] >= limit_per_language:
            continue
        seen[record.language] += 1
        selected.append(record)
    return selected


def generate_synthetic_corrections(
    ground_truth_dir: Path = DEFAULT_GROUND_TRUTH_DIR,
    *,
    synthetic_dir: Path = DEFAULT_SYNTHETIC_DIR,
    limit_per_language: int | None = None,
    sleep_seconds: float = 4.0,
    extractor: Extractor | None = None,
) -> tuple[list[Correction], CorrectionCounts]:
    """Diff the Phase 2 agent's output against ground truth and capture a `Correction` per disagreement.

    See the module docstring for what this is a stand-in for, and why.

    Args:
        ground_truth_dir: Directory of reference-label records.
        synthetic_dir: Directory of `<call_id>.txt` transcripts.
        limit_per_language: Cap on calls processed per `Language`, applied
            before any LLM calls. `None` processes every ground-truth record.
        sleep_seconds: Delay between agent runs, to respect provider rate limits.
        extractor: How to turn a transcript into a `CallAnalysis`. Defaults
            to the Phase 2 grounded agent (`extract_via_graph`); tests inject
            a fake to avoid real LLM calls.

    Returns:
        The captured `Correction` records, and a per-language tally of what
        happened to every call considered.
    """
    run_extractor = extractor or extract_via_graph
    records = _select_records(load_ground_truth(ground_truth_dir), limit_per_language=limit_per_language)

    Base.metadata.create_all(get_engine())
    with get_session() as session:
        agent_id = _get_or_create_synthetic_agent(session)

    corrections: list[Correction] = []
    counts = CorrectionCounts()

    for index, record in enumerate(records, start=1):
        transcript_path = synthetic_dir / f"{record.call_id}.txt"
        if not transcript_path.is_file():
            counts.failed[record.language] += 1
            logger.error("[%d/%d] %s: transcript missing", index, len(records), record.call_id)
            continue

        try:
            agent_output = run_extractor(transcript_path)
        except Exception as exc:
            counts.failed[record.language] += 1
            logger.error("[%d/%d] %s: agent extraction failed: %r", index, len(records), record.call_id, exc)
            continue

        counts.compared[record.language] += 1
        diff_path = _first_difference(agent_output, record)

        if diff_path is None:
            counts.matched[record.language] += 1
            logger.info("[%d/%d] %s: agent matches ground truth", index, len(records), record.call_id)
        else:
            with get_session() as session:
                _persist_analysis_record(
                    session,
                    agent_output,
                    agent_id=agent_id,
                    transcript=transcript_path.read_text(encoding="utf-8"),
                )
            correction = capture_correction(
                agent_output,
                record,
                error_location=diff_path,
                reviewer_id=REVIEWER_ID,
                note="auto-generated from ground-truth diff, not a real QA review",
            )
            corrections.append(correction)
            counts.corrected[record.language] += 1
            logger.info("[%d/%d] %s: correction at %s", index, len(records), record.call_id, diff_path)

        if index < len(records):
            time.sleep(sleep_seconds)

    return corrections, counts


def _render_report(counts: CorrectionCounts) -> str:
    """A per-language table: compared, corrections captured, matched, and failed — never blended."""
    header = f"{'language':<8} | {'compared':>8} | {'corrected':>9} | {'matched':>7} | {'failed':>6}"
    lines = [header, "-" * len(header)]
    for language in (Language.AR, Language.EN, Language.MIXED):
        lines.append(
            f"{language.value:<8} | {counts.compared[language]:>8} | {counts.corrected[language]:>9} | "
            f"{counts.matched[language]:>7} | {counts.failed[language]:>6}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GROUND_TRUTH_DIR)
    parser.add_argument("--synthetic-dir", type=Path, default=DEFAULT_SYNTHETIC_DIR)
    parser.add_argument(
        "--limit-per-language",
        type=int,
        default=None,
        help="Cap calls processed per language category (default: all).",
    )
    parser.add_argument("--sleep-seconds", type=float, default=4.0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _corrections, counts = generate_synthetic_corrections(
        args.ground_truth_dir,
        synthetic_dir=args.synthetic_dir,
        limit_per_language=args.limit_per_language,
        sleep_seconds=args.sleep_seconds,
    )
    print(_render_report(counts))


if __name__ == "__main__":
    main()
