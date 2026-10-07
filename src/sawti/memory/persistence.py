"""The memory loop against the database: corrections in, persisted rules out.

Phase 6.2. Phase 4 ran induction inside one experiment process with an
in-memory store; the service needs it durable. This module is the one
implementation both callers use — the Celery task after a review
(`sawti.api.tasks.induce_rules_task`) and the backfill over existing
corrections (`scripts/induce_rules_from_corrections.py`):

    for each correction not yet induced (oldest first):
        candidate = induce_rule([correction])           # phase 4, unchanged
        insert_rule(store, candidate)                    # conflict check, unchanged
        persist the rule if inserted; mark the correction induced either way
    consolidate(store) and sync the merge to the table   # phase 4, unchanged

Per correction, the rule (if any) and the correction's `induced_at` commit
together, so an interruption — a transient provider error, the daily quota —
loses at most the correction in flight, and a re-run resumes after it.

A Postgres advisory lock serializes runs: two concurrent reviews must not
each check conflicts against a store that is missing the other's new rule.

Rule text is redacted before it is stored. Rules are injected into prompts,
and an induced rule is model output derived from corrections; the redactor
is a cheap guarantee that nothing PII-shaped rides along.
"""

from __future__ import annotations

import contextlib
import logging
import uuid
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime

import sqlalchemy as sa
from sqlalchemy.orm import Session

from sawti.db.models import MemoryRuleRecord, ReviewerAction
from sawti.db.session import get_engine, get_session
from sawti.memory.conflict import insert_rule
from sawti.memory.consolidate import consolidate
from sawti.memory.induction import induce_rule
from sawti.memory.rule_schema import MemoryRule
from sawti.memory.store import Embedder, MemoryStore
from sawti.privacy.redaction import redact
from sawti.schemas import Correction

logger = logging.getLogger(__name__)

#: Arbitrary, fixed key for `pg_advisory_lock` — "the memory store is being written".
_INDUCTION_LOCK_KEY = 0x5A_77_1D_01

OUTCOME_INSERTED = "inserted"
OUTCOME_CONFLICT = "conflict"


@dataclass
class InductionReport:
    """What one run did, per language category of the corrections it processed."""

    processed: Counter[str] = field(default_factory=Counter)
    inserted: Counter[str] = field(default_factory=Counter)
    conflicts: Counter[str] = field(default_factory=Counter)
    merged_away: int = 0
    active_rules_after: int = 0


@contextlib.contextmanager
def memory_write_lock() -> Iterator[None]:
    """Hold the Postgres advisory lock that serializes every writer of `memory_rules`.

    Induction (`induce_and_store`) and maintenance such as
    `scripts/reconsolidate_memory_rules.py` both take it, so a review's
    induction never runs against a store that is being re-clustered.
    """
    with get_engine().connect() as lock_conn:
        lock_conn.execute(sa.text("SELECT pg_advisory_lock(:k)"), {"k": _INDUCTION_LOCK_KEY})
        try:
            yield
        finally:
            lock_conn.execute(sa.text("SELECT pg_advisory_unlock(:k)"), {"k": _INDUCTION_LOCK_KEY})


def to_memory_rule(row: MemoryRuleRecord) -> MemoryRule:
    """A `memory_rules` row as the phase 4 schema."""
    return MemoryRule(
        id=row.id,
        rule_text=row.rule_text,
        source_correction_ids=[uuid.UUID(str(i)) for i in row.source_correction_ids],
        created_at=row.created_at,
        success_count=row.success_count,
        failure_count=row.failure_count,
        retired=row.retired,
    )


def _to_row(rule: MemoryRule) -> MemoryRuleRecord:
    return MemoryRuleRecord(
        id=rule.id,
        rule_text=rule.rule_text,
        source_correction_ids=[str(i) for i in rule.source_correction_ids],
        created_at=rule.created_at,
        success_count=rule.success_count,
        failure_count=rule.failure_count,
        retired=rule.retired,
    )


def load_active_rules(session: Session) -> list[MemoryRule]:
    """Every non-retired persisted rule."""
    return [to_memory_rule(row) for row in session.query(MemoryRuleRecord).filter_by(retired=False)]


def pending_corrections(session: Session, *, submission_id: uuid.UUID | None = None) -> list[Correction]:
    """Corrections not yet induced, oldest first — all of them, or one review submission's."""
    query = session.query(ReviewerAction).filter(ReviewerAction.induced_at.is_(None))
    if submission_id is not None:
        query = query.filter(ReviewerAction.review_submission_id == submission_id)
    return [Correction.model_validate(row.payload) for row in query.order_by(ReviewerAction.created_at)]


def _mark_induced(session: Session, correction_id: uuid.UUID, outcome: str) -> None:
    session.execute(
        sa.update(ReviewerAction)
        .where(ReviewerAction.id == correction_id)
        .values(induced_at=datetime.now(UTC), induction_outcome=outcome)
    )


def _sync_consolidation(session: Session, before: list[MemoryRule], after: list[MemoryRule]) -> int:
    """Persist `consolidate()`'s result: new (merged) rules inserted, merged-away rules retired."""
    before_ids = {rule.id for rule in before}
    after_ids = {rule.id for rule in after if not rule.retired}
    for rule in after:
        if rule.id not in before_ids:
            session.add(_to_row(rule))
    merged_away = before_ids - after_ids
    if merged_away:
        session.execute(
            sa.update(MemoryRuleRecord).where(MemoryRuleRecord.id.in_(merged_away)).values(retired=True)
        )
    return len(merged_away)


def induce_and_store(
    *,
    submission_id: uuid.UUID | None = None,
    embed: Embedder | None = None,
    on_correction: Callable[[Correction, str], None] | None = None,
) -> InductionReport:
    """Run the memory loop over pending corrections and persist the rules.

    Args:
        submission_id: Only this review submission's corrections; None = every
            pending correction (the backfill).
        embed: Embedder for the conflict check, consolidation and the store;
            None = the real local model. Tests inject a fake.
        on_correction: Called after each correction is handled, with its
            outcome — the backfill uses it to report progress.

    Returns:
        Counts per language category of the processed corrections.

    Raises:
        Whatever the provider raises (transient errors included). Everything
        handled before the error is already committed.
    """
    report = InductionReport()
    with memory_write_lock():
        with get_session() as session:
            pending = pending_corrections(session, submission_id=submission_id)
            store = MemoryStore(embed=embed)
            store.replace_all(load_active_rules(session))
        for correction in pending:
            language = correction.original.language.value
            candidate = induce_rule([correction])
            candidate = candidate.model_copy(update={"rule_text": redact(candidate.rule_text).redacted_text})
            outcome = insert_rule(store, candidate, embed=embed)
            with get_session() as session:
                if outcome.inserted:
                    session.add(_to_row(candidate))
                status = OUTCOME_INSERTED if outcome.inserted else OUTCOME_CONFLICT
                _mark_induced(session, correction.id, status)
            report.processed[language] += 1
            (report.inserted if outcome.inserted else report.conflicts)[language] += 1
            if not outcome.inserted:
                logger.warning(
                    "induction: rule from correction %s conflicts with %s; not stored (needs a human)",
                    correction.id,
                    [str(rule.id) for rule in outcome.conflicts],
                )
            if on_correction is not None:
                on_correction(correction, status)

        if pending:
            before = store.all_active_rules()
            after = consolidate(before, embed=embed)
            with get_session() as session:
                report.merged_away = _sync_consolidation(session, before, after)
        with get_session() as session:
            report.active_rules_after = session.query(MemoryRuleRecord).filter_by(retired=False).count()
    return report
