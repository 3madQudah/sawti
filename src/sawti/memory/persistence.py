"""The memory loop against the database: corrections in, persisted rules out.

Phase 6.2. Phase 4 ran induction inside one experiment process with an
in-memory store; the service needs it durable. This module is the one
implementation both callers use — the Celery task after a review
(`sawti.api.tasks.induce_rules_task`) and the backfill over existing
corrections (`scripts/induce_rules_from_corrections.py`):

    for each correction not yet induced (oldest first):
        candidate = induce_rule([correction])           # phase 4, unchanged
        insert_rule(store, candidate)                    # conflict check, unchanged
        persist the rule if inserted, or record the conflict (phase 6.4);
        mark the correction induced either way
    consolidate(store) and sync the merge to the table   # phase 4, minus held rules

A refused candidate is a `memory_conflicts` row a human resolves through
`resolve_conflict` (docs/09-DECISIONS.md, 2026-10-10). While it is open, the
rules it names are held out of consolidation (phase 7.1): merging one away
would leave the conflict pointing at a retired rule, so `keep_new` and
`discard` would retire nothing and the merged rule would not be linked to it.

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
from typing import Literal

import sqlalchemy as sa
from sqlalchemy.orm import Session

from sawti.db.models import MemoryConflict, MemoryRuleRecord, ReviewerAction
from sawti.db.session import get_engine, get_session
from sawti.memory import lifecycle
from sawti.memory.conflict import ConflictDetail, insert_rule
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


class MemoryStoreBusyError(RuntimeError):
    """Another writer holds the memory write lock (a non-blocking `memory_write_lock`)."""


@contextlib.contextmanager
def memory_write_lock(*, blocking: bool = True) -> Iterator[None]:
    """Hold the Postgres advisory lock that serializes every writer of `memory_rules`.

    Induction (`induce_and_store`), conflict resolution (`resolve_conflict`)
    and maintenance such as `scripts/reconsolidate_memory_rules.py` all take
    it, so a review's induction never runs against a store that is being
    re-clustered.

    Args:
        blocking: Wait for the lock (default). False raises
            `MemoryStoreBusyError` at once instead — for an HTTP request,
            which must not hang behind a minutes-long induction run.
    """
    with get_engine().connect() as lock_conn:
        if blocking:
            lock_conn.execute(sa.text("SELECT pg_advisory_lock(:k)"), {"k": _INDUCTION_LOCK_KEY})
        elif not lock_conn.execute(
            sa.text("SELECT pg_try_advisory_lock(:k)"), {"k": _INDUCTION_LOCK_KEY}
        ).scalar():
            raise MemoryStoreBusyError("the memory store is being updated; try again shortly")
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


def conflict_payload(details: list[ConflictDetail]) -> list[dict[str, object]]:
    """`memory_conflicts.conflicting` for `details`. The judge's reasoning is redacted: it is model output."""
    return [
        {
            "rule_id": str(detail.rule.id),
            "similarity": round(detail.similarity, 4),
            "reasoning": redact(detail.reasoning).redacted_text,
        }
        for detail in details
    ]


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
                else:
                    session.add(
                        MemoryConflict(
                            reviewer_action_id=correction.id,
                            candidate_rule_id=candidate.id,
                            candidate_text=candidate.rule_text,
                            conflicting=conflict_payload(outcome.conflict_details),
                            detected_at=datetime.now(UTC),
                        )
                    )
                status = OUTCOME_INSERTED if outcome.inserted else OUTCOME_CONFLICT
                _mark_induced(session, correction.id, status)
            report.processed[language] += 1
            (report.inserted if outcome.inserted else report.conflicts)[language] += 1
            if not outcome.inserted:
                logger.warning(
                    "induction: rule from correction %s conflicts with %s; recorded for a human",
                    correction.id,
                    [str(rule.id) for rule in outcome.conflicts],
                )
            if on_correction is not None:
                on_correction(correction, status)

        if pending:
            report.merged_away = _consolidate_and_sync(store.all_active_rules(), embed=embed)
        with get_session() as session:
            report.active_rules_after = session.query(MemoryRuleRecord).filter_by(retired=False).count()
    return report


def rules_held_by_open_conflicts(session: Session) -> set[uuid.UUID]:
    """Ids of the existing rules named by every unresolved conflict."""
    rows = session.query(MemoryConflict.conflicting).filter(MemoryConflict.resolution.is_(None))
    return {uuid.UUID(entry["rule_id"]) for (conflicting,) in rows for entry in conflicting}


def _consolidate_and_sync(before: list[MemoryRule], *, embed: Embedder | None) -> int:
    """Consolidate `before`, minus the rules an open conflict names, and persist the result.

    Held rules are neither clustered nor retired; they become eligible again
    once their conflict is resolved. Callers hold the memory write lock, so no
    conflict can be recorded or resolved between reading the held set and
    writing the merge.
    """
    with get_session() as session:
        held = rules_held_by_open_conflicts(session)
    eligible = [rule for rule in before if rule.id not in held]
    if len(eligible) < len(before):
        logger.info("consolidation: %d rule(s) held out by open conflicts", len(before) - len(eligible))
    after = consolidate(eligible, embed=embed)
    with get_session() as session:
        return _sync_consolidation(session, eligible, after)


def consolidate_store(*, embed: Embedder | None = None) -> int:
    """Consolidate every active persisted rule, as `induce_and_store` does after a run.

    Queued after a conflict resolution that activates a rule
    (`sawti.api.tasks.consolidate_rules_task`): that rule has to go through
    consolidation like any other new rule.

    Returns:
        How many rules were merged away (retired).
    """
    with memory_write_lock():
        with get_session() as session:
            active = load_active_rules(session)
        return _consolidate_and_sync(active, embed=embed) if active else 0


# --- Conflict resolution (phase 6.4) -------------------------------------------

ConflictAction = Literal["keep_existing", "keep_new", "discard", "not_a_conflict"]

#: Which actions retire the conflicting existing rules, and which activate the candidate.
_RETIRES_EXISTING: frozenset[str] = frozenset({"keep_new", "discard"})
_ACTIVATES_CANDIDATE: frozenset[str] = frozenset({"keep_new", "not_a_conflict"})


class ConflictNotFoundError(LookupError):
    """No memory conflict with that id."""


class ConflictAlreadyResolvedError(RuntimeError):
    """The conflict already has a resolution; a second one would contradict it."""


@dataclass(frozen=True)
class ConflictResolution:
    """What resolving one conflict changed in the store."""

    conflict_id: uuid.UUID
    action: ConflictAction
    resolved_by: str
    resolved_at: datetime
    #: The candidate's new rule id, if the action made it an active rule.
    activated_rule_id: uuid.UUID | None
    #: Existing rules this resolution retired (those already retired are not repeated).
    retired_rule_ids: list[uuid.UUID]

    @property
    def needs_consolidation(self) -> bool:
        """An activated candidate is a new rule, and every new rule goes through consolidation."""
        return self.activated_rule_id is not None


def _write_rule(row: MemoryRuleRecord, rule: MemoryRule) -> None:
    """Copy a lifecycle function's result back onto its row."""
    row.retired = rule.retired
    row.success_count = rule.success_count
    row.failure_count = rule.failure_count


def resolve_conflict(
    conflict_id: uuid.UUID, action: ConflictAction, *, reviewer_id: str
) -> ConflictResolution:
    """Apply a human's decision on a memory conflict, through the rule lifecycle.

    `keep_existing`: candidate dropped. `keep_new`: conflicting rules retired,
    candidate activated. `discard`: neither (candidate dropped, conflicting
    rules retired). `not_a_conflict`: the detector was wrong; candidate
    activated, existing rules untouched. The resolution, its reviewer and time
    commit with the rule changes, under the memory write lock (not waited for).

    Raises:
        ConflictNotFoundError, ConflictAlreadyResolvedError, MemoryStoreBusyError.
    """
    now = datetime.now(UTC)
    with memory_write_lock(blocking=False), get_session() as session:
        conflict = session.get(MemoryConflict, conflict_id, with_for_update=True)
        if conflict is None:
            raise ConflictNotFoundError(str(conflict_id))
        if conflict.resolution is not None:
            raise ConflictAlreadyResolvedError(f"{conflict_id} was resolved as {conflict.resolution}")

        retired: list[uuid.UUID] = []
        if action in _RETIRES_EXISTING:
            ids = [uuid.UUID(entry["rule_id"]) for entry in conflict.conflicting]
            for row in session.query(MemoryRuleRecord).filter(MemoryRuleRecord.id.in_(ids)):
                if not row.retired:
                    _write_rule(row, lifecycle.retire(to_memory_rule(row)))
                    retired.append(row.id)

        activated = None
        if action in _ACTIVATES_CANDIDATE:
            candidate = MemoryRule(
                id=conflict.candidate_rule_id,
                rule_text=conflict.candidate_text,
                source_correction_ids=[conflict.reviewer_action_id],
                created_at=now,
                retired=True,
            )
            session.add(_to_row(lifecycle.activate(candidate)))
            activated = candidate.id

        conflict.resolution, conflict.resolved_by, conflict.resolved_at = action, reviewer_id, now
    return ConflictResolution(
        conflict_id=conflict_id,
        action=action,
        resolved_by=reviewer_id,
        resolved_at=now,
        activated_rule_id=activated,
        retired_rule_ids=sorted(retired),
    )
