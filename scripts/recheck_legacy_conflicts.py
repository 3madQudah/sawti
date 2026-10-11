"""Re-check the memory conflicts recorded before `memory_conflicts` existed.

Phase 6.4 (docs/09-DECISIONS.md, 2026-10-10). Before migration 0005 a refused
candidate left only `reviewer_actions.induction_outcome = 'conflict'`. Its
text, the rules it clashed with and the judge's reason were discarded, so
those conflicts cannot be shown or resolved. This script re-opens exactly
those corrections (`induced_at`, `induction_outcome` cleared) and runs the
normal memory loop over them (`induce_and_store`, throttled for the Gemini
free tier, as `induce_rules_from_corrections.py` does). Each one then either
conflicts again and is recorded in `memory_conflicts`, or no longer conflicts
with the current store and is inserted like any other rule.

Induction is an LLM call: the new candidates are not word-for-word the ones
refused on 2026-10-07.

Dry run by default (lists what it would re-open). `--apply` writes. It refuses
to run while other corrections are pending, so it can never induce anything
but the legacy conflicts. If the daily quota stops it part-way, the rest stay
pending: resume with `scripts/induce_rules_from_corrections.py`.

Take a `pg_dump` first.

Usage:
    uv run python scripts/recheck_legacy_conflicts.py            # dry run
    uv run python scripts/recheck_legacy_conflicts.py --apply
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from sawti.db.models import MemoryConflict, ReviewerAction
from sawti.db.session import get_session
from sawti.memory.persistence import OUTCOME_CONFLICT


def legacy_conflicts(session: Session) -> list[ReviewerAction]:
    """Corrections marked `conflict` with no `memory_conflicts` row, oldest first."""
    recorded = sa.select(MemoryConflict.reviewer_action_id)
    return (
        session.query(ReviewerAction)
        .filter(ReviewerAction.induction_outcome == OUTCOME_CONFLICT, ReviewerAction.id.not_in(recorded))
        .order_by(ReviewerAction.created_at)
        .all()
    )


def pending_count(session: Session) -> int:
    """Corrections not yet induced."""
    return session.query(ReviewerAction).filter(ReviewerAction.induced_at.is_(None)).count()


def reopen(session: Session, ids: list[uuid.UUID]) -> None:
    """Make these corrections pending again, so `induce_and_store` picks them up."""
    session.execute(
        sa.update(ReviewerAction)
        .where(ReviewerAction.id.in_(ids))
        .values(induced_at=None, induction_outcome=None)
    )


def describe(actions: list[ReviewerAction]) -> list[dict[str, Any]]:
    """What the dry run prints: ids and languages, no text."""
    return [
        {
            "correction_id": str(a.id),
            "language": a.payload["original"]["language"],
            "created_at": str(a.created_at),
        }
        for a in actions
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--apply", action="store_true", help="re-open and re-induce (LLM calls)")
    parser.add_argument("--min-interval", type=float, default=4.5, help="seconds between LLM calls")
    args = parser.parse_args()

    with get_session() as session:
        legacy = legacy_conflicts(session)
        print(json.dumps({"legacy_conflicts": describe(legacy)}, indent=2))
        if not args.apply or not legacy:
            return 0
        if (pending := pending_count(session)) > 0:
            print(
                f"{pending} corrections are already pending; finish them first "
                "(scripts/induce_rules_from_corrections.py)",
                file=sys.stderr,
            )
            return 2
        reopen(session, [a.id for a in legacy])
    print(f"re-opened {len(legacy)}; running the memory loop over them")

    from induce_rules_from_corrections import main as run_backfill

    sys.argv = [sys.argv[0], "--min-interval", str(args.min_interval)]
    return run_backfill()


if __name__ == "__main__":
    raise SystemExit(main())
