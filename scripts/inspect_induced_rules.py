"""Inspect induced rules against their source corrections.

Phase 4: manual quality-review aid for `sawti.memory.induction.induce_rule()`.

Not a test. `tests/memory/test_induction.py` only checks structural
properties (ids tracked, single vs. multi-correction merge) — whether the
induced *text* is actually a good, checkable rule is a judgment call, and
this script exists to make that judgment call inspectable rather than
implicit. It runs `induce_rule()` against the real synthetic corrections
captured by `scripts/generate_synthetic_corrections.py` (reviewer_id
`synthetic-day1` — see that script's docstring for what "synthetic" means
here) and renders a markdown table of induced rule text next to its source
correction(s), for both single corrections and merged groups.

Corrections are grouped by a normalized `error_location` (list indices and
bracketed criteria stripped, so `commitments[0].deadline` and
`rubric_scores[empathy]` group with their peers) — this is a convenience for
picking a representative sample to inspect, not a claim from
`sawti.memory.induction` about how rules should be merged in production;
`sawti.memory.consolidate` (Stage 6) does that job for real, by embedding
similarity.

Usage:
    uv run python scripts/inspect_induced_rules.py
    uv run python scripts/inspect_induced_rules.py --out /tmp/induced_rules.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sawti.db.models import ReviewerAction
from sawti.db.session import get_session
from sawti.memory.diff import topic_for
from sawti.memory.induction import induce_rule
from sawti.memory.rule_schema import MemoryRule
from sawti.schemas import Correction

REVIEWER_ID = "synthetic-day1"

#: How many individual, single-correction examples to show per group,
#: alongside that group's one merged rule.
SINGLES_PER_GROUP = 4


def _load_synthetic_corrections() -> list[Correction]:
    """Every `Correction` captured by the synthetic-corrections script, deserialized from its payload."""
    with get_session() as session:
        rows = session.query(ReviewerAction).filter_by(reviewer_id=REVIEWER_ID).all()
        return [Correction.model_validate(row.payload) for row in rows]


def _render_row(kind: str, rule: MemoryRule, corrections: list[Correction]) -> str:
    call_ids = ", ".join(f"`{correction.call_id}`" for correction in corrections)
    locations = ", ".join(sorted({correction.error_location for correction in corrections}))
    rule_text = rule.rule_text.replace("|", "\\|")
    return f"| {kind} | {rule_text} | {call_ids} | {locations} |"


def build_table(corrections: list[Correction], *, singles_per_group: int = SINGLES_PER_GROUP) -> str:
    """Group corrections, induce a rule per group and per sampled single, render as markdown."""
    groups: dict[str, list[Correction]] = {}
    for correction in corrections:
        groups.setdefault(topic_for(correction.error_location), []).append(correction)

    rows: list[str] = []
    for _key, group in sorted(groups.items()):
        merged_rule = induce_rule(group)
        rows.append(_render_row(f"merged ({len(group)})", merged_rule, group))
        for single in group[:singles_per_group]:
            single_rule = induce_rule([single])
            rows.append(_render_row("single", single_rule, [single]))

    header = "| kind | induced rule | source call(s) | error_location(s) |"
    separator = "| --- | --- | --- | --- |"
    return "\n".join([header, separator, *rows])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="Also write the table to this file.")
    args = parser.parse_args(argv)

    corrections = _load_synthetic_corrections()
    if not corrections:
        raise SystemExit(
            f"No synthetic corrections found (reviewer_id={REVIEWER_ID!r}). "
            "Run scripts/generate_synthetic_corrections.py first."
        )

    table = build_table(corrections)
    print(table)
    if args.out is not None:
        args.out.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
