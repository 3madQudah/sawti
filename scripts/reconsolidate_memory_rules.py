"""Undo the single-linkage over-merge and re-consolidate `memory_rules` with complete linkage.

Phase 6.2 fix (docs/09-DECISIONS.md, 2026-10-07). The backfill's one
consolidation pass ran single-linkage and chained 81 of 92 rules into one.
This reverses it from the rows still in the table — nothing was deleted, the
merged-away rules are only `retired`:

  1. snapshot the current (pre-fix) state to the plan file;
  2. restore the originals: un-retire every single-source rule (one per
     induced correction) and retire the multi-source rules the old
     consolidation created;
  3. cluster the restored rules with today's complete linkage and save the
     plan (the clusters) *before* any LLM call;
  4. merge each multi-member cluster — one LLM call for its text, via the
     same `_merge_cluster` consolidation uses — and commit it on its own:
     insert the merged rule, retire its members.

Steps 1–3 run once (they are skipped if the plan file exists). Step 4 is
resumable: a cluster whose members are already retired is done, so a re-run
after a quota stop continues with the next one. The whole run holds the
memory store's advisory lock. Gemini calls are throttled like the backfill.

Usage:
    uv run python scripts/reconsolidate_memory_rules.py
    uv run python scripts/reconsolidate_memory_rules.py --report-only
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sqlalchemy as sa
from induce_rules_from_corrections import _Throttled, report

from sawti.db.models import MemoryRuleRecord, ReviewerAction
from sawti.db.session import get_session
from sawti.llm.provider import get_llm_provider, is_transient_provider_error
from sawti.memory import consolidate
from sawti.memory.consolidate import DEFAULT_SIMILARITY_THRESHOLD, _cluster_indices, _merge_cluster
from sawti.memory.persistence import _to_row, memory_write_lock, to_memory_rule
from sawti.memory.store import _default_embed

PLAN_PATH = Path("data/memory_backfill/2026-10-07_reconsolidation_plan.json")


def _groups_of_active_rules(session: Any) -> list[list[str]]:
    """Each active rule as the list of single-source rules it stands for (its lineage)."""
    rows = session.query(MemoryRuleRecord).all()
    single = [r for r in rows if len(r.source_correction_ids) == 1]
    groups = []
    for rule in (r for r in rows if not r.retired):
        sources = {str(i) for i in rule.source_correction_ids}
        groups.append([str(s.id) for s in single if str(s.source_correction_ids[0]) in sources])
    return groups


def cluster_stats(
    groups: list[list[str]], text_of: dict[str, str], language_of: dict[str, str]
) -> dict[str, Any]:
    """Size distribution and, per multi-member cluster, its min pairwise cosine and languages."""
    clusters = []
    for members in sorted(groups, key=len, reverse=True):
        if len(members) < 2:
            continue
        vectors = np.array([_default_embed(text_of[m]) for m in members])
        pairs = [float(vectors[i] @ vectors[j]) for i, j in itertools.combinations(range(len(members)), 2)]
        clusters.append(
            {
                "size": len(members),
                "min_pairwise_cosine": round(min(pairs), 3),
                "median_pairwise_cosine": round(float(np.median(pairs)), 3),
                "languages": dict(sorted(Counter(language_of[m] for m in members).items())),
            }
        )
    return {
        "active_rules": len(groups),
        "size_distribution": dict(sorted(Counter(len(g) for g in groups).items())),
        "multi_member_clusters": clusters,
    }


def _lookups(session: Any) -> tuple[dict[str, str], dict[str, str]]:
    language_by_correction = {
        str(a.id): a.payload["original"]["language"] for a in session.query(ReviewerAction)
    }
    text_of, language_of = {}, {}
    for rule in session.query(MemoryRuleRecord).filter(
        sa.func.jsonb_array_length(MemoryRuleRecord.source_correction_ids) == 1
    ):
        text_of[str(rule.id)] = rule.rule_text
        language_of[str(rule.id)] = language_by_correction[str(rule.source_correction_ids[0])]
    return text_of, language_of


def prepare() -> dict[str, Any]:
    """Steps 1-3: snapshot, restore the originals, plan the complete-linkage clusters."""
    with get_session() as session:
        text_of, language_of = _lookups(session)
        pre_fix = cluster_stats(_groups_of_active_rules(session), text_of, language_of)
        pre_fix["per_language"] = report()["active_rules_per_language"]
        merged = [
            r for r in session.query(MemoryRuleRecord) if len(r.source_correction_ids) > 1 and not r.retired
        ]
        session.execute(
            sa.update(MemoryRuleRecord)
            .where(sa.func.jsonb_array_length(MemoryRuleRecord.source_correction_ids) == 1)
            .values(retired=False)
        )
        for rule in merged:
            rule.retired = True
    with get_session() as session:
        originals = (
            session.query(MemoryRuleRecord)
            .filter_by(retired=False)
            .order_by(MemoryRuleRecord.created_at)
            .all()
        )
        ids = [str(r.id) for r in originals]
        clusters = _cluster_indices(
            [_default_embed(r.rule_text) for r in originals], DEFAULT_SIMILARITY_THRESHOLD
        )
    plan = {
        "created_at": datetime.now(UTC).isoformat(),
        "linkage": "complete",
        "threshold": DEFAULT_SIMILARITY_THRESHOLD,
        "pre_fix": pre_fix,
        "retired_single_linkage_merges": [str(r.id) for r in merged],
        "restored_originals": len(ids),
        "clusters": [[ids[i] for i in cluster] for cluster in clusters],
    }
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLAN_PATH.write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def merge_pending(plan: dict[str, Any]) -> int:
    """Step 4: merge every multi-member cluster not yet merged; one commit per cluster."""
    merged = 0
    for members in plan["clusters"]:
        if len(members) < 2:
            continue
        with get_session() as session:
            rows = (
                session.query(MemoryRuleRecord)
                .filter(MemoryRuleRecord.id.in_([uuid.UUID(m) for m in members]))
                .all()
            )
            if all(row.retired for row in rows):
                continue  # done in an earlier run
            rules = [to_memory_rule(row) for row in rows]
        result = _merge_cluster(rules)
        with get_session() as session:
            session.add(_to_row(result))
            session.execute(
                sa.update(MemoryRuleRecord)
                .where(MemoryRuleRecord.id.in_([uuid.UUID(m) for m in members]))
                .values(retired=True)
            )
        merged += 1
        print(f"merged {len(members)} -> {result.id}: {result.rule_text}")
    return merged


def post_fix_stats(plan: dict[str, Any]) -> dict[str, Any]:
    with get_session() as session:
        text_of, language_of = _lookups(session)
    stats = cluster_stats(plan["clusters"], text_of, language_of)
    stats["per_language"] = report()["active_rules_per_language"]
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--min-interval", type=float, default=4.5)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    if not args.report_only:
        throttled = _Throttled(get_llm_provider(), args.min_interval)
        consolidate.get_llm_provider = lambda: throttled  # type: ignore[attr-defined]
        with memory_write_lock():
            plan = json.loads(PLAN_PATH.read_text()) if PLAN_PATH.exists() else prepare()
            for attempt in range(6):
                try:
                    merge_pending(plan)
                    break
                except Exception as exc:
                    if "PerDay" in str(exc):
                        print("daily quota reached; re-run to resume.", file=sys.stderr)
                        return 2
                    if is_transient_provider_error(exc) and attempt < 5:
                        print(f"transient error ({exc!r}); waiting 65s", file=sys.stderr)
                        time.sleep(65)
                        continue
                    raise
    plan = json.loads(PLAN_PATH.read_text())
    result = {"pre_fix": plan["pre_fix"], "post_fix": post_fix_stats(plan), "report": report()}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
