"""Estimate the cluster sizes phase 4's per-batch consolidation produced, under both linkages.

Phase 6.2 follow-up to the single-linkage over-merge. Phase 4's five-batch run
consolidated after every batch but never persisted its rules or its clusters,
so its max cluster size cannot be read back. This replays its *scale* with
zero LLM calls:

  * inputs: the 58 most recent `five-batch-experiment` corrections (the
    recorded run's count), in creation order, split into phase 4's recorded
    batch sizes 10 / 8 / 15 / 10 / 15;
  * rule text: the rule the 2026-10-07 backfill induced from each of those
    corrections (a re-induction — not phase 4's own text, which is lost);
    conflict-rejected corrections add no rule, as in phase 4;
  * after each batch: cluster the active set; a multi-member cluster is
    replaced by its medoid (the member most similar to the rest) standing in
    for the LLM-written merge text.

Real embeddings (`sawti.memory.store`'s model). Reported per batch: the
active-set size consolidated, the max cluster size, and the active count
after — to compare with phase 4's recorded 7 / 6 / 10 / 10 / 14, which is the
check on whether the replay is faithful enough to say anything.

Usage:
    uv run python scripts/simulate_phase4_consolidation.py
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np

from sawti.memory.consolidate import DEFAULT_SIMILARITY_THRESHOLD

PHASE4_BATCH_SIZES = (10, 8, 15, 10, 15)
PHASE4_ACTIVE_AFTER = (7, 6, 10, 10, 14)

Vector = np.ndarray[Any, np.dtype[np.float64]]
Clusterer = Callable[[list[Vector], float], list[list[int]]]


def single_linkage(embeddings: list[Vector], threshold: float) -> list[list[int]]:
    """The pre-fix algorithm: connected components of the similarity >= threshold graph."""
    n = len(embeddings)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if float(embeddings[i] @ embeddings[j]) >= threshold:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def replay(
    batches: list[list[Vector]], cluster: Clusterer, threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> list[dict[str, int]]:
    """Per batch: add the batch's rules, cluster, keep one medoid per cluster."""
    active: list[Vector] = []
    out = []
    for batch in batches:
        active = active + batch
        clusters = cluster(active, threshold)
        survivors = []
        for members in clusters:
            vectors = np.array([active[i] for i in members])
            medoid = int(np.argmax((vectors @ vectors.T).sum(axis=1)))
            survivors.append(active[members[medoid]])
        out.append(
            {
                "consolidated": len(active),
                "max_cluster": max(len(c) for c in clusters),
                "active_after": len(survivors),
            }
        )
        active = survivors
    return out


def main() -> int:
    from sawti.db.models import MemoryRuleRecord, ReviewerAction
    from sawti.db.session import get_session
    from sawti.memory.consolidate import _cluster_indices
    from sawti.memory.store import _default_embed

    with get_session() as session:
        actions = (
            session.query(ReviewerAction)
            .filter_by(reviewer_id="five-batch-experiment")
            .order_by(ReviewerAction.created_at)
            .all()
        )[-sum(PHASE4_BATCH_SIZES) :]
        rule_by_source = {
            str(rule.source_correction_ids[0]): rule.rule_text
            for rule in session.query(MemoryRuleRecord)
            if len(rule.source_correction_ids) == 1
        }
        batches: list[list[Vector]] = []
        start = 0
        for size in PHASE4_BATCH_SIZES:
            texts = [
                rule_by_source[str(a.id)]
                for a in actions[start : start + size]
                if str(a.id) in rule_by_source
            ]
            batches.append([np.asarray(_default_embed(t)) for t in texts])
            start += size

    result = {
        "threshold": DEFAULT_SIMILARITY_THRESHOLD,
        "phase4_recorded_active_after": list(PHASE4_ACTIVE_AFTER),
        "single_linkage": replay(batches, single_linkage),
        "complete_linkage": replay(batches, _cluster_indices),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
