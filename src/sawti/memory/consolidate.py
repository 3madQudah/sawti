"""Merge near-duplicate memory rules.

Phase 4: agent memory maintenance.

Run by hand for now (`consolidate(store.all_active_rules())`, then re-add
the result) — a Celery periodic task to run this automatically is
nice-to-have, not required for this stage; `sawti.api.tasks` is itself
still a Phase 6 stub with nothing wired up to schedule against.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import numpy as np
from pydantic import BaseModel, Field

from sawti.llm.provider import get_llm_provider, run_with_timeout
from sawti.memory.rule_schema import MemoryRule
from sawti.memory.store import Embedder, _default_embed

# A judgment call, calibrated by hand against real induced rules (not
# guessed) — see docs/09-DECISIONS.md for the measurements. With
# DEFAULT_EMBEDDING_MODEL, two rules stating the same principle in different
# words scored 0.61-0.75 cosine similarity; two rules merely on the same
# topic but giving genuinely different guidance (the case
# sawti.memory.conflict's "same topic isn't a conflict" distinction also
# warns about) scored ~0.58 — close enough to the near-duplicate range that
# no threshold separates the two cleanly. 0.7 sits above the same-topic
# score and below most near-duplicate pairs measured, erring toward missing
# a real near-duplicate over collapsing two genuinely distinct rules into
# one — the more expensive mistake, since a merge is not easily undone.
DEFAULT_SIMILARITY_THRESHOLD = 0.7

_CONSOLIDATE_SYSTEM_PROMPT = (
    "You merge near-duplicate rules for a bilingual (Arabic/English) "
    "contact-center call analysis agent into ONE rule that preserves all of "
    "their intent. These rules were judged as saying essentially the same "
    "thing, possibly in different words or with a different emphasis. "
    "Produce a single, clear, general rule_text that a QA engineer would "
    "recognize as covering everything the originals covered. Prefer an "
    "existing rule's phrasing when it is already clear, rather than "
    "inventing new wording for its own sake — the goal is one rule, not a "
    "new rule. One or two sentences."
)


class _MergedRule(BaseModel):
    """Transport shape for the model's response — just the merged rule text."""

    rule_text: str = Field(..., min_length=1)


def _cosine_similarity(
    left: np.ndarray[Any, np.dtype[np.float64]], right: np.ndarray[Any, np.dtype[np.float64]]
) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _cluster_indices(
    embeddings: list[np.ndarray[Any, np.dtype[np.float64]]], threshold: float
) -> list[list[int]]:
    """Single-linkage clustering: connected components of the "similarity >= threshold" graph.

    Simple union-find over all pairs — the rule sets this operates on
    (a store's active rules) are small enough that O(n^2) pairwise
    comparison is not a concern.
    """
    n = len(embeddings)
    parent = list(range(n))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    for i in range(n):
        for j in range(i + 1, n):
            if _cosine_similarity(embeddings[i], embeddings[j]) >= threshold:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def _dedupe_preserving_order(ids: list[UUID]) -> list[UUID]:
    seen: set[UUID] = set()
    result: list[UUID] = []
    for value in ids:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _merge_cluster(cluster: list[MemoryRule]) -> MemoryRule:
    """Merge a cluster of >= 2 near-duplicate rules into one, via LLMProvider."""
    prompt = "\n".join(f"Rule {index}: {rule.rule_text}" for index, rule in enumerate(cluster, start=1))
    merged = run_with_timeout(
        get_llm_provider().structured_complete(
            prompt,
            response_model=_MergedRule,
            system=_CONSOLIDATE_SYSTEM_PROMPT,
            temperature=0.2,
        )
    )

    source_ids: list[UUID] = []
    for rule in cluster:
        source_ids.extend(rule.source_correction_ids)

    # A judgment call, stated explicitly: counters are summed rather than
    # reset, so a cluster's combined track record survives the merge
    # instead of a freshly-created rule starting back at 0/0.
    return MemoryRule(
        rule_text=merged.rule_text,
        source_correction_ids=_dedupe_preserving_order(source_ids),
        created_at=datetime.now(UTC),
        success_count=sum(rule.success_count for rule in cluster),
        failure_count=sum(rule.failure_count for rule in cluster),
        retired=False,
    )


def consolidate(
    rules: list[MemoryRule],
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    embed: Embedder | None = None,
) -> list[MemoryRule]:
    """Merge near-duplicate rules in `rules` into single, generalized rules.

    Retired rules pass through unchanged and are never clustered — merging
    a rule that already failed its lifecycle check (`sawti.memory.lifecycle`)
    into an active one would revive it by the back door and muddy the
    survivor's counters with a rule that was already judged not to help.

    Args:
        rules: Candidate rules to deduplicate/merge. Order is not
            significant; the output order is not the input order.
        similarity_threshold: Cosine similarity at or above which two
            rules are treated as near-duplicates. See
            `DEFAULT_SIMILARITY_THRESHOLD`'s comment for the reasoning.
        embed: How to turn rule text into a vector for clustering. Defaults
            to the same local multilingual model `sawti.memory.store` uses;
            tests inject a fast, deterministic fake.

    Returns:
        The consolidated rule set: one entry per cluster (unchanged if the
        cluster has one rule, merged if it has more than one), plus every
        retired rule from the input, untouched.
    """
    active = [rule for rule in rules if not rule.retired]
    retired = [rule for rule in rules if rule.retired]

    if len(active) <= 1:
        return active + retired

    embed_fn = embed or _default_embed
    embeddings = [embed_fn(rule.rule_text) for rule in active]
    clusters = _cluster_indices(embeddings, similarity_threshold)

    consolidated: list[MemoryRule] = []
    for indices in clusters:
        cluster = [active[index] for index in indices]
        consolidated.append(cluster[0] if len(cluster) == 1 else _merge_cluster(cluster))

    return consolidated + retired
