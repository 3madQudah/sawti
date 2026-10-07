"""Tests for `sawti.memory.consolidate`.

Phase 4: mirrors `src/sawti/memory/consolidate.py`.

Uses the same deterministic fake embedder pattern as
`tests/memory/test_store.py` for clustering, plus a scripted fake
LLMProvider for the merge call — no real model calls.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pytest

from sawti.memory.consolidate import consolidate
from sawti.memory.rule_schema import MemoryRule

_VOCAB = ["deadline", "flag", "refund", "unrelated"]


def _fake_embed(text: str) -> np.ndarray:
    words = set(text.lower().split())
    return np.array([1.0 if word in words else 0.0 for word in _VOCAB])


def _rule(
    rule_text: str,
    *,
    source_correction_ids: list[UUID] | None = None,
    success_count: int = 0,
    failure_count: int = 0,
    retired: bool = False,
) -> MemoryRule:
    return MemoryRule(
        rule_text=rule_text,
        created_at=datetime.now(UTC),
        source_correction_ids=source_correction_ids or [],
        success_count=success_count,
        failure_count=failure_count,
        retired=retired,
    )


class _ScriptedMergeProvider:
    """Returns a fixed merged rule_text for every structured_complete call."""

    def __init__(self, merged_text: str = "merged rule text") -> None:
        self._merged_text = merged_text
        self.prompts: list[str] = []

    async def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        raise AssertionError("consolidate must use structured_complete, not complete")

    async def structured_complete(
        self, prompt: str, *, response_model: Any, system: str | None = None, **kwargs: Any
    ) -> Any:
        self.prompts.append(prompt)
        return response_model(rule_text=self._merged_text)


def _patch_provider(monkeypatch: pytest.MonkeyPatch, provider: _ScriptedMergeProvider) -> None:
    monkeypatch.setattr("sawti.memory.consolidate.get_llm_provider", lambda: provider)


def test_consolidate_merges_near_duplicate_rules_into_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """consolidate() merges rules with highly similar rule_text into a single rule."""
    first_id, second_id = uuid4(), uuid4()
    a = _rule("deadline flag required", source_correction_ids=[first_id], success_count=2, failure_count=1)
    b = _rule("deadline flag required", source_correction_ids=[second_id], success_count=1, failure_count=0)
    _patch_provider(monkeypatch, _ScriptedMergeProvider("merged: flag missing deadlines"))

    result = consolidate([a, b], embed=_fake_embed)

    assert len(result) == 1
    merged = result[0]
    assert merged.rule_text == "merged: flag missing deadlines"
    assert set(merged.source_correction_ids) == {first_id, second_id}
    assert merged.success_count == 3
    assert merged.failure_count == 1
    assert merged.retired is False


def test_consolidate_leaves_dissimilar_rules_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """consolidate() does not merge rules that are not near-duplicates."""
    a = _rule("deadline flag required")
    b = _rule("refund unrelated topic")
    provider = _ScriptedMergeProvider()
    _patch_provider(monkeypatch, provider)

    result = consolidate([a, b], embed=_fake_embed)

    assert {rule.rule_text for rule in result} == {a.rule_text, b.rule_text}
    assert provider.prompts == []  # no merge call was needed


def test_consolidate_leaves_retired_rules_untouched_and_unclustered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retired rule passes through unchanged, even if near-duplicate to an active rule."""
    active = _rule("deadline flag required")
    retired = _rule("deadline flag required", retired=True)
    provider = _ScriptedMergeProvider()
    _patch_provider(monkeypatch, provider)

    result = consolidate([active, retired], embed=_fake_embed)

    assert active in result
    assert retired in result
    assert provider.prompts == []  # nothing to merge: only one active rule


# --- Complete linkage (2026-10-07) -----------------------------------------------

from sawti.memory.consolidate import _cluster_indices  # noqa: E402


def _unit(angle_degrees: float) -> np.ndarray:
    radians = np.radians(angle_degrees)
    return np.array([np.cos(radians), np.sin(radians)])


def test_a_chain_does_not_become_one_cluster() -> None:
    """A~B and B~C clear the threshold, A~C does not: A and C must not share a cluster.

    Single linkage (the pre-fix behavior) put all three together. At 40 degrees
    apart, cos = 0.77 >= 0.7; A to C is 80 degrees, cos = 0.17.
    """
    a, b, c = _unit(0), _unit(40), _unit(80)
    assert float(a @ b) >= 0.7 and float(b @ c) >= 0.7 and float(a @ c) < 0.7
    clusters = _cluster_indices([a, b, c], 0.7)
    assert not any({0, 2} <= set(cluster) for cluster in clusters)
    assert sorted(len(cluster) for cluster in clusters) == [1, 2]


def test_every_pair_inside_every_cluster_clears_the_threshold() -> None:
    """The defining property, on a long chain where single linkage would make one cluster."""
    embeddings = [_unit(10 * i) for i in range(12)]
    for cluster in _cluster_indices(embeddings, 0.9):
        for i in cluster:
            for j in cluster:
                assert float(embeddings[i] @ embeddings[j]) >= 0.9 - 1e-9


def test_near_duplicates_still_merge_and_isolated_rules_stay_alone() -> None:
    embeddings = [_unit(0), _unit(5), _unit(10), _unit(90)]
    assert _cluster_indices(embeddings, 0.7) == [[0, 1, 2], [3]]
