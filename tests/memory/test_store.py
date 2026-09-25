"""Tests for `sawti.memory.store`.

Phase 4: mirrors `src/sawti/memory/store.py`.

Uses a small, deterministic fake embedder throughout — a real
sentence-transformers model works (see `scripts/inspect_induced_rules.py`
for that in practice via `sawti.memory.induction`), but downloading one on
every test run would make the suite slow and network-dependent. The fake
embeds each text as a one-hot bag-of-words vector over a fixed vocabulary,
so cosine similarity is exactly keyword overlap — fully predictable for
ordering assertions.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from sawti.memory.rule_schema import MemoryRule
from sawti.memory.store import MemoryStore

_VOCAB = ["refund", "deadline", "compliance", "identity", "rubric", "unrelated"]


def _fake_embed(text: str) -> np.ndarray:
    words = set(text.lower().split())
    return np.array([1.0 if word in words else 0.0 for word in _VOCAB])


def _rule(rule_text: str, *, retired: bool = False) -> MemoryRule:
    return MemoryRule(rule_text=rule_text, created_at=datetime.now(UTC), retired=retired)


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(embed=_fake_embed)


def test_retrieve_returns_at_most_top_k_rules(store: MemoryStore) -> None:
    """MemoryStore.retrieve() never returns more than top_k rules."""
    for i in range(8):
        store.add(_rule(f"rule {i} refund"))

    results = store.retrieve("refund", top_k=3)

    assert len(results) == 3


def test_retrieve_defaults_to_top_5(store: MemoryStore) -> None:
    """MemoryStore.retrieve() defaults top_k to 5 when not specified."""
    for i in range(8):
        store.add(_rule(f"rule {i} refund"))

    results = store.retrieve("refund")

    assert len(results) == 5


def test_retrieve_excludes_retired_rules(store: MemoryStore) -> None:
    """MemoryStore.retrieve() never returns rules with retired=True."""
    active = _rule("refund deadline")
    retired = _rule("refund deadline", retired=True)
    store.add(active)
    store.add(retired)

    results = store.retrieve("refund deadline", top_k=5)

    assert active in results
    assert retired not in results


def test_retrieve_orders_results_by_relevance_descending(store: MemoryStore) -> None:
    """MemoryStore.retrieve() returns the best-matching rule first."""
    exact = _rule("refund deadline compliance")
    partial = _rule("refund unrelated unrelated")
    unrelated = _rule("unrelated unrelated unrelated")
    store.add(unrelated)
    store.add(exact)
    store.add(partial)

    results = store.retrieve("refund deadline compliance", top_k=3)

    assert [rule.rule_text for rule in results] == [
        exact.rule_text,
        partial.rule_text,
        unrelated.rule_text,
    ]


def test_retrieve_returns_fewer_than_top_k_when_not_enough_rules_stored(store: MemoryStore) -> None:
    """MemoryStore.retrieve() returns fewer than top_k rather than padding, when few rules exist."""
    store.add(_rule("refund deadline"))

    results = store.retrieve("refund deadline", top_k=5)

    assert len(results) == 1


def test_add_replaces_a_rule_sharing_an_existing_id(store: MemoryStore) -> None:
    """add() called again with the same id replaces the stored rule and its embedding."""
    rule = _rule("refund deadline")
    store.add(rule)

    updated = rule.model_copy(update={"rule_text": "compliance identity"})
    store.add(updated)

    results = store.retrieve("compliance identity", top_k=1)

    assert results == [updated]


def test_replace_all_replaces_the_stores_entire_contents(store: MemoryStore) -> None:
    """replace_all() drops every prior rule, keeping only what it was given."""
    store.add(_rule("refund deadline"))
    store.add(_rule("compliance identity"))
    replacement = _rule("unrelated")

    store.replace_all([replacement])

    assert store.all_active_rules() == [replacement]
