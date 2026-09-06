"""Tests for `sawti.memory.store`.

Phase 4: mirrors `src/sawti/memory/store.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 4")
def test_retrieve_returns_at_most_top_k_rules() -> None:
    """MemoryStore.retrieve() never returns more than top_k rules."""


@pytest.mark.skip(reason="phase 4")
def test_retrieve_defaults_to_top_5() -> None:
    """MemoryStore.retrieve() defaults top_k to 5 when not specified."""


@pytest.mark.skip(reason="phase 4")
def test_retrieve_excludes_retired_rules() -> None:
    """MemoryStore.retrieve() never returns rules with retired=True."""


@pytest.mark.skip(reason="phase 4")
def test_retrieve_orders_results_by_relevance_descending() -> None:
    """MemoryStore.retrieve() returns the best-matching rule first."""
