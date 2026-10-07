"""Tests for `scripts/simulate_phase4_consolidation.py`'s replay (phase 6.2)."""

from __future__ import annotations

import numpy as np
from simulate_phase4_consolidation import replay, single_linkage

from sawti.memory.consolidate import _cluster_indices


def _unit(degrees: float) -> np.ndarray:
    radians = np.radians(degrees)
    return np.array([np.cos(radians), np.sin(radians)])


def test_replay_shows_single_linkage_chaining_and_complete_linkage_not() -> None:
    """One batch holding a chain A~B~C (A~C dissimilar): single linkage makes one cluster of 3."""
    batches = [[_unit(0), _unit(40), _unit(80)]]
    assert replay(batches, single_linkage) == [{"consolidated": 3, "max_cluster": 3, "active_after": 1}]
    assert replay(batches, _cluster_indices) == [{"consolidated": 3, "max_cluster": 2, "active_after": 2}]


def test_replay_carries_survivors_into_the_next_batch() -> None:
    batches = [[_unit(0), _unit(5)], [_unit(90)]]
    out = replay(batches, _cluster_indices)
    assert [b["consolidated"] for b in out] == [2, 2]  # 1 survivor + 1 new
    assert [b["active_after"] for b in out] == [1, 2]
