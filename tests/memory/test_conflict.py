"""Tests for `sawti.memory.conflict`.

Phase 4: mirrors `src/sawti/memory/conflict.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 4")
def test_detect_conflict_finds_contradictory_existing_rule() -> None:
    """detect_conflict() returns existing rules that contradict the candidate rule."""


@pytest.mark.skip(reason="phase 4")
def test_detect_conflict_returns_empty_when_no_conflict_exists() -> None:
    """detect_conflict() returns an empty list when the candidate rule is compatible with existing rules."""


@pytest.mark.skip(reason="phase 4")
def test_detect_conflict_runs_before_store_insert() -> None:
    """Conflict detection is invoked before a candidate rule is inserted into MemoryStore."""
