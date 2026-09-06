"""Tests for `sawti.memory.consolidate`.

Phase 4: mirrors `src/sawti/memory/consolidate.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 4")
def test_consolidate_merges_near_duplicate_rules_into_one() -> None:
    """consolidate() merges rules with highly similar rule_text into a single rule."""


@pytest.mark.skip(reason="phase 4")
def test_consolidate_leaves_dissimilar_rules_untouched() -> None:
    """consolidate() does not merge rules that are not near-duplicates."""
