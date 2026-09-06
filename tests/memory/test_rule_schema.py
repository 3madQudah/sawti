"""Tests for `sawti.memory.rule_schema`.

Phase 4: mirrors `src/sawti/memory/rule_schema.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 4")
def test_memory_rule_requires_nonempty_rule_text() -> None:
    """MemoryRule rejects construction with an empty rule_text."""


@pytest.mark.skip(reason="phase 4")
def test_memory_rule_defaults_to_not_retired() -> None:
    """MemoryRule.retired defaults to False on creation."""
