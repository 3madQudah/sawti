"""Tests for `sawti.memory.rule_schema`.

Phase 4: mirrors `src/sawti/memory/rule_schema.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from sawti.memory.rule_schema import MemoryRule


def test_memory_rule_requires_nonempty_rule_text() -> None:
    """MemoryRule rejects construction with an empty rule_text."""
    with pytest.raises(ValidationError):
        MemoryRule(rule_text="", created_at=datetime.now(UTC))


def test_memory_rule_defaults_to_not_retired() -> None:
    """MemoryRule.retired defaults to False on creation."""
    rule = MemoryRule(
        rule_text="Flag refund promises that lack a stated timeframe as incomplete.",
        created_at=datetime.now(UTC),
    )
    assert rule.retired is False
