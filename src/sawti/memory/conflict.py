"""Detect conflicts between a candidate memory rule and existing stored rules.

Phase 4: agent memory — runs before insert.
"""

from __future__ import annotations

from sawti.memory.rule_schema import MemoryRule


def detect_conflict(candidate: MemoryRule, existing: list[MemoryRule]) -> list[MemoryRule]:
    """Return existing rules that conflict with `candidate`.

    Must run before a new rule is inserted into the store.

    Args:
        candidate: The newly induced rule, not yet stored.
        existing: Currently stored rules to check against.

    Returns:
        The subset of `existing` that conflicts with `candidate`.

    Raises:
        NotImplementedError: Until conflict detection is implemented.
    """
    # TODO(phase-4): use LLMProvider or similarity+contradiction check to find conflicts.
    raise NotImplementedError
