"""Merge near-duplicate memory rules.

Phase 4: agent memory maintenance.
"""

from __future__ import annotations

from sawti.memory.rule_schema import MemoryRule


def consolidate(rules: list[MemoryRule]) -> list[MemoryRule]:
    """Merge near-duplicate rules in `rules` into single, generalized rules.

    Args:
        rules: Candidate rules to deduplicate/merge.

    Returns:
        The consolidated rule set.

    Raises:
        NotImplementedError: Until consolidation is implemented.
    """
    # TODO(phase-4): cluster by embedding similarity and merge each cluster via LLMProvider.
    raise NotImplementedError
