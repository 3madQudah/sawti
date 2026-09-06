"""Induce a general memory rule from one or more specific corrections.

Phase 4: agent memory.
"""

from __future__ import annotations

from sawti.memory.rule_schema import MemoryRule
from sawti.schemas import Correction


def induce_rule(corrections: list[Correction]) -> MemoryRule:
    """Generalize one or more corrections into a reusable `MemoryRule`.

    Args:
        corrections: One or more `Correction` records to generalize from.

    Returns:
        The induced `MemoryRule`.

    Raises:
        NotImplementedError: Until induction is implemented.
    """
    # TODO(phase-4): use LLMProvider to abstract corrections into a general, checkable rule.
    raise NotImplementedError
