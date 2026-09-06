"""Memory rule lifecycle: success tracking and auto-retirement.

Phase 4: agent memory maintenance.
"""

from __future__ import annotations

from sawti.memory.rule_schema import MemoryRule


def record_outcome(rule: MemoryRule, *, success: bool) -> MemoryRule:
    """Update a rule's success/failure counters after it was applied.

    Args:
        rule: The rule that was applied.
        success: Whether applying the rule led to a correct outcome.

    Returns:
        The updated rule.

    Raises:
        NotImplementedError: Until outcome tracking is implemented.
    """
    # TODO(phase-4): increment success_count/failure_count and persist.
    raise NotImplementedError


def maybe_retire(
    rule: MemoryRule, *, min_applications: int = 10, min_success_rate: float = 0.5
) -> MemoryRule:
    """Retire `rule` if it has underperformed over enough applications.

    Args:
        rule: The rule to evaluate.
        min_applications: Minimum number of applications before retirement is considered.
        min_success_rate: Minimum success rate required to keep the rule active.

    Returns:
        The rule, with `retired` set if it no longer meets the bar.

    Raises:
        NotImplementedError: Until auto-retirement is implemented.
    """
    # TODO(phase-4): compute success rate over (success_count + failure_count) applications
    # and set rule.retired when below min_success_rate past min_applications.
    raise NotImplementedError
