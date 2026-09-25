"""Memory rule lifecycle: success tracking and auto-retirement.

Phase 4: agent memory maintenance.

Pure functions, deliberately: neither touches `sawti.memory.store` or the
database. Both return a new `MemoryRule` rather than mutating `rule` in
place (`BaseModel.model_copy`, not attribute assignment) — consistent with
every other schema in this project being treated as an immutable value, and
it keeps these two functions trivially composable and testable without a
store or a session. A caller that wants the change to stick calls
`MemoryStore.add()` (or persists the rule elsewhere) with the returned
value; that wiring is the caller's job, not this module's.
"""

from __future__ import annotations

from sawti.memory.rule_schema import MemoryRule


def record_outcome(rule: MemoryRule, *, success: bool) -> MemoryRule:
    """Update a rule's success/failure counters after it was applied.

    Args:
        rule: The rule that was applied.
        success: Whether applying the rule led to a correct outcome.

    Returns:
        A new `MemoryRule` with `success_count` or `failure_count`
        incremented by one; `rule` itself is left unchanged.
    """
    return rule.model_copy(
        update={
            "success_count": rule.success_count + (1 if success else 0),
            "failure_count": rule.failure_count + (0 if success else 1),
        }
    )


def maybe_retire(
    rule: MemoryRule, *, min_applications: int = 10, min_success_rate: float = 0.5
) -> MemoryRule:
    """Retire `rule` if it has underperformed over enough applications.

    Args:
        rule: The rule to evaluate.
        min_applications: Minimum number of applications (`success_count +
            failure_count`) before retirement is considered at all — a rule
            applied twice hasn't had a fair chance yet.
        min_success_rate: Minimum `success_count / applications` required
            to keep the rule active, once `min_applications` is reached.

    Returns:
        `rule` unchanged if it hasn't reached `min_applications` yet or
        still clears `min_success_rate`; otherwise a new `MemoryRule` with
        `retired=True`. Never un-retires an already-retired rule.
    """
    applications = rule.success_count + rule.failure_count
    if applications == 0 or applications < min_applications:
        return rule

    success_rate = rule.success_count / applications
    if success_rate < min_success_rate:
        return rule.model_copy(update={"retired": True})
    return rule
