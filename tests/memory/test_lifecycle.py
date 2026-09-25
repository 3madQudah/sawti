"""Tests for `sawti.memory.lifecycle`.

Phase 4: mirrors `src/sawti/memory/lifecycle.py`.

Pure counter math — no LLM, no store, no monkeypatching needed.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sawti.memory.lifecycle import maybe_retire, record_outcome
from sawti.memory.rule_schema import MemoryRule


def _rule(*, success_count: int = 0, failure_count: int = 0, retired: bool = False) -> MemoryRule:
    return MemoryRule(
        rule_text="placeholder rule",
        created_at=datetime.now(UTC),
        success_count=success_count,
        failure_count=failure_count,
        retired=retired,
    )


def test_record_outcome_increments_success_count_on_success() -> None:
    """record_outcome(success=True) increments the rule's success_count."""
    rule = _rule(success_count=2, failure_count=1)

    updated = record_outcome(rule, success=True)

    assert updated.success_count == 3
    assert updated.failure_count == 1


def test_record_outcome_increments_failure_count_on_failure() -> None:
    """record_outcome(success=False) increments the rule's failure_count."""
    rule = _rule(success_count=2, failure_count=1)

    updated = record_outcome(rule, success=False)

    assert updated.success_count == 2
    assert updated.failure_count == 2


def test_record_outcome_does_not_mutate_the_original_rule() -> None:
    """record_outcome() returns a new MemoryRule rather than mutating its input."""
    rule = _rule(success_count=0, failure_count=0)

    record_outcome(rule, success=True)

    assert rule.success_count == 0
    assert rule.failure_count == 0


def test_maybe_retire_retires_rule_below_min_success_rate_after_min_applications() -> None:
    """maybe_retire() sets retired=True once a rule underperforms past min_applications."""
    rule = _rule(success_count=3, failure_count=7)  # 10 applications, 0.3 success rate

    updated = maybe_retire(rule, min_applications=10, min_success_rate=0.5)

    assert updated.retired is True


def test_maybe_retire_keeps_rule_active_before_min_applications_reached() -> None:
    """maybe_retire() does not retire a rule that hasn't reached min_applications yet."""
    rule = _rule(success_count=0, failure_count=3)  # only 3 applications, well below min

    updated = maybe_retire(rule, min_applications=10, min_success_rate=0.5)

    assert updated.retired is False


def test_maybe_retire_keeps_rule_active_when_success_rate_meets_threshold() -> None:
    """maybe_retire() does not retire a rule that clears min_success_rate past min_applications."""
    rule = _rule(success_count=8, failure_count=2)  # 10 applications, 0.8 success rate

    updated = maybe_retire(rule, min_applications=10, min_success_rate=0.5)

    assert updated.retired is False


def test_maybe_retire_never_un_retires_an_already_retired_rule() -> None:
    """A rule that clears the bar again is not automatically un-retired."""
    rule = _rule(success_count=8, failure_count=2, retired=True)

    updated = maybe_retire(rule, min_applications=10, min_success_rate=0.5)

    assert updated.retired is True
