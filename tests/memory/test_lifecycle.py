"""Tests for `sawti.memory.lifecycle`.

Phase 4: mirrors `src/sawti/memory/lifecycle.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 4")
def test_record_outcome_increments_success_count_on_success() -> None:
    """record_outcome(success=True) increments the rule's success_count."""


@pytest.mark.skip(reason="phase 4")
def test_record_outcome_increments_failure_count_on_failure() -> None:
    """record_outcome(success=False) increments the rule's failure_count."""


@pytest.mark.skip(reason="phase 4")
def test_maybe_retire_retires_rule_below_min_success_rate_after_min_applications() -> None:
    """maybe_retire() sets retired=True once a rule underperforms past min_applications."""


@pytest.mark.skip(reason="phase 4")
def test_maybe_retire_keeps_rule_active_before_min_applications_reached() -> None:
    """maybe_retire() does not retire a rule that hasn't reached min_applications yet."""
