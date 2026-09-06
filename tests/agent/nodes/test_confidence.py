"""Tests for `sawti.agent.nodes.confidence`.

Phase 2: mirrors `src/sawti/agent/nodes/confidence.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 2")
def test_confidence_sets_requires_human_review_true_below_threshold() -> None:
    """confidence() sets requires_human_review=True when the computed score is below threshold."""


@pytest.mark.skip(reason="phase 2")
def test_confidence_sets_requires_human_review_false_above_threshold() -> None:
    """confidence() sets requires_human_review=False when the computed score meets threshold."""


@pytest.mark.skip(reason="phase 2")
def test_confidence_reads_threshold_from_settings_not_a_hardcoded_value() -> None:
    """confidence() compares against sawti.config.get_settings().confidence_threshold."""
