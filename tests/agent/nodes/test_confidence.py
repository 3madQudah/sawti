"""Tests for `sawti.agent.nodes.confidence`.

Phase 2, stage 3: mirrors `src/sawti/agent/nodes/confidence.py`.

The boundary case matters most: a score sitting exactly on the threshold must
resolve one way, deterministically, and that direction must be tested rather
than assumed.
"""

from __future__ import annotations

import pytest

from sawti.agent.nodes.confidence import compute_confidence, confidence
from sawti.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Drop the cached Settings singleton so threshold monkeypatching takes effect."""
    get_settings.cache_clear()


async def test_confidence_flags_human_review_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """confidence() sets requires_human_review=True when the computed score is below threshold."""
    monkeypatch.setenv("SAWTI_CONFIDENCE_THRESHOLD", "0.7")

    update = await confidence({"grounding_coverage": 0.4})

    assert update["confidence"] == 0.4
    assert update["requires_human_review"] is True
    assert update["review_status"] == "awaiting_human"


async def test_confidence_allows_auto_pass_at_or_above_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """confidence() sets requires_human_review=False when the computed score meets threshold."""
    monkeypatch.setenv("SAWTI_CONFIDENCE_THRESHOLD", "0.7")

    update = await confidence({"grounding_coverage": 0.95})

    assert update["requires_human_review"] is False
    assert update["review_status"] == "auto_passed"


async def test_confidence_exactly_on_the_threshold_auto_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Boundary: a score equal to the threshold passes, because the test is `< threshold`.

    This is the case that silently flips if someone "tidies" the comparison
    operator, so it is pinned explicitly rather than left to inference.
    """
    monkeypatch.setenv("SAWTI_CONFIDENCE_THRESHOLD", "0.7")

    update = await confidence({"grounding_coverage": 0.7})

    assert update["confidence"] == 0.7
    assert update["requires_human_review"] is False


async def test_confidence_just_below_the_threshold_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Boundary, other side: a hair under the threshold goes to a human."""
    monkeypatch.setenv("SAWTI_CONFIDENCE_THRESHOLD", "0.7")

    update = await confidence({"grounding_coverage": 0.6999})

    assert update["requires_human_review"] is True


async def test_confidence_compares_against_the_configured_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """confidence() compares against sawti.config.get_settings().confidence_threshold."""
    monkeypatch.setenv("SAWTI_CONFIDENCE_THRESHOLD", "0.95")

    update = await confidence({"grounding_coverage": 0.9})

    # 0.9 auto-passes at the default 0.7 but must escalate at 0.95.
    assert update["requires_human_review"] is True


def test_default_threshold_is_the_documented_value() -> None:
    """The named constant, not a magic number: Settings carries the project default."""
    assert Settings().confidence_threshold == 0.7


def test_compute_confidence_uses_grounding_coverage() -> None:
    """Confidence is the measured grounding coverage, not a model self-rating."""
    assert compute_confidence({"grounding_coverage": 0.25}) == 0.25


def test_compute_confidence_scores_an_errored_run_as_zero() -> None:
    """Failure path: a run that errored is never confident, whatever its coverage says.

    Extraction dying before proposing anything leaves coverage at a vacuous 1.0;
    without this guard a broken run would auto-pass.
    """
    assert compute_confidence({"grounding_coverage": 1.0, "error": "provider exhausted"}) == 0.0


def test_compute_confidence_treats_missing_coverage_as_zero() -> None:
    """Failure path: if `ground` never ran, assume nothing rather than the best."""
    assert compute_confidence({}) == 0.0
