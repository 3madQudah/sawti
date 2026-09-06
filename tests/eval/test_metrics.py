"""Tests for `sawti.eval.metrics`.

Phase 1: mirrors `src/sawti/eval/metrics.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 1")
def test_accuracy_by_category_returns_a_mapping_not_a_float() -> None:
    """accuracy_by_category() returns dict[Language, float], never a single blended float."""


@pytest.mark.skip(reason="phase 1")
def test_accuracy_by_category_only_includes_languages_present_in_data() -> None:
    """accuracy_by_category() omits Language keys that have no examples in the input."""


@pytest.mark.skip(reason="phase 1")
def test_rubric_agreement_by_category_scores_each_language_independently() -> None:
    """rubric_agreement_by_category() computes agreement separately per Language."""


@pytest.mark.skip(reason="phase 1")
def test_grounding_precision_by_category_flags_ungrounded_claims_per_language() -> None:
    """grounding_precision_by_category() reports precision separately per Language."""
