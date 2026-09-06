"""Tests for `sawti.agent.nodes.score`.

Phase 2: mirrors `src/sawti/agent/nodes/score.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 2")
def test_score_produces_a_rubric_score_per_criterion() -> None:
    """score() produces exactly one RubricScore per rubric criterion."""


@pytest.mark.skip(reason="phase 2")
def test_score_requires_evidence_for_every_rubric_score() -> None:
    """score() never emits a RubricScore without a grounded evidence quote."""
