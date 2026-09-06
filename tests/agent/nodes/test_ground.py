"""Tests for `sawti.agent.nodes.ground`.

Phase 2: mirrors `src/sawti/agent/nodes/ground.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 2")
def test_ground_drops_claims_whose_evidence_text_does_not_match_transcript_span() -> None:
    """ground() removes any claim whose evidence.text differs from the transcript at its offsets."""


@pytest.mark.skip(reason="phase 2")
def test_ground_retains_claims_with_verified_evidence() -> None:
    """ground() keeps claims whose evidence quote verifiably matches the transcript."""


@pytest.mark.skip(reason="phase 2")
def test_ground_never_downgrades_an_unsupported_claim_to_low_confidence() -> None:
    """ground() drops unsupported claims outright rather than marking them low-confidence."""
