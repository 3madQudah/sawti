"""Tests for `sawti.agent.nodes.extract`.

Phase 2: mirrors `src/sawti/agent/nodes/extract.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 2")
def test_extract_populates_raw_extraction_from_redacted_transcript() -> None:
    """extract() calls the LLM provider with the redacted transcript, not the raw one."""


@pytest.mark.skip(reason="phase 2")
def test_extract_incorporates_retrieved_memory_rules_into_the_prompt() -> None:
    """extract() injects top-5 retrieved memory rules into the extraction prompt."""
