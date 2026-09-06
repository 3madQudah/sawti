"""Tests for `sawti.agent.state`.

Phase 2: mirrors `src/sawti/agent/state.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 2")
def test_agent_state_accepts_partial_keys() -> None:
    """AgentState, being total=False, can be constructed with only a subset of keys."""


@pytest.mark.skip(reason="phase 2")
def test_agent_state_analysis_key_holds_validated_call_analysis() -> None:
    """AgentState['analysis'], once set, holds a validated CallAnalysis instance."""
