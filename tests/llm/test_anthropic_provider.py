"""Tests for `sawti.llm.anthropic_provider`.

Phase 1: mirrors `src/sawti/llm/anthropic_provider.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 1")
def test_complete_returns_text_from_anthropic_response() -> None:
    """AnthropicProvider.complete() returns the text content of the API response."""


@pytest.mark.skip(reason="phase 1")
def test_structured_complete_returns_validated_response_model_instance() -> None:
    """AnthropicProvider.structured_complete() returns an instance of the requested response_model."""


@pytest.mark.skip(reason="phase 1")
def test_structured_complete_raises_on_schema_violation() -> None:
    """AnthropicProvider.structured_complete() raises when the model output fails validation."""
