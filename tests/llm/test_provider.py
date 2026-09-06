"""Tests for `sawti.llm.provider`.

Phase 1: mirrors `src/sawti/llm/provider.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 1")
def test_get_llm_provider_returns_anthropic_provider_when_configured() -> None:
    """get_llm_provider() returns an AnthropicProvider when SAWTI_LLM_PROVIDER=anthropic."""


@pytest.mark.skip(reason="phase 1")
def test_get_llm_provider_returns_vllm_provider_when_configured() -> None:
    """get_llm_provider() returns a VLLMProvider when SAWTI_LLM_PROVIDER=vllm."""


@pytest.mark.skip(reason="phase 1")
def test_get_llm_provider_never_requires_call_site_changes_to_swap_providers() -> None:
    """Switching SAWTI_LLM_PROVIDER changes the returned provider without editing call sites."""
