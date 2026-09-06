"""Tests for `sawti.llm.gemini_provider`.

Phase 1: mirrors `src/sawti/llm/gemini_provider.py`. The `google-genai`
client is always mocked here — these tests never make a real API call.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, ValidationError

from sawti.llm.gemini_provider import GeminiProvider


class _Widget(BaseModel):
    """Minimal response_model used to exercise structured_complete()."""

    name: str
    count: int


def _provider_with_mock_client(response_text: str) -> tuple[GeminiProvider, MagicMock]:
    fake_response = MagicMock()
    fake_response.text = response_text
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = AsyncMock(return_value=fake_response)

    provider = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")
    provider._client = fake_client
    return provider, fake_client


async def test_complete_returns_text_from_gemini_response() -> None:
    """GeminiProvider.complete() returns the text content of the API response."""
    provider, fake_client = _provider_with_mock_client("Agent: Hello.\nCustomer: Hi.")

    result = await provider.complete("say hello", system="be nice")

    assert result == "Agent: Hello.\nCustomer: Hi."
    fake_client.aio.models.generate_content.assert_awaited_once()
    _, kwargs = fake_client.aio.models.generate_content.await_args
    assert kwargs["model"] == "gemini-2.5-flash"
    assert kwargs["contents"] == "say hello"


async def test_structured_complete_returns_validated_response_model_instance() -> None:
    """GeminiProvider.structured_complete() returns a validated instance of response_model."""
    provider, fake_client = _provider_with_mock_client(json.dumps({"name": "widget", "count": 3}))

    result = await provider.structured_complete("describe a widget", response_model=_Widget)

    assert isinstance(result, _Widget)
    assert result == _Widget(name="widget", count=3)
    _, kwargs = fake_client.aio.models.generate_content.await_args
    config = kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == _Widget.model_json_schema()


async def test_structured_complete_raises_on_schema_violation() -> None:
    """GeminiProvider.structured_complete() raises when the model output fails validation."""
    provider, _ = _provider_with_mock_client(json.dumps({"name": "widget"}))  # missing required "count"

    with pytest.raises(ValidationError):
        await provider.structured_complete("describe a widget", response_model=_Widget)
