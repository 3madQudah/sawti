"""Tests for `sawti.llm.vllm_provider`, mocked at the HTTP level.

Phase 6.3. A `httpx.MockTransport` stands in for `vllm serve`'s
OpenAI-compatible API, so these check the actual request body the server
would receive (guided decoding, LoRA model name, thinking off) and the parsing
of its actual response shape.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import openai
import pytest
from pydantic import BaseModel, Field, ValidationError

from sawti.llm.provider import is_transient_provider_error
from sawti.llm.usage import capture_usage
from sawti.llm.vllm_provider import VLLMProvider


class _Answer(BaseModel):
    summary: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0)


def _completion(content: str, *, model: str = "sawti-qlora", finish: str = "stop") -> dict[str, Any]:
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 34, "total_tokens": 154},
    }


def _provider(handler: Any, **kwargs: Any) -> tuple[VLLMProvider, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    def _handle(request: httpx.Request) -> httpx.Response:
        seen.append({"path": request.url.path, "body": json.loads(request.content)})
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(_handle))
    return VLLMProvider("http://vllm.test/v1", "sawti-qlora", http_client=client, **kwargs), seen


async def test_structured_complete_sends_guided_decoding_lora_model_and_no_thinking() -> None:
    provider, seen = _provider(
        lambda r: httpx.Response(200, json=_completion('{"summary": "ok", "score": 0.5}'))
    )
    answer = await provider.structured_complete("transcript", response_model=_Answer, system="be exact")

    assert answer == _Answer(summary="ok", score=0.5)
    body = seen[0]["body"]
    assert seen[0]["path"] == "/v1/chat/completions"
    assert body["model"] == "sawti-qlora"
    assert body["messages"] == [
        {"role": "system", "content": "be exact"},
        {"role": "user", "content": "transcript"},
    ]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == _Answer.model_json_schema()
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["temperature"] == 0.0 and body["max_tokens"] == 4096


async def test_usage_is_reported_from_the_response() -> None:
    provider, _ = _provider(lambda r: httpx.Response(200, json=_completion('{"summary": "ok", "score": 1}')))
    with capture_usage() as usage:
        await provider.structured_complete("t", response_model=_Answer)
    assert [(u.prompt_tokens, u.completion_tokens, u.model) for u in usage] == [(120, 34, "sawti-qlora")]


async def test_output_that_violates_the_schema_raises_validation_error() -> None:
    """The grammar fixes structure, not field constraints: score 7 still fails `le=1.0`."""
    provider, _ = _provider(lambda r: httpx.Response(200, json=_completion('{"summary": "ok", "score": 7}')))
    with pytest.raises(ValidationError):
        await provider.structured_complete("t", response_model=_Answer)


async def test_truncated_generation_is_a_validation_error() -> None:
    provider, _ = _provider(
        lambda r: httpx.Response(200, json=_completion('{"summary": "o', finish="length"))
    )
    with pytest.raises(ValidationError):
        await provider.structured_complete("t", response_model=_Answer)


async def test_complete_returns_the_text_and_omits_the_schema() -> None:
    provider, seen = _provider(lambda r: httpx.Response(200, json=_completion("hello")))
    assert await provider.complete("hi") == "hello"
    assert "response_format" not in seen[0]["body"]


async def test_the_base_model_arm_names_the_base_model_and_can_send_no_template_kwargs() -> None:
    seen: list[dict[str, Any]] = []

    def _handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_completion("x", model="Qwen/Qwen3-8B"))

    provider = VLLMProvider(
        "http://vllm.test/v1",
        "Qwen/Qwen3-8B",
        chat_template_kwargs={},
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handle)),
    )
    await provider.complete("hi")
    assert seen[0]["model"] == "Qwen/Qwen3-8B"
    assert "chat_template_kwargs" not in seen[0]


@pytest.mark.parametrize("status", [429, 503])
async def test_rate_limits_and_overload_are_transient(status: int) -> None:
    provider, _ = _provider(lambda r: httpx.Response(status, json={"error": {"message": "busy"}}))
    with pytest.raises(openai.APIStatusError) as caught:
        await provider.complete("hi")
    assert is_transient_provider_error(caught.value)


async def test_a_bad_request_is_not_transient() -> None:
    provider, _ = _provider(lambda r: httpx.Response(400, json={"error": {"message": "bad schema"}}))
    with pytest.raises(openai.BadRequestError) as caught:
        await provider.complete("hi")
    assert not is_transient_provider_error(caught.value)


async def test_a_dead_server_is_transient() -> None:
    def _refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider, _ = _provider(_refuse)
    with pytest.raises(openai.APIConnectionError) as caught:
        await provider.complete("hi")
    assert is_transient_provider_error(caught.value)
