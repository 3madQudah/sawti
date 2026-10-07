"""Unit tests for `sawti.observability.tracing` (phase 6.3)."""

from __future__ import annotations

from typing import Any

import pytest
from fake_langfuse import FakeLangfuse
from pydantic import BaseModel

from sawti.llm.provider import LLMProvider
from sawti.llm.usage import record_usage
from sawti.observability import tracing

PII = "0791234567"


class _Out(BaseModel):
    text: str


class _Echo(LLMProvider):
    model_name = "echo-1"

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        record_usage(11, 5, model="echo-1")
        return prompt

    async def structured_complete(self, prompt: str, *, response_model: Any, **kwargs: Any) -> Any:
        record_usage(11, 5, model="echo-1")
        return response_model(text=prompt)


class _Boom(_Echo):
    async def complete(self, prompt: str, **kwargs: Any) -> str:
        raise RuntimeError("provider down")


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeLangfuse:
    client = FakeLangfuse()
    monkeypatch.setattr(tracing, "get_client", lambda: client)
    return client


def test_scrub_redacts_strings_at_any_depth_and_leaves_numbers() -> None:
    value = {"a": [f"call me on {PII}", {"b": (PII,)}], "n": 3, "none": None}
    assert tracing.scrub(value) == {
        "a": ["call me on [NATIONAL_ID]", {"b": ["[NATIONAL_ID]"]}],
        "n": 3,
        "none": None,
    }


async def test_a_generation_carries_latency_tokens_provider_model_language_and_call_id(
    fake: FakeLangfuse,
) -> None:
    with tracing.call_trace("call-1", "mixed", name="t"):
        out = await tracing.TracedProvider(_Echo(), provider_name="echo").structured_complete(
            "hello", response_model=_Out, system="sys"
        )
    assert out == _Out(text="hello")
    [trace] = fake.of("trace")
    assert trace.payload["session_id"] == "call-1"
    [gen] = fake.of("generation")
    assert gen.payload["name"] == "llm:_Out"
    assert gen.payload["model"] == "echo-1"
    assert gen.payload["usage"] == {"input": 11, "output": 5, "unit": "TOKENS"}
    meta = gen.payload["metadata"]
    assert (meta["provider"], meta["call_id"], meta["language"]) == ("echo", "call-1", "mixed")
    assert meta["latency_s"] >= 0
    assert gen.payload["output"] == {"text": "hello"}


async def test_raw_pii_handed_to_a_provider_never_reaches_the_trace(fake: FakeLangfuse) -> None:
    """The backstop: even a call site that passed raw text would be redacted before tracing."""
    with tracing.call_trace("call-2", "en", name="t"):
        await tracing.TracedProvider(_Echo(), provider_name="echo").complete(f"my id is {PII}", system=PII)
    assert PII not in fake.everything_sent()


async def test_a_failing_call_is_traced_as_an_error_and_still_raises(fake: FakeLangfuse) -> None:
    with pytest.raises(RuntimeError, match="provider down"), tracing.call_trace("c", "ar", name="t"):
        await tracing.TracedProvider(_Boom(), provider_name="boom").complete("x")
    [gen] = fake.of("generation")
    assert gen.payload["level"] == "ERROR" and "provider down" in gen.payload["status_message"]


async def test_tracing_off_is_a_transparent_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracing, "get_client", lambda: None)
    with tracing.call_trace("c", "ar", name="t"):
        assert await tracing.TracedProvider(_Echo(), provider_name="echo").complete("x") == "x"
    tracing.flush()  # no client: nothing to do, must not raise


async def test_a_broken_tracing_backend_never_breaks_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Broken:
        def trace(self, **_: Any) -> Any:
            raise ConnectionError("langfuse unreachable")

        def generation(self, **_: Any) -> Any:
            raise ConnectionError("langfuse unreachable")

        def flush(self) -> None:
            raise ConnectionError("langfuse unreachable")

    monkeypatch.setattr(tracing, "get_client", lambda: _Broken())
    with tracing.call_trace("c", "ar", name="t"):
        assert await tracing.TracedProvider(_Echo(), provider_name="echo").complete("x") == "x"
    tracing.flush()


def test_client_is_off_without_keys_and_points_at_the_configured_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sawti.config import Settings

    tracing.get_client.cache_clear()
    monkeypatch.setattr(tracing, "get_settings", lambda: Settings(_env_file=None))  # type: ignore[call-arg]
    assert tracing.get_client() is None

    built: dict[str, Any] = {}

    class _Langfuse:
        def __init__(self, **kwargs: Any) -> None:
            built.update(kwargs)

    import langfuse

    monkeypatch.setattr(langfuse, "Langfuse", _Langfuse)
    configured = Settings(_env_file=None, LANGFUSE_PUBLIC_KEY="pk", LANGFUSE_SECRET_KEY="sk")  # type: ignore[call-arg]
    monkeypatch.setattr(tracing, "get_settings", lambda: configured)
    tracing.get_client.cache_clear()
    try:
        assert tracing.get_client() is not None
        assert built["host"] == "http://localhost:3000"  # self-hosted by default, never Cloud
        assert built["mask"](data=f"x {PII}") == "x [NATIONAL_ID]"
    finally:
        tracing.get_client.cache_clear()
