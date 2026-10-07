"""Tests for `sawti.llm.fake_provider` and transient-error classification in `sawti.llm.provider`.

Phase 6.2.
"""

from __future__ import annotations

import httpx
import pytest

from sawti.agent.nodes.ground import is_grounded
from sawti.llm.fake_provider import FakeProvider
from sawti.llm.provider import TransientProviderError, get_llm_provider, is_transient_provider_error
from sawti.observability.tracing import TracedProvider
from sawti.schemas import ExtractionProposal

TRANSCRIPT = "Agent: I will call you.\nCustomer: Fine.\nAgent: Bye now.\nAgent: Third line.\n"


async def test_fake_provider_quotes_agent_lines_verbatim_plus_unsupported_claims() -> None:
    proposal = await FakeProvider(ungrounded_claims=2).structured_complete(
        TRANSCRIPT, response_model=ExtractionProposal
    )
    grounded = [is_grounded(TRANSCRIPT, c) for c in proposal.commitments]
    assert grounded == [True, True, False, False]
    assert proposal.sentiment_trajectory.points[0].quote.text == "Fine."


def test_fake_provider_is_selectable_by_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from sawti.config import get_settings

    monkeypatch.setenv("SAWTI_LLM_PROVIDER", "fake")
    get_settings.cache_clear()
    try:
        provider = get_llm_provider()
        assert isinstance(provider, TracedProvider)  # phase 6.3: every provider comes traced
        assert isinstance(provider.inner, FakeProvider)
    finally:
        get_settings.cache_clear()


class _CodeError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


class _StatusCodeError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(status_code)
        self.status_code = status_code


@pytest.mark.parametrize(
    "exc",
    [
        TransientProviderError("x"),
        TimeoutError(),
        ConnectionResetError(),
        httpx.ConnectError("refused"),
        _CodeError(429),
        _CodeError(503),
        _StatusCodeError(429),
    ],
)
def test_transient_errors_are_recognized(exc: BaseException) -> None:
    assert is_transient_provider_error(exc)


@pytest.mark.parametrize(
    "exc",
    [RuntimeError("429 in the text is not enough"), ValueError(), _CodeError(400), _StatusCodeError(401)],
)
def test_everything_else_is_not_transient(exc: BaseException) -> None:
    assert not is_transient_provider_error(exc)


async def test_provider_override_is_scoped_to_its_context() -> None:
    """Phase 6.3: the benchmark's injection point; concurrent tasks each see their own override."""
    import asyncio

    from sawti.llm.provider import provider_override

    async def which(tag: int) -> int:
        with provider_override(FakeProvider(ungrounded_claims=tag)):
            await asyncio.sleep(0.01 * (3 - tag))
            return get_llm_provider()._ungrounded  # type: ignore[attr-defined]

    assert await asyncio.gather(*(which(t) for t in range(3))) == [0, 1, 2]
    assert isinstance(get_llm_provider(), TracedProvider)  # back to the configured provider
