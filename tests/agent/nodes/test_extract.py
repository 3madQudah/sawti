"""Tests for `sawti.agent.nodes.extract`.

Phase 2, stage 2: mirrors `src/sawti/agent/nodes/extract.py`.

The LLM provider is mocked throughout — these tests are about what the node
sends, what it stores, and how it fails, never about model quality.
"""

from __future__ import annotations

import pytest

from sawti.agent.nodes import extract as extract_module
from sawti.agent.nodes.extract import EXTRACTION_SYSTEM_PROMPT, extract
from sawti.schemas import Commitment, ExtractionProposal, Quote

RAW = "Agent: Call me on +962 79 123 4567.\nCustomer: Fine.\n"


class _FakeProvider:
    """Stand-in for an LLMProvider that records its call and returns a fixed proposal."""

    def __init__(self, proposal: ExtractionProposal | None = None, error: Exception | None = None) -> None:
        """Store what this fake should return, or the error it should raise."""
        self.proposal = proposal
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def structured_complete(self, prompt: str, **kwargs: object) -> ExtractionProposal:
        """Record the call, then return the canned proposal or raise the canned error."""
        self.calls.append({"prompt": prompt, **kwargs})
        if self.error is not None:
            raise self.error
        assert self.proposal is not None
        return self.proposal


def _proposal() -> ExtractionProposal:
    """A minimal proposal with one commitment whose offsets are deliberately unverified."""
    quote = Quote(text="Fine.", speaker="Customer", start_char=0, end_char=5)
    return ExtractionProposal(
        summary="Customer agreed.",
        commitments=[Commitment(evidence=quote, promised_by="Agent", description="Call back.")],
    )


@pytest.fixture
def fake_provider(monkeypatch: pytest.MonkeyPatch) -> _FakeProvider:
    """Patch `get_llm_provider` in the extract module and hand back the fake."""
    provider = _FakeProvider(proposal=_proposal())
    monkeypatch.setattr(extract_module, "get_llm_provider", lambda: provider)
    return provider


async def test_extract_populates_raw_extraction_from_redacted_transcript(
    fake_provider: _FakeProvider,
) -> None:
    """extract() calls the LLM provider with the redacted transcript, not the raw one."""
    redacted = "Agent: Call me on [PHONE].\nCustomer: Fine.\n"
    update = await extract({"transcript": RAW, "redacted_transcript": redacted})

    assert fake_provider.calls[0]["prompt"] == redacted
    assert "+962" not in str(fake_provider.calls[0]["prompt"])
    assert update["raw_extraction"]["summary"] == "Customer agreed."
    assert update["summary"] == "Customer agreed."
    assert len(update["commitments"]) == 1


async def test_extract_redacts_before_calling_the_model_when_caller_did_not(
    fake_provider: _FakeProvider,
) -> None:
    """The PII rule holds even if the caller forgot: raw phone numbers never reach the model."""
    update = await extract({"transcript": RAW})

    sent = str(fake_provider.calls[0]["prompt"])
    assert "+962 79 123 4567" not in sent
    assert "[PHONE]" in sent
    # The redacted text is written back to state so `ground` checks the right string.
    assert update["redacted_transcript"] == sent


async def test_extract_requests_verbatim_evidence_and_best_guess_offsets(
    fake_provider: _FakeProvider,
) -> None:
    """The system prompt must ask for copied text and admit offsets will be wrong."""
    await extract({"transcript": RAW})

    system = str(fake_provider.calls[0]["system"])
    assert system == EXTRACTION_SYSTEM_PROMPT
    assert "CHARACTER-FOR-CHARACTER COPY" in system
    assert "start_char" in system and "end_char" in system
    assert "your guess will often be wrong" in system


async def test_extract_asks_for_the_extraction_proposal_schema(fake_provider: _FakeProvider) -> None:
    """Extraction targets the proposal shape, not CallAnalysis — no confidence decided yet."""
    await extract({"transcript": RAW})

    requested = fake_provider.calls[0]["response_model"]
    assert set(requested.model_fields) == set(ExtractionProposal.model_fields)
    assert "confidence" not in requested.model_fields


async def test_extract_repairs_offset_arithmetic_but_keeps_the_models_position_claim() -> None:
    """A model whose end_char arithmetic is wrong must not lose the whole call.

    Real Gemini output estimates spans rather than counting them, so end_char
    routinely disagrees with len(text). That is a fact about the quote's own
    fields, not a claim about the transcript, so it is repaired — while
    start_char, which *is* a claim, is carried through untouched for ground()
    to check.
    """

    class _BadArithmeticProvider:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def structured_complete(self, prompt: str, **kwargs: object) -> object:
            self.calls.append({"prompt": prompt, **kwargs})
            model = kwargs["response_model"]
            return model.model_validate(
                {
                    "summary": "Customer agreed.",
                    "commitments": [
                        {
                            "evidence": {
                                "text": "Fine.",
                                "speaker": "Customer",
                                "start_char": 41,
                                # Wrong: 99 - 41 != len("Fine.")
                                "end_char": 99,
                            },
                            "promised_by": "Agent",
                            "description": "Call back.",
                        }
                    ],
                }
            )

    provider = _BadArithmeticProvider()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(extract_module, "get_llm_provider", lambda: provider)
    try:
        update = await extract({"transcript": RAW})
    finally:
        monkeypatch.undo()

    quote = update["commitments"][0].evidence
    assert quote.start_char == 41
    assert quote.end_char == 41 + len("Fine.")


async def test_extract_records_provider_failure_as_error_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure path: a provider error becomes `error` in state so the graph can escalate.

    Raising here would abort the run; an error routes to human review, which is
    what the project rule requires of anything the system could not verify.
    """
    provider = _FakeProvider(error=RuntimeError("provider exhausted"))
    monkeypatch.setattr(extract_module, "get_llm_provider", lambda: provider)

    update = await extract({"transcript": RAW})

    assert "provider exhausted" in update["error"]
    assert "commitments" not in update


async def test_extract_reports_missing_transcript_as_an_error() -> None:
    """Failure path: nothing to extract from is an error, not an empty success."""
    update = await extract({"call_id": "call-001"})

    assert "no transcript" in update["error"]


async def test_extract_does_not_retrieve_memory_rules_in_phase_2(fake_provider: _FakeProvider) -> None:
    """Memory retrieval is phase 4; the prompt carries no injected rules today.

    `sawti.memory.store` raises NotImplementedError, so calling it would crash
    the graph. This test pins the deliberate absence so it is a decision, not drift.
    """
    await extract({"transcript": RAW})

    assert fake_provider.calls[0]["system"] == EXTRACTION_SYSTEM_PROMPT
