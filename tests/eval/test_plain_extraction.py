"""Tests for `sawti.eval.plain_extraction`.

Phase 1: mirrors `src/sawti/eval/plain_extraction.py`. These pin the
properties that make this path a usable baseline: a minimal prompt, no
content retries, and no quiet correction of what the model returned.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sawti.data.ground_truth import RUBRIC_CRITERIA
from sawti.eval.plain_extraction import run_plain_extraction
from sawti.schemas import Language

TRANSCRIPT = (
    "Agent: Hello, you have reached support, my name is Rami.\n"
    "Customer: I was charged twice this month and I want a refund.\n"
    "Agent: I will process the refund today and send you a confirmation.\n"
    "Customer: Thank you, that helps.\n"
)

GROUNDED_QUOTE = "I will process the refund today"
INVENTED_QUOTE = "I'll sort that out for you immediately"


def _response(
    *,
    quote_text: str = GROUNDED_QUOTE,
    sentiment: list[tuple[float, float]] | None = None,
) -> dict:
    points = sentiment if sentiment is not None else [(-0.6, 0.0), (0.5, 16.0)]
    return {
        "summary": "Double charge; refund promised.",
        "commitments": [
            {"quote": quote_text, "promised_by": "Agent", "description": "process the refund today"}
        ],
        "compliance_flags": [],
        "rubric_scores": [
            {"quote": quote_text, "criterion": criterion, "score": 0.7, "justification": "ok"}
            for criterion in RUBRIC_CRITERIA
        ],
        "sentiment_points": [
            {"quote": quote_text, "speaker": "Customer", "score": score, "timestamp_sec": ts}
            for score, ts in points
        ],
        "confidence": 0.8,
    }


class _StubProvider:
    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    async def complete(self, prompt, *, system=None, **kwargs):  # pragma: no cover - unused
        raise AssertionError("the baseline must use structured_complete")

    async def structured_complete(self, prompt, *, response_model, system=None, **kwargs):
        self.prompts.append(prompt)
        self.systems.append(system)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response_model.model_validate(response)


@pytest.fixture
def transcript(tmp_path: Path) -> Path:
    path = tmp_path / "call_0000_ar.txt"
    path.write_text(TRANSCRIPT, encoding="utf-8")
    return path


def _install(monkeypatch, provider) -> None:
    monkeypatch.setattr("sawti.eval.plain_extraction.get_llm_provider", lambda: provider)
    monkeypatch.setattr("sawti.eval.plain_extraction.time.sleep", lambda _seconds: None)


def test_run_plain_extraction_returns_a_validated_call_analysis(transcript, monkeypatch):
    """The baseline returns a CallAnalysis with id and language from the filename."""
    _install(monkeypatch, _StubProvider([_response()]))

    analysis = run_plain_extraction(transcript)

    assert analysis.call_id == "call_0000_ar"
    assert analysis.language is Language.AR
    assert len(analysis.rubric_scores) == len(RUBRIC_CRITERIA)


def test_plain_prompt_carries_no_coaching(transcript, monkeypatch):
    """The baseline prompt names the fields and criteria and nothing else.

    If this test starts failing because the prompt grew guidance, that is the
    safeguard working — the baseline is supposed to stay uncoached.
    """
    provider = _StubProvider([_response()])
    _install(monkeypatch, provider)

    run_plain_extraction(transcript)
    prompt = provider.prompts[0]
    instruction = prompt.replace(TRANSCRIPT, "")

    for criterion in RUBRIC_CRITERIA:
        assert criterion in instruction
    for coaching in ("verbatim", "character-for-character", "EXACT", "REJECTED", "Jordan"):
        assert coaching not in instruction
    assert len(instruction) < 500


def test_run_plain_extraction_does_not_retry_on_ungrounded_quotes(transcript, monkeypatch):
    """An invented quote is returned as-is — exactly one model call, no re-prompt."""
    provider = _StubProvider([_response(quote_text=INVENTED_QUOTE)])
    _install(monkeypatch, provider)

    analysis = run_plain_extraction(transcript)

    assert len(provider.prompts) == 1
    assert analysis.commitments[0].evidence.text == INVENTED_QUOTE


def test_ungrounded_quotes_are_preserved_for_the_metric_to_catch(transcript, monkeypatch):
    """Placeholder offsets never disguise an invented quote as a grounded one."""
    _install(monkeypatch, _StubProvider([_response(quote_text=INVENTED_QUOTE)]))

    analysis = run_plain_extraction(transcript)
    evidence = analysis.commitments[0].evidence

    # Text is untouched, and the transcript does not contain it — which is what
    # grounding_precision_by_category checks. The offsets are not a claim.
    assert evidence.text not in TRANSCRIPT
    assert TRANSCRIPT[evidence.start_char : evidence.end_char] != evidence.text


def test_grounded_quotes_get_real_offsets(transcript, monkeypatch):
    """When the quote is real, the recorded offsets slice back to it exactly."""
    _install(monkeypatch, _StubProvider([_response(quote_text=GROUNDED_QUOTE)]))

    evidence = run_plain_extraction(transcript).commitments[0].evidence

    assert TRANSCRIPT[evidence.start_char : evidence.end_char] == GROUNDED_QUOTE
    assert evidence.speaker == "Agent"


def test_run_plain_extraction_retries_provider_errors_only(transcript, monkeypatch):
    """A transport error is retried; the prompt sent on the retry is unchanged."""
    provider = _StubProvider([RuntimeError("429 rate limited"), _response()])
    _install(monkeypatch, provider)

    analysis = run_plain_extraction(transcript)

    assert len(provider.prompts) == 2
    assert provider.prompts[0] == provider.prompts[1]
    assert analysis.summary


def test_run_plain_extraction_raises_when_output_violates_the_schema(transcript, monkeypatch):
    """Out-of-order sentiment points fail as a real baseline failure, not silently sorted."""
    _install(monkeypatch, _StubProvider([_response(sentiment=[(0.5, 16.0), (-0.6, 0.0)])]))

    with pytest.raises(ValidationError):
        run_plain_extraction(transcript)
