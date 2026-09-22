"""Tests for `sawti.eval.experiments.grounded_graph`.

Phase 2: mirrors `src/sawti/eval/experiments/grounded_graph.py`.

The provider is stubbed throughout — this checks that the experiment drives the
graph and assembles its state correctly, not that the model is any good.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sawti.agent.nodes import extract as extract_module
from sawti.eval.experiments.grounded_graph import GateStats, extract_via_graph
from sawti.schemas import Commitment, ExtractionProposal, Language, Quote

TRANSCRIPT = (
    "Agent: Good morning, this is Sara from Orange.\n"
    "Customer: My bill is wrong again.\n"
    "Agent: I will refund the difference by Sunday.\n"
)
QUOTE_TEXT = "I will refund the difference by Sunday."


class _StubProvider:
    """Returns one commitment, with honest or dishonest offsets."""

    def __init__(self, *, grounded: bool) -> None:
        """Record whether this stub's offsets should actually match the transcript."""
        self.grounded = grounded

    async def structured_complete(self, prompt: str, **kwargs: object) -> ExtractionProposal:
        """Return a single-commitment proposal, grounded or paraphrased."""
        text = QUOTE_TEXT if self.grounded else "I will refund you first thing tomorrow."
        start = prompt.index(text) if self.grounded else 0
        quote = Quote(text=text, speaker="Agent", start_char=start, end_char=start + len(text))
        return ExtractionProposal(
            summary="Refund promised.",
            commitments=[Commitment(evidence=quote, promised_by="Agent", description="Refund.")],
        )


@pytest.fixture
def transcript(tmp_path: Path) -> Path:
    """A synthetic transcript named so call_id and language can be inferred."""
    path = tmp_path / "call_0000_ar.txt"
    path.write_text(TRANSCRIPT, encoding="utf-8")
    return path


def _use(monkeypatch: pytest.MonkeyPatch, *, grounded: bool) -> None:
    """Point the extraction node at a stub provider."""
    monkeypatch.setattr(extract_module, "get_llm_provider", lambda: _StubProvider(grounded=grounded))


def test_extract_via_graph_returns_a_validated_analysis(
    transcript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A grounded run yields a CallAnalysis with call_id and language from the filename."""
    _use(monkeypatch, grounded=True)

    analysis = extract_via_graph(transcript)

    assert analysis.call_id == "call_0000_ar"
    assert analysis.language is Language.AR
    assert len(analysis.commitments) == 1
    assert analysis.confidence == 1.0
    assert analysis.requires_human_review is False


def test_extract_via_graph_emits_no_unsupported_claims(
    transcript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of phase 2: a paraphrased quote never reaches the emitted analysis."""
    _use(monkeypatch, grounded=False)

    analysis = extract_via_graph(transcript)

    assert analysis.commitments == []
    assert analysis.confidence == 0.0
    assert analysis.requires_human_review is True


def test_gate_stats_separate_model_behaviour_from_pipeline_filtering(
    transcript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GateStats records what was proposed, so a 0.0 emitted rate is not mistaken for a better model."""
    _use(monkeypatch, grounded=False)
    stats = GateStats()

    extract_via_graph(transcript, stats=stats)

    assert stats.calls == 1
    assert stats.proposed == 1
    assert stats.rejected == 1
    assert stats.escalated == 1
    assert stats.rejection_rate(Language.AR) == 1.0


def test_gate_stats_report_zero_rejection_for_a_clean_run(
    transcript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fully grounded run proposes claims and rejects none."""
    _use(monkeypatch, grounded=True)
    stats = GateStats()

    extract_via_graph(transcript, stats=stats)

    assert stats.proposed == 1
    assert stats.rejected == 0
    assert stats.escalated == 0
    assert stats.rejection_rate(Language.AR) == 0.0


def test_gate_stats_rejection_rate_is_zero_when_nothing_was_proposed() -> None:
    """Failure path: no proposals must not divide by zero."""
    assert GateStats().rejection_rate(Language.EN) == 0.0


def test_extract_via_graph_raises_when_the_graph_errored(
    transcript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure path: a provider failure surfaces as an extraction failure, not a padded result.

    run_eval excludes it from the metrics and reports it by count, so a shrinking
    denominator can never pass for a rising score.
    """

    class _Broken:
        async def structured_complete(self, prompt: str, **kwargs: object) -> ExtractionProposal:
            raise RuntimeError("provider exhausted")

    monkeypatch.setattr(extract_module, "get_llm_provider", lambda: _Broken())

    with pytest.raises(RuntimeError, match="graph run failed"):
        extract_via_graph(transcript)
