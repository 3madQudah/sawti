"""Tests for `sawti.memory.induction`.

Phase 4: mirrors `src/sawti/memory/induction.py`.

These only check structural properties (ids tracked, single vs.
multi-correction merge) with a scripted fake provider — no real LLM call,
and no assertion about whether the induced text is actually a *good* rule.
That is judged by inspection; see `scripts/inspect_induced_rules.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from sawti.memory.induction import induce_rule
from sawti.schemas import CallAnalysis, Correction, Language


def _analysis(call_id: str, *, language: Language = Language.EN) -> CallAnalysis:
    return CallAnalysis(
        call_id=call_id,
        language=language,
        summary="placeholder summary",
        confidence=0.9,
        requires_human_review=False,
    )


def _correction(call_id: str, *, error_location: str = "commitments[0].deadline") -> Correction:
    return Correction(
        call_id=call_id,
        original=_analysis(call_id),
        corrected=_analysis(call_id),
        error_location=error_location,
        reviewer_id="qa-1",
        timestamp=datetime.now(UTC),
    )


class _FakeProvider:
    """A scripted LLMProvider stub — returns a fixed rule_text, records prompts."""

    def __init__(self, rule_text: str = "Flag commitments without a stated deadline as incomplete.") -> None:
        self._rule_text = rule_text
        self.prompts: list[str] = []

    async def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        raise AssertionError("induce_rule must use structured_complete, not complete")

    async def structured_complete(
        self, prompt: str, *, response_model: Any, system: str | None = None, **kwargs: Any
    ) -> Any:
        self.prompts.append(prompt)
        return response_model(rule_text=self._rule_text)


def _patch_provider(monkeypatch: pytest.MonkeyPatch, provider: _FakeProvider) -> None:
    monkeypatch.setattr("sawti.memory.induction.get_llm_provider", lambda: provider)


def test_induce_rule_generalizes_a_single_correction(monkeypatch: pytest.MonkeyPatch) -> None:
    """induce_rule() produces a MemoryRule referencing its single source correction."""
    correction = _correction("call_0000_en")
    _patch_provider(monkeypatch, _FakeProvider())

    rule = induce_rule([correction])

    assert rule.rule_text
    assert rule.source_correction_ids == [correction.id]


def test_induce_rule_generalizes_across_multiple_similar_corrections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """induce_rule() merges multiple related corrections into one general rule."""
    corrections = [
        _correction("call_0000_en"),
        _correction("call_0001_ar"),
        _correction("call_0002_mixed"),
    ]
    provider = _FakeProvider()
    _patch_provider(monkeypatch, provider)

    rule = induce_rule(corrections)

    assert rule.rule_text
    assert len(rule.source_correction_ids) == 3
    # One induction call for the whole batch, not one per correction.
    assert len(provider.prompts) == 1


def test_induce_rule_tracks_all_source_correction_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """induce_rule() records every input correction's id in source_correction_ids."""
    corrections = [_correction(f"call_{i:04d}_en") for i in range(4)]
    _patch_provider(monkeypatch, _FakeProvider())

    rule = induce_rule(corrections)

    assert set(rule.source_correction_ids) == {correction.id for correction in corrections}


def test_induce_rule_raises_on_empty_corrections(monkeypatch: pytest.MonkeyPatch) -> None:
    """induce_rule() refuses to generalize from nothing."""
    _patch_provider(monkeypatch, _FakeProvider())

    with pytest.raises(ValueError, match="at least one"):
        induce_rule([])
