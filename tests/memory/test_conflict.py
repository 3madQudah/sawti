"""Tests for `sawti.memory.conflict`.

Phase 4: mirrors `src/sawti/memory/conflict.py`.

Uses the same deterministic fake embedder pattern as
`tests/memory/test_store.py` for similarity narrowing, plus a scripted fake
LLMProvider for the contradiction judgment — no real model calls.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest

from sawti.memory.conflict import detect_conflict, insert_rule
from sawti.memory.rule_schema import MemoryRule
from sawti.memory.store import MemoryStore

_VOCAB = ["deadline", "flag", "refund", "unrelated"]


def _fake_embed(text: str) -> np.ndarray:
    words = set(text.lower().split())
    return np.array([1.0 if word in words else 0.0 for word in _VOCAB])


def _rule(rule_text: str) -> MemoryRule:
    return MemoryRule(rule_text=rule_text, created_at=datetime.now(UTC))


class _ScriptedConflictProvider:
    """Returns a queued contradicts=<bool> verdict per structured_complete call, in order."""

    def __init__(self, verdicts: list[bool]) -> None:
        self._verdicts = list(verdicts)
        self.prompts: list[str] = []

    async def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        raise AssertionError("conflict detection must use structured_complete, not complete")

    async def structured_complete(
        self, prompt: str, *, response_model: Any, system: str | None = None, **kwargs: Any
    ) -> Any:
        self.prompts.append(prompt)
        return response_model(contradicts=self._verdicts.pop(0), reasoning="scripted")


def _patch_provider(monkeypatch: pytest.MonkeyPatch, provider: _ScriptedConflictProvider) -> None:
    monkeypatch.setattr("sawti.memory.conflict.get_llm_provider", lambda: provider)


def test_detect_conflict_finds_contradictory_existing_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """detect_conflict() returns existing rules that contradict the candidate rule."""
    candidate = _rule("deadline flag required")
    contradictory = _rule("deadline flag never required")
    _patch_provider(monkeypatch, _ScriptedConflictProvider([True]))

    result = detect_conflict(candidate, [contradictory], embed=_fake_embed)

    assert result == [contradictory]


def test_detect_conflict_returns_empty_when_no_conflict_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    """detect_conflict() returns an empty list when the candidate rule is compatible with existing rules."""
    candidate = _rule("deadline flag required")
    compatible = _rule("deadline flag required")
    _patch_provider(monkeypatch, _ScriptedConflictProvider([False]))

    result = detect_conflict(candidate, [compatible], embed=_fake_embed)

    assert result == []


def test_detect_conflict_short_circuits_with_no_existing_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    """detect_conflict() makes no LLM call when there is nothing to compare against."""
    provider = _ScriptedConflictProvider([])
    _patch_provider(monkeypatch, provider)

    result = detect_conflict(_rule("deadline flag required"), [], embed=_fake_embed)

    assert result == []
    assert provider.prompts == []


def test_detect_conflict_runs_before_store_insert(monkeypatch: pytest.MonkeyPatch) -> None:
    """Conflict detection is invoked before a candidate rule is inserted into MemoryStore."""
    store = MemoryStore(embed=_fake_embed)
    existing = _rule("deadline flag required")
    store.add(existing)
    candidate = _rule("deadline flag never required")
    _patch_provider(monkeypatch, _ScriptedConflictProvider([True]))

    outcome = insert_rule(store, candidate, embed=_fake_embed)

    assert outcome.inserted is False
    assert outcome.conflicts == [existing]
    # The candidate must never have reached the store.
    assert candidate not in store.all_active_rules()
    assert existing in store.all_active_rules()


def test_insert_rule_inserts_when_no_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    """insert_rule() adds the candidate to the store when detect_conflict() finds nothing."""
    store = MemoryStore(embed=_fake_embed)
    existing = _rule("refund unrelated")
    store.add(existing)
    candidate = _rule("deadline flag required")
    _patch_provider(monkeypatch, _ScriptedConflictProvider([False]))

    outcome = insert_rule(store, candidate, embed=_fake_embed)

    assert outcome.inserted is True
    assert outcome.conflicts == []
    assert candidate in store.all_active_rules()
