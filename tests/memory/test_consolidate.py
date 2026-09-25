"""Tests for `sawti.memory.consolidate`.

Phase 4: mirrors `src/sawti/memory/consolidate.py`.

Uses the same deterministic fake embedder pattern as
`tests/memory/test_store.py` for clustering, plus a scripted fake
LLMProvider for the merge call — no real model calls.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pytest

from sawti.memory.consolidate import consolidate
from sawti.memory.rule_schema import MemoryRule

_VOCAB = ["deadline", "flag", "refund", "unrelated"]


def _fake_embed(text: str) -> np.ndarray:
    words = set(text.lower().split())
    return np.array([1.0 if word in words else 0.0 for word in _VOCAB])


def _rule(
    rule_text: str,
    *,
    source_correction_ids: list[UUID] | None = None,
    success_count: int = 0,
    failure_count: int = 0,
    retired: bool = False,
) -> MemoryRule:
    return MemoryRule(
        rule_text=rule_text,
        created_at=datetime.now(UTC),
        source_correction_ids=source_correction_ids or [],
        success_count=success_count,
        failure_count=failure_count,
        retired=retired,
    )


class _ScriptedMergeProvider:
    """Returns a fixed merged rule_text for every structured_complete call."""

    def __init__(self, merged_text: str = "merged rule text") -> None:
        self._merged_text = merged_text
        self.prompts: list[str] = []

    async def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        raise AssertionError("consolidate must use structured_complete, not complete")

    async def structured_complete(
        self, prompt: str, *, response_model: Any, system: str | None = None, **kwargs: Any
    ) -> Any:
        self.prompts.append(prompt)
        return response_model(rule_text=self._merged_text)


def _patch_provider(monkeypatch: pytest.MonkeyPatch, provider: _ScriptedMergeProvider) -> None:
    monkeypatch.setattr("sawti.memory.consolidate.get_llm_provider", lambda: provider)


def test_consolidate_merges_near_duplicate_rules_into_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """consolidate() merges rules with highly similar rule_text into a single rule."""
    first_id, second_id = uuid4(), uuid4()
    a = _rule("deadline flag required", source_correction_ids=[first_id], success_count=2, failure_count=1)
    b = _rule("deadline flag required", source_correction_ids=[second_id], success_count=1, failure_count=0)
    _patch_provider(monkeypatch, _ScriptedMergeProvider("merged: flag missing deadlines"))

    result = consolidate([a, b], embed=_fake_embed)

    assert len(result) == 1
    merged = result[0]
    assert merged.rule_text == "merged: flag missing deadlines"
    assert set(merged.source_correction_ids) == {first_id, second_id}
    assert merged.success_count == 3
    assert merged.failure_count == 1
    assert merged.retired is False


def test_consolidate_leaves_dissimilar_rules_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """consolidate() does not merge rules that are not near-duplicates."""
    a = _rule("deadline flag required")
    b = _rule("refund unrelated topic")
    provider = _ScriptedMergeProvider()
    _patch_provider(monkeypatch, provider)

    result = consolidate([a, b], embed=_fake_embed)

    assert {rule.rule_text for rule in result} == {a.rule_text, b.rule_text}
    assert provider.prompts == []  # no merge call was needed


def test_consolidate_leaves_retired_rules_untouched_and_unclustered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retired rule passes through unchanged, even if near-duplicate to an active rule."""
    active = _rule("deadline flag required")
    retired = _rule("deadline flag required", retired=True)
    provider = _ScriptedMergeProvider()
    _patch_provider(monkeypatch, provider)

    result = consolidate([active, retired], embed=_fake_embed)

    assert active in result
    assert retired in result
    assert provider.prompts == []  # nothing to merge: only one active rule
