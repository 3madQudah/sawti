"""Vector store for memory rules, with top-5 retrieval at inference time.

Phase 4: agent memory.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any
from uuid import UUID

import numpy as np

from sawti.memory.rule_schema import MemoryRule

# No embedding provider is configured anywhere in sawti.config (checked) —
# this is the simplest option that needs no new infra: a local
# sentence-transformers model, embedded and compared in-process, rather than
# pgvector (a new Postgres extension plus a migration) or an external
# embedding API (a new provider, network dependency, and per-call cost, for
# every add()/retrieve()). See docs/09-DECISIONS.md for the rejected
# alternatives in full.
#
# Multilingual, not English-only: this project is Arabic/English/
# code-switched throughout, and `sawti.memory.induction` always writes
# `rule_text` in English, while a `retrieve()` query (a transcript, or
# context derived from one) may be Arabic or mixed. An English-only model
# would degrade exactly the cross-lingual matching this store exists to do.
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

#: Turns text into a vector. `MemoryStore`'s default (`_default_embed`) loads
#: a real model; tests inject a fast, deterministic fake instead.
Embedder = Callable[[str], "np.ndarray[Any, np.dtype[np.float64]]"]


@lru_cache
def _load_model(model_name: str) -> Any:
    """Lazily load and memoize a sentence-transformers model.

    Imported and instantiated only on first real use, so importing this
    module — or constructing a `MemoryStore` with an injected `embed` — never
    triggers a model download.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _default_embed(
    text: str, *, model_name: str = DEFAULT_EMBEDDING_MODEL
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Embed `text` with a local, multilingual sentence-transformers model."""
    model = _load_model(model_name)
    embedding = model.encode(text, normalize_embeddings=True)
    return np.asarray(embedding, dtype=np.float64)


def _cosine_similarity(
    left: np.ndarray[Any, np.dtype[np.float64]], right: np.ndarray[Any, np.dtype[np.float64]]
) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


class MemoryStore:
    """Vector-backed store of induced `MemoryRule` records.

    Embedding and similarity search are in-process (see module docstring),
    so a `MemoryStore` instance's index does not survive the process.
    Callers needing durability persist `MemoryRule` records elsewhere (the
    rules themselves, not the index — `sawti.memory.capture`'s DB) and
    rebuild the index with `add()` on startup.
    """

    def __init__(self, *, embed: Embedder | None = None) -> None:
        """
        Args:
            embed: How to turn text into a vector. Defaults to a local
                multilingual sentence-transformers model
                (`DEFAULT_EMBEDDING_MODEL`), loaded lazily on first use.
                Tests inject a fast, deterministic fake to avoid downloading
                a real model.
        """
        self._embed: Embedder = embed or _default_embed
        self._rules: dict[UUID, MemoryRule] = {}
        self._embeddings: dict[UUID, np.ndarray[Any, np.dtype[np.float64]]] = {}

    def add(self, rule: MemoryRule) -> None:
        """Embed `rule.rule_text` and store it, indexed by `rule.id`.

        Args:
            rule: The rule to store. Calling `add()` again with a rule
                sharing an existing `id` replaces it, embedding included.
        """
        self._rules[rule.id] = rule
        self._embeddings[rule.id] = self._embed(rule.rule_text)

    def retrieve(self, query: str, *, top_k: int = 5) -> list[MemoryRule]:
        """Retrieve the `top_k` most relevant, non-retired rules for `query`.

        Args:
            query: Context to match rules against (e.g. the current transcript).
            top_k: Maximum number of rules to return. Defaults to 5.

        Returns:
            The most relevant rules, best match first. Fewer than `top_k`
            when fewer than `top_k` non-retired rules are stored.
        """
        candidates = [rule for rule in self._rules.values() if not rule.retired]
        if not candidates:
            return []

        query_embedding = self._embed(query)
        scored = [
            (_cosine_similarity(query_embedding, self._embeddings[rule.id]), rule) for rule in candidates
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [rule for _, rule in scored[:top_k]]

    def all_active_rules(self) -> list[MemoryRule]:
        """Every non-retired rule currently stored, in no particular order.

        Added for `sawti.memory.conflict.insert_rule()`, which needs the
        full active set to check a candidate against — not just the top-k
        for one query the way `retrieve()` narrows things down.
        """
        return [rule for rule in self._rules.values() if not rule.retired]

    def replace_all(self, rules: list[MemoryRule]) -> None:
        """Replace the store's entire contents with `rules`, re-embedding each.

        Added for `sawti.eval.experiments.five_batch`'s between-batch
        consolidation step: `sawti.memory.consolidate.consolidate()` returns
        a new rule list (merged clusters plus untouched singles) meant to
        *become* the store's contents, not sit alongside the pre-merge ones.
        """
        self._rules.clear()
        self._embeddings.clear()
        for rule in rules:
            self.add(rule)
