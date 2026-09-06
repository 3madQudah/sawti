"""Vector store for memory rules, with top-5 retrieval at inference time.

Phase 4: agent memory.
"""

from __future__ import annotations

from sawti.memory.rule_schema import MemoryRule


class MemoryStore:
    """Vector-backed store of induced `MemoryRule` records."""

    def add(self, rule: MemoryRule) -> None:
        """Persist a new rule and index it for retrieval.

        Args:
            rule: The rule to store.

        Raises:
            NotImplementedError: Until storage is implemented.
        """
        # TODO(phase-4): embed rule.rule_text and upsert into the vector index.
        raise NotImplementedError

    def retrieve(self, query: str, *, top_k: int = 5) -> list[MemoryRule]:
        """Retrieve the `top_k` most relevant, non-retired rules for `query`.

        Args:
            query: Context to match rules against (e.g. the current transcript).
            top_k: Maximum number of rules to return. Defaults to 5.

        Returns:
            The most relevant rules, best match first.

        Raises:
            NotImplementedError: Until retrieval is implemented.
        """
        # TODO(phase-4): embed query and run top-k similarity search, excluding retired rules.
        raise NotImplementedError
