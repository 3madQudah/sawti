"""PII redaction — runs before any transcript text reaches a model.

Phase 1: privacy layer, required upstream of every LLM call.
"""

from __future__ import annotations

from pydantic import BaseModel


class RedactionResult(BaseModel):
    """Output of a redaction pass over a piece of text."""

    redacted_text: str
    original_length: int
    redaction_count: int


def redact(text: str) -> RedactionResult:
    """Redact PII (names, phone numbers, national IDs, card numbers, etc.) from `text`.

    Must run before `text` is passed to any `LLMProvider` call, per project
    rule: "PII redaction runs before any text reaches a model."

    Args:
        text: Raw transcript or utterance text, possibly containing PII.

    Returns:
        A `RedactionResult` with the redacted text and redaction stats.

    Raises:
        NotImplementedError: Until redaction rules are implemented.
    """
    # TODO(phase-1): implement PII detection/redaction (regex + NER) for ar/en/mixed text.
    raise NotImplementedError
