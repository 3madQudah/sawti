"""The one path from a raw transcript to the text the service stores.

Phase 6.4. `POST /calls` and the ASR worker (`sawti.api.asr_tasks`) both call
`redact_transcript`, so a transcript that arrives as audio is redacted and
validated exactly like one that arrives as text. The raw text goes in and
only the redacted text comes out. Callers keep the raw text in a local
variable and nowhere else: not in the database, graph state, a trace or a log
(docs/09-DECISIONS.md, 2026-10-06, decision 1; 2026-10-10).
"""

from __future__ import annotations

from sawti.api.views import split_turns
from sawti.privacy.redaction import RedactionResult, redact


def redact_transcript(raw: str) -> RedactionResult:
    """Redact `raw` and check that every line of the result is a speaker turn.

    Returns:
        The redaction result: the redacted text and how many spans were replaced.

    Raises:
        UnattributableLineError: A line has no `Agent:` / `Customer:` prefix.
            The review screen needs speaker turns, and the parser does not
            guess. Its message quotes the offending line, already redacted.
    """
    redaction = redact(raw)
    split_turns(redaction.redacted_text, "transcript")
    return redaction
