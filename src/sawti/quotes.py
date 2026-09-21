"""Locating verbatim quotes in a transcript, and the exact offsets they imply.

Shared by everything that has to turn a piece of quote *text* into a real
`sawti.schemas.Quote`:

  * `scripts/find_quote.py` — the human-facing CLI.
  * `sawti.data.ground_truth.generate_reference_labels` — LLM-proposed quote
    text, located programmatically because a model must never be trusted to
    compute character offsets.
  * `sawti.eval.metrics.grounding_precision_by_category` — checking whether a
    prediction's evidence actually appears in the transcript at all.

The rule this module exists to enforce: offsets are *computed from the
transcript*, never asserted by a model or counted by hand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SPEAKER_RE = re.compile(r"^(Agent|Customer):\s*")

# Used when a quote is located on a line with no "Agent:"/"Customer:" prefix.
# `Quote.speaker` has min_length=1, so it needs *some* value; this one is
# visibly a non-answer rather than a wrong attribution.
UNKNOWN_SPEAKER = "Unknown"


@dataclass(frozen=True)
class QuoteMatch:
    """One occurrence of a searched-for substring within a transcript.

    Attributes:
        line_number: 1-indexed line number of the line containing the match.
        speaker: Speaker parsed from the containing line's "Agent:"/"Customer:"
            prefix, or None if the line doesn't start with one.
        start_char: Inclusive start offset into the full transcript text.
        end_char: Exclusive end offset into the full transcript text.
        line_text: Full text of the line containing the match, for context.
    """

    line_number: int
    speaker: str | None
    start_char: int
    end_char: int
    line_text: str


def _parse_speaker(line_text: str) -> str | None:
    """Parse the speaker label from a line's "Agent:"/"Customer:" prefix, if present."""
    match = _SPEAKER_RE.match(line_text)
    return match.group(1) if match else None


def find_quote_matches(text: str, substring: str) -> list[QuoteMatch]:
    """Find every non-overlapping occurrence of `substring` in `text`.

    Args:
        text: Full transcript text to search (e.g. a synthetic call file's contents).
        substring: Exact, verbatim text to find. Must be non-empty.

    Returns:
        One `QuoteMatch` per occurrence, in order of appearance. Empty if
        `substring` does not appear at all.

    Raises:
        ValueError: If `substring` is empty.
    """
    if not substring:
        raise ValueError("substring must be non-empty")

    matches: list[QuoteMatch] = []
    start = 0
    while True:
        start_char = text.find(substring, start)
        if start_char == -1:
            break
        end_char = start_char + len(substring)
        line_start = text.rfind("\n", 0, start_char) + 1
        newline_idx = text.find("\n", start_char)
        line_end = newline_idx if newline_idx != -1 else len(text)
        line_text = text[line_start:line_end]
        line_number = text.count("\n", 0, start_char) + 1
        matches.append(
            QuoteMatch(
                line_number=line_number,
                speaker=_parse_speaker(line_text),
                start_char=start_char,
                end_char=end_char,
                line_text=line_text,
            )
        )
        start = end_char  # non-overlapping: resume search after this match
    return matches


def is_verbatim(text: str, substring: str) -> bool:
    """Return whether `substring` appears verbatim in `text` at least once.

    An empty `substring` is never verbatim evidence, so it returns False
    rather than raising — callers here are scoring untrusted model output,
    not validating human input.
    """
    return bool(substring) and substring in text
