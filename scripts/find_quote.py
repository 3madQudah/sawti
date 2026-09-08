"""Find the exact character offsets of a verbatim quote in a call transcript.

Hand-counting `start_char`/`end_char` for every `Quote`/evidence required by
`sawti.schemas` is impractical — this CLI computes them instead.

Usage:
    uv run python scripts/find_quote.py data/synthetic/call_0000_ar.txt "بدي أسكر الحساب"

Offsets are computed against the transcript file's full text, exactly as
`Quote.start_char`/`end_char` are defined: 0-indexed into the full text.

If the substring appears more than once, every match is printed with its
line number so a human can pick the right one — this never guesses which
one was meant. If it appears zero times, that is reported clearly: evidence
must be verbatim, so a zero-match result is likely a typo or paraphrase.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

_SPEAKER_RE = re.compile(r"^(Agent|Customer):\s*")


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


def _print_match(substring: str, match: QuoteMatch) -> None:
    """Print one match's details plus a paste-ready JSON evidence snippet."""
    print(f"  line {match.line_number}: {match.line_text}")
    print(f"  speaker: {match.speaker!r}")
    print(f"  start_char: {match.start_char}")
    print(f"  end_char: {match.end_char}")
    paste_ready = {
        "text": substring,
        "speaker": match.speaker,
        "start_char": match.start_char,
        "end_char": match.end_char,
    }
    print(f"  paste-ready: {json.dumps(paste_ready, ensure_ascii=False)}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path, help="Path to a transcript .txt file")
    parser.add_argument("substring", help="Exact, verbatim substring to locate")
    args = parser.parse_args()

    text = args.transcript.read_text(encoding="utf-8")
    matches = find_quote_matches(text, args.substring)

    if not matches:
        print(f"NOT FOUND: {args.substring!r} does not appear verbatim in {args.transcript}.")
        print("Evidence must be an exact verbatim quote — check for a typo or paraphrase.")
        raise SystemExit(1)

    if len(matches) == 1:
        print(f"1 match found in {args.transcript}:")
        _print_match(args.substring, matches[0])
        return

    print(f"{len(matches)} matches found in {args.transcript} — pick one, do not guess:")
    for index, match in enumerate(matches, start=1):
        print(f"\n--- match {index} ---")
        _print_match(args.substring, match)


if __name__ == "__main__":
    main()
