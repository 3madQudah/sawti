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

The matching itself lives in `sawti.quotes` so that the programmatic callers
(reference-label generation, grounding-precision scoring) locate quotes with
exactly the same logic this CLI shows a human. `QuoteMatch` and
`find_quote_matches` are re-exported here for convenience.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sawti.quotes import QuoteMatch, find_quote_matches

__all__ = ["QuoteMatch", "find_quote_matches", "main"]


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
