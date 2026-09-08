"""Tests for `scripts/find_quote.py`.

Phase 1 item 3: quote-offset helper for ground-truth authoring.
"""

from __future__ import annotations

import pytest
from find_quote import find_quote_matches


def test_find_quote_matches_unique_match() -> None:
    """A substring appearing exactly once returns exactly one QuoteMatch."""
    text = "Agent: Hello there.\nCustomer: I have a billing issue.\n"

    matches = find_quote_matches(text, "billing issue")

    assert len(matches) == 1
    match = matches[0]
    assert match.line_number == 2
    assert match.speaker == "Customer"
    assert text[match.start_char : match.end_char] == "billing issue"


def test_find_quote_matches_multiple_matches() -> None:
    """A substring appearing more than once returns one QuoteMatch per occurrence, in order."""
    text = "Agent: I understand.\nCustomer: You never understand anything.\n"

    matches = find_quote_matches(text, "understand")

    assert len(matches) == 2
    assert matches[0].line_number == 1
    assert matches[0].speaker == "Agent"
    assert matches[1].line_number == 2
    assert matches[1].speaker == "Customer"
    for match in matches:
        assert text[match.start_char : match.end_char] == "understand"


def test_find_quote_matches_zero_matches() -> None:
    """A substring that never appears returns an empty list."""
    text = "Agent: Hello there.\nCustomer: I have a billing issue.\n"

    assert find_quote_matches(text, "refund approved") == []


def test_find_quote_matches_arabic_text() -> None:
    """find_quote_matches() works correctly on Arabic-script transcript text."""
    text = "Agent: أهلاً بك، كيف أقدر أساعدك؟\nCustomer: بدي أسكر الحساب فوراً.\n"

    matches = find_quote_matches(text, "بدي أسكر الحساب")

    assert len(matches) == 1
    match = matches[0]
    assert match.line_number == 2
    assert match.speaker == "Customer"
    assert text[match.start_char : match.end_char] == "بدي أسكر الحساب"


def test_find_quote_matches_rejects_empty_substring() -> None:
    """find_quote_matches() rejects an empty search substring rather than matching everything."""
    with pytest.raises(ValueError, match="non-empty"):
        find_quote_matches("Agent: hi.", "")
