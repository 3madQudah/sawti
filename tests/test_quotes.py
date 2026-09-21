"""Tests for `sawti.quotes`.

Phase 1: mirrors `src/sawti/quotes.py`. The quote-locating logic shared by
`scripts/find_quote.py`, reference-label generation, and grounding-precision
scoring — all three must agree on what "verbatim" means.
"""

from __future__ import annotations

import pytest

from sawti.quotes import find_quote_matches, is_verbatim

TRANSCRIPT = "Agent: Hello there.\nCustomer: I have a billing issue.\n"
ARABIC_TRANSCRIPT = "Agent: أهلاً بك، كيف أقدر أساعدك؟\nCustomer: بدي أسكر الحساب فوراً.\n"


def test_find_quote_matches_unique_match_reports_real_offsets():
    """A unique substring yields one match whose offsets slice back to the same text."""
    matches = find_quote_matches(TRANSCRIPT, "billing issue")

    assert len(matches) == 1
    match = matches[0]
    assert match.line_number == 2
    assert match.speaker == "Customer"
    assert TRANSCRIPT[match.start_char : match.end_char] == "billing issue"


def test_find_quote_matches_returns_every_occurrence_in_order():
    """Repeated substrings yield one match each, in order of appearance."""
    text = "Agent: I understand.\nCustomer: You never understand anything.\n"

    matches = find_quote_matches(text, "understand")

    assert [m.line_number for m in matches] == [1, 2]
    assert [m.speaker for m in matches] == ["Agent", "Customer"]


def test_find_quote_matches_arabic_script():
    """Offsets are correct for Arabic-script text, not just ASCII."""
    matches = find_quote_matches(ARABIC_TRANSCRIPT, "بدي أسكر الحساب")

    assert len(matches) == 1
    match = matches[0]
    assert match.speaker == "Customer"
    assert ARABIC_TRANSCRIPT[match.start_char : match.end_char] == "بدي أسكر الحساب"


def test_find_quote_matches_absent_substring_is_empty():
    """A substring that never appears returns no matches rather than guessing."""
    assert find_quote_matches(TRANSCRIPT, "refund approved") == []


def test_find_quote_matches_speaker_is_none_without_a_prefix():
    """A line with no Agent:/Customer: prefix reports speaker None, not a guess."""
    matches = find_quote_matches("no speaker prefix here\n", "prefix")

    assert len(matches) == 1
    assert matches[0].speaker is None


def test_find_quote_matches_rejects_empty_substring():
    """An empty search substring is rejected rather than matching everything."""
    with pytest.raises(ValueError, match="non-empty"):
        find_quote_matches(TRANSCRIPT, "")


def test_is_verbatim_distinguishes_present_from_absent():
    """is_verbatim() is True only for text that really appears in the transcript."""
    assert is_verbatim(TRANSCRIPT, "billing issue") is True
    assert is_verbatim(TRANSCRIPT, "Billing Issue") is False
    assert is_verbatim(TRANSCRIPT, "a billing problem") is False


def test_is_verbatim_empty_substring_is_not_evidence():
    """An empty quote is never verbatim evidence, and does not raise."""
    assert is_verbatim(TRANSCRIPT, "") is False
