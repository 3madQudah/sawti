"""Tests for `sawti.privacy.redaction`.

Phase 1 module, tested in phase 2 when `sawti.agent.nodes.extract` made it
load-bearing. Scope is the MVP regex redactor — structured identifiers only.
"""

from __future__ import annotations

import pytest

from sawti.privacy.redaction import redact


def test_redact_removes_phone_numbers() -> None:
    """redact() replaces phone numbers in the text with a redaction marker."""
    result = redact("Agent: Call me on +962 79 123 4567 tomorrow.")

    assert "+962 79 123 4567" not in result.redacted_text
    assert "[PHONE]" in result.redacted_text
    assert result.redaction_count == 1


def test_redact_removes_emails_and_national_ids() -> None:
    """Structured identifiers with distinctive shape are caught."""
    result = redact("My ID is 9801234567 and my email is sara.q@example.com")

    assert "[NATIONAL_ID]" in result.redacted_text
    assert "[EMAIL]" in result.redacted_text
    assert "9801234567" not in result.redacted_text
    assert "sara.q@example.com" not in result.redacted_text


def test_redact_removes_payment_card_numbers() -> None:
    """A grouped 16-digit card is redacted as a card, not split across patterns."""
    result = redact("Customer: the card is 4111 1111 1111 1111.")

    assert "[CARD]" in result.redacted_text
    assert "4111" not in result.redacted_text


def test_redact_handles_arabic_indic_digits() -> None:
    """Arabic-Indic digits are redacted too — the ar transcripts use them."""
    result = redact("Agent: رقمي ٠٧٩١٢٣٤٥٦٧ تفضل")

    assert "٠٧٩١٢٣٤٥٦٧" not in result.redacted_text
    assert result.redaction_count == 1


def test_redact_removes_pii_in_code_switched_text() -> None:
    """redact() correctly redacts PII in mixed Arabic/English (code-switched) text."""
    result = redact("Customer: تمام، my email is ali@example.com و رقمي +962791234567")

    assert "ali@example.com" not in result.redacted_text
    assert "+962791234567" not in result.redacted_text
    # Non-PII Arabic text is untouched.
    assert "تمام" in result.redacted_text


def test_redact_result_reports_accurate_redaction_count() -> None:
    """RedactionResult.redaction_count matches the number of PII spans removed."""
    result = redact("a@b.com and c@d.com and +962791234567")

    assert result.redaction_count == 3


def test_redact_preserves_non_pii_text() -> None:
    """redact() leaves text with no PII unchanged."""
    text = "Agent: Good morning, how can I help you today?\nCustomer: My bill is wrong.\n"
    result = redact(text)

    assert result.redacted_text == text
    assert result.redaction_count == 0
    assert result.original_length == len(text)


@pytest.mark.skip(reason="needs NER — MVP redactor is regex-only, see docs/09-DECISIONS.md")
def test_redact_removes_names_in_arabic_and_english() -> None:
    """redact() detects and removes personal names in both Arabic and English text.

    Deliberately still skipped. Personal names have no regex shape; catching
    them needs NER, which is logged as future work rather than silently
    pretended-to. Leaving this test red-but-skipped keeps the gap visible.
    """
