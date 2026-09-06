"""Tests for `sawti.privacy.redaction`.

Phase 1: mirrors `src/sawti/privacy/redaction.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 1")
def test_redact_removes_phone_numbers() -> None:
    """redact() replaces phone numbers in the text with a redaction marker."""


@pytest.mark.skip(reason="phase 1")
def test_redact_removes_names_in_arabic_and_english() -> None:
    """redact() detects and removes personal names in both Arabic and English text."""


@pytest.mark.skip(reason="phase 1")
def test_redact_removes_pii_in_code_switched_text() -> None:
    """redact() correctly redacts PII in mixed Arabic/English (code-switched) text."""


@pytest.mark.skip(reason="phase 1")
def test_redact_result_reports_accurate_redaction_count() -> None:
    """RedactionResult.redaction_count matches the number of PII spans removed."""


@pytest.mark.skip(reason="phase 1")
def test_redact_preserves_non_pii_text() -> None:
    """redact() leaves text with no PII unchanged."""
