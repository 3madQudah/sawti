"""Tests for `sawti.privacy.redaction`.

Phase 1 module, tested in phase 2 when `sawti.agent.nodes.extract` made it
load-bearing. Scope is the MVP regex redactor — structured identifiers only.
"""

from __future__ import annotations

import re

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


# --- Phone formats (found in production, 2026-10-11: an uploaded en call kept
# "555-019-8234" in its redacted transcript and sent it to the model). Every
# case asserts the whole number is gone, not only that a placeholder appeared.

_PHONE_FORMATS = [
    # 3-3-4 and 3-4, the formats the old patterns missed.
    "555-019-8234",
    "555 019 8234",
    "555.019.8234",
    "(555) 019-8234",
    "019-8234",
    "512 3456",
    "+1 555 019 8234",
    "+1 (555) 019-8234",
    # Jordanian mobile and landline, national format.
    "079 123 4567",
    "079-123-4567",
    "07 9123 4567",
    "0791 234 567",
    "079 1234567",
    "06 512 3456",
    "065123456",
    "0-7-9-1-2-3-4-5-6-7",
    # Jordanian, international format.
    "+962 79 123 4567",
    "+962-79-123-4567",
    "+962791234567",
    "+962 7 9123 4567",
    "00962 79 123 4567",
    "00962791234567",
    # Arabic-Indic digits.
    "٠٧٩ ١٢٣ ٤٥٦٧",
    "٠٧٩-١٢٣-٤٥٦٧",
    "٠٧ ٩١٢٣ ٤٥٦٧",
    "+٩٦٢ ٧٩ ١٢٣ ٤٥٦٧",
    "٥٥٥-٠١٩-٨٢٣٤",
]


@pytest.mark.parametrize("number", _PHONE_FORMATS)
def test_redact_removes_every_common_phone_format(number: str) -> None:
    result = redact(f"Customer: my number is {number}, thanks.")

    assert result.redacted_text == "Customer: my number is [PHONE], thanks."
    assert result.redaction_count == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # ar
        ("Customer: رقمي ٠٧٩ ١٢٣ ٤٥٦٧ تفضل", "Customer: رقمي [PHONE] تفضل"),
        ("Customer: رقم الخط هو 079-123-4567", "Customer: رقم الخط هو [PHONE]"),
        # A conjunction written onto the number is a word character; `\b` would miss it.
        ("Customer: ورقمي و٠٧٩ ١٢٣ ٤٥٦٧", "Customer: ورقمي و[PHONE]"),
        # en: the line from the production call.
        (
            "Customer: with your account? Sure. The phone number is 555-019-8234. Thank you, David.",
            "Customer: with your account? Sure. The phone number is [PHONE]. Thank you, David.",
        ),
        # mixed
        ("Customer: تمام، my number is 07 9123 4567 و شكرا", "Customer: تمام، my number is [PHONE] و شكرا"),
        ("Customer: خذ الرقم +962 7 9123 4567 please", "Customer: خذ الرقم [PHONE] please"),
    ],
)
def test_redact_removes_phones_in_ar_en_and_mixed_turns(text: str, expected: str) -> None:
    assert redact(text).redacted_text == expected


def test_an_international_number_is_one_phone_not_a_card_with_a_stray_plus() -> None:
    """Before the fix the card pattern took the 12 digits after `+` and left `+[CARD]`."""
    assert redact("call +962791234567 now").redacted_text == "call [PHONE] now"


def test_a_bare_ten_digit_run_is_still_read_as_a_national_id() -> None:
    """Unchanged precedence: the more sensitive reading wins (docs/09-DECISIONS.md)."""
    assert redact("رقمي 0791234567").redacted_text == "رقمي [NATIONAL_ID]"


def test_a_dictated_run_is_not_half_redacted_from_an_inner_zero() -> None:
    """A dictated 10-digit ID must not lose only its tail to the 0-led phone pattern."""
    text = "Customer: رقم الهوية 1-0-7-1-2-3-4-5-6-7"
    result = redact(text).redacted_text

    assert result == text or not re.search(r"\d", result)


@pytest.mark.parametrize(
    "text",
    [
        "Agent: your balance of $145 is due on the 12th.",
        "Customer: I paid 1,500,000 JD last year.",
        "Agent: the bill dated 12/10/2026 at 10:30.",
        "Agent: the record shows 3 bills in 2026 and 12 in 2025.",
        "Customer: call 911 if it sparks.",
        "Agent: reference 88542, PIN 123456.",
    ],
)
def test_redact_leaves_amounts_dates_times_and_short_numbers(text: str) -> None:
    assert redact(text).redacted_text == text


# --- Catch-all: any other run of 7+ digits (decided 2026-10-11, after the
# specific patterns left account numbers, IDs and dictated numbers behind).


@pytest.mark.parametrize(
    "number",
    [
        "4589210",  # 7 contiguous: the shortest run caught
        "458921037",  # 9-digit account number
        "4589-2103",  # dddd-dddd account number
        "458-921037",
        "458-9-2103",
        "4 5 8 9 2 1 0 3 7",  # dictated, spaces
        "1-0-7-1-2-3-4-5-6-7",  # dictated, dashes, no leading zero
        "١٢٣٤٥٦٧",  # Arabic-Indic
        "١-٢-٣-٤-٥-٦-٧-٨-٩",
        "123456789012345678901",  # 21 digits: no specific pattern fits
    ],
)
def test_the_catch_all_redacts_any_other_run_of_seven_or_more_digits(number: str) -> None:
    result = redact(f"Customer: رقم الحساب {number} تمام")

    assert result.redacted_text == "Customer: رقم الحساب [NUMBER] تمام"
    assert result.redaction_count == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The specific labels still win over the catch-all.
        ("call 079 123 4567", "call [PHONE]"),
        ("id 9801234567", "id [NATIONAL_ID]"),
        ("card 4111 1111 1111 1111", "card [CARD]"),
        ("phone 555-019-8234", "phone [PHONE]"),
        # Each run is its own span; a comma, slash or dot ends a run.
        ("account 458921037 and line 4589210", "account [NUMBER] and line [NUMBER]"),
        ("1,500,000 or 12/10/2026", "1,500,000 or 12/10/2026"),
        ("version 458.921.037", "version 458.921.037"),
        # Six digits is not a run.
        ("PIN 123 456", "PIN 123 456"),
    ],
)
def test_the_catch_all_runs_last_and_only_on_seven_or_more_digits(text: str, expected: str) -> None:
    assert redact(text).redacted_text == expected


def test_the_catch_all_also_takes_dash_written_dates_a_known_cost() -> None:
    """Spaces and dashes join a run, so a dash-written date of 8 digits is redacted."""
    assert redact("due 2026-10-15").redacted_text == "due [NUMBER]"


@pytest.mark.skip(reason="needs NER — MVP redactor is regex-only, see docs/09-DECISIONS.md")
def test_redact_removes_names_in_arabic_and_english() -> None:
    """redact() detects and removes personal names in both Arabic and English text.

    Deliberately still skipped. Personal names have no regex shape; catching
    them needs NER, which is logged as future work rather than silently
    pretended-to. Leaving this test red-but-skipped keeps the gap visible.
    """


# --- Never redacted: record ids and machine timestamps (found 2026-10-11 via the
# tracing test: the Langfuse mask runs `redact()` on every trace string, and the
# digit patterns matched inside UUIDs and ISO timestamps).


@pytest.mark.parametrize(
    "value",
    [
        "05469959-c243-48f4-8cbe-b8c527e3c59c",  # first group all digits
        "8b768603-7b79-4fa8-a67e-d231086cb91c",  # digits run across a dash
        "a1b2c3d4-1234-5678-9abc-def012345678",  # two all-digit groups
        "12345678-1234-1234-1234-123456789012",  # every group all digits
        "2026-10-11T15:48:02.407611+00:00",
        "2026-10-11T15:48:02Z",
        "2026-10-11T15:48",
    ],
)
def test_uuids_and_iso_timestamps_pass_through_untouched(value: str) -> None:
    result = redact(f"call_id={value} done")

    assert result.redacted_text == f"call_id={value} done"
    assert result.redaction_count == 0


def test_pii_next_to_a_uuid_or_timestamp_is_still_redacted() -> None:
    text = "05469959-c243-48f4-8cbe-b8c527e3c59c at 2026-10-11T15:48:02Z: call 079 123 4567, acct 458921037"

    assert redact(text).redacted_text == (
        "05469959-c243-48f4-8cbe-b8c527e3c59c at 2026-10-11T15:48:02Z: call [PHONE], acct [NUMBER]"
    )
