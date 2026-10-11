"""PII redaction — runs before any transcript text reaches a model.

Phase 1 module, implemented in phase 2 because `sawti.agent.nodes.extract`
sends transcript text to a cloud LLM and the project rule "PII redaction runs
before any text reaches a model" is not optional.

Scope, stated plainly: this is an **MVP regex redactor**, sized for the
synthetic Jordanian contact-center dataset in `data/synthetic/`. It catches
structured identifiers — phone numbers, emails, national IDs, card numbers,
IBANs — because those have shape, and shape is what a regex can see.

It does **not** catch unstructured PII: personal names, addresses, employers,
anything that is only identifiable in context. That needs NER, and NER is
noted as a future improvement in `docs/09-DECISIONS.md`. Do not read this
module as "the transcript is now anonymous"; read it as "the obvious
identifiers are gone before the text leaves the process".

Redaction is length-preserving-ish but NOT offset-preserving: placeholders
differ in length from what they replace. That is why `AgentState` carries
both `transcript` and `redacted_transcript`, and why grounding checks quote
offsets against the redacted text — the string the model actually saw.
"""

from __future__ import annotations

import re

from pydantic import BaseModel


class RedactionResult(BaseModel):
    """Output of a redaction pass over a piece of text."""

    redacted_text: str
    original_length: int
    redaction_count: int


# Ordered: earlier patterns win, so a 16-digit card is not first eaten by the
# phone pattern. Each entry is (placeholder, compiled pattern).
#
# Python 3's `\d` is Unicode-aware, so it matches Arabic-Indic digits
# (U+0660-U+0669) as well as ASCII ones. The synthetic `ar` transcripts use the
# two interchangeably, so this is load-bearing rather than incidental.
_DIGIT = r"\d"
# One optional separator between digit groups, as people say and ASR writes
# phone numbers: a space, a dash or a dot. Never a comma or slash (amounts, dates).
_SEP = r"[ .\-]?"
# Phone patterns are bounded by "no digit", not `\b`: an Arabic conjunction or
# preposition written onto the number (و٠٧٩…, ل٠٧٩…) is a word character, so
# `\b` would refuse the match there. The `+` in the lookbehind stops a pattern
# starting inside an international number.
_NOT_AFTER_DIGIT = r"(?<![\d+])"
_NOT_BEFORE_DIGIT = r"(?!\d)"

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Email: standard-enough shape, checked before anything digit-based so an
    # address containing digits is not partially eaten.
    ("[EMAIL]", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # IBAN: JO + 2 check digits + 4 bank chars + up to 18 alphanumerics.
    ("[IBAN]", re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[A-Z0-9]{10,20}\b")),
    # Phone, international: `+` or `00`, a 1-3 digit country code, then 2-5
    # groups of 1-4 digits (+962 7 9123 4567, +962791234567, 00962 79 123 4567,
    # +1 (555) 019-8234). Before the card pattern, which would otherwise read
    # the 12 digits after `+` as a card and leave the `+` behind.
    (
        "[PHONE]",
        re.compile(
            rf"{_NOT_AFTER_DIGIT}(?:\+|00){_DIGIT}{{1,3}}(?:{_SEP}\(?{_DIGIT}{{1,4}}\)?){{2,5}}{_NOT_BEFORE_DIGIT}"
        ),
    ),
    # Payment card: 13-19 digits, optionally grouped by spaces or dashes.
    ("[CARD]", re.compile(rf"\b{_DIGIT}{{4}}(?:[ -]?{_DIGIT}{{4}}){{2,4}}\b")),
    # Jordanian national ID: exactly 10 consecutive digits.
    ("[NATIONAL_ID]", re.compile(rf"\b{_DIGIT}{{10}}\b")),
    # Phone, Jordanian national format: a leading zero (ASCII or Arabic-Indic)
    # and 8-9 more digits, with an optional separator between any two digits:
    # 079 123 4567, 07 9123 4567, 0791 234 567, 06 512 3456, 0-7-9-1-2-3-...
    # Deliberately after national ID, so a bare 10-digit run is read as an ID,
    # which is the more sensitive of the two.
    (
        "[PHONE]",
        # The extra lookbehind keeps it from starting at a zero in the middle of
        # a longer dictated run (1-0-7-…), which would redact the tail and leave
        # the head behind under the wrong label.
        re.compile(rf"{_NOT_AFTER_DIGIT}(?<!\d[ .\-])[0٠](?:{_SEP}{_DIGIT}){{8,9}}{_NOT_BEFORE_DIGIT}"),
    ),
    # Phone, 3-3-4 with separators or a bracketed area code: 555-019-8234,
    # 555 019 8234, 555.019.8234, (555) 019-8234. The contiguous form is ten
    # digits, already caught as a national ID above.
    (
        "[PHONE]",
        re.compile(
            rf"{_NOT_AFTER_DIGIT}\(?{_DIGIT}{{3}}\)?{_SEP}{_DIGIT}{{3}}[ .\-]{_DIGIT}{{4}}{_NOT_BEFORE_DIGIT}"
        ),
    ),
    # Phone, local 3-4 with a separator: 019-8234, 512 3456.
    ("[PHONE]", re.compile(rf"{_NOT_AFTER_DIGIT}{_DIGIT}{{3}}[ .\-]{_DIGIT}{{4}}{_NOT_BEFORE_DIGIT}")),
)


def redact(text: str) -> RedactionResult:
    """Redact structured PII from `text` before it is sent to any model.

    Must run before `text` is passed to any `LLMProvider` call, per project
    rule: "PII redaction runs before any text reaches a model."

    Patterns are applied in a fixed order (see `_PATTERNS`) so that the more
    sensitive interpretation of an ambiguous digit run wins — a bare 10-digit
    number is treated as a national ID rather than a phone number.

    Args:
        text: Raw transcript or utterance text, possibly containing PII.

    Returns:
        A `RedactionResult` with the redacted text and redaction stats. Text
        containing no recognized PII comes back unchanged with a count of 0.
    """
    redacted = text
    count = 0

    for placeholder, pattern in _PATTERNS:
        # `subn` reports how many replacements happened, which is the stat the
        # caller needs; a plain `sub` would force a second pass to count.
        redacted, replacements = pattern.subn(placeholder, redacted)
        count += replacements

    return RedactionResult(
        redacted_text=redacted,
        original_length=len(text),
        redaction_count=count,
    )
