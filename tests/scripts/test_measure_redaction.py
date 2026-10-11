"""`scripts/measure_redaction.py`: redaction before vs after, per language, with no PII in the output."""

from __future__ import annotations

import json
import re
from pathlib import Path

import measure_redaction
import pytest


def _redact_phones_with_dashes(text: str) -> tuple[str, int]:
    return re.subn(r"\d{3}-\d{3}-\d{4}", "[PHONE]", text)


def _redact_nothing(text: str) -> tuple[str, int]:
    return text, 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("call 555-019-8234 now", ["555-019-8234"]),
        ("الرقم ٠٧٩ ١٢٣ ٤٥٦٧ تفضل", ["٠٧٩ ١٢٣ ٤٥٦٧"]),
        ("dictated 1-0-7-1-2-3-4", ["1-0-7-1-2-3-4"]),
        ("account 458921037.", ["458921037"]),
        # Six digits, an amount with commas, a date: not runs of 7+.
        ("PIN 123456, paid 1,500,000 on 12/10/2026", []),
        # A placeholder ends a run.
        ("[PHONE] 1234", []),
    ],
)
def test_digit_runs(text: str, expected: list[str]) -> None:
    assert measure_redaction.digit_runs(text) == expected


def test_shape_keeps_the_format_and_drops_the_number() -> None:
    assert measure_redaction.shape("555-019-8234") == "ddd-ddd-dddd"
    assert measure_redaction.shape("٠٧٩ ١٢٣") == "ddd ddd"


def test_load_transcripts_reads_language_from_the_file_name(tmp_path: Path) -> None:
    (tmp_path / "call_0001_en.txt").write_text("Agent: hi\n", encoding="utf-8")
    (tmp_path / "call_0002_mixed.txt").write_text("Agent: مرحبا\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    rows = measure_redaction.load_transcripts(tmp_path)

    assert [(cid, lang) for cid, lang, _ in rows] == [("call_0001_en", "en"), ("call_0002_mixed", "mixed")]


def test_compare_reports_each_language_separately() -> None:
    rows = [
        ("call_0001_en", "en", "Customer: It's 555-019-8234 and account 458921037."),
        ("call_0002_ar", "ar", "Customer: رقمي 079-123-4567"),
        ("call_0003_mixed", "mixed", "Agent: no numbers"),
    ]

    report = measure_redaction.compare(rows, before=_redact_nothing, after=_redact_phones_with_dashes)

    en = report["en"]
    assert en["calls"] == 1
    assert en["runs_in_raw_text"] == 2
    assert (en["before"]["spans"], en["after"]["spans"]) == (0, 1)
    assert en["after"]["by_placeholder"] == {"[PHONE]": 1}
    assert (en["before"]["runs_left"], en["after"]["runs_left"]) == (2, 1)
    assert en["after"]["runs_left_by_shape"] == {"ddddddddd": 1}
    assert en["after"]["calls_with_runs_left"] == ["call_0001_en"]
    assert en["calls_changed"] == ["call_0001_en"]
    assert report["ar"]["after"]["runs_left"] == 0
    assert report["mixed"]["calls_changed"] == []


def test_the_report_carries_no_digits_of_any_number() -> None:
    rows = [("call_0001_en", "en", "Customer: It's 555-019-8234 and account 458921037.")]
    report = measure_redaction.compare(rows, before=_redact_nothing, after=_redact_phones_with_dashes)
    text = json.dumps(report, ensure_ascii=False)

    assert "555" not in text
    assert "458921037" not in text


def test_redactor_from_source_wraps_that_modules_redact() -> None:
    source = (
        "from types import SimpleNamespace\n"
        "def redact(text):\n"
        "    out = text.replace('x', '[X]')\n"
        "    return SimpleNamespace(redacted_text=out, redaction_count=text.count('x'))\n"
    )
    redactor = measure_redaction.redactor_from_source(source)

    assert redactor("axbx") == ("a[X]b[X]", 2)


def test_redactor_at_rev_loads_the_module_from_git(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    class _Done:
        stdout = "def redact(text):\n    raise AssertionError('not called here')\n"

    def _run(cmd: list[str], **_: object) -> _Done:
        calls.append(cmd)
        return _Done()

    monkeypatch.setattr(measure_redaction.subprocess, "run", _run)

    measure_redaction.redactor_at_rev("abc123")

    assert calls == [["git", "show", "abc123:src/sawti/privacy/redaction.py"]]
