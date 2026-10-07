"""Tests for `scripts/measure_service.py`'s pure parts (selection and statistics).

Phase 6.2. The docker/HTTP orchestration itself is exercised by running it;
what is tested here is what decides the reported numbers.
"""

from __future__ import annotations

import pytest
from measure_service import _seconds, pick_transcripts, summarize

from sawti.schemas import Language


def _detail(
    language: str, queued: str, started: str, finished: str, *, attempts: int = 1, status: str = "completed"
):
    return {
        "language": language,
        "queued_at": f"2026-10-07T10:00:{queued}+00:00",
        "started_at": f"2026-10-07T10:00:{started}+00:00",
        "finished_at": f"2026-10-07T10:00:{finished}+00:00",
        "attempts": attempts,
        "status": status,
    }


def test_pick_transcripts_interleaves_languages_and_skips_unparseable_calls() -> None:
    picked = pick_transcripts(2)
    assert [language for language, _, _ in picked] == [Language.AR, Language.EN, Language.MIXED] * 2
    # call_0003_mixed has a line with no speaker prefix (the phase 1 defect).
    assert "call_0003_mixed" not in {stem for _, stem, _ in picked}


def test_summarize_reports_each_language_separately() -> None:
    details = [
        _detail("ar", "00", "01", "06"),
        _detail("ar", "00", "06", "11", attempts=2),
        _detail("en", "00", "11", "16", status="failed"),
    ]
    summary = summarize(details)
    assert set(summary["per_language"]) == {"ar", "en"}
    ar = summary["per_language"]["ar"]
    assert ar["queue_wait_p50"] == pytest.approx(3.5)
    assert ar["processing_p50"] == pytest.approx(5.0)
    assert ar["retry_rate"] == 0.5 and ar["failure_rate"] == 0.0
    assert summary["per_language"]["en"]["failure_rate"] == 1.0
    assert summary["wall_seconds"] == 16.0
    assert summary["throughput_per_min"] == pytest.approx(3 / 16 * 60)


def test_seconds_tolerates_missing_timestamps() -> None:
    assert _seconds(None, "2026-10-07T10:00:00+00:00") is None
