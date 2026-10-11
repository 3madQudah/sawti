"""`scripts/measure_asr_latency.py`: which calls it measures and how it summarizes them (phase 6.4)."""

from __future__ import annotations

import measure_asr_latency as measure
import pytest


def _entry(call_id: str, language: str) -> dict[str, str]:
    return {"call_id": call_id, "language": language}


def test_selects_the_first_calls_by_id_per_language_and_a_separate_warmup() -> None:
    manifest = {
        "entries": [
            _entry("call_0003_mixed", "mixed"),
            _entry("call_0000_ar", "ar"),
            _entry("call_0001_ar", "ar"),
            _entry("call_0002_en", "en"),
            _entry("call_0004_ar", "ar"),
        ]
    }
    chosen, warmup = measure.select_calls(manifest, per_language=1)
    assert {lang: [e["call_id"] for e in calls] for lang, calls in chosen.items()} == {
        "ar": ["call_0000_ar"],
        "en": ["call_0002_en"],
        "mixed": ["call_0003_mixed"],
    }
    assert warmup["call_id"] == "call_0001_ar", "the warm-up call is never one of the measured ones"


def test_summary_pools_time_over_audio() -> None:
    rows = [
        {"audio_s": 60.0, "total_s": 30.0, "transcribe_s": 10.0, "diarize_s": 18.0, "rtf": 0.5},
        {"audio_s": 120.0, "total_s": 120.0, "transcribe_s": 20.0, "diarize_s": 95.0, "rtf": 1.0},
    ]
    summary = measure.summarize(rows)
    assert summary["n"] == 2
    assert summary["rtf_pooled"] == pytest.approx(150 / 180, abs=1e-3)
    assert summary["rtf_whisper_pooled"] == pytest.approx(30 / 180, abs=1e-3)
    assert summary["rtf_diarization_pooled"] == pytest.approx(113 / 180, abs=1e-3)
    assert summary["processing_s_median"] == 75.0
    assert summary["rtf_per_call"] == [0.5, 1.0]
