"""`sawti.asr.pipeline.transcribe_call`: decode, Whisper, diarize, label (phase 6.4).

Whisper, ffmpeg and pyannote are stubbed; what is tested is the wiring: one
decode feeds diarization, the forced language comes from the category, roles
come from the first-speaker convention, and the raw text never shows in a repr.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sawti.asr import pipeline
from sawti.asr.diarization import DiarizedSegment
from sawti.asr.transcribe import Transcript, TranscriptSegment

PHONE = "0791234567"


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    waveform = np.zeros(16_000 * 30, dtype=np.float32)  # 30 s

    def fake_transcribe(path: Path, *, category: str) -> Transcript:
        seen["category"] = category
        return Transcript(
            call_id=path.stem,
            language="en",
            segments=[
                TranscriptSegment(text="Hello, how can I help?", start_sec=0.0, end_sec=2.0),
                TranscriptSegment(text=f"Call me on {PHONE}.", start_sec=2.5, end_sec=4.0),
                TranscriptSegment(text="Sure.", start_sec=4.5, end_sec=5.0),
            ],
        )

    def fake_diarize(path: Path, *, pipeline: Any, waveform: Any) -> list[DiarizedSegment]:
        seen["pipeline"], seen["waveform"] = pipeline, waveform
        return [
            DiarizedSegment(speaker="SPEAKER_01", start_sec=0.0, end_sec=2.2),
            DiarizedSegment(speaker="SPEAKER_00", start_sec=2.3, end_sec=4.2),
            DiarizedSegment(speaker="SPEAKER_01", start_sec=4.3, end_sec=5.0),
        ]

    monkeypatch.setattr(pipeline, "load_audio", lambda path: waveform)
    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline, "diarize", fake_diarize)
    seen["decoded"] = waveform
    return seen


def test_transcribe_call_labels_turns_and_times_each_stage(stubs: dict[str, Any], tmp_path: Path) -> None:
    sentinel = object()

    result = pipeline.transcribe_call(tmp_path / "call.wav", category="mixed", diarization_pipeline=sentinel)

    assert result.raw_transcript == (
        f"Agent: Hello, how can I help?\nCustomer: Call me on {PHONE}.\nAgent: Sure.\n"
    )
    assert stubs["category"] == "mixed"
    assert stubs["pipeline"] is sentinel
    assert stubs["waveform"] is stubs["decoded"], "one decode serves Whisper's duration and pyannote"
    assert result.audio_seconds == pytest.approx(30.0)
    assert result.decoded_language == "en"
    timings = result.timings
    assert timings.total_s >= timings.decode_s + timings.transcribe_s + timings.diarize_s - 1e-6
    assert result.realtime_factor == pytest.approx(timings.total_s / 30.0)


def test_the_raw_transcript_never_appears_in_a_repr(stubs: dict[str, Any], tmp_path: Path) -> None:
    result = pipeline.transcribe_call(tmp_path / "call.wav", category="en")
    assert PHONE not in repr(result) and PHONE not in str(result)


def test_silence_skips_diarization_and_yields_an_empty_transcript(
    stubs: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pipeline, "transcribe", lambda path, *, category: Transcript(call_id="x", language="ar", segments=[])
    )

    result = pipeline.transcribe_call(tmp_path / "call.wav", category="ar")

    assert result.raw_transcript == ""
    assert "pipeline" not in stubs, "nothing to attribute, so pyannote is not run"
