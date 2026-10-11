"""One call's audio to a speaker-labelled transcript: decode, Whisper, diarize, label.

Phase 6.4: the ASR worker (`sawti.api.asr_tasks`) and the latency
measurement (`scripts/measure_asr_latency.py`) both call `transcribe_call`, so
the measured pipeline is the one the service runs.

**The result is raw, unredacted text: PII.** It exists only in memory, on its
way to `sawti.api.ingest.redact_transcript`. Its `repr` hides the text, so
logging an `AsrResult` by accident cannot leak it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from sawti.asr.diarization import (
    SAMPLE_RATE,
    assign_roles_by_first_speaker,
    diarize,
    merge_transcript_with_speakers,
    render_labelled_transcript,
)
from sawti.asr.transcribe import transcribe


class AsrTimings(BaseModel):
    """Wall-clock seconds per stage. No text."""

    decode_s: float = Field(..., ge=0.0)
    transcribe_s: float = Field(..., ge=0.0)
    diarize_s: float = Field(..., ge=0.0)
    total_s: float = Field(..., ge=0.0)


class AsrResult(BaseModel):
    """A transcribed call: the raw labelled transcript (PII), the audio length, the timings."""

    #: `Agent: ...` / `Customer: ...` lines, unredacted. Never store, log or trace it.
    raw_transcript: str = Field(..., repr=False)
    audio_seconds: float = Field(..., ge=0.0)
    #: The language Whisper decoded with (the forced one).
    decoded_language: str
    timings: AsrTimings

    @property
    def realtime_factor(self) -> float:
        """Processing time / audio duration (below 1 = faster than real time)."""
        return self.timings.total_s / self.audio_seconds if self.audio_seconds else 0.0


def load_audio(audio_path: Path) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Decode any ffmpeg-readable file (wav/mp3/m4a) to 16 kHz mono float32, via the ffmpeg CLI."""
    from whisper.audio import load_audio as whisper_load_audio

    audio: np.ndarray[Any, np.dtype[np.float32]] = whisper_load_audio(str(audio_path), sr=SAMPLE_RATE)
    return audio


def transcribe_call(audio_path: Path, *, category: str, diarization_pipeline: Any | None = None) -> AsrResult:
    """Transcribe and diarize one call, and label its turns.

    Whisper's language is forced from `category`
    (`Settings.whisper_language_for`, the Phase 3 decision). Speakers come from
    pyannote, constrained to two, and the roles from the first-speaker-is-agent
    convention (`assign_roles_by_first_speaker`), which is wrong for any call
    the customer opens.

    Args:
        audio_path: The call's audio file.
        category: The call's `sawti.schemas.Language` value.
        diarization_pipeline: A loaded pyannote pipeline to reuse; None loads one.

    Returns:
        The raw labelled transcript, which is empty if Whisper heard no speech,
        plus timings.

    Raises:
        FileNotFoundError, `DiarizationUnavailableError`, or whatever Whisper,
        ffmpeg or pyannote raise.
    """
    started = time.perf_counter()
    waveform = load_audio(audio_path)
    decoded = time.perf_counter()
    transcript = transcribe(audio_path, category=category)
    transcribed = time.perf_counter()
    speakers = (
        diarize(audio_path, pipeline=diarization_pipeline, waveform=waveform) if transcript.segments else []
    )
    diarized = time.perf_counter()

    labelled = assign_roles_by_first_speaker(merge_transcript_with_speakers(transcript.segments, speakers))
    return AsrResult(
        raw_transcript=render_labelled_transcript(labelled),
        audio_seconds=len(waveform) / SAMPLE_RATE,
        decoded_language=transcript.language,
        timings=AsrTimings(
            decode_s=decoded - started,
            transcribe_s=transcribed - decoded,
            diarize_s=diarized - transcribed,
            total_s=time.perf_counter() - started,
        ),
    )
