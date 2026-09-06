"""Whisper-based transcription, with Arabic forced as the recognition language.

Phase 1: ASR pipeline entry point.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    """A single ASR-produced segment of a transcript."""

    text: str
    start_sec: float
    end_sec: float


class Transcript(BaseModel):
    """Full transcription result for one call recording."""

    call_id: str
    segments: list[TranscriptSegment]
    language: str


def transcribe(audio_path: Path, *, language: str = "ar") -> Transcript:
    """Transcribe `audio_path` using Whisper with `language` forced.

    Args:
        audio_path: Path to the call audio file.
        language: Language code to force Whisper to recognize (default "ar").

    Returns:
        The resulting `Transcript`.

    Raises:
        NotImplementedError: Until the Whisper integration is implemented.
    """
    # TODO(phase-1): run Whisper (model per sawti.config.Settings.whisper_model) with
    # language="ar" forced, and assemble a Transcript from the resulting segments.
    raise NotImplementedError
