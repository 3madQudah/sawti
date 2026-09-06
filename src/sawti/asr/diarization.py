"""Speaker diarization for call recordings.

Phase 1: ASR pipeline, speaker attribution.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from sawti.asr.transcribe import TranscriptSegment


class DiarizedSegment(BaseModel):
    """A time-bounded segment attributed to a single speaker."""

    speaker: str
    start_sec: float
    end_sec: float


class SpeakerTranscriptSegment(BaseModel):
    """A transcript segment annotated with a speaker label."""

    text: str
    start_sec: float
    end_sec: float
    speaker: str


def diarize(audio_path: Path) -> list[DiarizedSegment]:
    """Run speaker diarization over `audio_path`.

    Args:
        audio_path: Path to the call audio file.

    Returns:
        Diarized segments, ordered by start time.

    Raises:
        NotImplementedError: Until the diarization integration is implemented.
    """
    # TODO(phase-1): run pyannote.audio diarization pipeline and return segments.
    raise NotImplementedError


def merge_transcript_with_speakers(
    transcript_segments: list[TranscriptSegment], diarized_segments: list[DiarizedSegment]
) -> list[SpeakerTranscriptSegment]:
    """Assign speaker labels to transcript segments using diarization output.

    Args:
        transcript_segments: ASR segments (text + timing) without speaker labels.
        diarized_segments: Diarized speaker segments to align against.

    Returns:
        Transcript segments annotated with a speaker label each.

    Raises:
        NotImplementedError: Until alignment is implemented.
    """
    # TODO(phase-1): align by timestamp overlap and attach speaker labels.
    raise NotImplementedError
