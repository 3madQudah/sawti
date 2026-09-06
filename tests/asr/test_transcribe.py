"""Tests for `sawti.asr.transcribe`.

Phase 1: mirrors `src/sawti/asr/transcribe.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 1")
def test_transcribe_forces_arabic_language_by_default() -> None:
    """transcribe() calls Whisper with language='ar' when no override is given."""


@pytest.mark.skip(reason="phase 1")
def test_transcribe_returns_segments_ordered_by_start_time() -> None:
    """transcribe() returns TranscriptSegments in non-decreasing start_sec order."""


@pytest.mark.skip(reason="phase 1")
def test_transcribe_raises_on_missing_audio_file() -> None:
    """transcribe() raises when audio_path does not exist."""
