"""Tests for `sawti.asr.diarization`.

Phase 1: mirrors `src/sawti/asr/diarization.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 1")
def test_diarize_returns_segments_ordered_by_start_time() -> None:
    """diarize() returns DiarizedSegments in non-decreasing start_sec order."""


@pytest.mark.skip(reason="phase 1")
def test_merge_transcript_with_speakers_assigns_speaker_to_every_segment() -> None:
    """merge_transcript_with_speakers() attaches a speaker label to every transcript segment."""


@pytest.mark.skip(reason="phase 1")
def test_merge_transcript_with_speakers_uses_max_overlap_when_ambiguous() -> None:
    """merge_transcript_with_speakers() assigns the speaker with greatest time overlap."""
