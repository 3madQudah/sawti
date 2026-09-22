"""Tests for `sawti.asr.diarization`.

Phase 3: mirrors `src/sawti/asr/diarization.py`.

pyannote's pretrained pipeline is gated on HuggingFace, so `diarize()` is
tested against a stub pipeline rather than the real weights: these tests pin
the contract — ordering, the token error path, and the overlap-based
alignment — not pyannote's clustering quality, which is measured against real
audio in `eval_results.md`.
"""

from pathlib import Path
from typing import Any

import pytest

from sawti.asr import diarization as diarization_module
from sawti.asr.diarization import (
    DiarizationUnavailableError,
    DiarizedSegment,
    SpeakerTranscriptSegment,
    assign_roles_by_first_speaker,
    diarize,
    merge_transcript_with_speakers,
)
from sawti.asr.transcribe import TranscriptSegment


class _FakeTurn:
    """Stands in for a pyannote `Segment`."""

    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class _FakeAnnotation:
    """Stands in for a pyannote `Annotation`."""

    def __init__(self, tracks: list[tuple[float, float, str]]) -> None:
        self._tracks = tracks

    def itertracks(self, yield_label: bool = False) -> Any:
        for start, end, speaker in self._tracks:
            yield _FakeTurn(start, end), None, speaker


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    """An existing (empty) file standing in for a call recording."""
    path = tmp_path / "call_0000_ar.wav"
    path.write_bytes(b"")
    return path


@pytest.fixture
def stub_pipeline(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the gated pipeline loader with a stub returning fixed tracks."""
    recorded: dict[str, Any] = {}
    # Deliberately out of order, to prove diarize() sorts.
    tracks = [(5.0, 9.0, "SPEAKER_01"), (0.0, 4.0, "SPEAKER_00")]

    def fake_pipeline(path: str, num_speakers: int | None = None) -> _FakeAnnotation:
        recorded["path"] = path
        recorded["num_speakers"] = num_speakers
        return _FakeAnnotation(tracks)

    def fake_load(pipeline_name: str, token: str | None) -> Any:
        recorded["pipeline_name"] = pipeline_name
        recorded["token"] = token
        return fake_pipeline

    monkeypatch.setattr(diarization_module, "_load_pipeline", fake_load)
    return recorded


class TestDiarize:
    """The pipeline-facing half."""

    def test_diarize_returns_segments_ordered_by_start_time(
        self, audio_file: Path, stub_pipeline: dict[str, Any]
    ) -> None:
        """diarize() returns DiarizedSegments in non-decreasing start_sec order."""
        segments = diarize(audio_file)

        starts = [segment.start_sec for segment in segments]
        assert starts == sorted(starts)
        assert [segment.speaker for segment in segments] == ["SPEAKER_00", "SPEAKER_01"]

    def test_diarize_constrains_to_two_speakers_by_default(
        self, audio_file: Path, stub_pipeline: dict[str, Any]
    ) -> None:
        """A contact-centre call is an agent and a customer."""
        diarize(audio_file)

        assert stub_pipeline["num_speakers"] == 2

    def test_diarize_raises_on_missing_audio_file(
        self, tmp_path: Path, stub_pipeline: dict[str, Any]
    ) -> None:
        """diarize() raises when audio_path does not exist."""
        with pytest.raises(FileNotFoundError):
            diarize(tmp_path / "does_not_exist.wav")

    def test_missing_token_raises_with_actionable_instructions(
        self, audio_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gated-weights failure names both repos that need accepting."""
        with pytest.raises(DiarizationUnavailableError) as error:
            diarization_module._load_pipeline("pyannote/speaker-diarization-3.1", None)

        message = str(error.value)
        assert "HF_TOKEN" in message
        assert "segmentation-3.0" in message
        assert "speaker-diarization-3.1" in message


class TestMergeTranscriptWithSpeakers:
    """The alignment half."""

    def test_assigns_speaker_to_every_segment(self) -> None:
        """merge_transcript_with_speakers() attaches a speaker to every segment."""
        transcript = [
            TranscriptSegment(text="hello", start_sec=0.0, end_sec=3.0),
            TranscriptSegment(text="hi back", start_sec=5.0, end_sec=8.0),
        ]
        diarized = [
            DiarizedSegment(speaker="SPEAKER_00", start_sec=0.0, end_sec=4.0),
            DiarizedSegment(speaker="SPEAKER_01", start_sec=4.5, end_sec=9.0),
        ]

        merged = merge_transcript_with_speakers(transcript, diarized)

        assert len(merged) == len(transcript)
        assert all(segment.speaker for segment in merged)
        assert [segment.speaker for segment in merged] == ["SPEAKER_00", "SPEAKER_01"]

    def test_uses_max_overlap_when_ambiguous(self) -> None:
        """A segment straddling two speakers takes the one it overlaps most."""
        # 0..10; SPEAKER_00 covers 0..3 (3s overlap), SPEAKER_01 covers 3..10 (7s).
        transcript = [TranscriptSegment(text="straddles", start_sec=0.0, end_sec=10.0)]
        diarized = [
            DiarizedSegment(speaker="SPEAKER_00", start_sec=0.0, end_sec=3.0),
            DiarizedSegment(speaker="SPEAKER_01", start_sec=3.0, end_sec=10.0),
        ]

        merged = merge_transcript_with_speakers(transcript, diarized)

        assert merged[0].speaker == "SPEAKER_01"

    def test_preserves_text_and_timing(self) -> None:
        """Alignment annotates; it must not alter the transcript."""
        transcript = [TranscriptSegment(text="verbatim", start_sec=1.5, end_sec=2.5)]
        diarized = [DiarizedSegment(speaker="SPEAKER_00", start_sec=0.0, end_sec=4.0)]

        merged = merge_transcript_with_speakers(transcript, diarized)

        assert merged[0].text == "verbatim"
        assert merged[0].start_sec == 1.5
        assert merged[0].end_sec == 2.5

    def test_segment_overlapping_nothing_falls_back_to_nearest(self) -> None:
        """Text over diarized silence is still attributed, not dropped."""
        transcript = [TranscriptSegment(text="orphan", start_sec=20.0, end_sec=21.0)]
        diarized = [
            DiarizedSegment(speaker="SPEAKER_00", start_sec=0.0, end_sec=4.0),
            DiarizedSegment(speaker="SPEAKER_01", start_sec=15.0, end_sec=18.0),
        ]

        merged = merge_transcript_with_speakers(transcript, diarized)

        assert len(merged) == 1
        assert merged[0].speaker == "SPEAKER_01"

    def test_empty_transcript_returns_empty(self) -> None:
        """Nothing to attribute is not an error."""
        diarized = [DiarizedSegment(speaker="SPEAKER_00", start_sec=0.0, end_sec=4.0)]

        assert merge_transcript_with_speakers([], diarized) == []

    def test_empty_diarization_with_transcript_raises(self) -> None:
        """There is no speaker to assign; failing loudly beats inventing one."""
        transcript = [TranscriptSegment(text="hello", start_sec=0.0, end_sec=3.0)]

        with pytest.raises(ValueError, match="diarized_segments is empty"):
            merge_transcript_with_speakers(transcript, [])


class TestAssignRolesByFirstSpeaker:
    """Mapping anonymous clusters onto contact-centre roles."""

    def test_earliest_speaker_becomes_agent(self) -> None:
        """The convention: the agent opens the call."""
        segments = [
            SpeakerTranscriptSegment(text="a", start_sec=0.0, end_sec=2.0, speaker="SPEAKER_01"),
            SpeakerTranscriptSegment(text="b", start_sec=3.0, end_sec=5.0, speaker="SPEAKER_00"),
        ]

        roled = assign_roles_by_first_speaker(segments)

        assert [segment.speaker for segment in roled] == ["Agent", "Customer"]

    def test_every_other_cluster_becomes_customer(self) -> None:
        """A third cluster is folded into Customer rather than left anonymous."""
        segments = [
            SpeakerTranscriptSegment(text="a", start_sec=0.0, end_sec=1.0, speaker="SPEAKER_00"),
            SpeakerTranscriptSegment(text="b", start_sec=2.0, end_sec=3.0, speaker="SPEAKER_01"),
            SpeakerTranscriptSegment(text="c", start_sec=4.0, end_sec=5.0, speaker="SPEAKER_02"),
        ]

        roled = assign_roles_by_first_speaker(segments)

        assert [segment.speaker for segment in roled] == ["Agent", "Customer", "Customer"]

    def test_empty_input_returns_empty(self) -> None:
        """No segments, no roles."""
        assert assign_roles_by_first_speaker([]) == []
