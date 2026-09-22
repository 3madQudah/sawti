"""Tests for `sawti.eval.experiments.asr_propagation`.

Phase 3: mirrors `src/sawti/eval/experiments/asr_propagation.py`.

Whisper and the LLM are both out of scope here. These cover the glue that
decides whether the propagation delta is trustworthy: that ASR text is
re-rendered into the same speaker-labelled shape the clean corpus uses, and
that WER aggregates by words rather than by call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sawti.asr.transcribe import TranscriptSegment
from sawti.data.generate_audio import AudioManifestEntry, AudioSegment
from sawti.eval.experiments.asr_propagation import (
    CallWer,
    TranscriptionReport,
    _hypothesis_from_transcript,
    _oracle_speaker_segments,
    _render_labelled_transcript,
)
from sawti.schemas import Language


def _entry(segments: list[AudioSegment]) -> AudioManifestEntry:
    """A manifest entry wrapping `segments`."""
    return AudioManifestEntry(
        call_id="call_0000_ar",
        language=Language.AR,
        audio_path="call_0000_ar.wav",
        reference_transcript_path="call_0000_ar.txt",
        reference_analysis_path=None,
        duration_sec=segments[-1].end_sec if segments else 0.0,
        voices={"Agent": "v1", "Customer": "v2"},
        segments=segments,
    )


class TestOracleSpeakerSegments:
    """The manifest timeline reshaped as diarizer output."""

    def test_preserves_speakers_and_timings(self) -> None:
        """Oracle attribution must be exactly the manifest's own timeline."""
        entry = _entry(
            [
                AudioSegment(speaker="Agent", start_sec=0.0, end_sec=2.0, text="a", source_line=1),
                AudioSegment(
                    speaker="Customer", start_sec=2.3, end_sec=4.0, text="b", source_line=2
                ),
            ]
        )

        oracle = _oracle_speaker_segments(entry)

        assert [segment.speaker for segment in oracle] == ["Agent", "Customer"]
        assert [(segment.start_sec, segment.end_sec) for segment in oracle] == [
            (0.0, 2.0),
            (2.3, 4.0),
        ]


class TestRenderLabelledTranscript:
    """Re-rendering ASR output into the clean corpus's line format."""

    def test_labels_each_segment_with_its_speaker(self) -> None:
        """Output matches the 'Agent: ...' / 'Customer: ...' shape."""
        entry = _entry(
            [
                AudioSegment(speaker="Agent", start_sec=0.0, end_sec=2.0, text="x", source_line=1),
                AudioSegment(
                    speaker="Customer", start_sec=2.3, end_sec=4.0, text="y", source_line=2
                ),
            ]
        )
        asr = [
            TranscriptSegment(text="hello there", start_sec=0.1, end_sec=1.9),
            TranscriptSegment(text="hi back", start_sec=2.4, end_sec=3.9),
        ]

        rendered = _render_labelled_transcript(entry, asr)

        assert rendered == "Agent: hello there\nCustomer: hi back\n"

    def test_merges_consecutive_segments_from_one_speaker(self) -> None:
        """Whisper splits a turn into several segments; the turn is one line.

        Without this, one spoken turn becomes several 'Agent:' lines and the
        ASR transcript has a visibly different shape from the clean one —
        which would show up in the delta as if it were ASR error.
        """
        entry = _entry(
            [AudioSegment(speaker="Agent", start_sec=0.0, end_sec=6.0, text="x", source_line=1)]
        )
        asr = [
            TranscriptSegment(text="first part", start_sec=0.0, end_sec=2.0),
            TranscriptSegment(text="second part", start_sec=2.0, end_sec=4.0),
            TranscriptSegment(text="third part", start_sec=4.0, end_sec=6.0),
        ]

        rendered = _render_labelled_transcript(entry, asr)

        assert rendered == "Agent: first part second part third part\n"

    def test_skips_empty_segments(self) -> None:
        """A silent segment must not produce a bare 'Agent:' line."""
        entry = _entry(
            [AudioSegment(speaker="Agent", start_sec=0.0, end_sec=4.0, text="x", source_line=1)]
        )
        asr = [
            TranscriptSegment(text="real text", start_sec=0.0, end_sec=2.0),
            TranscriptSegment(text="   ", start_sec=2.0, end_sec=4.0),
        ]

        rendered = _render_labelled_transcript(entry, asr)

        assert rendered == "Agent: real text\n"

    def test_speaker_changes_start_a_new_line(self) -> None:
        """Alternating speakers stay on separate lines."""
        entry = _entry(
            [
                AudioSegment(speaker="Agent", start_sec=0.0, end_sec=2.0, text="x", source_line=1),
                AudioSegment(
                    speaker="Customer", start_sec=2.0, end_sec=4.0, text="y", source_line=2
                ),
                AudioSegment(speaker="Agent", start_sec=4.0, end_sec=6.0, text="z", source_line=3),
            ]
        )
        asr = [
            TranscriptSegment(text="one", start_sec=0.5, end_sec=1.5),
            TranscriptSegment(text="two", start_sec=2.5, end_sec=3.5),
            TranscriptSegment(text="three", start_sec=4.5, end_sec=5.5),
        ]

        rendered = _render_labelled_transcript(entry, asr)

        assert rendered.splitlines() == ["Agent: one", "Customer: two", "Agent: three"]


class TestTranscriptionReport:
    """Corpus-level WER aggregation."""

    def _call(self, language: str, wer: float, words: int) -> CallWer:
        return CallWer(
            call_id=f"call_{language}",
            language=language,
            wer=wer,
            wer_raw=wer + 0.1,
            reference_words=words,
            hypothesis_words=words,
            audio_sec=10.0,
            transcribe_sec=5.0,
        )

    def test_wer_is_word_weighted_not_call_averaged(self) -> None:
        """A 10-word call must outweigh a 1-word call.

        Call-averaged would give 0.5; word-weighted gives (1*1 + 0*10)/11.
        """
        report = TranscriptionReport(calls=[self._call("ar", 1.0, 1), self._call("ar", 0.0, 10)])

        assert report.wer_by_category()[Language.AR] == pytest.approx(1 / 11)

    def test_reports_per_language_never_blended(self) -> None:
        """The project's standing rule, enforced at the report boundary."""
        report = TranscriptionReport(
            calls=[self._call("ar", 0.1, 10), self._call("mixed", 0.5, 10)]
        )

        result = report.wer_by_category()

        assert set(result) == {Language.AR, Language.MIXED}
        assert result[Language.AR] == pytest.approx(0.1)
        assert result[Language.MIXED] == pytest.approx(0.5)

    def test_raw_mode_reports_the_unnormalized_rate(self) -> None:
        """Both normalized and raw WER stay available from one report."""
        report = TranscriptionReport(calls=[self._call("ar", 0.2, 10)])

        assert report.wer_by_category(normalize=False)[Language.AR] == pytest.approx(0.3)

    def test_realtime_factor_is_audio_over_wall_clock(self) -> None:
        """10s of audio in 5s is 2x realtime."""
        report = TranscriptionReport(calls=[self._call("ar", 0.0, 10)])

        assert report.realtime_factor() == pytest.approx(2.0)

    def test_empty_report_is_safe(self) -> None:
        """No calls means no categories and no division by zero."""
        report = TranscriptionReport()

        assert report.wer_by_category() == {}
        assert report.realtime_factor() == 0.0


class TestTranscriptionResume:
    """Re-scoring an existing transcript instead of re-transcribing it."""

    def test_hypothesis_is_recovered_from_a_labelled_transcript(self) -> None:
        """Stripping the prefixes must leave exactly the recognized words."""
        text = "Agent: hello there\nCustomer: hi back\n"

        assert _hypothesis_from_transcript(text) == "hello there hi back"

    def test_blank_lines_are_ignored(self) -> None:
        """A trailing newline must not add an empty token."""
        assert _hypothesis_from_transcript("Agent: one\n\nCustomer: two\n") == "one two"

    def test_resume_skips_transcription_and_still_scores(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An existing transcript is re-scored without touching Whisper."""
        from sawti.data.generate_audio import AudioManifest
        from sawti.eval.experiments import asr_propagation as module

        def fail_if_called(*args: object, **kwargs: object) -> None:
            raise AssertionError("resume must not call transcribe()")

        monkeypatch.setattr(module, "transcribe", fail_if_called)

        entry = _entry(
            [
                AudioSegment(
                    speaker="Agent", start_sec=0.0, end_sec=1.0, text="hello there", source_line=1
                )
            ]
        )
        asr_dir = tmp_path / "asr"
        asr_dir.mkdir()
        asr_dir.joinpath("call_0000_ar.txt").write_text("Agent: hello there\n", encoding="utf-8")

        report = module.transcribe_corpus(
            AudioManifest(generated_at="2026-09-22T00:00:00+00:00", entries=[entry]),
            output_dir=asr_dir,
        )

        assert len(report.calls) == 1
        assert report.calls[0].wer == 0.0
        assert report.calls[0].transcribe_sec == 0.0

    def test_realtime_factor_excludes_resumed_calls(self) -> None:
        """A resumed call has no timing and must not be counted as free speed."""
        timed = CallWer(
            call_id="a", language="ar", wer=0.0, wer_raw=0.0, reference_words=10,
            hypothesis_words=10, audio_sec=10.0, transcribe_sec=5.0,
        )
        resumed = CallWer(
            call_id="b", language="ar", wer=0.0, wer_raw=0.0, reference_words=10,
            hypothesis_words=10, audio_sec=100.0, transcribe_sec=0.0,
        )

        report = TranscriptionReport(calls=[timed, resumed])

        assert report.realtime_factor() == pytest.approx(2.0)

    def test_resumed_calls_still_count_toward_wer(self) -> None:
        """Excluding them from timing must not exclude them from accuracy."""
        resumed = CallWer(
            call_id="b", language="mixed", wer=0.5, wer_raw=0.6, reference_words=10,
            hypothesis_words=10, audio_sec=100.0, transcribe_sec=0.0,
        )

        report = TranscriptionReport(calls=[resumed])

        assert report.wer_by_category()[Language.MIXED] == pytest.approx(0.5)


class TestExtractWithRetry:
    """Surviving transient provider errors on a 150-call run."""

    def test_returns_on_first_success(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A healthy provider means exactly one attempt."""
        from sawti.eval.experiments import asr_propagation as module

        calls = {"n": 0}
        sentinel = object()

        def fake_extract(path: Path, *, stats: object = None) -> object:
            calls["n"] += 1
            return sentinel

        monkeypatch.setattr(module, "extract_via_graph", fake_extract)

        assert module.extract_with_retry(tmp_path / "call_0000_ar.txt") is sentinel
        assert calls["n"] == 1

    def test_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A 503 on the first attempt must not drop the call."""
        from sawti.eval.experiments import asr_propagation as module

        calls = {"n": 0}
        sentinel = object()

        def flaky(path: Path, *, stats: object = None) -> object:
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("503 UNAVAILABLE")
            return sentinel

        monkeypatch.setattr(module, "extract_via_graph", flaky)

        result = module.extract_with_retry(tmp_path / "c.txt", backoff_sec=0.0)

        assert result is sentinel
        assert calls["n"] == 3

    def test_reraises_last_error_after_exhausting_attempts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A permanently failing provider still fails, with its own error."""
        from sawti.eval.experiments import asr_propagation as module

        def always_fails(path: Path, *, stats: object = None) -> object:
            raise RuntimeError("503 UNAVAILABLE")

        monkeypatch.setattr(module, "extract_via_graph", always_fails)

        with pytest.raises(RuntimeError, match="503"):
            module.extract_with_retry(tmp_path / "c.txt", attempts=2, backoff_sec=0.0)

    def test_gate_stats_counted_once_despite_retries(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The bug this guards: extract_via_graph tallies before it raises.

        Without the scratch-and-merge, a call that failed twice would
        contribute its proposed/rejected counts three times.
        """
        from sawti.eval.experiments import asr_propagation as module
        from sawti.eval.experiments.grounded_graph import GateStats
        from sawti.schemas import Language

        calls = {"n": 0}

        def flaky(path: Path, *, stats: GateStats | None = None) -> object:
            calls["n"] += 1
            if stats is not None:
                stats.record(Language.AR, proposed=10, rejected=2, escalated=False)
            if calls["n"] < 3:
                raise RuntimeError("503")
            return object()

        monkeypatch.setattr(module, "extract_via_graph", flaky)
        real = GateStats()

        module.extract_with_retry(tmp_path / "c.txt", stats=real, backoff_sec=0.0)

        assert real.calls == 1
        assert real.proposed == 10
        assert real.rejected == 2
        assert real.proposed_by_language[Language.AR] == 10
