"""Tests for `sawti.eval.experiments.asr_language_forcing`.

Phase 3: mirrors `src/sawti/eval/experiments/asr_language_forcing.py`.

Whisper is stubbed. These cover the experiment's bookkeeping — that both arms
are actually run with different language settings, and that the per-category
summary stays per-category — not the WER values themselves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sawti.asr.transcribe import AUTO_DETECT, Transcript, TranscriptSegment
from sawti.eval.experiments import asr_language_forcing as forcing_module
from sawti.eval.experiments.asr_language_forcing import (
    CallComparison,
    ForcingReport,
    compare_call,
    summarize,
)
from sawti.schemas import Language


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    """An existing file standing in for a call recording."""
    path = tmp_path / "call_0003_mixed.wav"
    path.write_bytes(b"")
    return path


@pytest.fixture
def transcribe_spy(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """Record the `language` each transcribe() arm was called with."""
    seen: list[str | None] = []

    def fake_transcribe(path: Path, *, language: str | None = None, model: str | None = None):
        seen.append(language)
        detected = "en" if language == AUTO_DETECT else str(language)
        return Transcript(
            call_id=Path(path).stem,
            language=detected,
            segments=[TranscriptSegment(text="hello there", start_sec=0.0, end_sec=1.0)],
        )

    monkeypatch.setattr(forcing_module, "transcribe", fake_transcribe)
    return seen


class TestCompareCall:
    """One call, both arms."""

    def test_runs_both_a_forced_and_a_detected_arm(
        self, audio_file: Path, transcribe_spy: list[str | None]
    ) -> None:
        """The comparison is only meaningful if both arms actually differ."""
        compare_call(audio_file, "hello there", Language.MIXED)

        assert transcribe_spy == ["ar", AUTO_DETECT]

    def test_records_what_the_detected_arm_detected(
        self, audio_file: Path, transcribe_spy: list[str | None]
    ) -> None:
        """A mixed call decoded as 'en' is the failure mode being looked for."""
        comparison = compare_call(audio_file, "hello there", Language.MIXED)

        assert comparison.forced_language == "ar"
        assert comparison.detected_language == "en"

    def test_honours_an_alternative_forced_language(
        self, audio_file: Path, transcribe_spy: list[str | None]
    ) -> None:
        """The forced arm is configurable, for checking ar/en for regressions."""
        compare_call(audio_file, "hello there", Language.EN, forced_language="en")

        assert transcribe_spy[0] == "en"

    def test_scores_both_normalized_and_raw(
        self, audio_file: Path, transcribe_spy: list[str | None]
    ) -> None:
        """Raw WER is carried alongside so normalization cannot hide anything."""
        comparison = compare_call(audio_file, "hello there", Language.MIXED)

        assert comparison.forced_wer == 0.0
        assert comparison.forced_wer_raw == 0.0


class TestDetectedLanguageCounts:
    """What the detection arm actually chose."""

    def _comparison(self, language: Language, detected: str) -> CallComparison:
        return CallComparison(
            call_id="c",
            language=language,
            forced_language="ar",
            detected_language=detected,
            forced_wer=0.4,
            detected_wer=0.3,
            forced_wer_raw=0.5,
            detected_wer_raw=0.4,
            forced_sec=1.0,
            detected_sec=1.0,
        )

    def test_counts_detections_per_category(self) -> None:
        """Detection spread is the qualitative half of this experiment."""
        report = ForcingReport(
            comparisons=[
                self._comparison(Language.MIXED, "en"),
                self._comparison(Language.MIXED, "en"),
                self._comparison(Language.MIXED, "ar"),
            ]
        )

        assert report.detected_language_counts()[Language.MIXED] == {"en": 2, "ar": 1}

    def test_keeps_categories_separate(self) -> None:
        """ar detections must not be pooled with mixed ones."""
        report = ForcingReport(
            comparisons=[
                self._comparison(Language.AR, "ar"),
                self._comparison(Language.MIXED, "en"),
            ]
        )

        counts = report.detected_language_counts()

        assert counts[Language.AR] == {"ar": 1}
        assert counts[Language.MIXED] == {"en": 1}


class TestSummarize:
    """The per-category rollup."""

    def test_reports_each_category_separately(self) -> None:
        """Never one blended number — the project's standing rule."""
        report = ForcingReport(
            comparisons=[
                CallComparison(
                    call_id="a", language=Language.MIXED, forced_language="ar",
                    detected_language="en", forced_wer=0.4, detected_wer=0.2,
                    forced_wer_raw=0.5, detected_wer_raw=0.3, forced_sec=1.0, detected_sec=1.0,
                ),
                CallComparison(
                    call_id="b", language=Language.AR, forced_language="ar",
                    detected_language="ar", forced_wer=0.1, detected_wer=0.1,
                    forced_wer_raw=0.2, detected_wer_raw=0.2, forced_sec=1.0, detected_sec=1.0,
                ),
            ]
        )

        summary: dict[str, Any] = summarize(report)  # type: ignore[assignment]

        assert set(summary) == {"mixed", "ar"}
        assert summary["mixed"]["forced_wer"] == 0.4
        assert summary["mixed"]["detected_wer"] == 0.2
        assert summary["ar"]["calls"] == 1

    def test_omits_categories_with_no_calls(self) -> None:
        """A category that was not run must not appear as a zero."""
        report = ForcingReport(
            comparisons=[
                CallComparison(
                    call_id="a", language=Language.MIXED, forced_language="ar",
                    detected_language="en", forced_wer=0.4, detected_wer=0.2,
                    forced_wer_raw=0.5, detected_wer_raw=0.3, forced_sec=1.0, detected_sec=1.0,
                )
            ]
        )

        assert set(summarize(report)) == {"mixed"}

    def test_empty_report_summarizes_to_nothing(self) -> None:
        """No calls, no categories."""
        assert summarize(ForcingReport(comparisons=[])) == {}


class TestCompareAgainstRecordedForced:
    """Reusing an already-measured forced arm instead of re-transcribing it."""

    def test_runs_only_the_detection_arm(
        self, audio_file: Path, transcribe_spy: list[str | None]
    ) -> None:
        """The forced arm must not be transcribed again."""
        forcing_module.compare_against_recorded_forced(
            audio_file, "hello there", Language.MIXED, forced_wer=0.4, forced_wer_raw=0.5
        )

        assert transcribe_spy == [AUTO_DETECT]

    def test_carries_the_recorded_forced_numbers_through(
        self, audio_file: Path, transcribe_spy: list[str | None]
    ) -> None:
        """The reused figures appear unchanged in the comparison."""
        comparison = forcing_module.compare_against_recorded_forced(
            audio_file, "hello there", Language.MIXED, forced_wer=0.4, forced_wer_raw=0.5
        )

        assert comparison.forced_wer == 0.4
        assert comparison.forced_wer_raw == 0.5
        assert comparison.forced_sec == 0.0

    def test_still_scores_the_detection_arm_itself(
        self, audio_file: Path, transcribe_spy: list[str | None]
    ) -> None:
        """The detected arm is measured here, not reused."""
        comparison = forcing_module.compare_against_recorded_forced(
            audio_file, "hello there", Language.MIXED, forced_wer=0.4, forced_wer_raw=0.5
        )

        assert comparison.detected_language == "en"
        assert comparison.detected_wer == 0.0
