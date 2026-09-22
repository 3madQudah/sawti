"""Tests for `sawti.asr.transcribe`.

Phase 3: mirrors `src/sawti/asr/transcribe.py`.

Whisper itself is stubbed out here. These tests pin the contract this module
owns — that the language is forced rather than detected, that segments come
back ordered, that a missing file fails cleanly — not Whisper's accuracy, which
is measured against real audio in `sawti.eval.wer` and `eval_results.md`.
"""

from pathlib import Path
from typing import Any

import pytest

from sawti.asr import transcribe as transcribe_module
from sawti.asr.transcribe import AUTO_DETECT, Transcript, transcribe
from sawti.config import get_settings
from sawti.schemas import Language


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    """An existing (empty) file standing in for a call recording."""
    path = tmp_path / "call_0000_ar.wav"
    path.write_bytes(b"")
    return path


@pytest.fixture
def whisper_spy(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace both Whisper backends with a recording stub.

    Returns a dict that captures the arguments the backend was called with.
    """
    recorded: dict[str, Any] = {}
    result: dict[str, Any] = {
        "language": "ar",
        "segments": [
            {"text": "second", "start": 5.0, "end": 9.0},
            {"text": "first", "start": 0.0, "end": 4.0},
        ],
    }

    def fake_runner(audio_path: Path, model: str, language: str | None) -> dict[str, Any]:
        recorded["audio_path"] = audio_path
        recorded["model"] = model
        recorded["language"] = language
        return result

    monkeypatch.setattr(transcribe_module, "_run_mlx", fake_runner)
    monkeypatch.setattr(transcribe_module, "_run_openai_whisper", fake_runner)
    recorded["result"] = result
    return recorded


def test_transcribe_forces_arabic_language_by_default(
    audio_file: Path, whisper_spy: dict[str, Any]
) -> None:
    """transcribe() calls Whisper with language='ar' when no override is given."""
    transcribe(audio_file)

    assert whisper_spy["language"] == "ar"


def test_transcribe_passes_explicit_language_override(
    audio_file: Path, whisper_spy: dict[str, Any]
) -> None:
    """An explicit `language` is forwarded to the backend as the forced language."""
    transcribe(audio_file, language="en")

    assert whisper_spy["language"] == "en"


def test_auto_detect_passes_none_so_whisper_detects(
    audio_file: Path, whisper_spy: dict[str, Any]
) -> None:
    """AUTO_DETECT is the only path that stops forcing a language."""
    transcribe(audio_file, language=AUTO_DETECT)

    assert whisper_spy["language"] is None


def test_transcribe_returns_segments_ordered_by_start_time(
    audio_file: Path, whisper_spy: dict[str, Any]
) -> None:
    """transcribe() returns TranscriptSegments in non-decreasing start_sec order."""
    result = transcribe(audio_file)

    starts = [segment.start_sec for segment in result.segments]
    assert starts == sorted(starts)
    assert [segment.text for segment in result.segments] == ["first", "second"]


def test_transcribe_raises_on_missing_audio_file(tmp_path: Path) -> None:
    """transcribe() raises when audio_path does not exist."""
    with pytest.raises(FileNotFoundError):
        transcribe(tmp_path / "does_not_exist.wav")


def test_transcribe_uses_configured_model_by_default(
    audio_file: Path, whisper_spy: dict[str, Any]
) -> None:
    """The model name comes from Settings rather than being hardcoded."""
    from sawti.config import get_settings

    transcribe(audio_file)

    assert whisper_spy["model"] == get_settings().whisper_model


def test_transcribe_model_override_wins(audio_file: Path, whisper_spy: dict[str, Any]) -> None:
    """An explicit `model` overrides the configured default."""
    transcribe(audio_file, model="tiny")

    assert whisper_spy["model"] == "tiny"


def test_call_id_is_taken_from_the_audio_filename(
    audio_file: Path, whisper_spy: dict[str, Any]
) -> None:
    """call_id lets a Transcript be paired back to its manifest entry."""
    result = transcribe(audio_file)

    assert result.call_id == "call_0000_ar"


def test_transcript_text_joins_segments_in_order() -> None:
    """Transcript.text is the WER hypothesis: segments joined in order."""
    transcript = Transcript(
        call_id="call_0000_ar",
        language="ar",
        segments=[
            transcribe_module.TranscriptSegment(text=" first ", start_sec=0.0, end_sec=1.0),
            transcribe_module.TranscriptSegment(text="", start_sec=1.0, end_sec=2.0),
            transcribe_module.TranscriptSegment(text="second", start_sec=2.0, end_sec=3.0),
        ],
    )

    assert transcript.text == "first second"


class TestPerCategoryForcing:
    """Forced language resolved per language category, not one global value."""

    def test_english_category_decodes_as_english(
        self, audio_file: Path, whisper_spy: dict[str, Any]
    ) -> None:
        """The whole point: en calls must not go through the Arabic decoder."""
        transcribe(audio_file, category=Language.EN)

        assert whisper_spy["language"] == "en"

    @pytest.mark.parametrize("category", [Language.AR, Language.MIXED])
    def test_arabic_and_mixed_stay_on_the_arabic_decoder(
        self, audio_file: Path, whisper_spy: dict[str, Any], category: Language
    ) -> None:
        """ar is unchanged, and mixed stays forced to ar deliberately."""
        transcribe(audio_file, category=category)

        assert whisper_spy["language"] == "ar"

    def test_explicit_language_still_wins_over_category(
        self, audio_file: Path, whisper_spy: dict[str, Any]
    ) -> None:
        """An explicit override beats the category default."""
        transcribe(audio_file, language="fr", category=Language.EN)

        assert whisper_spy["language"] == "fr"

    def test_auto_detect_still_wins_over_category(
        self, audio_file: Path, whisper_spy: dict[str, Any]
    ) -> None:
        """The forcing-comparison experiment must still be able to opt out."""
        transcribe(audio_file, language=AUTO_DETECT, category=Language.EN)

        assert whisper_spy["language"] is None

    def test_no_category_falls_back_to_the_global_setting(
        self, audio_file: Path, whisper_spy: dict[str, Any]
    ) -> None:
        """Callers that pass no category keep the previous behaviour."""
        transcribe(audio_file)

        assert whisper_spy["language"] == get_settings().whisper_language
