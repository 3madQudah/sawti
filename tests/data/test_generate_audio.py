"""Tests for `sawti.data.generate_audio`.

Phase 3: mirrors `src/sawti/data/generate_audio.py`.

The TTS provider is stubbed: these tests pin the part that matters for
evaluation correctness — that the manifest's speaker timeline lines up with
the audio actually written — not edge-tts's voice quality.
"""

from __future__ import annotations

import asyncio
import wave
from pathlib import Path

import pytest

from sawti.data import generate_audio as generate_audio_module
from sawti.data.generate_audio import (
    GAP_SEC,
    SAMPLE_RATE_HZ,
    VOICES,
    AudioManifestEntry,
    AudioSegment,
    language_of,
    synthesize_call,
)
from sawti.data.transcript_parser import Turn
from sawti.schemas import Language

#: Bytes-per-second of the mono PCM16 stream the generator writes.
_BYTES_PER_SEC = SAMPLE_RATE_HZ * 2


@pytest.fixture
def stub_tts(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Replace TTS + decode with a stub producing one second of silence per turn."""
    recorded: dict[str, list[str]] = {"voices": [], "texts": []}

    async def fake_synthesize(text: str, voice: str) -> bytes:
        recorded["texts"].append(text)
        recorded["voices"].append(voice)
        return b"fake-mp3"

    def fake_decode(mp3_bytes: bytes, ffmpeg: str) -> bytes:
        return b"\x00" * _BYTES_PER_SEC  # exactly 1.0s

    monkeypatch.setattr(generate_audio_module, "_synthesize_turn", fake_synthesize)
    monkeypatch.setattr(generate_audio_module, "_decode_to_pcm", fake_decode)
    monkeypatch.setattr(generate_audio_module, "_require_ffmpeg", lambda: "ffmpeg")
    return recorded


@pytest.fixture
def turns() -> list[Turn]:
    """A three-turn alternating call."""
    return [
        Turn(speaker="Agent", text="hello", source_line=1),
        Turn(speaker="Customer", text="hi back", source_line=2),
        Turn(speaker="Agent", text="goodbye", source_line=3),
    ]


class TestLanguageOf:
    """Call-id suffix to Language category."""

    @pytest.mark.parametrize(
        ("call_id", "expected"),
        [
            ("call_0000_ar", Language.AR),
            ("call_0006_en", Language.EN),
            ("call_0003_mixed", Language.MIXED),
        ],
    )
    def test_infers_language_from_suffix(self, call_id: str, expected: Language) -> None:
        """The corpus encodes the category in the filename."""
        assert language_of(call_id) is expected

    def test_unknown_suffix_raises(self) -> None:
        """An unrecognized category must not silently become a default."""
        with pytest.raises(ValueError):
            language_of("call_0000_klingon")


class TestSynthesizeCall:
    """Audio synthesis and the ground-truth timeline it emits."""

    def test_segment_boundaries_match_the_written_audio(
        self, tmp_path: Path, turns: list[Turn], stub_tts: dict[str, list[str]]
    ) -> None:
        """The manifest timeline must describe the file actually on disk.

        This is the assertion the whole diarization ground truth rests on: if
        these drift, every diarization number is scored against the wrong times.
        """
        entry = asyncio.run(synthesize_call("call_0000_ar", turns, tmp_path))

        with wave.open(str(tmp_path / "call_0000_ar.wav")) as handle:
            actual_duration = handle.getnframes() / handle.getframerate()
            assert handle.getframerate() == SAMPLE_RATE_HZ
            assert handle.getnchannels() == 1

        assert entry.duration_sec == pytest.approx(actual_duration)
        assert entry.segments[-1].end_sec <= actual_duration

    def test_segments_are_gap_separated_and_non_overlapping(
        self, tmp_path: Path, turns: list[Turn], stub_tts: dict[str, list[str]]
    ) -> None:
        """Each turn is 1.0s of stub audio, separated by exactly GAP_SEC."""
        entry = asyncio.run(synthesize_call("call_0000_ar", turns, tmp_path))

        assert [segment.start_sec for segment in entry.segments] == [
            pytest.approx(0.0),
            pytest.approx(1.0 + GAP_SEC),
            pytest.approx(2.0 + 2 * GAP_SEC),
        ]
        for earlier, later in zip(entry.segments, entry.segments[1:], strict=False):
            assert earlier.end_sec <= later.start_sec

    def test_speaker_timeline_preserves_turn_order_and_roles(
        self, tmp_path: Path, turns: list[Turn], stub_tts: dict[str, list[str]]
    ) -> None:
        """Ground-truth speakers follow the transcript, not completion order."""
        entry = asyncio.run(synthesize_call("call_0000_ar", turns, tmp_path))

        assert [segment.speaker for segment in entry.segments] == [
            "Agent",
            "Customer",
            "Agent",
        ]
        assert [segment.text for segment in entry.segments] == ["hello", "hi back", "goodbye"]

    def test_uses_the_two_distinct_voices_for_the_language(
        self, tmp_path: Path, turns: list[Turn], stub_tts: dict[str, list[str]]
    ) -> None:
        """Two acoustically distinct voices is what makes diarization meaningful."""
        asyncio.run(synthesize_call("call_0000_ar", turns, tmp_path))

        assert stub_tts["voices"] == [
            VOICES[Language.AR]["Agent"],
            VOICES[Language.AR]["Customer"],
            VOICES[Language.AR]["Agent"],
        ]
        assert len(set(stub_tts["voices"])) == 2

    def test_every_language_maps_to_two_distinct_voices(self) -> None:
        """No category may render both speakers with the same voice."""
        for language, voices in VOICES.items():
            assert set(voices) == {"Agent", "Customer"}, language
            assert voices["Agent"] != voices["Customer"], language

    def test_links_audio_to_its_reference_transcript(
        self, tmp_path: Path, turns: list[Turn], stub_tts: dict[str, list[str]]
    ) -> None:
        """The manifest is what pairs a WER hypothesis with the right reference."""
        entry = asyncio.run(synthesize_call("call_0000_ar", turns, tmp_path))

        assert entry.call_id == "call_0000_ar"
        assert entry.language is Language.AR
        assert entry.reference_transcript_path.endswith("call_0000_ar.txt")


class TestReferenceText:
    """The WER reference built from the manifest."""

    def test_joins_segment_text_in_order(self) -> None:
        """reference_text is the concatenated turn text, in call order."""
        entry = AudioManifestEntry(
            call_id="call_0000_ar",
            language=Language.AR,
            audio_path="a.wav",
            reference_transcript_path="a.txt",
            reference_analysis_path=None,
            duration_sec=3.0,
            voices={"Agent": "v1", "Customer": "v2"},
            segments=[
                AudioSegment(speaker="Agent", start_sec=0.0, end_sec=1.0, text="first", source_line=1),
                AudioSegment(
                    speaker="Customer", start_sec=1.0, end_sec=2.0, text="second", source_line=2
                ),
            ],
        )

        assert entry.reference_text == "first second"


class TestResumeAndRetry:
    """Surviving a long corpus run against a flaky network service."""

    def test_completed_calls_are_skipped_on_resume(
        self, tmp_path: Path, turns: list[Turn], stub_tts: dict[str, list[str]]
    ) -> None:
        """A second run re-synthesizes nothing that already finished."""
        audio_dir = tmp_path / "audio"
        synthetic_dir = tmp_path / "synthetic"
        synthetic_dir.mkdir()
        synthetic_dir.joinpath("call_0000_ar.txt").write_text(
            "Agent: hello\nCustomer: hi back\nAgent: goodbye\n", encoding="utf-8"
        )

        first = asyncio.run(
            generate_audio_module.generate_audio_corpus(
                ["call_0000_ar"], synthetic_dir=synthetic_dir, audio_dir=audio_dir
            )
        )
        calls_after_first = len(stub_tts["texts"])

        second = asyncio.run(
            generate_audio_module.generate_audio_corpus(
                ["call_0000_ar"], synthetic_dir=synthetic_dir, audio_dir=audio_dir
            )
        )

        assert len(stub_tts["texts"]) == calls_after_first, "resume re-synthesized a finished call"
        assert [entry.call_id for entry in second.entries] == ["call_0000_ar"]
        assert second.entries[0].duration_sec == first.entries[0].duration_sec

    def test_no_resume_forces_resynthesis(
        self, tmp_path: Path, turns: list[Turn], stub_tts: dict[str, list[str]]
    ) -> None:
        """`resume=False` ignores the sidecar."""
        audio_dir = tmp_path / "audio"
        synthetic_dir = tmp_path / "synthetic"
        synthetic_dir.mkdir()
        synthetic_dir.joinpath("call_0000_ar.txt").write_text("Agent: hello\n", encoding="utf-8")

        asyncio.run(
            generate_audio_module.generate_audio_corpus(
                ["call_0000_ar"], synthetic_dir=synthetic_dir, audio_dir=audio_dir
            )
        )
        calls_after_first = len(stub_tts["texts"])

        asyncio.run(
            generate_audio_module.generate_audio_corpus(
                ["call_0000_ar"],
                synthetic_dir=synthetic_dir,
                audio_dir=audio_dir,
                resume=False,
            )
        )

        assert len(stub_tts["texts"]) > calls_after_first

    def test_resume_reruns_a_call_whose_audio_was_deleted(
        self, tmp_path: Path, stub_tts: dict[str, list[str]]
    ) -> None:
        """A manifest must never point at audio that is not there."""
        audio_dir = tmp_path / "audio"
        synthetic_dir = tmp_path / "synthetic"
        synthetic_dir.mkdir()
        synthetic_dir.joinpath("call_0000_ar.txt").write_text("Agent: hello\n", encoding="utf-8")

        asyncio.run(
            generate_audio_module.generate_audio_corpus(
                ["call_0000_ar"], synthetic_dir=synthetic_dir, audio_dir=audio_dir
            )
        )
        (audio_dir / "call_0000_ar.wav").unlink()
        calls_after_first = len(stub_tts["texts"])

        manifest = asyncio.run(
            generate_audio_module.generate_audio_corpus(
                ["call_0000_ar"], synthetic_dir=synthetic_dir, audio_dir=audio_dir
            )
        )

        assert len(stub_tts["texts"]) > calls_after_first
        assert (audio_dir / "call_0000_ar.wav").is_file()
        assert manifest.entries[0].call_id == "call_0000_ar"

    def test_synthesize_retries_transient_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dropped connection is retried rather than killing the run."""
        monkeypatch.setattr(generate_audio_module, "TTS_BACKOFF_SEC", 0.0)
        attempts = {"count": 0}

        class _FakeCommunicate:
            def __init__(self, text: str, voice: str) -> None:
                pass

            async def stream(self):  # type: ignore[no-untyped-def]
                attempts["count"] += 1
                if attempts["count"] < 3:
                    raise ConnectionError("Cannot connect to host speech.platform.bing.com:443")
                yield {"type": "audio", "data": b"ok"}

        monkeypatch.setitem(
            __import__("sys").modules, "edge_tts", type("M", (), {"Communicate": _FakeCommunicate})
        )

        result = asyncio.run(generate_audio_module._synthesize_turn("hello", "voice"))

        assert result == b"ok"
        assert attempts["count"] == 3

    def test_synthesize_gives_up_after_max_attempts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A permanently unreachable endpoint fails loudly, naming the cause."""
        monkeypatch.setattr(generate_audio_module, "TTS_BACKOFF_SEC", 0.0)

        class _AlwaysFails:
            def __init__(self, text: str, voice: str) -> None:
                pass

            async def stream(self):  # type: ignore[no-untyped-def]
                raise ConnectionError("DNS failure")
                yield  # pragma: no cover - unreachable, makes this an async generator

        monkeypatch.setitem(
            __import__("sys").modules, "edge_tts", type("M", (), {"Communicate": _AlwaysFails})
        )

        with pytest.raises(RuntimeError, match="failed after"):
            asyncio.run(generate_audio_module._synthesize_turn("hello", "voice"))
