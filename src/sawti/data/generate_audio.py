"""Synthesize call audio from the Phase 1 transcripts, with speaker ground truth.

Phase 3: builds the audio corpus that `sawti.asr` is evaluated against.

There is no recorded audio for this project — the Phase 1 corpus is text only.
This module renders each transcript to speech so Whisper and the diarizer have
something real to run against, and, critically, emits the *reference* speaker
timeline alongside it.

How the ground truth stays exact
--------------------------------
Each turn is synthesized as its own audio file and the turns are then
concatenated sample-by-sample at a fixed sample rate. Segment boundaries are
therefore computed from exact frame counts rather than estimated from the
mixed-down file, so `AudioSegment.start_sec` / `end_sec` are accurate to the
sample. A fixed `GAP_SEC` of digital silence separates turns, which is what
gives the diarizer an acoustic boundary to find.

What this audio is and is not
-----------------------------
It is synthetic speech, not recorded telephony: no channel noise, no codec
artifacts, no crosstalk, no overlapping speech, and exactly one voice per
speaker for the whole corpus. Diarization scored on it is therefore an
optimistic bound — the two speakers in every call differ in gender, which is
the easiest case a diarizer ever sees. WER is likewise optimistic relative to
real call-center audio. Both caveats are recorded in `docs/09-DECISIONS.md`
and repeated next to the numbers in `eval_results.md`.

Voice choice is per language category; see `VOICES` and `docs/09-DECISIONS.md`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import subprocess
import wave
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from sawti.data.transcript_parser import Speaker, Turn, parse_transcript
from sawti.schemas import Language

logger = logging.getLogger(__name__)

DEFAULT_SYNTHETIC_DIR = Path("data/synthetic")
DEFAULT_GROUND_TRUTH_DIR = Path("data/ground_truth")
DEFAULT_AUDIO_DIR = Path("data/audio")
DEFAULT_MANIFEST_PATH = DEFAULT_AUDIO_DIR / "manifest.json"

#: Whisper resamples everything to 16 kHz mono internally; emitting it directly
#: avoids a lossy round trip and keeps frame arithmetic simple.
SAMPLE_RATE_HZ = 16_000
SAMPLE_WIDTH_BYTES = 2
CHANNELS = 1

#: Digital silence inserted between turns, in seconds. Long enough to give the
#: diarizer a real boundary, short enough to stay conversationally plausible.
GAP_SEC = 0.3

#: Transient-failure policy for the TTS endpoint. edge-tts talks to a public
#: Microsoft service over the network; a corpus run is long enough that a DNS
#: blip or a dropped connection part-way through is expected rather than
#: exceptional, and losing an hour of completed work to one is not acceptable.
TTS_MAX_ATTEMPTS = 5
TTS_BACKOFF_SEC = 2.0

#: Voice per language category and speaker role.
#:
#: `ar` uses the Jordanian (ar-JO) pair, matching the dialect the Phase 1
#: corpus was generated in. `mixed` uses the multilingual voices because the
#: code-switched turns carry Arabic script and English in the same sentence,
#: which the ar-JO and plain en-US voices both mangle.
VOICES: dict[Language, dict[Speaker, str]] = {
    Language.AR: {"Agent": "ar-JO-TaimNeural", "Customer": "ar-JO-SanaNeural"},
    Language.EN: {"Agent": "en-US-AndrewNeural", "Customer": "en-US-AvaNeural"},
    Language.MIXED: {
        "Agent": "en-US-AndrewMultilingualNeural",
        "Customer": "en-US-AvaMultilingualNeural",
    },
}


class AudioSegment(BaseModel):
    """One synthesized turn, with its exact position in the concatenated audio."""

    speaker: Speaker
    start_sec: float
    end_sec: float
    text: str
    source_line: int


class AudioManifestEntry(BaseModel):
    """Everything needed to evaluate one synthesized call.

    Ties the audio to the reference transcript it was rendered from, the Phase
    1/2 reference analysis for the same call, and the true speaker timeline.
    """

    call_id: str
    language: Language
    audio_path: str
    reference_transcript_path: str
    reference_analysis_path: str | None
    duration_sec: float
    sample_rate_hz: int = SAMPLE_RATE_HZ
    voices: dict[str, str]
    segments: list[AudioSegment]

    @property
    def reference_text(self) -> str:
        """Concatenated turn text, in order — the WER reference for this call."""
        return " ".join(segment.text for segment in self.segments)


class AudioManifest(BaseModel):
    """The synthesized audio corpus, mapping audio to its reference material."""

    generated_at: str
    tts_provider: str = "edge-tts"
    sample_rate_hz: int = SAMPLE_RATE_HZ
    gap_sec: float = GAP_SEC
    entries: list[AudioManifestEntry] = Field(default_factory=list)


def language_of(call_id: str) -> Language:
    """Infer the language category from a `call_NNNN_<lang>` identifier.

    Args:
        call_id: Call identifier, e.g. `call_0003_mixed`.

    Returns:
        The `Language` category named by the identifier's suffix.

    Raises:
        ValueError: If the suffix is not a known `Language`.
    """
    suffix = call_id.rsplit("_", 1)[-1]
    return Language(suffix)


def _require_ffmpeg() -> str:
    """Return the ffmpeg executable path, or raise a actionable error."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. It decodes the TTS output into PCM. "
            "Install it with `brew install ffmpeg`."
        )
    return ffmpeg


def _decode_to_pcm(mp3_bytes: bytes, ffmpeg: str) -> bytes:
    """Decode MP3 bytes to raw mono PCM16 at `SAMPLE_RATE_HZ`.

    Args:
        mp3_bytes: Encoded audio as returned by the TTS provider.
        ffmpeg: Path to the ffmpeg executable.

    Returns:
        Raw little-endian PCM16 frames, without a WAV header.

    Raises:
        RuntimeError: If ffmpeg fails to decode the input.
    """
    result = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE_HZ), "-ac", str(CHANNELS),
            "pipe:1",
        ],
        input=mp3_bytes,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode TTS audio: {result.stderr.decode()[:400]}")
    return result.stdout


async def _synthesize_turn(text: str, voice: str) -> bytes:
    """Synthesize one turn to MP3 bytes via edge-tts.

    Args:
        text: The utterance to speak.
        voice: edge-tts voice short name, e.g. `ar-JO-TaimNeural`.

    Returns:
        MP3-encoded audio bytes.

    Raises:
        RuntimeError: If the provider returns no audio for the turn.
    """
    import edge_tts

    last_error: Exception | None = None
    for attempt in range(1, TTS_MAX_ATTEMPTS + 1):
        try:
            communicate = edge_tts.Communicate(text, voice)
            buffer = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buffer.extend(chunk["data"])
            if buffer:
                return bytes(buffer)
            last_error = RuntimeError(
                f"edge-tts returned no audio for voice={voice} text={text[:60]!r}"
            )
        except Exception as error:  # network faults surface as many exception types
            last_error = error

        if attempt < TTS_MAX_ATTEMPTS:
            delay = TTS_BACKOFF_SEC * (2 ** (attempt - 1))
            logger.warning(
                "TTS attempt %d/%d failed (%s); retrying in %.0fs",
                attempt, TTS_MAX_ATTEMPTS, last_error, delay,
            )
            await asyncio.sleep(delay)

    raise RuntimeError(
        f"edge-tts failed after {TTS_MAX_ATTEMPTS} attempts for voice={voice}: {last_error}"
    )


def _write_wav(path: Path, pcm: bytes) -> None:
    """Write raw PCM16 frames to `path` as a mono WAV at `SAMPLE_RATE_HZ`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH_BYTES)
        handle.setframerate(SAMPLE_RATE_HZ)
        handle.writeframes(pcm)


async def synthesize_call(
    call_id: str,
    turns: list[Turn],
    audio_dir: Path,
    *,
    concurrency: int = 6,
) -> AudioManifestEntry:
    """Synthesize one call's audio and compute its true speaker timeline.

    Turns are synthesized concurrently but concatenated strictly in order, so
    segment offsets follow the transcript regardless of completion order.

    Args:
        call_id: Call identifier, e.g. `call_0003_mixed`.
        turns: Parsed speaker turns, in order.
        audio_dir: Directory to write `<call_id>.wav` into.
        concurrency: Maximum simultaneous TTS requests for this call.

    Returns:
        The manifest entry describing the written audio.
    """
    ffmpeg = _require_ffmpeg()
    language = language_of(call_id)
    voices = VOICES[language]

    semaphore = asyncio.Semaphore(concurrency)

    async def render(turn: Turn) -> bytes:
        async with semaphore:
            mp3 = await _synthesize_turn(turn.text, voices[turn.speaker])
        return _decode_to_pcm(mp3, ffmpeg)

    pcm_per_turn = await asyncio.gather(*(render(turn) for turn in turns))

    gap = b"\x00" * (int(GAP_SEC * SAMPLE_RATE_HZ) * SAMPLE_WIDTH_BYTES)
    frames_per_sec = SAMPLE_RATE_HZ * SAMPLE_WIDTH_BYTES

    combined = bytearray()
    segments: list[AudioSegment] = []
    for turn, pcm in zip(turns, pcm_per_turn, strict=True):
        start_sec = len(combined) / frames_per_sec
        combined.extend(pcm)
        end_sec = len(combined) / frames_per_sec
        segments.append(
            AudioSegment(
                speaker=turn.speaker,
                start_sec=round(start_sec, 4),
                end_sec=round(end_sec, 4),
                text=turn.text,
                source_line=turn.source_line,
            )
        )
        combined.extend(gap)

    audio_path = audio_dir / f"{call_id}.wav"
    _write_wav(audio_path, bytes(combined))

    reference_analysis = DEFAULT_GROUND_TRUTH_DIR / f"{call_id}.json"
    return AudioManifestEntry(
        call_id=call_id,
        language=language,
        audio_path=str(audio_path),
        reference_transcript_path=str(DEFAULT_SYNTHETIC_DIR / f"{call_id}.txt"),
        reference_analysis_path=str(reference_analysis) if reference_analysis.exists() else None,
        duration_sec=round(len(combined) / frames_per_sec, 4),
        voices=dict(voices),
        segments=segments,
    )


def _load_completed(progress_path: Path, audio_dir: Path) -> dict[str, AudioManifestEntry]:
    """Read previously completed entries from the progress sidecar.

    An entry counts as completed only if its audio file is still on disk, so a
    deleted wav is re-synthesized rather than silently referenced by a manifest
    that points at nothing.

    Args:
        progress_path: JSONL sidecar written during the run.
        audio_dir: Directory the audio was written into.

    Returns:
        Completed entries by call id. Empty if the sidecar does not exist.
    """
    if not progress_path.is_file():
        return {}

    completed: dict[str, AudioManifestEntry] = {}
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = AudioManifestEntry.model_validate_json(line)
        except ValueError:
            logger.warning("skipping unreadable progress line in %s", progress_path)
            continue
        if audio_dir.joinpath(f"{entry.call_id}.wav").is_file():
            completed[entry.call_id] = entry
    return completed


async def generate_audio_corpus(
    call_ids: list[str],
    *,
    synthetic_dir: Path = DEFAULT_SYNTHETIC_DIR,
    audio_dir: Path = DEFAULT_AUDIO_DIR,
    progress_path: Path | None = None,
    resume: bool = True,
) -> AudioManifest:
    """Synthesize audio for `call_ids` and return the resulting manifest.

    Calls are processed one at a time (turns within a call run concurrently) to
    keep the load on the TTS endpoint modest and failures easy to attribute.

    Each completed call is appended to a JSONL progress sidecar before the next
    one starts. A corpus run takes the better part of an hour against a public
    network service, so it will sometimes die part-way; with the sidecar a
    re-run resumes instead of re-synthesizing everything, and the manifest is
    still assembled from complete records.

    Args:
        call_ids: Call identifiers to synthesize.
        synthetic_dir: Directory holding `<call_id>.txt` transcripts.
        audio_dir: Directory to write audio into.
        progress_path: JSONL sidecar for per-call progress. Defaults to
            `<audio_dir>/.progress.jsonl`.
        resume: Skip calls already recorded in the sidecar with audio on disk.

    Returns:
        An `AudioManifest` covering every successfully synthesized call, in
        `call_ids` order.
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    progress_path = progress_path or audio_dir / ".progress.jsonl"

    completed = _load_completed(progress_path, audio_dir) if resume else {}
    if completed:
        logger.info("resuming: %d call(s) already synthesized", len(completed))

    for index, call_id in enumerate(call_ids, start=1):
        if call_id in completed:
            logger.info("[%d/%d] %s: already done, skipping", index, len(call_ids), call_id)
            continue

        raw = synthetic_dir.joinpath(f"{call_id}.txt").read_text(encoding="utf-8")
        turns = parse_transcript(raw, call_id)
        entry = await synthesize_call(call_id, turns, audio_dir)
        completed[call_id] = entry

        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(entry.model_dump_json() + "\n")

        logger.info(
            "[%d/%d] %s: %d turns, %.1fs", index, len(call_ids), call_id, len(turns), entry.duration_sec
        )

    return AudioManifest(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        entries=[completed[call_id] for call_id in call_ids if call_id in completed],
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize call audio from Phase 1 transcripts.")
    parser.add_argument("--synthetic-dir", type=Path, default=DEFAULT_SYNTHETIC_DIR)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Synthesize only the first N calls.")
    parser.add_argument("--progress", type=Path, default=None, help="JSONL progress sidecar path.")
    parser.add_argument(
        "--no-resume", action="store_true", help="Re-synthesize calls even if already completed."
    )
    parser.add_argument(
        "--calls", nargs="*", default=None, help="Explicit call ids to synthesize (overrides --limit)."
    )
    parser.add_argument(
        "--per-language",
        type=int,
        default=None,
        help="Synthesize the first N calls of each language category.",
    )
    return parser.parse_args(argv)


def _select_call_ids(args: argparse.Namespace) -> list[str]:
    """Resolve the CLI selection flags to an ordered list of call ids."""
    all_ids = sorted(path.stem for path in args.synthetic_dir.glob("*.txt"))
    if args.calls:
        return list(args.calls)
    if args.per_language:
        selected: list[str] = []
        for language in Language:
            matching = [call_id for call_id in all_ids if language_of(call_id) is language]
            selected.extend(matching[: args.per_language])
        return selected
    if args.limit:
        return all_ids[: args.limit]
    return all_ids


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: synthesize audio and write the manifest."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    call_ids = _select_call_ids(args)

    manifest = asyncio.run(
        generate_audio_corpus(
            call_ids,
            synthetic_dir=args.synthetic_dir,
            audio_dir=args.audio_dir,
            progress_path=args.progress,
            resume=not args.no_resume,
        )
    )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    total_sec = sum(entry.duration_sec for entry in manifest.entries)
    logger.info(
        "Wrote %d calls (%.1f min of audio) and manifest -> %s",
        len(manifest.entries), total_sec / 60, args.manifest,
    )


if __name__ == "__main__":
    main()
