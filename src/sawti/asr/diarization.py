"""Speaker diarization for call recordings.

Phase 3: ASR pipeline, speaker attribution.

Whisper returns *what* was said with timings but not *who* said it. A QA rubric
that scores agent behaviour needs the distinction — "did the agent apologize"
is unanswerable from an unattributed transcript. This module runs pyannote's
pretrained diarization pipeline over the audio and aligns its speaker turns
onto the ASR segments by time overlap.

Gated weights
-------------
pyannote's pretrained pipelines are gated on HuggingFace. Running `diarize()`
requires a token in `HF_TOKEN` (see `.env.example`) belonging to an account
that has accepted the licence conditions on **both**
`pyannote/segmentation-3.0` and `pyannote/speaker-diarization-3.1` — accepting
only the pipeline repo is the usual cause of a 401 here. `diarize()` raises
`DiarizationUnavailableError` with those instructions rather than failing deep
inside the pyannote loader.

Role assignment
---------------
pyannote emits anonymous cluster labels (`SPEAKER_00`, `SPEAKER_01`) with no
inherent meaning; which cluster is the agent is not something acoustics can
tell you. `assign_roles_by_first_speaker` applies the contact-centre
convention that the agent opens the call. That is a convention, not a
detection, and it is wrong for any call the customer opens — it is used for
the diarization sanity check in `eval_results.md` and flagged as a known
limitation there.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from sawti.asr.transcribe import TranscriptSegment
from sawti.config import get_settings

logger = logging.getLogger(__name__)

#: The gated pretrained pipeline this module targets.
DEFAULT_PIPELINE = "pyannote/speaker-diarization-3.1"

_TOKEN_HELP = (
    "pyannote's pretrained pipelines are gated. Set HF_TOKEN in .env to a "
    "HuggingFace read token from an account that has accepted the licence on "
    "BOTH https://huggingface.co/pyannote/segmentation-3.0 and "
    "https://huggingface.co/pyannote/speaker-diarization-3.1"
)


class DiarizationUnavailableError(RuntimeError):
    """Raised when the gated pyannote pipeline cannot be loaded."""


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


def _load_pipeline(pipeline_name: str, token: str | None) -> Any:
    """Load the pretrained pyannote pipeline, or raise with instructions.

    Args:
        pipeline_name: HuggingFace pipeline identifier.
        token: HuggingFace access token, or `None`.

    Returns:
        The loaded `pyannote.audio.Pipeline`.

    Raises:
        DiarizationUnavailableError: If pyannote is missing, the token is
            absent, or the gated weights are refused.
    """
    if not token:
        raise DiarizationUnavailableError(f"HF_TOKEN is not set. {_TOKEN_HELP}")

    try:
        from pyannote.audio import Pipeline
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise DiarizationUnavailableError(
            "pyannote.audio is not installed. Install the ASR extra: uv sync --extra asr"
        ) from error

    try:
        pipeline = Pipeline.from_pretrained(pipeline_name, use_auth_token=token)
    except Exception as error:  # pyannote surfaces bare HTTP errors from the hub
        raise DiarizationUnavailableError(
            f"Could not load {pipeline_name}: {error}. {_TOKEN_HELP}"
        ) from error

    if pipeline is None:
        # pyannote returns None rather than raising when the licence is unaccepted.
        raise DiarizationUnavailableError(f"{pipeline_name} returned no pipeline. {_TOKEN_HELP}")
    return pipeline


def diarize(
    audio_path: Path,
    *,
    pipeline_name: str = DEFAULT_PIPELINE,
    num_speakers: int | None = 2,
) -> list[DiarizedSegment]:
    """Run speaker diarization over `audio_path`.

    Args:
        audio_path: Path to the call audio file.
        pipeline_name: Pretrained pyannote pipeline to use.
        num_speakers: Expected speaker count, passed to the pipeline as a
            constraint. Defaults to 2 — a contact-centre call is an agent and
            a customer — which is a much easier problem than free clustering.
            Pass `None` to let the pipeline decide.

    Returns:
        Diarized segments, ordered by start time.

    Raises:
        FileNotFoundError: If `audio_path` does not exist.
        DiarizationUnavailableError: If the gated pipeline cannot be loaded.
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    pipeline = _load_pipeline(pipeline_name, get_settings().hf_token)
    annotation = pipeline(str(audio_path), num_speakers=num_speakers)

    segments = [
        DiarizedSegment(speaker=str(speaker), start_sec=float(turn.start), end_sec=float(turn.end))
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    segments.sort(key=lambda segment: segment.start_sec)
    return segments


def _overlap_sec(
    first_start: float, first_end: float, second_start: float, second_end: float
) -> float:
    """Length of the temporal intersection of two intervals, 0.0 if disjoint."""
    return max(0.0, min(first_end, second_end) - max(first_start, second_start))


def merge_transcript_with_speakers(
    transcript_segments: list[TranscriptSegment], diarized_segments: list[DiarizedSegment]
) -> list[SpeakerTranscriptSegment]:
    """Assign speaker labels to transcript segments using diarization output.

    Each ASR segment takes the speaker it shares the most time with. ASR and
    diarization segment the audio independently, so their boundaries rarely
    coincide and a segment routinely touches two speakers; greatest overlap is
    the standard resolution, and it is what the ambiguous case must use.

    A segment overlapping nothing — Whisper emitting text over a stretch the
    diarizer called silence — falls back to the nearest speaker turn by
    midpoint distance, so every segment is attributed. Losing a turn entirely
    would silently drop content from the transcript the agent is scored on.

    Args:
        transcript_segments: ASR segments (text + timing) without speaker labels.
        diarized_segments: Diarized speaker segments to align against.

    Returns:
        Transcript segments annotated with a speaker label each, in input
        order. Empty if `transcript_segments` is empty.

    Raises:
        ValueError: If `diarized_segments` is empty while there are transcript
            segments to attribute — there is no speaker to assign.
    """
    if not transcript_segments:
        return []
    if not diarized_segments:
        raise ValueError("Cannot assign speakers: diarized_segments is empty")

    merged: list[SpeakerTranscriptSegment] = []
    for segment in transcript_segments:
        best = max(
            diarized_segments,
            key=lambda candidate: _overlap_sec(
                segment.start_sec, segment.end_sec, candidate.start_sec, candidate.end_sec
            ),
        )
        overlap = _overlap_sec(
            segment.start_sec, segment.end_sec, best.start_sec, best.end_sec
        )
        if overlap <= 0.0:
            midpoint = (segment.start_sec + segment.end_sec) / 2
            best = min(
                diarized_segments,
                key=lambda candidate: abs(
                    midpoint - (candidate.start_sec + candidate.end_sec) / 2
                ),
            )
        merged.append(
            SpeakerTranscriptSegment(
                text=segment.text,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                speaker=best.speaker,
            )
        )
    return merged


def assign_roles_by_first_speaker(
    segments: list[SpeakerTranscriptSegment],
) -> list[SpeakerTranscriptSegment]:
    """Relabel anonymous diarization clusters as `Agent` / `Customer`.

    Applies the contact-centre convention that the agent speaks first: the
    cluster owning the earliest segment becomes `Agent`, every other cluster
    becomes `Customer`. This is a convention, not speaker identification, and
    it mislabels any call the customer opens.

    Args:
        segments: Speaker-annotated segments carrying pyannote cluster labels.

    Returns:
        The same segments with `speaker` remapped to `Agent` or `Customer`.
    """
    if not segments:
        return []

    first_speaker = min(segments, key=lambda segment: segment.start_sec).speaker
    return [
        segment.model_copy(
            update={"speaker": "Agent" if segment.speaker == first_speaker else "Customer"}
        )
        for segment in segments
    ]
