"""Whisper-based transcription, with Arabic forced as the recognition language.

Phase 3: ASR pipeline entry point.

Why the language is forced
--------------------------
`sawti.config.Settings.whisper_language` defaults to `"ar"` and this module
passes it to Whisper explicitly rather than letting Whisper detect. Whisper
detects language from the first 30 seconds only, so on a code-switched call
that opens in English it commits the whole call to English decoding and
transliterates the Arabic that follows into Latin script — which destroys the
verbatim quotes the Phase 2 `ground` node depends on. The measured cost of
auto-detection on this corpus is recorded in `eval_results.md`.

Pass `language=AUTO_DETECT` to opt into detection; that path exists so the
comparison above can be measured, not because it is a supported mode.

Backends
--------
Two are supported behind one interface:

- `mlx-whisper`, on Apple Silicon, which runs the model on the GPU via MLX.
- `openai-whisper`, everywhere else.

The default model is `large-v3-turbo` rather than `large-v3`; on the 8 GB M1
this project is developed on, turbo transcribes at roughly 2x realtime where
CPU `large-v3` manages well under 0.1x. That trade is deliberate and recorded
in `docs/09-DECISIONS.md`.
"""

from __future__ import annotations

import logging
import platform
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from sawti.config import get_settings

logger = logging.getLogger(__name__)

#: Sentinel for `language`: let Whisper detect instead of forcing a language.
AUTO_DETECT = "auto"

#: Whisper model name -> MLX community repo holding the converted weights.
_MLX_REPOS: dict[str, str] = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
}


class TranscriptSegment(BaseModel):
    """A single ASR-produced segment of a transcript."""

    text: str
    start_sec: float
    end_sec: float


class Transcript(BaseModel):
    """Full transcription result for one call recording."""

    call_id: str
    segments: list[TranscriptSegment]
    language: str

    @property
    def text(self) -> str:
        """The full transcript as one whitespace-joined string."""
        return " ".join(segment.text.strip() for segment in self.segments if segment.text.strip())


def _use_mlx() -> bool:
    """Whether the MLX backend is available on this machine."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def _run_mlx(audio_path: Path, model: str, language: str | None) -> dict[str, Any]:
    """Transcribe via mlx-whisper, returning Whisper's raw result dict."""
    import mlx_whisper

    repo = _MLX_REPOS.get(model, model)
    result: dict[str, Any] = mlx_whisper.transcribe(
        str(audio_path), path_or_hf_repo=repo, language=language, verbose=False
    )
    return result


def _run_openai_whisper(audio_path: Path, model: str, language: str | None) -> dict[str, Any]:
    """Transcribe via openai-whisper, returning Whisper's raw result dict."""
    import whisper

    loaded = whisper.load_model(model)
    result: dict[str, Any] = loaded.transcribe(str(audio_path), language=language, verbose=False)
    return result


def transcribe(
    audio_path: Path,
    *,
    language: str | None = None,
    model: str | None = None,
    category: str | None = None,
) -> Transcript:
    """Transcribe `audio_path` using Whisper with `language` forced.

    Args:
        audio_path: Path to the call audio file.
        language: Language code to force Whisper to recognize. `None` resolves
            from settings — see `category`. Pass `AUTO_DETECT` to let Whisper
            detect the language instead of forcing one.
        model: Whisper model name. `None` uses `Settings.whisper_model`.
        category: The call's `sawti.schemas.Language` category. When given and
            `language` is `None`, the forced language is
            `Settings.whisper_language_for(category)` — so English calls decode
            as English while `ar` and `mixed` stay on the Arabic decoder.
            Without it the global `Settings.whisper_language` applies, which
            sends English audio through the Arabic decoder; that is what
            produced `call_0070_en`. See `docs/09-DECISIONS.md`.

    Returns:
        The resulting `Transcript`, segments ordered by start time. Its
        `language` field reports the language actually decoded — the forced
        code, or what Whisper detected under `AUTO_DETECT`.

    Raises:
        FileNotFoundError: If `audio_path` does not exist.
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    settings = get_settings()
    model_name = model or settings.whisper_model
    if language is not None:
        requested = language
    elif category is not None:
        requested = settings.whisper_language_for(category)
    else:
        requested = settings.whisper_language
    forced = None if requested == AUTO_DETECT else requested

    runner = _run_mlx if _use_mlx() else _run_openai_whisper
    logger.debug(
        "transcribing %s (model=%s, language=%s, backend=%s)",
        audio_path.name, model_name, forced or AUTO_DETECT, runner.__name__,
    )
    result = runner(audio_path, model_name, forced)

    segments = [
        TranscriptSegment(
            text=str(segment.get("text", "")).strip(),
            start_sec=float(segment.get("start", 0.0)),
            end_sec=float(segment.get("end", 0.0)),
        )
        for segment in result.get("segments", [])
    ]
    segments.sort(key=lambda segment: segment.start_sec)

    return Transcript(
        call_id=audio_path.stem,
        segments=segments,
        language=str(result.get("language") or forced or AUTO_DETECT),
    )
