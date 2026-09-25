"""Application settings, loaded from environment variables and `.env`.

Phase 0: foundational configuration used by every other module.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide configuration.

    All values are overridable via environment variables or a `.env` file
    (see `.env.example` for the full list). Nothing in this codebase should
    read `os.environ` directly — go through `get_settings()` instead, so
    provider/config swaps never require editing call sites.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )

    # --- App ---
    env: str = Field(default="development", alias="SAWTI_ENV")
    log_level: str = Field(default="INFO", alias="SAWTI_LOG_LEVEL")

    # --- LLM provider ---
    llm_provider: Literal["anthropic", "vllm", "gemini"] = Field(
        default="anthropic", alias="SAWTI_LLM_PROVIDER"
    )
    llm_model: str = Field(default="claude-sonnet-5", alias="SAWTI_LLM_MODEL")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    vllm_base_url: str = Field(default="http://localhost:8001/v1", alias="VLLM_BASE_URL")
    vllm_model: str | None = Field(default=None, alias="VLLM_MODEL")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    # "gemini-flash-lite-latest" is a stable alias to the current free-tier flash-lite
    # model — pinned model ids (e.g. "gemini-2.5-flash") get retired for new API keys,
    # and "gemini-flash-latest" was observed returning persistent 503s under load.
    gemini_model: str = Field(default="gemini-flash-lite-latest", alias="SAWTI_GEMINI_MODEL")

    # --- Confidence / review routing ---
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, alias="SAWTI_CONFIDENCE_THRESHOLD")

    # --- Postgres ---
    database_url: str = Field(
        default="postgresql+psycopg://sawti:sawti@localhost:5432/sawti",
        alias="DATABASE_URL",
    )

    # --- Redis / Celery ---
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    celery_broker_url: str = Field(default="redis://localhost:6379/0", alias="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/1", alias="CELERY_RESULT_BACKEND")

    # --- Langfuse ---
    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")

    # --- ASR ---
    # "large-v3-turbo" rather than "large-v3": on the 8 GB M1 this project is
    # developed on, turbo runs at ~2x realtime against well under 0.1x for
    # large-v3 on CPU, which is the difference between a 2-hour and a multi-day
    # corpus run. Accuracy cost is measured in eval_results.md; rationale in
    # docs/09-DECISIONS.md. Override with WHISPER_MODEL=large-v3 on a box that
    # can carry it.
    whisper_model: str = Field(default="large-v3-turbo", alias="WHISPER_MODEL")
    # Forced, never auto-detected — Whisper detects from the first 30s only and
    # mis-commits code-switched calls. See sawti.asr.transcribe.
    #
    # This is the *fallback*, applied to any language category without an entry
    # in `whisper_language_overrides`. It is deliberately not the only knob: a
    # single global forced language sent every English call through the Arabic
    # decoder, which mostly worked but produced one total loss
    # (`call_0070_en`, 615 words of Arabic repetition-loop output, WER 1.000).
    # See docs/09-DECISIONS.md.
    whisper_language: str = Field(default="ar", alias="WHISPER_LANGUAGE")
    # Per-language-category forced language, overriding `whisper_language`.
    # Keys are `sawti.schemas.Language` values. The default encodes the decision
    # of 2026-09-22: `en` decodes as English, while `ar` and `mixed` both stay
    # on the Arabic decoder. Set as JSON, e.g.
    # WHISPER_LANGUAGE_OVERRIDES='{"en": "en", "mixed": "ar"}'
    whisper_language_overrides: dict[str, str] = Field(
        default_factory=lambda: {"en": "en"}, alias="WHISPER_LANGUAGE_OVERRIDES"
    )
    # Required for pyannote's gated diarization pipelines. Accept the licences
    # at pyannote/segmentation-3.0 and pyannote/speaker-diarization-3.1 first.
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")

    # --- Fine-tuning (Phase 5) ---
    # PROJECT_BRIEF.md names "Qwen3-8B" without an exact checkpoint; confirmed
    # explicitly (not guessed — see docs/09-DECISIONS.md, 2026-09-25) as the
    # instruction/chat-tuned variant, not `Qwen/Qwen3-8B-Base`: its chat
    # template matches how `scripts/build_finetune_dataset.py` frames each
    # training pair (instruction + input -> corrected output). Consumed by
    # `scripts/train_qlora.py` on a CUDA host (Colab), never loaded on this
    # Mac dev environment.
    finetune_base_model: str = Field(default="Qwen/Qwen3-8B", alias="SAWTI_FINETUNE_BASE_MODEL")

    def whisper_language_for(self, category: str) -> str:
        """Return the forced Whisper language for a language category.

        Args:
            category: A `sawti.schemas.Language` value, e.g. `"en"`. A
                `Language` member may be passed directly; its `.value` is
                used, since `str(Language.EN)` is `"Language.EN"` rather
                than `"en"`.

        Returns:
            The override for `category` if one is configured, else
            `whisper_language`.
        """
        key = getattr(category, "value", category)
        return self.whisper_language_overrides.get(key, self.whisper_language)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide `Settings` singleton, constructed on first call."""
    return Settings()
