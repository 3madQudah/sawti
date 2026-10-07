"""Application settings, loaded from environment variables and `.env`.

Phase 0: foundational configuration used by every other module.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
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
    # "fake" is `sawti.llm.fake_provider`: deterministic, no network. For the
    # 6.2 container e2e and load measurement, so neither spends provider quota
    # nor measures a free-tier rate limit instead of this system.
    llm_provider: Literal["anthropic", "vllm", "gemini", "fake"] = Field(
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

    # --- Service (phase 6.2) ---
    api_host: str = Field(default="127.0.0.1", alias="SAWTI_API_HOST")
    api_port: int = Field(default=8000, alias="SAWTI_API_PORT")
    # Shared key checked by `sawti.api.deps`. NOT authentication — see
    # docs/09-DECISIONS.md 2026-10-06 (decision 3). Unset means every
    # protected route refuses (fail closed), never "open".
    api_key: SecretStr | None = Field(default=None, alias="SAWTI_API_KEY")
    # Browser origins allowed to call the API (the 6.4 dashboard). JSON list.
    cors_origins: list[str] = Field(default_factory=list, alias="SAWTI_CORS_ORIGINS")
    # Celery retry policy for transient provider errors (429 / quota / 5xx).
    # Exponential: backoff * 2**attempt, capped at backoff_max.
    task_max_retries: int = Field(default=5, ge=0, alias="SAWTI_TASK_MAX_RETRIES")
    task_retry_backoff_seconds: float = Field(default=10.0, ge=0.0, alias="SAWTI_TASK_RETRY_BACKOFF_SECONDS")
    task_retry_backoff_max_seconds: float = Field(
        default=300.0, ge=0.0, alias="SAWTI_TASK_RETRY_BACKOFF_MAX_SECONDS"
    )
    # Reaper (Celery beat): calls stuck `queued` (no message enqueued for this
    # long) or `processing` (claimed this long ago, worker presumably gone).
    # The processing timeout must exceed one task attempt (provider timeout
    # 120 s); the queued one must exceed the longest retry backoff.
    reaper_interval_seconds: float = Field(default=60.0, gt=0.0, alias="SAWTI_REAPER_INTERVAL_SECONDS")
    reaper_queued_timeout_seconds: float = Field(
        default=900.0, gt=0.0, alias="SAWTI_REAPER_QUEUED_TIMEOUT_SECONDS"
    )
    reaper_processing_timeout_seconds: float = Field(
        default=900.0, gt=0.0, alias="SAWTI_REAPER_PROCESSING_TIMEOUT_SECONDS"
    )
    # Top-k memory rules folded into `extract`'s prompt, as in phase 4.
    memory_top_k: int = Field(default=5, ge=0, alias="SAWTI_MEMORY_TOP_K")
    # Load the embedding model when a worker process starts, so the first task
    # does not pay for it and cold start is measurable on its own.
    worker_preload_embeddings: bool = Field(default=True, alias="SAWTI_WORKER_PRELOAD_EMBEDDINGS")
    # `fake` provider knobs: simulated model latency, and how many unsupported
    # claims it proposes (1 -> coverage 2/3 -> escalates at the 0.7 default).
    fake_llm_latency_seconds: float = Field(default=0.0, ge=0.0, alias="SAWTI_FAKE_LLM_LATENCY_SECONDS")
    fake_llm_ungrounded_claims: int = Field(default=1, ge=0, alias="SAWTI_FAKE_LLM_UNGROUNDED_CLAIMS")

    # --- Langfuse ---
    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    # Self-hosted only (phase 6.3): the default is the local v2 server from
    # docker-compose.yml, never Langfuse Cloud — traces carry (redacted)
    # transcript text, and Cloud would be a third party. Tracing is off unless
    # both keys are set.
    langfuse_host: str = Field(default="http://localhost:3000", alias="LANGFUSE_HOST")

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
