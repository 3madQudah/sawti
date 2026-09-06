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
    whisper_model: str = Field(default="large-v3", alias="WHISPER_MODEL")
    whisper_language: str = Field(default="ar", alias="WHISPER_LANGUAGE")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide `Settings` singleton, constructed on first call."""
    return Settings()
