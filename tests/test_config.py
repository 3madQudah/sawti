"""Tests for `sawti.config`.

Phase 0: mirrors `src/sawti/config.py`.
"""

from __future__ import annotations

from sawti.config import Settings, get_settings


def test_settings_defaults_confidence_threshold_when_unset(monkeypatch) -> None:
    """Settings.confidence_threshold defaults to 0.7 when no env var is set."""
    monkeypatch.delenv("SAWTI_CONFIDENCE_THRESHOLD", raising=False)
    settings = Settings(_env_file=None)
    assert settings.confidence_threshold == 0.7


def test_settings_reads_confidence_threshold_from_env(monkeypatch) -> None:
    """Settings.confidence_threshold is overridable via SAWTI_CONFIDENCE_THRESHOLD."""
    monkeypatch.setenv("SAWTI_CONFIDENCE_THRESHOLD", "0.9")
    settings = Settings(_env_file=None)
    assert settings.confidence_threshold == 0.9


def test_settings_defaults_llm_provider_to_anthropic(monkeypatch) -> None:
    """Settings.llm_provider defaults to 'anthropic' when no env var is set."""
    monkeypatch.delenv("SAWTI_LLM_PROVIDER", raising=False)
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "anthropic"


def test_get_settings_returns_cached_singleton() -> None:
    """get_settings() returns the same Settings instance on repeated calls."""
    assert get_settings() is get_settings()
