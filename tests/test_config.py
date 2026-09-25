"""Tests for `sawti.config`.

Phase 0: mirrors `src/sawti/config.py`.
"""

from __future__ import annotations

import pytest

from sawti.config import Settings, get_settings
from sawti.schemas import Language


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


def test_settings_defaults_finetune_base_model_to_qwen3_8b_instruct(monkeypatch) -> None:
    """Confirmed explicitly with the user, 2026-09-25 — not a guess. See docs/09-DECISIONS.md."""
    monkeypatch.delenv("SAWTI_FINETUNE_BASE_MODEL", raising=False)
    settings = Settings(_env_file=None)
    assert settings.finetune_base_model == "Qwen/Qwen3-8B"


def test_settings_reads_finetune_base_model_from_env(monkeypatch) -> None:
    """Overridable, not a magic string baked into scripts/train_qlora.py."""
    monkeypatch.setenv("SAWTI_FINETUNE_BASE_MODEL", "Qwen/Qwen3-8B-Base")
    settings = Settings(_env_file=None)
    assert settings.finetune_base_model == "Qwen/Qwen3-8B-Base"


class TestWhisperLanguageForCategory:
    """Per-language-category forced Whisper language."""

    def test_english_overrides_the_global_default(self) -> None:
        """The 2026-09-22 decision: en decodes as English, not Arabic."""
        assert Settings().whisper_language_for(Language.EN) == "en"

    @pytest.mark.parametrize("category", [Language.AR, Language.MIXED])
    def test_arabic_and_mixed_use_the_global_default(self, category: Language) -> None:
        """ar and mixed both stay on the Arabic decoder."""
        settings = Settings()

        assert settings.whisper_language_for(category) == settings.whisper_language

    def test_accepts_a_plain_string_category(self) -> None:
        """Callers holding a bare 'en' need not construct the enum."""
        assert Settings().whisper_language_for("en") == "en"

    def test_enum_member_resolves_by_value_not_repr(self) -> None:
        """Guards a real bug: str(Language.EN) is 'Language.EN', not 'en'.

        Resolving by repr silently fell through to the global fallback, which
        is exactly the behaviour this setting exists to fix.
        """
        assert Settings().whisper_language_for(Language.EN) == Settings().whisper_language_for("en")

    def test_unknown_category_falls_back_to_the_global_default(self) -> None:
        """An unmapped category must not raise mid-corpus."""
        settings = Settings()

        assert settings.whisper_language_for("klingon") == settings.whisper_language

    def test_overrides_are_configurable_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mapping is settings, not a hardcoded table."""
        monkeypatch.setenv("WHISPER_LANGUAGE_OVERRIDES", '{"mixed": "en"}')

        assert Settings().whisper_language_for(Language.MIXED) == "en"
