"""Tests for `sawti.data.generate_calls`.

Phase 1: mirrors `src/sawti/data/generate_calls.py`. The LLM provider is
always mocked here — these tests never make a real API call.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from sawti.data.generate_calls import (
    _find_dialect_violations,
    _generate_with_retry,
    generate_batch,
    generate_call,
)
from sawti.schemas import Language


def _mock_provider(transcript: str = "Agent: Hello.\nCustomer: Hi, I have a question.") -> MagicMock:
    provider = MagicMock()
    provider.complete = AsyncMock(return_value=transcript)
    return provider


def test_generate_call_produces_nonempty_transcript(monkeypatch) -> None:
    """generate_call() returns a non-empty transcript string."""
    provider = _mock_provider()
    monkeypatch.setattr("sawti.data.generate_calls.get_llm_provider", lambda: provider)

    transcript = generate_call(Language.EN, seed=1)

    assert transcript
    assert transcript == provider.complete.return_value
    provider.complete.assert_awaited_once()


def test_generate_call_is_reproducible_with_seed(monkeypatch) -> None:
    """generate_call() with the same seed produces the same transcript."""
    provider = _mock_provider()
    monkeypatch.setattr("sawti.data.generate_calls.get_llm_provider", lambda: provider)

    generate_call(Language.MIXED, seed=42)
    generate_call(Language.MIXED, seed=42)

    first_call, second_call = provider.complete.await_args_list
    assert first_call == second_call


def test_generate_batch_writes_expected_number_of_files(monkeypatch, tmp_path: Path) -> None:
    """generate_batch() writes exactly n transcript files to output_dir."""
    provider = _mock_provider()
    monkeypatch.setattr("sawti.data.generate_calls.get_llm_provider", lambda: provider)

    output_dir = tmp_path / "synthetic"
    paths = generate_batch(10, output_dir=output_dir, sleep_seconds=0)

    assert len(paths) == 10
    assert all(path.exists() for path in paths)
    assert len(list(output_dir.glob("*.txt"))) == 10


def test_generate_batch_respects_language_mix_proportions(monkeypatch, tmp_path: Path) -> None:
    """generate_batch() distributes generated calls across languages per language_mix."""
    provider = _mock_provider()
    monkeypatch.setattr("sawti.data.generate_calls.get_llm_provider", lambda: provider)

    output_dir = tmp_path / "synthetic"
    language_mix = {Language.AR: 0.5, Language.EN: 0.3, Language.MIXED: 0.2}
    paths = generate_batch(10, output_dir=output_dir, language_mix=language_mix, sleep_seconds=0)

    assert len(paths) == 10
    counts = {lang: sum(1 for path in paths if path.stem.endswith(f"_{lang.value}")) for lang in language_mix}
    assert counts == {Language.AR: 5, Language.EN: 3, Language.MIXED: 2}


def test_generate_batch_creates_output_dir_if_missing(monkeypatch, tmp_path: Path) -> None:
    """generate_batch() creates output_dir when it does not already exist."""
    provider = _mock_provider()
    monkeypatch.setattr("sawti.data.generate_calls.get_llm_provider", lambda: provider)

    output_dir = tmp_path / "does" / "not" / "exist"
    assert not output_dir.exists()

    generate_batch(2, output_dir=output_dir, sleep_seconds=0)

    assert output_dir.is_dir()


@pytest.mark.parametrize("language", [Language.AR, Language.EN, Language.MIXED])
def test_generate_call_speaker_turns_survive_mock_roundtrip(monkeypatch, language: Language) -> None:
    """generate_call() returns the provider's transcript verbatim (modulo whitespace)."""
    transcript = "Agent: كيف بقدر أساعدك؟\nCustomer: I have an issue with my bill."
    provider = _mock_provider(transcript)
    monkeypatch.setattr("sawti.data.generate_calls.get_llm_provider", lambda: provider)

    result = generate_call(language, seed=7)

    assert result == transcript


def test_find_dialect_violations_clean_arabic_string_has_none() -> None:
    """_find_dialect_violations() returns [] for a clean, all-Arabic-script transcript."""
    text = "Agent: أهلاً بك، كيف أقدر أساعدك اليوم؟\nCustomer: عندي مشكلة بالفاتورة الشهرية."
    assert _find_dialect_violations(text, Language.AR) == []


def test_find_dialect_violations_clean_mixed_string_has_none() -> None:
    """_find_dialect_violations() returns [] for a clean code-switched transcript."""
    text = (
        "Agent: Ahlan, welcome to Zain support. كيف أقدر أساعدك اليوم؟\n"
        "Customer: I have an issue with my bill, بس مش فاهم ليش زادت القيمة."
    )
    assert _find_dialect_violations(text, Language.MIXED) == []


def test_find_dialect_violations_flags_egyptian_marker() -> None:
    """_find_dialect_violations() flags banned Egyptian-dialect words like "فندم"."""
    text = "Agent: تفضل يا فندم، كيف أقدر أساعدك؟"
    violations = _find_dialect_violations(text, Language.AR)
    assert any("فندم" in v for v in violations)


def test_find_dialect_violations_flags_arabizi() -> None:
    """_find_dialect_violations() flags Arabizi (Latin-letter transliterated Arabic)."""
    text = "Agent: Ahlan, kif ba'dar a'awnak today?"
    violations = _find_dialect_violations(text, Language.MIXED)
    assert any("ba'dar" in v for v in violations)


def test_find_dialect_violations_flags_mixed_script_glitch_word() -> None:
    """_find_dialect_violations() flags a single token mixing Arabic and Latin script."""
    text = "Agent: the l'flوس will be refunded within 3 days."
    violations = _find_dialect_violations(text, Language.MIXED)
    assert any("l'fl" in v for v in violations)


def test_find_dialect_violations_does_not_flag_english_contractions() -> None:
    """_find_dialect_violations() does not misflag ordinary English contractions as Arabizi."""
    text = "Agent: I don't think that's right, but I'll check — we're on it."
    assert _find_dialect_violations(text, Language.MIXED) == []


def test_generate_with_retry_retries_on_dialect_violation_then_succeeds(monkeypatch) -> None:
    """_generate_with_retry() retries when generate_call() returns a violating transcript."""
    bad = "Agent: تفضل يا فندم، كيف أقدر أساعدك؟"
    good = "Agent: تفضل يا سيدي، كيف أقدر أساعدك؟"
    results = iter([bad, good])
    monkeypatch.setattr(
        "sawti.data.generate_calls.generate_call",
        lambda language, *, seed: next(results),
    )

    result = _generate_with_retry(Language.AR, seed=1, max_retries=3, initial_delay=0.0)

    assert result == good


def test_generate_with_retry_raises_when_violations_persist(monkeypatch) -> None:
    """_generate_with_retry() raises, naming the violations, once retries are exhausted."""
    bad = "Agent: تفضل يا فندم، كيف أقدر أساعدك؟"
    monkeypatch.setattr(
        "sawti.data.generate_calls.generate_call",
        lambda language, *, seed: bad,
    )

    with pytest.raises(ValueError, match="dialect violations"):
        _generate_with_retry(Language.AR, seed=1, max_retries=2, initial_delay=0.0)
