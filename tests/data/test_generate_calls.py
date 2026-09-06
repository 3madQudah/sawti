"""Tests for `sawti.data.generate_calls`.

Phase 1: mirrors `src/sawti/data/generate_calls.py`. The LLM provider is
always mocked here — these tests never make a real API call.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from sawti.data.generate_calls import generate_batch, generate_call
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
