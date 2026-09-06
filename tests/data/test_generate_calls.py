"""Tests for `sawti.data.generate_calls`.

Phase 1: mirrors `src/sawti/data/generate_calls.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 1")
def test_generate_call_produces_nonempty_transcript() -> None:
    """generate_call() returns a non-empty transcript string."""


@pytest.mark.skip(reason="phase 1")
def test_generate_call_is_reproducible_with_seed() -> None:
    """generate_call() with the same seed produces the same transcript."""


@pytest.mark.skip(reason="phase 1")
def test_generate_batch_writes_expected_number_of_files() -> None:
    """generate_batch() writes exactly n transcript files to output_dir."""


@pytest.mark.skip(reason="phase 1")
def test_generate_batch_respects_language_mix_proportions() -> None:
    """generate_batch() distributes generated calls across languages per language_mix."""
