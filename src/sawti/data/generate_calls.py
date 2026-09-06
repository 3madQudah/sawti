"""Synthetic bilingual call transcript generation for development and eval.

Phase 1: synthetic data generation.
"""

from __future__ import annotations

from pathlib import Path

from sawti.schemas import Language


def generate_call(language: Language, *, seed: int | None = None) -> str:
    """Generate a single synthetic call transcript.

    Args:
        language: Target language category for the generated call.
        seed: Optional RNG seed for reproducibility.

    Returns:
        The generated transcript text.

    Raises:
        NotImplementedError: Until generation is implemented.
    """
    # TODO(phase-1): generate a synthetic transcript via LLMProvider, tagged by language.
    raise NotImplementedError


def generate_batch(
    n: int, *, output_dir: Path, language_mix: dict[Language, float] | None = None
) -> list[Path]:
    """Generate a batch of synthetic call transcripts and write them to `output_dir`.

    Args:
        n: Number of calls to generate.
        output_dir: Directory to write generated transcripts into.
        language_mix: Optional proportion of each language category; defaults to an even split.

    Returns:
        Paths to the written transcript files.

    Raises:
        NotImplementedError: Until batch generation is implemented.
    """
    # TODO(phase-1): loop generate_call() per language_mix proportions and persist to output_dir.
    raise NotImplementedError
