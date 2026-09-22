"""Tests for `sawti.data.transcript_parser`.

Phase 3: mirrors `src/sawti/data/transcript_parser.py`.
"""

from pathlib import Path

import pytest

from sawti.data.transcript_parser import (
    SPEAKER_REPAIRS,
    UnattributableLineError,
    parse_transcript,
)

SYNTHETIC_DIR = Path("data/synthetic")


def test_parses_well_formed_transcript_in_order() -> None:
    """Prefixed lines become turns in source order with prefixes stripped."""
    raw = "Agent: Hello there\nCustomer: Hi back\nAgent: Goodbye\n"

    turns = parse_transcript(raw, "call_9999_en")

    assert [turn.speaker for turn in turns] == ["Agent", "Customer", "Agent"]
    assert [turn.text for turn in turns] == ["Hello there", "Hi back", "Goodbye"]
    assert [turn.source_line for turn in turns] == [1, 2, 3]


def test_blank_lines_are_skipped_without_shifting_line_numbers() -> None:
    """Blank lines produce no turn but still consume a source line number."""
    turns = parse_transcript("Agent: One\n\nCustomer: Two\n", "call_9999_en")

    assert [turn.source_line for turn in turns] == [1, 3]


def test_unprefixed_line_without_repair_raises() -> None:
    """An unattributable line is fatal rather than silently guessed."""
    raw = "Agent: Hello\nthis line lost its prefix\n"

    with pytest.raises(UnattributableLineError, match="line 2"):
        parse_transcript(raw, "call_9999_en")


def test_repair_attributes_unprefixed_line() -> None:
    """A recorded repair supplies the speaker for a bare line."""
    raw = SYNTHETIC_DIR.joinpath("call_0003_mixed.txt").read_text(encoding="utf-8")

    turns = parse_transcript(raw, "call_0003_mixed")
    repaired = next(turn for turn in turns if turn.source_line == 5)

    assert repaired.speaker == "Agent"
    assert repaired.text.startswith("Let me check the system")


def test_repair_strips_corrupted_prefix_word() -> None:
    """A corrupted prefix ('Component: ') is removed from the turn text."""
    raw = SYNTHETIC_DIR.joinpath("call_0072_en.txt").read_text(encoding="utf-8")

    turns = parse_transcript(raw, "call_0072_en")
    repaired = next(turn for turn in turns if turn.source_line == 5)

    assert repaired.speaker == "Agent"
    assert not repaired.text.startswith("Component")
    assert repaired.text.startswith("Thank you for trying")


def test_repair_to_none_drops_non_speech_metadata_line() -> None:
    """The 'Category: Account Closure' prompt leak is dropped, not synthesized."""
    raw = SYNTHETIC_DIR.joinpath("call_0148_mixed.txt").read_text(encoding="utf-8")

    turns = parse_transcript(raw, "call_0148_mixed")

    assert all(turn.source_line != 5 for turn in turns)
    assert all("Account Closure" not in turn.text for turn in turns)


def test_alternating_repair_block_is_not_collapsed_to_one_speaker() -> None:
    """call_0036_mixed's seven lost prefixes alternate rather than continue."""
    raw = SYNTHETIC_DIR.joinpath("call_0036_mixed.txt").read_text(encoding="utf-8")

    turns = parse_transcript(raw, "call_0036_mixed")
    block = [turn.speaker for turn in turns if 5 <= turn.source_line <= 11]

    assert block == ["Agent", "Customer", "Agent", "Customer", "Agent", "Customer", "Agent"]


@pytest.mark.parametrize("path", sorted(SYNTHETIC_DIR.glob("*.txt")), ids=lambda p: p.stem)
def test_every_synthetic_transcript_parses(path: Path) -> None:
    """The whole Phase 1 corpus parses with no unattributable line."""
    turns = parse_transcript(path.read_text(encoding="utf-8"), path.stem)

    assert turns, f"{path.stem} produced no turns"
    assert all(turn.text for turn in turns)


def test_repair_table_only_references_real_defective_lines() -> None:
    """Every repair targets a line that genuinely lacks a valid prefix.

    Guards against the table drifting out of sync with the corpus — a stale
    entry would silently relabel a line that is actually fine.
    """
    for call_id, repairs in SPEAKER_REPAIRS.items():
        lines = SYNTHETIC_DIR.joinpath(f"{call_id}.txt").read_text(encoding="utf-8").splitlines()
        for line_number in repairs:
            line = lines[line_number - 1].strip()
            assert not line.startswith(("Agent:", "Customer:")), (
                f"{call_id} line {line_number} already has a valid prefix; "
                "the repair entry is stale"
            )
