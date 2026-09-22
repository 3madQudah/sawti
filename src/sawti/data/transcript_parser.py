"""Parse speaker-labeled synthetic transcripts into ordered speaker turns.

Phase 3: shared by audio synthesis (`sawti.data.generate_audio`) and by the
diarization ground truth it writes into the audio manifest.

Why this module exists separately from the TTS code: the *true speaker per
turn* is the reference signal a diarization metric is scored against. If this
parser guesses, the diarization number measures the guess, not the diarizer.
So it never guesses — a line it cannot attribute raises rather than defaulting
to "same speaker as the previous line".

Known Phase 1 data defect
-------------------------
Sixteen lines across ten calls left the Phase 1 generator without a usable
`Agent:` / `Customer:` prefix. Three carry a corrupted prefix word
("Commissioner:", "Component:", a half-retracted "Cousin... عفواً، Customer:"),
one is a prompt-metadata leak that is not speech at all ("Category: Account
Closure"), and the rest simply lost the prefix. `call_0036_mixed` lost seven
consecutive prefixes, where the turns alternate — so the otherwise tempting
"a bare line continues the previous speaker" rule would be wrong for nearly
half of the affected lines.

`SPEAKER_REPAIRS` therefore records a hand-verified attribution per affected
line, each one read against its surrounding turns. The defect is left in
`data/synthetic/` rather than patched in place: that corpus is the frozen
input to the Phase 1 and Phase 2 numbers already recorded in
`eval_results.md`, and editing it would silently invalidate the comparison
this phase exists to make. See `docs/09-DECISIONS.md`.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

Speaker = Literal["Agent", "Customer"]

#: A line that already carries a well-formed speaker prefix.
_PREFIXED_LINE = re.compile(r"^(Agent|Customer):\s*(.*)$")


class _Repair(BaseModel):
    """A hand-verified attribution for one defective transcript line.

    Attributes:
        speaker: The true speaker, or `None` to drop the line as non-speech.
        strip: Literal corrupted prefix to remove from the line before use.
    """

    speaker: Speaker | None
    strip: str = ""


class Turn(BaseModel):
    """One speaker turn of a call transcript."""

    speaker: Speaker
    text: str
    source_line: int = Field(description="1-based line number in the source transcript.")


#: Hand-verified repairs, keyed by call id then 1-based source line number.
#: Every entry was read in the context of the turns around it; the reasoning
#: for each is recorded in `docs/09-DECISIONS.md`.
SPEAKER_REPAIRS: dict[str, dict[int, _Repair]] = {
    # Agent asks for the order number; the next line answers with it.
    "call_0003_mixed": {5: _Repair(speaker="Agent")},
    # Agent walks the customer through the next setup step.
    "call_0011_mixed": {7: _Repair(speaker="Agent")},
    # Agent reads the account back and concedes the erroneous fee.
    "call_0019_mixed": {5: _Repair(speaker="Agent")},
    # Seven consecutive prefixes lost from an alternating exchange.
    "call_0036_mixed": {
        5: _Repair(speaker="Agent"),  # "Let me check the system for you"
        6: _Repair(speaker="Customer"),  # "Akid, take your time."
        7: _Repair(speaker="Agent"),  # reports the customs delay
        8: _Repair(speaker="Customer"),  # asks for a delivery date
        9: _Repair(speaker="Agent"),  # gives the delivery date
        10: _Repair(speaker="Customer"),  # thanks the agent
        11: _Repair(speaker="Agent"),  # closes the call
    },
    # Corrupted prefix word; the turn is the agent offering to help.
    "call_0064_en": {3: _Repair(speaker="Agent", strip="Commissioner: ")},
    # Generator self-corrects mid-line ("Cousin... sorry, Customer:").
    "call_0067_mixed": {6: _Repair(speaker="Customer", strip="Cousin... عفواً، Customer: ")},
    # Corrupted prefix word; the agent credits the customer's own troubleshooting.
    "call_0072_en": {5: _Repair(speaker="Agent", strip="Component: ")},
    # Agent looks up the order and explains the customs delay.
    "call_0135_mixed": {5: _Repair(speaker="Agent")},
    # Prompt-metadata leak, not speech. Dropped, so it is never synthesized.
    "call_0148_mixed": {5: _Repair(speaker=None)},
    # Agent confirms the fault is on the operator's side.
    "call_0149_mixed": {7: _Repair(speaker="Agent")},
}


class UnattributableLineError(ValueError):
    """Raised when a transcript line has no speaker prefix and no recorded repair."""


def parse_transcript(raw: str, call_id: str) -> list[Turn]:
    """Parse `raw` transcript text into ordered speaker turns.

    Args:
        raw: Full transcript text, one turn per line, `Agent:`/`Customer:` prefixed.
        call_id: Call identifier, used to look up `SPEAKER_REPAIRS`.

    Returns:
        Turns in source order. Lines repaired to `None` are omitted.

    Raises:
        UnattributableLineError: If a non-blank line carries no speaker prefix
            and no repair is recorded for it. Deliberately fatal: silently
            guessing a speaker would corrupt the diarization ground truth.
    """
    repairs = SPEAKER_REPAIRS.get(call_id, {})
    turns: list[Turn] = []

    for line_number, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        match = _PREFIXED_LINE.match(stripped)
        if match:
            speaker, text = match.group(1), match.group(2).strip()
            if text:
                turns.append(Turn(speaker=speaker, text=text, source_line=line_number))
            continue

        repair = repairs.get(line_number)
        if repair is None:
            raise UnattributableLineError(
                f"{call_id} line {line_number} has no speaker prefix and no recorded "
                f"repair in SPEAKER_REPAIRS: {stripped[:80]!r}"
            )
        if repair.speaker is None:
            continue

        text = stripped.removeprefix(repair.strip).strip()
        if text:
            turns.append(Turn(speaker=repair.speaker, text=text, source_line=line_number))

    return turns
