"""Capture a supervisor correction with its full surrounding context.

Phase 4: agent memory.
"""

from __future__ import annotations

from sawti.schemas import CallAnalysis, Correction


def capture_correction(
    original: CallAnalysis,
    corrected: CallAnalysis,
    *,
    error_location: str,
    reviewer_id: str,
    note: str | None = None,
) -> Correction:
    """Build a `Correction` record capturing a reviewer's edit and its full context.

    Args:
        original: The unreviewed agent output.
        corrected: The reviewer's corrected version.
        error_location: Field path or artifact id of what was wrong.
        reviewer_id: Identity of the QA reviewer making the correction.
        note: Optional free-text explanation from the reviewer.

    Returns:
        A validated `Correction` record, ready for induction.

    Raises:
        NotImplementedError: Until persistence/context capture is implemented.
    """
    # TODO(phase-4): stamp timestamp, persist via sawti.db.session, and return the Correction.
    raise NotImplementedError
