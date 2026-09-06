"""Load and validate human-reviewed ground truth labels used for evaluation.

Phase 1: ground truth loading, required by eval.runner from phase 1 onward.
"""

from __future__ import annotations

from pathlib import Path

from sawti.schemas import CallAnalysis


def load_ground_truth(path: Path) -> list[CallAnalysis]:
    """Load reviewed ground-truth `CallAnalysis` records from `path`.

    Args:
        path: Directory or file containing ground truth records.

    Returns:
        Validated `CallAnalysis` instances.

    Raises:
        NotImplementedError: Until loading is implemented.
    """
    # TODO(phase-1): read ground truth files and validate each into CallAnalysis.
    raise NotImplementedError


def validate_ground_truth(records: list[CallAnalysis]) -> None:
    """Validate a set of ground-truth records for internal consistency.

    Args:
        records: Ground-truth `CallAnalysis` records to validate.

    Raises:
        NotImplementedError: Until validation rules are implemented.
    """
    # TODO(phase-1): check for duplicate call_ids, missing evidence, language mislabels, etc.
    raise NotImplementedError
