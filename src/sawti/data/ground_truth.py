"""Load and validate human-reviewed ground truth labels used for evaluation.

Phase 1: ground truth loading, required by eval.runner from phase 1 onward.

Ground-truth records live as one `*.json` file per call in `data/ground_truth/`,
each shaped like `sawti.schemas.CallAnalysis`. Authoring templates for these
files are produced by `scripts/generate_ground_truth_templates.py` and filled
in by hand (`scripts/find_quote.py` helps locate exact quote offsets); this
module only loads and validates the finished result.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import ValidationError

from sawti.schemas import CallAnalysis, Language

# The 7 QA rubric criteria every `CallAnalysis.rubric_scores` must cover.
# Fixed and agreed with the user — do not add, remove, or rename any of these.
RUBRIC_CRITERIA: tuple[str, ...] = (
    "professionalism",
    "empathy",
    "resolution_effectiveness",
    "communication_clarity",
    "call_closure",
    "accuracy_of_information",
    "procedure_adherence",
)

# Matches how `sawti.data.generate_calls.generate_batch` names synthetic
# transcripts (call_<idx>_<lang>.txt) — ground-truth call_ids mirror that
# stem, so the language category can be cross-checked from the id alone.
_CALL_ID_LANGUAGE_RE = re.compile(r"_(ar|en|mixed)$")


def _language_from_call_id(call_id: str) -> Language | None:
    """Infer the `Language` encoded in a call_id's `_ar`/`_en`/`_mixed` suffix.

    Returns None if `call_id` does not end with a recognized language suffix
    (e.g. a hand-picked call_id that doesn't follow the synthetic naming
    convention) — such records are skipped by the cross-check, not rejected.
    """
    match = _CALL_ID_LANGUAGE_RE.search(call_id)
    return Language(match.group(1)) if match else None


def load_ground_truth(path: Path) -> list[CallAnalysis]:
    """Load and validate every `*.json` ground-truth record in `path`.

    Args:
        path: Directory containing ground-truth `*.json` files.

    Returns:
        Validated `CallAnalysis` instances, one per file, sorted by filename.

    Raises:
        ValueError: If any file is not valid JSON, or fails `CallAnalysis`
            validation (e.g. an authoring template that hasn't been filled
            in yet) — the error names the offending file and the cause.
    """
    records: list[CallAnalysis] = []
    for file_path in sorted(path.glob("*.json")):
        raw = file_path.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{file_path}: not valid JSON ({exc})") from exc
        try:
            records.append(CallAnalysis.model_validate(data))
        except ValidationError as exc:
            raise ValueError(f"{file_path}: failed CallAnalysis validation:\n{exc}") from exc
    return records


def validate_ground_truth(records: list[CallAnalysis]) -> None:
    """Validate a set of ground-truth records for internal consistency.

    Checks, across all records:
      * No two records share the same `call_id`.
      * Every record has a `RubricScore` for each of the 7 `RUBRIC_CRITERIA`.
      * Every record's `language` agrees with the language encoded in its own
        `call_id` (the `_ar`/`_en`/`_mixed` filename suffix), where present.

    Args:
        records: Ground-truth `CallAnalysis` records to validate.

    Raises:
        ValueError: Listing every problem found, not just the first.
    """
    problems: list[str] = []

    call_id_counts: dict[str, int] = {}
    for record in records:
        call_id_counts[record.call_id] = call_id_counts.get(record.call_id, 0) + 1
    for call_id, count in sorted(call_id_counts.items()):
        if count > 1:
            problems.append(f"duplicate call_id {call_id!r} appears {count} times")

    for record in records:
        present_criteria = {score.criterion for score in record.rubric_scores}
        missing = [c for c in RUBRIC_CRITERIA if c not in present_criteria]
        if missing:
            problems.append(f"call_id {record.call_id!r} is missing rubric criteria: {missing}")

        expected_language = _language_from_call_id(record.call_id)
        if expected_language is not None and record.language != expected_language:
            problems.append(
                f"call_id {record.call_id!r} encodes language {expected_language.value!r} "
                f"but record.language is {record.language.value!r}"
            )

    if problems:
        raise ValueError("ground truth validation failed:\n" + "\n".join(f"  - {p}" for p in problems))
