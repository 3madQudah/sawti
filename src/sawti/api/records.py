"""Graph state -> persisted records, and review verdicts -> `Correction`s. Pure functions.

Phase 6.2. Shared by the worker (`sawti.api.tasks`) and the review route; no
I/O here, so each mapping is tested on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sawti.api.schemas import EDITABLE_FIELDS, ClaimKind, ClaimVerdict, ReviewSubmission
from sawti.api.views import CLAIM_LISTS
from sawti.schemas import CallAnalysis, Claim, Language, RubricScore, SentimentTrajectory

#: `CallAnalysis.summary` must be non-empty. When extraction failed there is
#: no model summary to store; this says so instead of inventing one.
NO_ANALYSIS_SUMMARY = "No analysis: extraction failed. See the escalation reason."


def analysis_from_state(values: dict[str, Any], *, call_id: str, language: Language) -> CallAnalysis:
    """Assemble final (or suspended) graph state into the validated `CallAnalysis`.

    Unlike the eval harness's version, a run that recorded an `error` still
    yields a record — with no claims, confidence 0.0 and human review
    required — because in the service such a call goes to a human, and the
    review screen needs something to show.
    """
    error = values.get("error")
    return CallAnalysis(
        call_id=call_id,
        language=language,
        summary=values.get("summary") or NO_ANALYSIS_SUMMARY,
        commitments=[] if error else list(values.get("commitments", [])),
        compliance_flags=[] if error else list(values.get("compliance_flags", [])),
        rubric_scores=[] if error else list(values.get("rubric_scores", [])),
        sentiment_trajectory=values.get("sentiment_trajectory") or SentimentTrajectory(),
        confidence=0.0 if error else float(values.get("confidence", 0.0)),
        requires_human_review=True if error else bool(values.get("requires_human_review", True)),
    )


def escalation_reason(values: dict[str, Any], *, threshold: float) -> str | None:
    """A short, human-readable reason a call is in front of a reviewer; None if it auto-passed."""
    error = values.get("error")
    if error:
        return f"Analysis error: {error}"
    if not values.get("requires_human_review", True):
        return None
    rejected = len(values.get("rejected_claims", []))
    kept = sum(len(values.get(name, [])) for name, _ in CLAIM_LISTS)
    total = kept + rejected
    coverage = float(values.get("grounding_coverage", 0.0))
    return (
        f"{rejected} of {total} claims had no verbatim evidence "
        f"(grounding coverage {coverage:.0%}, threshold {threshold:.0%})"
    )


class InvalidVerdictError(ValueError):
    """A verdict that cannot be applied to its claim (the route answers 422)."""


@dataclass(frozen=True)
class PlannedCorrection:
    """One `Correction` a review submission produces: arguments for `capture_correction`."""

    error_location: str
    corrected: CallAnalysis
    note: str | None


def _error_location(kind: ClaimKind, index: int, claim: Claim, verdict: ClaimVerdict) -> str:
    """The `Correction.error_location` for a verdict, in `sawti.memory.induction`'s vocabulary.

    Induction renders only the field at `error_location` (see its
    `_describe_field`), so the path decides what the rule is learned from:

    * reject  -> the whole list (`commitments`, `compliance_flags`) — induction
      shows the count / the flagged rule ids before and after — or, for a
      rubric score, `rubric_scores[<criterion>]` (scored vs not scored).
    * correct -> the one item (`commitments[i]`, `compliance_flags[i]`), with
      every editable field rendered; for a rubric score, `rubric_scores[<criterion>]`
      (score only) or `rubric_scores[<criterion>].justification` when the
      justification changed.
    """
    list_name = {
        "commitment": "commitments",
        "compliance_flag": "compliance_flags",
        "rubric_score": "rubric_scores",
    }[kind]
    if isinstance(claim, RubricScore):
        base = f"rubric_scores[{claim.criterion}]"
        if verdict.changes is not None and "justification" in verdict.changes.model_fields_set:
            return f"{base}.justification"
        return base
    return list_name if verdict.verdict == "reject" else f"{list_name}[{index}]"


def plan_corrections(submission: ReviewSubmission, original: CallAnalysis) -> list[PlannedCorrection]:
    """Map per-claim verdicts onto `sawti.schemas.Correction`s (one per non-confirm verdict).

    Every planned `corrected` differs from `original` in a structured field —
    that difference is what `sawti.memory.induction` learns a rule from:

    * `reject`  -> `corrected` is `original` with that claim removed.
    * `correct` -> `corrected` is `original` with the reviewer's `changes`
      applied to that claim. Changes must apply to the claim's kind and must
      actually change it.
    * `confirm` -> nothing; it is recorded on the `ReviewSubmission` row only.

    Notes: the verdict's own note, then the submission-level note.

    Raises:
        InvalidVerdictError: A change that does not apply to the claim's kind,
            or a `correct` that leaves the claim as it was.
    """
    planned: list[PlannedCorrection] = []
    for verdict in submission.verdicts:
        if verdict.verdict == "confirm":
            continue
        list_name, index = _parse_claim_id(verdict.claim_id)
        kind = dict(CLAIM_LISTS)[list_name]
        claims = list(getattr(original, list_name))
        claim = claims[index]
        if verdict.verdict == "reject":
            del claims[index]
        else:
            assert verdict.changes is not None  # guaranteed by ClaimVerdict's validator
            not_applicable = verdict.changes.model_fields_set - EDITABLE_FIELDS[kind]
            if not_applicable:
                raise InvalidVerdictError(
                    f"{verdict.claim_id}: {', '.join(sorted(not_applicable))} cannot be changed on a {kind}"
                )
            changed = claim.model_copy(update=verdict.changes.as_update())
            changed = type(claim).model_validate(changed.model_dump())  # re-run field validation
            if changed == claim:
                raise InvalidVerdictError(f"{verdict.claim_id}: the changes leave the claim as it was")
            claims[index] = changed
        corrected = original.model_copy(update={list_name: claims})
        note = "\n\n".join(part for part in (verdict.note, submission.note) if part) or None
        planned.append(
            PlannedCorrection(
                error_location=_error_location(kind, index, claim, verdict),
                corrected=corrected,
                note=note,
            )
        )
    return planned


def _parse_claim_id(claim_id: str) -> tuple[str, int]:
    """`"commitments[2]"` -> `("commitments", 2)`."""
    list_name, _, rest = claim_id.partition("[")
    return list_name, int(rest.rstrip("]"))
