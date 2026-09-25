"""Induce a general memory rule from one or more specific corrections.

Phase 4: agent memory.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from sawti.llm.provider import get_llm_provider, run_with_timeout
from sawti.memory.rule_schema import MemoryRule
from sawti.schemas import CallAnalysis, Correction

# A judgment call, stated explicitly: this prompt asks for the *general
# principle* behind a correction, not a restatement of the specific call —
# see the module docstring's example. Structural properties (ids tracked,
# single vs. multi-correction merge) are what tests/memory/test_induction.py
# checks; whether the induced text is actually a *good* rule is judged by
# inspection (scripts/inspect_induced_rules.py), not asserted in tests.
_INDUCTION_SYSTEM_PROMPT = (
    "You distill QA reviewer corrections into reusable rules for a bilingual "
    "(Arabic/English) contact-center call analysis agent.\n"
    "\n"
    "Given one or more corrections — what the agent originally produced versus "
    "what a reviewer corrected it to — state the GENERAL PRINCIPLE the agent "
    "should follow to avoid repeating this class of mistake. Do not restate "
    "the specific call's content.\n"
    "\n"
    "Good: \"Flag commitments that promise an action without a stated "
    "timeframe as incomplete.\"\n"
    "Bad: \"call_0042's refund commitment was marked complete but shouldn't "
    "have been.\"\n"
    "\n"
    "Each correction below shows ONLY the one field the reviewer changed —\n"
    "not the whole analysis. Base the rule strictly on that field. Do not\n"
    "invent guidance about other fields (rubric scores, compliance flags,\n"
    "summaries, ...) that were not shown to you, even if they seem like\n"
    "plausible general advice.\n"
    "\n"
    "If more than one correction is given, find the single principle that "
    "covers all of them — do not just list each one separately. One or two "
    "sentences. No call ids, no mention of 'the reviewer' or 'this call'."
)


class _InducedRule(BaseModel):
    """Transport shape for the model's response — just the generalized rule text.

    `MemoryRule`'s other fields (`id`, `created_at`, `source_correction_ids`)
    are not the model's to decide; `induce_rule()` sets them itself.
    """

    rule_text: str = Field(..., min_length=1)


def _render_analysis(analysis: CallAnalysis) -> str:
    """Compact, QA-relevant summary of a *whole* analysis — the fallback for an unrecognized `error_location`.

    Deliberately omits free text (`summary`, descriptions, justifications):
    the model is being asked to generalize about *structure* (was a
    deadline given, which rules were flagged, how scores diverged), not to
    compare wording between two independently-authored analyses.

    `_describe_field` is preferred whenever `error_location` is recognized —
    showing the model the whole analysis when only one field actually
    changed invites it to invent guidance about fields nobody corrected.
    """
    commitments = (
        "; ".join(
            f"deadline={'given' if commitment.deadline else 'none'}"
            for commitment in analysis.commitments
        )
        or "none"
    )
    flags = ", ".join(flag.rule_id for flag in analysis.compliance_flags) or "none"
    scores = ", ".join(f"{score.criterion}={score.score:.2f}" for score in analysis.rubric_scores) or "none"
    return f"commitments=[{commitments}]; compliance_flags=[{flags}]; rubric_scores=[{scores}]"


_COMMITMENT_DEADLINE_RE = re.compile(r"^commitments\[(\d+)\]\.deadline$")
_RUBRIC_SCORE_RE = re.compile(r"^rubric_scores\[(.+)\]$")


def _describe_field(analysis: CallAnalysis, error_location: str) -> str:
    """Render just the one field named by `error_location`, not the whole analysis.

    Mirrors `scripts.generate_synthetic_corrections._first_difference`'s own
    vocabulary of paths, since that function is what produces these
    `error_location` values in the first place. Falls back to
    `_render_analysis` for a path outside that vocabulary — e.g. a real
    future reviewer's own field path — so induction never breaks on it, at
    the cost of that one correction's prompt being less targeted.
    """
    if error_location == "commitments":
        return f"{len(analysis.commitments)} commitment(s)"

    deadline_match = _COMMITMENT_DEADLINE_RE.match(error_location)
    if deadline_match:
        index = int(deadline_match.group(1))
        commitment = analysis.commitments[index] if index < len(analysis.commitments) else None
        return f"deadline={'given' if commitment and commitment.deadline else 'none'}"

    if error_location == "compliance_flags":
        flags = ", ".join(flag.rule_id for flag in analysis.compliance_flags) or "none"
        return f"flags=[{flags}]"

    rubric_match = _RUBRIC_SCORE_RE.match(error_location)
    if rubric_match:
        criterion = rubric_match.group(1)
        score = next((s.score for s in analysis.rubric_scores if s.criterion == criterion), None)
        return f"{criterion}={score:.2f}" if score is not None else f"{criterion}=not scored"

    return _render_analysis(analysis)


def _describe_correction(index: int, correction: Correction) -> str:
    """Render one correction as prompt text: what changed, at the field the reviewer actually flagged."""
    lines = [
        f"Correction {index} (error at `{correction.error_location}`):",
        f"  original:  {_describe_field(correction.original, correction.error_location)}",
        f"  corrected: {_describe_field(correction.corrected, correction.error_location)}",
    ]
    if correction.note:
        lines.append(f"  reviewer note: {correction.note}")
    return "\n".join(lines)


def _build_induction_prompt(corrections: list[Correction]) -> str:
    """Render every correction into one prompt, in a fixed, model-friendly order."""
    descriptions = [
        _describe_correction(index, correction) for index, correction in enumerate(corrections, start=1)
    ]
    return "\n\n".join(descriptions)


def induce_rule(corrections: list[Correction]) -> MemoryRule:
    """Generalize one or more corrections into a reusable `MemoryRule`.

    Args:
        corrections: One or more `Correction` records to generalize from.

    Returns:
        The induced `MemoryRule`, with `source_correction_ids` set from
        every input correction's `id`.

    Raises:
        ValueError: If `corrections` is empty — there is nothing to generalize from.
    """
    if not corrections:
        raise ValueError("induce_rule() needs at least one Correction to generalize from.")

    induced = run_with_timeout(
        get_llm_provider().structured_complete(
            _build_induction_prompt(corrections),
            response_model=_InducedRule,
            system=_INDUCTION_SYSTEM_PROMPT,
            temperature=0.2,
        )
    )

    return MemoryRule(
        rule_text=induced.rule_text,
        source_correction_ids=[correction.id for correction in corrections],
        created_at=datetime.now(UTC),
    )
