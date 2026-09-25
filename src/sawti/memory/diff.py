"""Find the first QA-relevant field where two analyses of the same call disagree.

Phase 4: shared by `scripts/generate_synthetic_corrections.py` (Stage 2) and
`sawti.eval.experiments.five_batch` (Stage 8) — both need the same
"is this call's agent output actually wrong, and where" judgment, and
duplicating it would let the two silently drift apart on what counts as a
disagreement worth a `Correction`.
"""

from __future__ import annotations

import re

from sawti.data.ground_truth import RUBRIC_CRITERIA
from sawti.schemas import CallAnalysis

_INDEX_RE = re.compile(r"\[[^\]]*\]")

# Rubric scores are 0.0-1.0. A judgment call, stated explicitly (project
# convention — see sawti.eval.metrics): a gap this size is treated as a real
# QA disagreement worth a correction, not the ordinary variance of two
# independently-written scores that both land in the same rough range.
RUBRIC_SCORE_TOLERANCE = 0.25


def first_difference(agent: CallAnalysis, truth: CallAnalysis) -> str | None:
    """Find the first QA-relevant field where `agent` and `truth` disagree.

    Deliberately narrow, and a judgment call stated explicitly: `summary`,
    commitment/flag descriptions, and rubric justifications are free text
    that two independently-authored analyses almost never match word for
    word even when they agree in substance. Diffing them directly would
    make every call "differ" on wording, which is not a QA-actionable
    correction and would flood induction with noise instead of the general
    principle it is supposed to find. Only structurally meaningful content
    is compared, in a fixed order:

      1. commitments — count, then (index-paired) whether a deadline was
         given at all.
      2. compliance_flags — the set of `rule_id`s raised.
      3. rubric_scores — each of the seven canonical criteria, when both
         sides scored it, differing by more than `RUBRIC_SCORE_TOLERANCE`.

    Args:
        agent: A model-produced analysis (the Phase 2 grounded agent's own
            output, with or without memory rules applied).
        truth: The corresponding ground-truth reference label.

    Returns:
        A field path in `sawti.schemas.Correction.error_location`'s style
        (e.g. `"commitments[0].deadline"`), or `None` if nothing comparable
        differs.
    """
    if len(agent.commitments) != len(truth.commitments):
        return "commitments"
    for index, (predicted, expected) in enumerate(zip(agent.commitments, truth.commitments, strict=True)):
        if bool(predicted.deadline) != bool(expected.deadline):
            return f"commitments[{index}].deadline"

    agent_flag_ids = {flag.rule_id for flag in agent.compliance_flags}
    truth_flag_ids = {flag.rule_id for flag in truth.compliance_flags}
    if agent_flag_ids != truth_flag_ids:
        return "compliance_flags"

    agent_scores = {score.criterion: score.score for score in agent.rubric_scores}
    truth_scores = {score.criterion: score.score for score in truth.rubric_scores}
    for criterion in RUBRIC_CRITERIA:
        if criterion not in agent_scores or criterion not in truth_scores:
            continue
        if abs(agent_scores[criterion] - truth_scores[criterion]) > RUBRIC_SCORE_TOLERANCE:
            return f"rubric_scores[{criterion}]"

    return None


def topic_for(error_location: str) -> str:
    """Normalize an `error_location` to its field, dropping list indices and bracketed criteria.

    `"commitments[0].deadline"` and `"commitments[2].deadline"` both become
    `"commitments.deadline"`; `"rubric_scores[empathy]"` becomes
    `"rubric_scores"`; a location with no brackets is returned unchanged.
    Two corrections (or a correction and a live call's own diff) are "about
    the same thing" when their topics match — used to judge whether a
    specific retrieved `sawti.memory.rule_schema.MemoryRule` actually helped
    on a call, in `sawti.eval.experiments.five_batch`, and to group
    corrections for inspection in `scripts/inspect_induced_rules.py`.
    """
    return _INDEX_RE.sub("", error_location)
