"""QA rubric scoring node — enforces the canonical criteria set over extracted scores.

Phase 2, stage 2.

This node does real work rather than re-asking a model. `extract` already
produces `RubricScore` entries, but it scores "the criteria you can evidence" —
it does not know, and is not told, that the project has a fixed, agreed set of
exactly seven criteria (`sawti.data.ground_truth.RUBRIC_CRITERIA`). Reconciling
free-form model output against that canonical set is a deterministic job, and
doing it here means the graph never issues a second LLM call that could disagree
with the first about the same transcript.

What is deliberately NOT done: inventing a score for a criterion the model did
not evidence. A fabricated `RubricScore` would need a fabricated quote, which is
exactly the failure grounding exists to prevent. Missing criteria are reported
as missing.
"""

from __future__ import annotations

import logging
from typing import Any

from sawti.agent.state import AgentState
from sawti.data.ground_truth import RUBRIC_CRITERIA
from sawti.schemas import RubricScore

logger = logging.getLogger(__name__)


def reconcile_rubric(scores: list[RubricScore]) -> tuple[list[RubricScore], list[str]]:
    """Reduce extracted rubric scores to at most one per canonical criterion.

    Role in the graph: the testable core of the scoring node.

    Two things get fixed here. Criteria the model invented (anything outside
    `RUBRIC_CRITERIA`) are discarded, because a score on a criterion nobody
    agreed to is not comparable against the reference labels. Duplicates are
    reduced to the first occurrence, because `CallAnalysis` consumers and
    `rubric_agreement_by_category` both key on criterion name and a second entry
    would silently shadow the first.

    Args:
        scores: Rubric scores as extracted — already grounded by `ground`.

    Returns:
        `(kept, missing)` — the canonical-order surviving scores, and the names
        of canonical criteria that were not scored at all.
    """
    by_criterion: dict[str, RubricScore] = {}
    for score in scores:
        if score.criterion not in RUBRIC_CRITERIA:
            continue
        # First occurrence wins: it is the one the model committed to before it
        # started repeating itself.
        by_criterion.setdefault(score.criterion, score)

    # Canonical order, not model order, so two runs over the same call produce
    # comparable output.
    kept = [by_criterion[name] for name in RUBRIC_CRITERIA if name in by_criterion]
    missing = [name for name in RUBRIC_CRITERIA if name not in by_criterion]
    return kept, missing


async def score(state: AgentState) -> dict[str, Any]:
    """Score the call against the QA rubric, producing grounded `RubricScore` entries.

    Role in the graph: runs after `ground`, so every score it sees is already
    evidence-verified. Its job is completeness and canonicality, not judgment —
    it decides which of the model's scores count, never what they should be.

    Args:
        state: Current agent state, containing grounded claims and the transcript.

    Returns:
        A partial state update with `rubric_scores` reconciled against the
        canonical seven criteria.
    """
    kept, missing = reconcile_rubric(list(state.get("rubric_scores", [])))

    if missing:
        # Not an error: a criterion can go unscored because its evidence failed
        # grounding, which is the system working. It is logged because an
        # incomplete rubric is worth seeing in a trace.
        logger.info("score: %s unscored for %s", ", ".join(missing), state.get("call_id"))

    return {"rubric_scores": kept}
