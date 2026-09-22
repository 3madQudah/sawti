"""Confidence estimation node — determines automated-verdict vs. human-review routing.

Phase 2, stage 3.

The formula is deliberately the simplest one that means something:

    confidence = grounding_coverage

That is the fraction of proposed claims whose evidence survived verification in
`ground`. It is chosen over a model self-rating for one reason: it is *measured*,
not asserted. A model asked "how confident are you?" produces a number with no
external referent, and the phase 1 reference generator already showed models are
happy to sound certain about quotes they paraphrased. Grounding coverage is
computed from the transcript, so it cannot be talked up.

It is also the number phase 2 is trying to move. Tying routing to it means the
system escalates exactly when it has been caught being unsupported — the failure
mode this phase exists to reduce.

MVP scope, stated so it is not mistaken for a finished model: this ignores
extraction completeness (a call that proposed one claim and grounded it scores
1.0, same as one that proposed twenty), rubric coverage, and any notion of model
self-certainty. Folding those in is future work — see `docs/09-DECISIONS.md`.
"""

from __future__ import annotations

import logging
from typing import Any

from sawti.agent.state import AgentState
from sawti.config import get_settings

logger = logging.getLogger(__name__)


def compute_confidence(state: AgentState) -> float:
    """Derive an overall confidence score for the analysis so far.

    Role in the graph: the testable core of the confidence node.

    A run that recorded an `error` scores 0.0 regardless of coverage. Partial
    output from a failed run is not evidence of a good analysis, and letting a
    vacuous 1.0 coverage (no claims proposed, because extraction died) mean
    "confident" would route a broken run straight to an automated verdict.

    Args:
        state: Current agent state, after `ground` has run.

    Returns:
        Confidence in [0.0, 1.0].
    """
    if state.get("error") is not None:
        return 0.0
    # Absent coverage means `ground` never ran — treat as no evidence of quality
    # rather than assuming the best.
    return float(state.get("grounding_coverage", 0.0))


async def confidence(state: AgentState) -> dict[str, Any]:
    """Compute an overall confidence score for the call analysis so far.

    Role in the graph: the branch point's input. It writes the two keys
    `route_after_confidence` reads, and nothing else decides routing.

    Per project rule, low-confidence results must route to human review, never
    to an automated verdict — this node sets `requires_human_review` accordingly
    for the graph's conditional edge to act on.

    The comparison is `>= threshold` for auto-pass, so a score sitting exactly on
    the threshold passes. The threshold is read from
    `sawti.config.Settings.confidence_threshold` on every call rather than
    captured at import, so an operator can retune it without a redeploy.

    Args:
        state: Current agent state, containing grounded claims and coverage.

    Returns:
        A partial state update with `confidence`, `requires_human_review`, and
        the human-review `review_status`.
    """
    threshold = get_settings().confidence_threshold
    score = compute_confidence(state)
    needs_human = score < threshold

    logger.info(
        "confidence for %s: %.3f (threshold %.2f) -> %s",
        state.get("call_id"),
        score,
        threshold,
        "human review" if needs_human else "auto-pass",
    )

    return {
        "confidence": score,
        "requires_human_review": needs_human,
        "review_status": "awaiting_human" if needs_human else "auto_passed",
    }
