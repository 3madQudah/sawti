"""Tests for `sawti.agent.nodes.score`.

Phase 2, stage 2: mirrors `src/sawti/agent/nodes/score.py`.
"""

from __future__ import annotations

from sawti.agent.nodes.score import reconcile_rubric, score
from sawti.data.ground_truth import RUBRIC_CRITERIA
from sawti.schemas import Quote, RubricScore

TRANSCRIPT = "Agent: Good morning, this is Sara from Orange.\n"


def _rubric(criterion: str, value: float = 0.6) -> RubricScore:
    """A grounded rubric score for `criterion`."""
    text = "Good morning, this is Sara from Orange."
    start = TRANSCRIPT.index(text)
    return RubricScore(
        evidence=Quote(text=text, speaker="Agent", start_char=start, end_char=start + len(text)),
        criterion=criterion,
        score=value,
        justification="Greeted and identified.",
    )


async def test_score_produces_exactly_one_rubric_score_per_rubric_criterion() -> None:
    """score() produces exactly one RubricScore per rubric criterion."""
    proposed = [_rubric(name) for name in RUBRIC_CRITERIA]
    update = await score({"rubric_scores": proposed})

    kept = update["rubric_scores"]
    assert len(kept) == len(RUBRIC_CRITERIA)
    assert [s.criterion for s in kept] == list(RUBRIC_CRITERIA)


def test_reconcile_rubric_drops_criteria_outside_the_canonical_set() -> None:
    """Failure path: a criterion the model invented is not comparable, so it is discarded."""
    kept, missing = reconcile_rubric([_rubric("professionalism"), _rubric("vibes")])

    assert [s.criterion for s in kept] == ["professionalism"]
    assert "vibes" not in {s.criterion for s in kept}
    assert len(missing) == len(RUBRIC_CRITERIA) - 1


def test_reconcile_rubric_collapses_duplicates_to_the_first_occurrence() -> None:
    """Failure path: a repeated criterion would silently shadow itself downstream."""
    kept, _ = reconcile_rubric([_rubric("empathy", 0.3), _rubric("empathy", 0.9)])

    assert len(kept) == 1
    assert kept[0].score == 0.3


def test_reconcile_rubric_reports_criteria_that_were_never_scored() -> None:
    """A criterion whose evidence failed grounding is reported missing, never invented."""
    kept, missing = reconcile_rubric([_rubric("call_closure")])

    assert len(kept) == 1
    assert set(missing) == set(RUBRIC_CRITERIA) - {"call_closure"}


def test_reconcile_rubric_returns_canonical_order_regardless_of_input_order() -> None:
    """Two runs over the same call must produce comparable output ordering."""
    shuffled = [_rubric(name) for name in reversed(RUBRIC_CRITERIA)]
    kept, _ = reconcile_rubric(shuffled)

    assert [s.criterion for s in kept] == list(RUBRIC_CRITERIA)


async def test_score_never_fabricates_a_score_for_an_unevidenced_criterion() -> None:
    """score() never emits a RubricScore without a grounded evidence quote.

    Nothing is invented to fill a gap — a fabricated score would need a
    fabricated quote, which is the exact failure grounding exists to prevent.
    """
    update = await score({"rubric_scores": []})

    assert update["rubric_scores"] == []
