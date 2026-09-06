"""Evaluation metrics — always computed and reported per language category.

Phase 1: eval foundation. Never add a function here that returns a single
blended float across languages; every metric returns a per-category mapping.
Structurally enforced by the return type: `dict[Language, float]`, not `float`.
"""

from __future__ import annotations

from sawti.schemas import CallAnalysis, Language


def accuracy_by_category(
    predictions: list[CallAnalysis], ground_truth: list[CallAnalysis]
) -> dict[Language, float]:
    """Compute accuracy, grouped by `Language` category.

    Args:
        predictions: Model-produced analyses.
        ground_truth: Corresponding reviewed ground-truth analyses.

    Returns:
        A mapping from each `Language` present in the data to its accuracy.
        Never a single blended float across all categories.

    Raises:
        NotImplementedError: Until scoring is implemented.
    """
    # TODO(phase-1): group predictions/ground_truth by language, score each group
    # independently, return the per-category mapping.
    raise NotImplementedError


def rubric_agreement_by_category(
    predictions: list[CallAnalysis], ground_truth: list[CallAnalysis]
) -> dict[Language, float]:
    """Compute rubric-score agreement with ground truth, grouped by `Language` category.

    Args:
        predictions: Model-produced analyses.
        ground_truth: Corresponding reviewed ground-truth analyses.

    Returns:
        A mapping from each `Language` present in the data to its agreement score.

    Raises:
        NotImplementedError: Until scoring is implemented.
    """
    # TODO(phase-1): group by language, compute rubric agreement per group.
    raise NotImplementedError


def grounding_precision_by_category(predictions: list[CallAnalysis]) -> dict[Language, float]:
    """Compute the fraction of claims whose evidence verifiably quotes the transcript, per category.

    Args:
        predictions: Model-produced analyses to check.

    Returns:
        A mapping from each `Language` present in the data to its grounding precision.

    Raises:
        NotImplementedError: Until scoring is implemented.
    """
    # TODO(phase-1): group by language, verify each claim's evidence, return per-category precision.
    raise NotImplementedError
