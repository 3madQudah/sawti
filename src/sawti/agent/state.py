"""LangGraph state definition shared by every node in the analysis graph.

Phase 2, stage 1: the graph's working state.

Design notes that the rest of phase 2 depends on:

  * This TypedDict is *working* state, not the output contract. Nodes read and
    write named keys here; the single value that leaves the graph as a result
    is `analysis`, validated into `CallAnalysis`. Everything else is scaffolding
    that exists so a later node can do its job.
  * `total=False` throughout: a node writes only the keys it owns, and LangGraph
    merges that partial dict into the running state. A node must therefore never
    assume a key written by a *later* node exists — read with `.get()` and a
    default, or you couple the node to the graph's execution order.
  * Claim-bearing keys hold validated Pydantic models (`Commitment`,
    `ComplianceFlag`, `RubricScore`), never raw dicts. `raw_extraction` is the
    one deliberate exception: it is the unvalidated LLM payload, and it exists
    precisely so that the boundary between "what the model said" and "what
    survived validation" is visible in state rather than hidden inside a node.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from sawti.schemas import (
    CallAnalysis,
    Claim,
    Commitment,
    ComplianceFlag,
    Language,
    RubricScore,
    SentimentTrajectory,
)

# Where a call sits in the human-review workflow. This is graph bookkeeping,
# not part of any persisted schema — `CallAnalysis.requires_human_review` is
# the durable record. It is a Literal rather than an Enum in `schemas.py`
# deliberately: adding a schema type for a value that never crosses the graph
# boundary would be inventing output contract we do not need.
#
#   pending        — no routing decision made yet (stages 1-2).
#   auto_passed    — confidence >= threshold; the graph may return a verdict.
#   awaiting_human — confidence < threshold; the graph interrupts (stage 4).
#   human_reviewed — a reviewer resumed the graph and supplied corrections.
ReviewStatus = Literal["pending", "auto_passed", "awaiting_human", "human_reviewed"]


class AgentState(TypedDict, total=False):
    """Mutable state threaded through the LangGraph analysis graph.

    Keys are grouped by the node that *writes* them. Any node may read any key
    written by a node upstream of it; nothing may read a key written downstream.
    """

    # --- Inputs: populated by the caller before the first node runs ---
    call_id: str
    # The raw transcript. Retained for audit and for offset provenance only.
    transcript: str
    # ar / en / mixed. Carried through the whole graph because every metric this
    # project reports is sliced by language — never blended into one number.
    language: Language
    # PII-stripped transcript. This is what the extraction node sends to the LLM,
    # which makes it — not `transcript` — the string a quote must be verbatim
    # against. `Quote.start_char/end_char` are defined as offsets into "the full
    # transcript text that was passed to the extracting node", so the grounding
    # node in stage 2 must check against this key to stay consistent with the
    # schema. Checking against `transcript` instead would reject every claim
    # whose quote happens to span a redacted span.
    redacted_transcript: str

    # --- Written by `extract` ---
    # Unvalidated LLM output, kept for debugging and for the eval harness to
    # compare "proposed" against "survived grounding".
    raw_extraction: dict[str, Any]
    summary: str
    commitments: list[Commitment]
    compliance_flags: list[ComplianceFlag]
    rubric_scores: list[RubricScore]
    sentiment_trajectory: SentimentTrajectory

    # --- Written by `ground` (stage 2) ---
    # Claims whose evidence did not appear verbatim in `redacted_transcript`.
    # They are removed from the lists above and parked here: dropped, never
    # down-weighted. This key exists so the phase 2 success criterion — that
    # unsupported claims measurably drop versus the phase 1 baseline — is
    # directly countable from a graph run rather than inferred afterwards.
    rejected_claims: list[Claim]
    # Fraction of extracted claims that survived grounding, in [0.0, 1.0].
    # Written by `ground` and consumed by `confidence` in stage 3.
    grounding_coverage: float

    # --- Written by `confidence` (stage 3) ---
    confidence: float
    requires_human_review: bool

    # --- Written by `confidence` and updated by `escalate` (stage 4) ---
    review_status: ReviewStatus
    # Whatever the QA reviewer supplied when resuming an interrupted run. Typed
    # as Any because the resume payload is the review API's contract, not the
    # graph's: phase 3 will send a `Correction`, a test may send a plain dict.
    reviewer_input: Any

    # --- Terminal ---
    # The one validated output contract. Assembled from the keys above once the
    # graph has finished routing.
    analysis: CallAnalysis
    # Set by any node that fails recoverably. A non-None `error` must never be
    # allowed to produce an automated verdict — it routes to human review like
    # any other low-confidence case.
    error: str | None
