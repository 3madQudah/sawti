"""Claim/commitment/sentiment extraction node — first stage of the analysis graph.

Phase 2, stage 2: the only node in the graph that talks to a model.

Everything this node produces is a *proposal*. The model is asked to copy
evidence verbatim and to guess character offsets, and it will get offsets wrong
regularly — that is expected, designed for, and precisely why `ground` runs
immediately downstream. Nothing here is treated as fact.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field, model_validator

from sawti.agent.state import AgentState
from sawti.data.ground_truth import RUBRIC_CRITERIA
from sawti.llm.provider import get_llm_provider
from sawti.privacy.redaction import redact
from sawti.schemas import (
    Commitment,
    ComplianceFlag,
    ExtractionProposal,
    Quote,
    RubricScore,
    SentimentPoint,
)

logger = logging.getLogger(__name__)

# The model is told to produce offsets even though it is bad at them. The
# alternative — asking only for quote text and locating it ourselves, as
# `sawti.data.ground_truth` does — would make `ground` a formality, because
# programmatically located offsets are correct by construction. Phase 2 exists
# to measure how often a model's own claims are unsupported, so the model has to
# be allowed to be wrong in a way we can detect.
EXTRACTION_SYSTEM_PROMPT = (
    "You are a QA analyst at a bilingual (Arabic/English) contact center in Jordan. "
    "Analyze the single call transcript given to you and report what it actually "
    "contains — never what a typical call of this kind usually contains.\n"
    "\n"
    "EVIDENCE RULES — these matter more than anything else:\n"
    "1. Every commitment, compliance flag, and rubric score MUST carry an "
    "`evidence` quote whose `text` is an EXACT, CHARACTER-FOR-CHARACTER COPY of a "
    "span from the transcript. Copy it; do not retype, translate, summarize, "
    "normalize spelling or punctuation, fix typos, or alter diacritics. For "
    "Arabic, reproduce the exact Arabic script as written.\n"
    "2. Do NOT include the 'Agent: ' or 'Customer: ' line prefix inside the quote "
    "text itself. Put the speaker in the `speaker` field instead.\n"
    "3. Keep each quote to a single clean span — one clause or one sentence — but "
    "long enough to be unambiguous.\n"
    "4. For each quote also give your best-guess `start_char` and `end_char`: the "
    "0-based character offsets of that span within the transcript text exactly as "
    "it was given to you, where `end_char - start_char` equals the length of your "
    "quote text. Counting characters is hard and your guess will often be wrong; "
    "give your honest best estimate anyway. Never alter the quote text to make "
    "your offsets look right — the text is what matters.\n"
    "5. If the transcript does not support a claim, omit the claim. An empty list "
    "is always better than an invented entry.\n"
    "\n"
    "WHAT TO REPORT:\n"
    "- summary: what the customer wanted, what the agent did, how it ended.\n"
    "- commitments: every promise by either party (callback, refund, cancellation, "
    "escalation, a fix within some time). One entry per promise.\n"
    "- compliance_flags: every point where the agent broke a contact center rule — "
    "failing to identify themselves or the company, failing to verify identity "
    "before discussing account details, rudeness, interrupting, improper "
    "disclosure, promising what they cannot deliver, closing without confirming "
    "resolution, ignoring an explicit request. Give a short snake_case rule_id and "
    "a severity.\n"
    "- rubric_scores: score EVERY ONE of these 7 criteria, using exactly these "
    f"names, one entry each, no extras and none missing: {', '.join(RUBRIC_CRITERIA)}. "
    "Each needs a justification and an evidence quote. Use the full 0.0-1.0 range; a "
    "merely adequate call is around 0.6-0.7, not 0.9.\n"
    "- sentiment_trajectory: 3 to 6 points tracking the CUSTOMER's sentiment across "
    "the call, in order, with increasing timestamp_sec (~8 seconds per turn)."
)


# --- Transport shapes -------------------------------------------------------
#
# The model is asked for character offsets it is demonstrably bad at computing.
# In practice it returns offsets whose *arithmetic* is wrong — end_char minus
# start_char not equal to len(text) — because it estimates spans rather than
# counting them, and Arabic text makes that worse.
#
# `Quote` rejects that outright, which means one bad offset pair fails the whole
# `ExtractionProposal` and loses every claim in the call, including the
# well-located ones. That is the wrong failure: arithmetic consistency is a fact
# about the quote's own fields, not a claim about the transcript, so there is
# nothing for `ground` to verify in it.
#
# These lax mirrors repair that arithmetic on the way in and leave the part that
# actually matters untouched: `start_char` is the model's positional claim, and
# it is carried through unchanged for `ground` to check against the transcript.
# A model that guesses the wrong position still gets its claim dropped — which
# is exactly the behaviour phase 2 exists to measure.


class _LaxQuote(Quote):
    """A `Quote` that repairs self-inconsistent offsets before validation.

    Role: transport only. Nothing outside this module holds a `_LaxQuote` — the
    proposal is re-validated into the strict schema before it enters state.
    """

    @model_validator(mode="before")
    @classmethod
    def _repair_end_char(cls, data: Any) -> Any:
        """Derive `end_char` from `start_char` and the text, discarding the model's guess."""
        if not isinstance(data, dict):
            return data
        text = data.get("text")
        if not isinstance(text, str) or not text:
            return data
        start = data.get("start_char")
        if not isinstance(start, int) or start < 0:
            start = 0
        # The model's positional claim is preserved; only the redundant end
        # offset is recomputed, because it carries no independent information.
        return {**data, "start_char": start, "end_char": start + len(text)}


class _LaxCommitment(Commitment):
    """A `Commitment` whose evidence tolerates model-supplied offset arithmetic."""

    evidence: _LaxQuote


class _LaxComplianceFlag(ComplianceFlag):
    """A `ComplianceFlag` whose evidence tolerates model-supplied offset arithmetic."""

    evidence: _LaxQuote


class _LaxRubricScore(RubricScore):
    """A `RubricScore` whose evidence tolerates model-supplied offset arithmetic."""

    evidence: _LaxQuote


class _LaxSentimentPoint(SentimentPoint):
    """A `SentimentPoint` whose quote tolerates model-supplied offset arithmetic."""

    quote: _LaxQuote


class _LaxSentimentTrajectory(BaseModel):
    """A trajectory over lax points.

    Standalone rather than a `SentimentTrajectory` subclass: `list` is invariant,
    so narrowing the element type in a subclass is not a valid override.
    """

    points: list[_LaxSentimentPoint] = Field(default_factory=list)


class _LaxExtractionProposal(BaseModel):
    """The shape actually requested from the provider.

    Field-for-field the same as `ExtractionProposal` — same names, same required
    evidence — differing only in tolerating offset arithmetic the model cannot
    get right. Standalone rather than a subclass because `list` is invariant.
    It is normalized into the strict `ExtractionProposal` immediately after
    parsing, so the strict schema stays the contract everything else sees.
    """

    summary: str = Field(..., min_length=1)
    commitments: list[_LaxCommitment] = Field(default_factory=list)
    compliance_flags: list[_LaxComplianceFlag] = Field(default_factory=list)
    rubric_scores: list[_LaxRubricScore] = Field(default_factory=list)
    sentiment_trajectory: _LaxSentimentTrajectory = Field(default_factory=_LaxSentimentTrajectory)


async def extract(state: AgentState) -> dict[str, Any]:
    """Extract candidate claims, commitments, and sentiment points from the transcript.

    Role in the graph: the entry node. It turns transcript text into structured
    *proposals* and guarantees the PII rule holds before any text leaves the
    process — if the caller did not supply `redacted_transcript`, this node
    redacts `transcript` itself rather than silently sending raw text to a
    cloud model.

    Memory-rule retrieval is deliberately absent. `sawti.memory.store` is phase 4
    and raises `NotImplementedError` today; wiring a call to it now would either
    crash the graph or need a stub that pretends rules exist. The extraction
    prompt is a plain analyst prompt until phase 4 gives it something real to
    inject.

    A provider failure is recorded in `error` rather than raised: an exception
    would abort the run, while an `error` routes through `route_after_confidence`
    to human review, which is the behaviour the project rule demands.

    Args:
        state: Current agent state; must contain `transcript` or `redacted_transcript`.

    Returns:
        A partial state update holding the raw proposal plus the parsed, *ungrounded*
        claim lists. Every quote in them is unverified until `ground` runs.
    """
    redacted = state.get("redacted_transcript")
    if not redacted:
        source = state.get("transcript", "")
        if not source:
            return {"error": "extract: no transcript or redacted_transcript in state"}
        # PII redaction before any text reaches a model — not negotiable.
        redacted = redact(source).redacted_text

    try:
        lax = await get_llm_provider().structured_complete(
            redacted,
            response_model=_LaxExtractionProposal,
            system=EXTRACTION_SYSTEM_PROMPT,
        )
        # Back to the strict contract before anything else sees it.
        proposal = ExtractionProposal.model_validate(lax.model_dump())
    except Exception as exc:
        logger.error("extract failed for %s: %r", state.get("call_id"), exc)
        return {"redacted_transcript": redacted, "error": f"extract: {exc!r}"}

    return {
        "redacted_transcript": redacted,
        # Kept verbatim so the eval harness can compare what the model *proposed*
        # against what survived grounding. That difference is the phase 2 metric.
        "raw_extraction": proposal.model_dump(mode="json"),
        "summary": proposal.summary,
        "commitments": proposal.commitments,
        "compliance_flags": proposal.compliance_flags,
        "rubric_scores": proposal.rubric_scores,
        "sentiment_trajectory": proposal.sentiment_trajectory,
    }
