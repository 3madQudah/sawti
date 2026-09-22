"""Quote verification node — rejects any claim whose evidence isn't a verbatim transcript match.

Phase 2, stage 2: the node the whole phase exists for.

The rule, and why it is this one:

    A claim is grounded if its quote **text** appears verbatim in the transcript.
    Its offsets are then **computed from the transcript**, never trusted from the
    model.

Normalization is exact — byte-for-byte, no whitespace or punctuation folding, no
Arabic diacritic stripping, no case folding. The extraction prompt tells the
model to copy, not to retype. Forgiving paraphrase, re-punctuation, or "fixed"
diacritics here would forgive exactly the behaviour that produces unsupported
claims, raising the pass rate without raising truthfulness.

What is deliberately *not* checked is the model's character offsets. Measured on
real Gemini output: quote text was verbatim 100% of the time, while the model's
`start_char` was correct only ~38% of the time, drifting further the deeper into
the transcript a quote sat. Offsets are an arithmetic artifact the model
estimates rather than counts; rejecting an honest claim for them is a false
rejection, not a caught hallucination. So the text is the claim under test, and
the offsets become derived facts — located with `sawti.quotes.find_quote_matches`,
the same helper the ground-truth authoring path uses for the same reason.

The consequence worth stating: a quote appearing more than once in a transcript
is anchored to its first occurrence. That can attribute a claim to the wrong
instance of a repeated phrase. It cannot make an unsupported claim look
supported, which is the failure this node exists to prevent.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from sawti.agent.state import AgentState
from sawti.quotes import UNKNOWN_SPEAKER, find_quote_matches
from sawti.schemas import Claim, Quote

logger = logging.getLogger(__name__)

ClaimT = TypeVar("ClaimT", bound=Claim)

# Claim-bearing state keys this node filters, named here for tests and readers
# (the node itself unpacks them explicitly — see `ground`). Sentiment points
# are excluded:
# `SentimentPoint` is not a `Claim` subclass, so it cannot be parked in
# `rejected_claims`, and a sentiment reading is an interpretation of the call
# rather than an assertion about it.
_CLAIM_KEYS = ("commitments", "compliance_flags", "rubric_scores")


def is_grounded(transcript: str, claim: Claim) -> bool:
    """Return whether `claim`'s evidence quote really appears in the transcript.

    Role in the graph: the single predicate the grounding node is built on,
    separated out so it can be tested directly and reused by the eval layer.

    Args:
        transcript: The exact text the extracting node showed the model.
        claim: Any `Claim` subclass instance.

    Returns:
        True if the quote's text occurs verbatim in `transcript`. The claim's own
        offsets are ignored — see the module docstring.
    """
    return bool(find_quote_matches(transcript, claim.evidence.text)) if claim.evidence.text else False


def locate_claim(transcript: str, claim: ClaimT) -> ClaimT | None:
    """Return `claim` with transcript-computed offsets, or None if it is unsupported.

    Role in the graph: turns a verified claim into a *correctly anchored* one. The
    returned copy's `start_char`/`end_char` are computed from the transcript and
    its speaker is parsed from the containing line, so nothing downstream ever
    relies on a number the model supplied.

    Args:
        transcript: The exact text the extracting node showed the model.
        claim: The claim to verify and anchor.

    Returns:
        A copy of `claim` with corrected evidence, or None when the quote text
        does not appear in the transcript at all.
    """
    matches = find_quote_matches(transcript, claim.evidence.text) if claim.evidence.text else []
    if not matches:
        return None
    # First occurrence: see the module docstring on repeated phrases.
    match = matches[0]
    located = Quote(
        text=claim.evidence.text,
        speaker=match.speaker or claim.evidence.speaker or UNKNOWN_SPEAKER,
        start_char=match.start_char,
        end_char=match.end_char,
    )
    return claim.model_copy(update={"evidence": located})


def _partition(transcript: str, claims: list[ClaimT]) -> tuple[list[ClaimT], list[ClaimT]]:
    """Split `claims` into (grounded-and-anchored, rejected) against `transcript`.

    Returns:
        Two lists: the surviving claims, each re-anchored to real transcript
        offsets, and the rejected originals, untouched. Order is preserved.
    """
    kept: list[ClaimT] = []
    dropped: list[ClaimT] = []
    for claim in claims:
        located = locate_claim(transcript, claim)
        if located is None:
            # Rejected claims are stored exactly as the model produced them —
            # they are evidence about the model, so they are not normalized.
            dropped.append(claim)
        else:
            kept.append(located)
    return kept, dropped


async def ground(state: AgentState) -> dict[str, Any]:
    """Verify every extracted claim's evidence quote against the source transcript.

    Role in the graph: the gate between "the model said so" and "the transcript
    shows it". It runs before scoring, compliance, and confidence, so no
    downstream node can ever read an unsupported claim.

    Any claim whose `evidence.text` does not appear verbatim anywhere in the
    transcript is moved to `rejected_claims` — not kept with a lower weight, not
    annotated, not scored down. Dropped. Surviving claims are re-anchored to
    offsets computed from the transcript.

    Args:
        state: Current agent state, containing the extraction node's proposals.

    Returns:
        A partial state update with the filtered and re-anchored claim lists, the
        rejects, and `grounding_coverage` — the fraction of proposals that survived.
    """
    # Offsets are defined against the text the model was shown, which is the
    # redacted transcript (see `AgentState.redacted_transcript`). Falling back to
    # the raw transcript keeps the node usable when redaction did not run.
    transcript = state.get("redacted_transcript") or state.get("transcript", "")

    # Each key is handled explicitly rather than in a loop over `_CLAIM_KEYS`:
    # a TypedDict only types its keys when they are literals, so a loop variable
    # would erase the element types and make the partition untyped.
    commitments, rejected_commitments = _partition(transcript, list(state.get("commitments", [])))
    flags, rejected_flags = _partition(transcript, list(state.get("compliance_flags", [])))
    scores, rejected_scores = _partition(transcript, list(state.get("rubric_scores", [])))

    rejected: list[Claim] = [*rejected_commitments, *rejected_flags, *rejected_scores]
    survived = len(commitments) + len(flags) + len(scores)
    total = survived + len(rejected)

    update: dict[str, Any] = {
        "commitments": commitments,
        "compliance_flags": flags,
        "rubric_scores": scores,
    }

    if rejected:
        logger.info(
            "ground: dropped %d/%d unsupported claim(s) for %s",
            len(rejected),
            total,
            state.get("call_id"),
        )

    update["rejected_claims"] = rejected
    # A call that proposed nothing has nothing unsupported in it. Scoring that as
    # 0.0 would route every silent transcript to a human for the wrong reason;
    # vacuous truth is the honest reading, and emptiness shows up in the claim
    # counts rather than being smuggled into the coverage number.
    update["grounding_coverage"] = survived / total if total else 1.0

    return update
