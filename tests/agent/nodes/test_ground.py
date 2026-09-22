"""Tests for `sawti.agent.nodes.ground`.

Phase 2, stage 2: mirrors `src/sawti/agent/nodes/ground.py`.

The failure paths are the point of this file: a claim quoting real text at the
wrong offsets must be dropped, not kept with a caveat.
"""

from __future__ import annotations

from sawti.agent.nodes.ground import ground, is_grounded, locate_claim
from sawti.agent.state import AgentState
from sawti.schemas import Commitment, ComplianceFlag, Quote, RubricScore, Severity

TRANSCRIPT = (
    "Agent: Good morning, this is Sara from Orange.\n"
    "Customer: My bill is wrong again.\n"
    "Agent: I will refund the difference by Sunday.\n"
)


def _located_quote(text: str, speaker: str = "Agent") -> Quote:
    """Build a quote whose offsets are computed from TRANSCRIPT — i.e. correct."""
    start = TRANSCRIPT.index(text)
    return Quote(text=text, speaker=speaker, start_char=start, end_char=start + len(text))


def _misplaced_quote(text: str, speaker: str = "Agent") -> Quote:
    """Build a quote whose text is real but whose offsets point somewhere else.

    The common real-world case: the model copied the text correctly and then
    guessed the offsets wrong. This is NOT an unsupported claim — grounding
    re-anchors it rather than dropping it.
    """
    return Quote(text=text, speaker=speaker, start_char=0, end_char=len(text))


def _commitment(quote: Quote) -> Commitment:
    """A commitment carrying `quote` as its evidence."""
    return Commitment(evidence=quote, promised_by="Agent", description="Refund the difference.")


def _flag(quote: Quote) -> ComplianceFlag:
    """A compliance flag carrying `quote` as its evidence."""
    return ComplianceFlag(
        evidence=quote,
        rule_id="no_identity_verification",
        severity=Severity.MEDIUM,
        description="No ID check.",
    )


def _rubric(quote: Quote, criterion: str = "empathy") -> RubricScore:
    """A rubric score carrying `quote` as its evidence."""
    return RubricScore(evidence=quote, criterion=criterion, score=0.6, justification="Adequate.")


def test_is_grounded_accepts_a_correctly_located_quote() -> None:
    """is_grounded() is True when the quote text occurs in the transcript."""
    assert is_grounded(TRANSCRIPT, _commitment(_located_quote("I will refund the difference by Sunday.")))


def test_is_grounded_accepts_verbatim_text_at_the_wrong_offsets() -> None:
    """Bad offsets are not an unsupported claim — the text is what is under test.

    Measured on real model output, quote text was verbatim ~100% of the time
    while start_char was right only ~38% of the time. Rejecting on offsets would
    discard honest claims wholesale; see the module docstring.
    """
    claim = _commitment(_misplaced_quote("I will refund the difference by Sunday."))
    assert claim.evidence.text in TRANSCRIPT
    assert is_grounded(TRANSCRIPT, claim) is True


def test_is_grounded_rejects_text_that_is_not_in_the_transcript_at_all() -> None:
    """Failure path: the model invented or paraphrased the quote."""
    invented = Quote(text="I will refund you today.", speaker="Agent", start_char=0, end_char=24)
    assert is_grounded(TRANSCRIPT, _commitment(invented)) is False


def test_is_grounded_rejects_a_paraphrase_without_normalizing_it() -> None:
    """Failure path: normalization is exact, so re-punctuated text is unsupported.

    Forgiving this would forgive the behaviour that produces unsupported claims.
    """
    paraphrased = Quote(
        text="I will refund the difference by sunday!", speaker="Agent", start_char=0, end_char=39
    )
    assert is_grounded(TRANSCRIPT, _commitment(paraphrased)) is False


def test_locate_claim_rewrites_offsets_to_the_real_transcript_span() -> None:
    """A verified claim comes back anchored to offsets computed from the transcript."""
    text = "I will refund the difference by Sunday."
    located = locate_claim(TRANSCRIPT, _commitment(_misplaced_quote(text)))

    assert located is not None
    assert TRANSCRIPT[located.evidence.start_char : located.evidence.end_char] == text
    assert located.evidence.speaker == "Agent"


def test_locate_claim_returns_none_for_an_unsupported_quote() -> None:
    """Failure path: nothing to anchor means nothing to keep."""
    invented = Quote(text="I will refund you today.", speaker="Agent", start_char=0, end_char=24)
    assert locate_claim(TRANSCRIPT, _commitment(invented)) is None


def test_locate_claim_anchors_a_repeated_phrase_to_its_first_occurrence() -> None:
    """Documented limitation: ambiguity resolves to the first match, deterministically."""
    transcript = "Agent: Thank you.\nCustomer: Thank you.\n"
    quote = Quote(text="Thank you.", speaker="Agent", start_char=0, end_char=10)
    located = locate_claim(transcript, _commitment(quote))

    assert located is not None
    assert located.evidence.start_char == transcript.index("Thank you.")


async def test_ground_retains_claims_with_verified_evidence() -> None:
    """ground() keeps claims whose evidence quote verifiably matches the transcript."""
    good = _commitment(_located_quote("I will refund the difference by Sunday."))
    state: AgentState = {"redacted_transcript": TRANSCRIPT, "commitments": [good]}

    update = await ground(state)

    assert update["commitments"] == [good]
    assert update["rejected_claims"] == []
    assert update["grounding_coverage"] == 1.0


async def test_ground_drops_claims_whose_evidence_is_not_in_the_transcript() -> None:
    """ground() removes any claim whose evidence.text does not occur in the transcript."""
    invented = Quote(text="I will refund you today.", speaker="Agent", start_char=0, end_char=24)
    bad = _commitment(invented)
    state: AgentState = {"redacted_transcript": TRANSCRIPT, "commitments": [bad]}

    update = await ground(state)

    assert update["commitments"] == []
    assert update["rejected_claims"] == [bad]
    assert update["grounding_coverage"] == 0.0


async def test_ground_repairs_offsets_rather_than_dropping_the_claim() -> None:
    """A verbatim quote with wrong offsets is kept and re-anchored, not rejected."""
    text = "I will refund the difference by Sunday."
    state: AgentState = {
        "redacted_transcript": TRANSCRIPT,
        "commitments": [_commitment(_misplaced_quote(text))],
    }

    update = await ground(state)

    assert update["rejected_claims"] == []
    assert update["grounding_coverage"] == 1.0
    evidence = update["commitments"][0].evidence
    assert TRANSCRIPT[evidence.start_char : evidence.end_char] == text


async def test_ground_never_downgrades_an_unsupported_claim_to_low_confidence() -> None:
    """ground() drops unsupported claims outright rather than marking them low-confidence.

    The rejected claim must be absent from the live lists entirely — not present
    with an altered score, weight, or annotation.
    """
    invented = Quote(text="The agent was rude throughout.", speaker="Agent", start_char=0, end_char=30)
    bad = _rubric(invented, criterion="accuracy_of_information")
    state: AgentState = {"redacted_transcript": TRANSCRIPT, "rubric_scores": [bad]}

    update = await ground(state)

    assert update["rubric_scores"] == []
    rejected = update["rejected_claims"]
    assert rejected == [bad]
    # Dropped, not mutated: the reject is the original object, score untouched.
    assert rejected[0].score == 0.6


async def test_ground_reports_partial_coverage_for_a_mixed_batch() -> None:
    """A batch that is half grounded reports coverage of exactly 0.5 across all claim types."""
    invented_a = Quote(text="We will call you back.", speaker="Agent", start_char=0, end_char=22)
    invented_b = Quote(text="The agent hung up.", speaker="Agent", start_char=0, end_char=18)
    good_commitment = _commitment(_located_quote("I will refund the difference by Sunday."))
    bad_commitment = _commitment(invented_a)
    good_flag = _flag(_located_quote("Good morning, this is Sara from Orange."))
    bad_rubric = _rubric(invented_b)

    state: AgentState = {
        "redacted_transcript": TRANSCRIPT,
        "commitments": [good_commitment, bad_commitment],
        "compliance_flags": [good_flag],
        "rubric_scores": [bad_rubric],
    }

    update = await ground(state)

    assert update["commitments"] == [good_commitment]
    assert update["compliance_flags"] == [good_flag]
    assert update["rubric_scores"] == []
    assert len(update["rejected_claims"]) == 2
    assert update["grounding_coverage"] == 0.5


async def test_ground_scores_empty_input_as_vacuously_grounded() -> None:
    """No claims means nothing unsupported: coverage is 1.0, not 0.0."""
    update = await ground({"redacted_transcript": TRANSCRIPT})

    assert update["rejected_claims"] == []
    assert update["grounding_coverage"] == 1.0


async def test_ground_checks_against_the_redacted_transcript_the_model_saw() -> None:
    """Offsets are verified against `redacted_transcript`, not the raw one.

    Redaction shifts offsets, so a quote located in the redacted text would fail
    against the raw transcript. Checking the wrong string would reject grounded
    claims wholesale.
    """
    redacted = "Agent: Call me on [PHONE] tomorrow.\n"
    raw = "Agent: Call me on +962 79 123 4567 tomorrow.\n"
    text = "Call me on [PHONE] tomorrow."
    start = redacted.index(text)
    quote = Quote(text=text, speaker="Agent", start_char=start, end_char=start + len(text))

    update = await ground(
        {"transcript": raw, "redacted_transcript": redacted, "commitments": [_commitment(quote)]}
    )

    assert len(update["commitments"]) == 1
    assert update["grounding_coverage"] == 1.0
