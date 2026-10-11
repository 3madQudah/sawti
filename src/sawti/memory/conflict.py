"""Detect conflicts between a candidate memory rule and existing stored rules.

Phase 4: agent memory — runs before insert.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from sawti.llm.provider import get_llm_provider, run_with_timeout
from sawti.memory.rule_schema import MemoryRule
from sawti.memory.store import Embedder, MemoryStore

# Two-stage, for cost: an LLM contradiction judgment for every pair would be
# O(existing rules) provider calls per insert. Similarity narrows the field
# first (reusing MemoryStore itself — see detect_conflict), and only the
# `similarity_top_k` most topically-similar existing rules get an actual
# LLM judgment. A rule with nothing in common with the candidate cannot
# contradict it, so nothing is lost by not asking about it.
DEFAULT_SIMILARITY_TOP_K = 5

_CONFLICT_SYSTEM_PROMPT = (
    "You judge whether two rules for a bilingual (Arabic/English) "
    "contact-center call analysis agent genuinely CONTRADICT each other — "
    "one instructs the agent toward X in some situation, the other toward "
    "an incompatible action in that SAME situation, such that a system "
    "trying to follow both would not know what to do.\n"
    "\n"
    "Being about the same topic is NOT a contradiction. Two rules can both "
    "concern, say, compliance flags or commitment deadlines without "
    "disagreeing — they may cover different cases, agree, or simply be "
    "redundant with each other. Only mark contradicts=true when following "
    "both rules together is impossible, or they give opposite guidance for "
    "the same case.\n"
    "\n"
    "Contradiction, e.g.: \"Flag a missing deadline as a compliance issue\" "
    "vs. \"A missing deadline is acceptable and should not be flagged.\"\n"
    "Not a contradiction, e.g.: \"Flag missing refund deadlines\" vs. "
    "\"Flag missing callback deadlines\" — different cases; both can be "
    "followed at once."
)


class _ConflictJudgment(BaseModel):
    """Transport shape for the model's response to one candidate/existing-rule pair."""

    contradicts: bool
    reasoning: str = Field(
        default="", description="One sentence on why — kept for a human reviewing a flagged conflict."
    )


def _judges_contradiction(candidate: MemoryRule, existing_rule: MemoryRule) -> _ConflictJudgment:
    prompt = f"Rule A (candidate): {candidate.rule_text}\nRule B (existing): {existing_rule.rule_text}"
    return run_with_timeout(
        get_llm_provider().structured_complete(
            prompt,
            response_model=_ConflictJudgment,
            system=_CONFLICT_SYSTEM_PROMPT,
            temperature=0.0,
        )
    )


@dataclass(frozen=True)
class ConflictDetail:
    """One existing rule judged to contradict a candidate, and the evidence for it.

    Phase 6.4: kept so a human resolving the conflict sees why it was flagged.
    Before, only the rule survived and the rest was discarded.
    """

    rule: MemoryRule
    #: Cosine similarity of the two rule texts (the narrowing stage's score).
    similarity: float
    #: The judge's one-sentence reason.
    reasoning: str


def detect_conflict_details(
    candidate: MemoryRule,
    existing: list[MemoryRule],
    *,
    similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
    embed: Embedder | None = None,
) -> list[ConflictDetail]:
    """`detect_conflict()`, keeping each conflict's similarity and the judge's reasoning.

    Same two stages and the same provider calls as `detect_conflict()`.

    Returns:
        One `ConflictDetail` per existing rule judged to contradict
        `candidate`, most similar first.
    """
    if not existing:
        return []

    narrowing_store = MemoryStore(embed=embed)
    for rule in existing:
        narrowing_store.add(rule)
    similar = narrowing_store.retrieve_scored(candidate.rule_text, top_k=similarity_top_k)

    details = []
    for similarity, rule in similar:
        judgment = _judges_contradiction(candidate, rule)
        if judgment.contradicts:
            details.append(ConflictDetail(rule=rule, similarity=similarity, reasoning=judgment.reasoning))
    return details


def detect_conflict(
    candidate: MemoryRule,
    existing: list[MemoryRule],
    *,
    similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
    embed: Embedder | None = None,
) -> list[MemoryRule]:
    """Return existing rules that genuinely contradict `candidate`.

    Must run before a new rule is inserted into the store — see
    `insert_rule()`, which wires this in.

    Two stages: `existing` is first narrowed to the `similarity_top_k` rules
    most topically similar to `candidate` (a throwaway `MemoryStore` built
    from `existing`, reusing Stage 4's embedding/similarity machinery rather
    than duplicating it), then each of those gets an actual LLM judgment of
    contradiction — not just topical similarity, which two compatible or
    even redundant rules can share. `detect_conflict_details()` does the
    same and keeps the evidence.

    Args:
        candidate: The newly induced rule, not yet stored.
        existing: Currently stored rules to check against.
        similarity_top_k: How many topically-similar rules to actually run
            an LLM contradiction check against. Bounds provider calls per
            insert rather than checking against every existing rule.
        embed: Passed through to the narrowing `MemoryStore`. Defaults to
            its own default (a real embedder); tests inject a fake.

    Returns:
        The subset of `existing` that an LLM judged as contradicting
        `candidate`, not merely topically related to it.
    """
    details = detect_conflict_details(candidate, existing, similarity_top_k=similarity_top_k, embed=embed)
    return [detail.rule for detail in details]


@dataclass
class InsertOutcome:
    """What happened when `insert_rule()` tried to insert a candidate."""

    inserted: bool
    candidate: MemoryRule
    conflicts: list[MemoryRule] = field(default_factory=list)
    #: The same conflicts with their similarity and the judge's reasoning (phase 6.4).
    conflict_details: list[ConflictDetail] = field(default_factory=list)


def insert_rule(
    store: MemoryStore,
    candidate: MemoryRule,
    *,
    similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
    embed: Embedder | None = None,
) -> InsertOutcome:
    """Insert `candidate` into `store`, unless it conflicts with an existing rule.

    This is the orchestration point that wires `detect_conflict()` in before
    `MemoryStore.add()` — deliberately a separate function rather than
    baking the check into `MemoryStore.add()` itself, so `add()` stays the
    low-level "embed and store" primitive Stage 4 already tests and Stage 6
    (`consolidate`) needs for merged/replacement writes that have already
    been reasoned about upstream. `insert_rule()` is the path a freshly
    induced candidate is meant to go through.

    **On a real conflict, the design decision — stated explicitly, since
    it's consequential: the candidate is NOT inserted, and the conflicting
    existing rule(s) are left untouched.** Neither the new rule nor the old
    one is silently preferred; both are surfaced (via the returned
    `InsertOutcome.conflicts`) for a human to resolve. This follows the same
    standing rule the rest of this project applies to any decision the
    system cannot make with confidence — a rejected `CallAnalysis` routes to
    human review rather than an automated verdict; a contradictory rule does
    the same rather than either silently overwriting institutional
    knowledge or silently admitting a rule that fights with one already in
    use. There is no review queue yet (the same Phase 6 gap as everywhere
    else in this project without one) — callers are responsible for
    surfacing `InsertOutcome.conflicts` somewhere a human will see it.

    Args:
        store: The store to insert into.
        candidate: The newly induced rule to insert.
        similarity_top_k: Forwarded to `detect_conflict()`.
        embed: Forwarded to `detect_conflict()`.

    Returns:
        An `InsertOutcome` recording whether the insert happened and, if
        not, which existing rules it conflicted with.
    """
    details = detect_conflict_details(
        candidate, store.all_active_rules(), similarity_top_k=similarity_top_k, embed=embed
    )
    if details:
        return InsertOutcome(
            inserted=False,
            candidate=candidate,
            conflicts=[detail.rule for detail in details],
            conflict_details=details,
        )

    store.add(candidate)
    return InsertOutcome(inserted=True, candidate=candidate)
