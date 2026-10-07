"""Request and response models for the HTTP API — the contract the 6.4 dashboard consumes.

Phase 6.2. Written before the routes, and shaped by the two screens they serve:

  * Review queue — a table of escalated calls with a language filter and
    per-language pending counts (`ReviewQueueResponse`).
  * Call review — the redacted transcript with evidence highlighted in it,
    a Confirm / Correct / Reject verdict per grounded claim, the claims
    grounding rejected, and the memory rules used (`CallDetail`); the verdicts
    come back as a `ReviewSubmission`.

Conventions every consumer can rely on:

  * All text is the *redacted* transcript; the raw text is never stored
    (docs/09-DECISIONS.md, 2026-10-06, decision 1).
  * Every character offset (`TranscriptTurn.start_char`,
    `GroundedClaim.start_char`/`end_char`) indexes the same string,
    `CallDetail.redacted_transcript`, so
    `redacted_transcript[claim.start_char:claim.end_char] == claim.quote`.
  * Per-language numbers are always one field per category (`PendingCounts`),
    never a blended total.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from sawti.data.transcript_parser import Speaker
from sawti.schemas import CallStatus, Language, Severity

#: Which `CallAnalysis` list a claim came from. The claim id is
#: `"<list>[<index>]"`, e.g. `"commitments[0]"` — the same path style as
#: `Correction.error_location`, and stable because a stored analysis is never
#: rewritten.
ClaimKind = Literal["commitment", "compliance_flag", "rubric_score"]

Verdict = Literal["confirm", "correct", "reject"]


# --- Submit -----------------------------------------------------------------


class CallSubmitRequest(BaseModel):
    """`POST /calls`: one call's transcript. Redacted on arrival; the raw text is not kept."""

    transcript: str = Field(
        ...,
        min_length=1,
        max_length=200_000,
        description="One turn per line, each prefixed `Agent:` or `Customer:`.",
    )
    language: Language
    agent_external_id: str = Field(..., min_length=1, max_length=255)


class CallSubmitResponse(BaseModel):
    """`202` body for `POST /calls`."""

    call_id: UUID


# --- Review queue -------------------------------------------------------------


class PendingCounts(BaseModel):
    """Calls awaiting review, per language category. Never a single blended number."""

    ar: int = Field(..., ge=0)
    en: int = Field(..., ge=0)
    mixed: int = Field(..., ge=0)


class ReviewQueueItem(BaseModel):
    """One row of the review queue table."""

    call_id: UUID
    agent_name: str
    language: Language
    grounding_coverage: float | None
    escalation_reason: str | None
    #: When the call entered its current status — for `awaiting_review`, when
    #: the analysis finished and the graph suspended.
    waiting_since: datetime | None


class ReviewQueueResponse(BaseModel):
    """`GET /reviews`: the filtered rows, oldest first, plus per-language pending counts.

    `pending` always covers all three categories and ignores the `language`
    filter, so the screen's counters do not change when the table is filtered.
    """

    items: list[ReviewQueueItem]
    pending: PendingCounts


# --- Call detail --------------------------------------------------------------


class TranscriptTurn(BaseModel):
    """One speaker turn of the redacted transcript."""

    index: int = Field(..., ge=0)
    speaker: Speaker
    #: Script-based, per turn: `ar` (Arabic script only), `en` (no Arabic
    #: script), `mixed` (both). The dashboard sets `dir="rtl"` for `ar`,
    #: `ltr` for `en`, and `auto` for `mixed`.
    language: Language
    text: str
    #: Offset of `text` in `CallDetail.redacted_transcript`.
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., gt=0)


class GroundedClaim(BaseModel):
    """A claim that survived grounding, located in the redacted transcript."""

    claim_id: str
    kind: ClaimKind
    quote: str
    speaker: str
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., gt=0)
    #: The turn containing `start_char`; None only if the quote starts outside
    #: every turn (e.g. inside a speaker prefix).
    turn_index: int | None
    #: The commitment / flag description, or the rubric justification.
    description: str
    # Kind-specific fields; None where they do not apply.
    promised_by: str | None = None
    deadline: datetime | None = None
    rule_id: str | None = None
    severity: Severity | None = None
    criterion: str | None = None
    score: float | None = None


class RejectedClaim(BaseModel):
    """A claim `ground` dropped: its quote is not verbatim in the transcript."""

    kind: ClaimKind
    #: The quote exactly as the model proposed it.
    attempted_quote: str
    speaker: str
    description: str


class MemoryRuleUsed(BaseModel):
    """A memory rule retrieved into `extract`'s prompt for this call."""

    id: UUID
    text: str


class CallDetail(BaseModel):
    """`GET /calls/{call_id}`: everything the call review screen renders.

    Analysis fields are None/empty until the analysis task has finished.
    """

    call_id: UUID
    status: CallStatus
    language: Language
    agent_name: str
    queued_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    pii_redacted_count: int
    #: Times the analysis task claimed the call; >1 means transient-error retries.
    attempts: int

    redacted_transcript: str
    turns: list[TranscriptTurn]

    summary: str | None = None
    confidence: float | None = None
    grounding_coverage: float | None = None
    confidence_threshold: float | None = None
    escalation_reason: str | None = None
    claims: list[GroundedClaim] = Field(default_factory=list)
    rejected_claims: list[RejectedClaim] = Field(default_factory=list)
    memory_rules: list[MemoryRuleUsed] = Field(default_factory=list)
    #: The submitted review, once there is one.
    review: ReviewSubmission | None = None


# --- Review submission ----------------------------------------------------------


#: Which `ClaimChanges` fields apply to which claim kind.
EDITABLE_FIELDS: dict[ClaimKind, frozenset[str]] = {
    "commitment": frozenset({"description", "promised_by", "deadline"}),
    "compliance_flag": frozenset({"rule_id", "severity", "description"}),
    "rubric_score": frozenset({"score", "justification"}),
}


class ClaimChanges(BaseModel):
    """The structured fields a `correct` verdict changes — only the ones the reviewer set.

    "Set" is literal: a field absent from the request is unchanged. For
    `deadline`, an explicit `null` removes the deadline; every other field
    must be given a value if present. Which fields apply depends on the
    claim's kind (`EDITABLE_FIELDS`); the route rejects the rest. The
    evidence quote is not editable: it is what the transcript says.
    """

    description: str | None = Field(default=None, min_length=1, max_length=2_000)
    promised_by: str | None = Field(default=None, min_length=1, max_length=255)
    deadline: datetime | None = None
    rule_id: str | None = Field(default=None, min_length=1, max_length=255)
    severity: Severity | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    justification: str | None = Field(default=None, min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def _only_deadline_may_be_null(self) -> ClaimChanges:
        """Explicit null means "clear", which only makes sense for the optional deadline."""
        for name in self.model_fields_set - {"deadline"}:
            if getattr(self, name) is None:
                raise ValueError(f"{name} cannot be null")
        if not self.model_fields_set:
            raise ValueError("changes must set at least one field")
        return self

    def as_update(self) -> dict[str, object]:
        """Just the fields the reviewer set, for `model_copy(update=...)`."""
        return {name: getattr(self, name) for name in self.model_fields_set}


class ClaimVerdict(BaseModel):
    """A reviewer's verdict on one grounded claim."""

    claim_id: str = Field(..., min_length=1)
    verdict: Verdict
    #: Required with, and only allowed with, `correct`: what the claim should say.
    changes: ClaimChanges | None = None
    #: Optional free-text explanation, any verdict.
    note: str | None = Field(default=None, max_length=10_000)

    @model_validator(mode="after")
    def _changes_only_with_correct(self) -> ClaimVerdict:
        """`correct` must change a structured field; the other verdicts change nothing."""
        if self.verdict == "correct" and self.changes is None:
            raise ValueError(f"{self.claim_id}: a 'correct' verdict needs structured changes")
        if self.verdict != "correct" and self.changes is not None:
            raise ValueError(f"{self.claim_id}: changes are only allowed with 'correct'")
        return self


class ReviewSubmission(BaseModel):
    """`POST /reviews/{call_id}/corrections`: one verdict per grounded claim.

    The route rejects the submission (422) unless the claim ids are exactly the
    call's grounded claims — none missing, none unknown — and every `correct`
    actually changes its claim. Mapping onto `sawti.schemas.Correction`: each
    `reject` / `correct` verdict becomes one `Correction` whose `corrected`
    differs from `original` in structured fields (`reject`: the claim is
    removed; `correct`: the changes are applied), at an `error_location`
    `sawti.memory.induction` can render (`sawti.api.records.plan_corrections`).
    `confirm` verdicts produce no `Correction`; the whole submission is stored
    as a `ReviewSubmission` row. See docs/09-DECISIONS.md, 2026-10-07.
    """

    verdicts: list[ClaimVerdict]
    note: str | None = Field(default=None, max_length=10_000)

    @model_validator(mode="after")
    def _one_verdict_per_claim(self) -> ReviewSubmission:
        """Reject duplicate claim ids — two verdicts for one claim is ambiguous."""
        seen: set[str] = set()
        for verdict in self.verdicts:
            if verdict.claim_id in seen:
                raise ValueError(f"duplicate verdict for {verdict.claim_id}")
            seen.add(verdict.claim_id)
        return self


class ReviewAccepted(BaseModel):
    """`202` body for `POST /reviews/{call_id}/corrections`."""

    call_id: UUID
    submission_id: UUID
    corrections_recorded: int


# --- Memory rules -----------------------------------------------------------------


class MemoryRuleView(BaseModel):
    """One active memory rule."""

    id: UUID
    text: str
    #: How many corrections it was induced (or merged) from.
    source_corrections: int
    created_at: datetime


class MemoryRuleListResponse(BaseModel):
    """`GET /memory/rules`: active rules, newest first."""

    rules: list[MemoryRuleView]


# --- Health ---------------------------------------------------------------------


class HealthResponse(BaseModel):
    """`GET /health`: overall status plus each dependency's."""

    status: Literal["ok", "degraded"]
    database: Literal["ok", "error"]
    redis: Literal["ok", "error"]


CallDetail.model_rebuild()
