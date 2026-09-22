"""Canonical Pydantic v2 schemas for every value that crosses an agent node boundary.

Phase 0: this is the source of truth for output shapes across all later phases.

Rules encoded here (do not weaken these when extending the module):
  * Every agent output is a validated Pydantic model — no raw dicts cross a
    node boundary.
  * Every `Claim` (and subclass) carries a non-empty verbatim `evidence`
    quote. A claim without a quote is invalid, not "low confidence".
  * `CallAnalysis.requires_human_review` is structurally forced to `True`
    whenever `confidence` falls below the configured threshold.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from itertools import pairwise
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from sawti.config import get_settings


class Language(str, Enum):
    """Language category for a transcript, utterance, or evaluation slice.

    Evaluation is always reported per member of this enum — `ar`, `en`,
    `mixed` — never as a single blended number. See `sawti.eval.metrics`.
    """

    AR = "ar"
    EN = "en"
    MIXED = "mixed"


class Severity(str, Enum):
    """Severity of a compliance violation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Quote(BaseModel):
    """A verbatim excerpt from a call transcript, anchored to exact character offsets.

    This is the unit of grounding for every extracted claim. `start_char`
    and `end_char` index into the full transcript text that was passed to
    the extracting node, so a quote can always be traced back to its exact
    source span.
    """

    text: str = Field(..., min_length=1, description="Verbatim text as it appears in the transcript.")
    speaker: str = Field(..., min_length=1, description="Speaker label/id assigned by diarization.")
    start_char: int = Field(..., ge=0, description="Inclusive start offset into the full transcript text.")
    end_char: int = Field(..., gt=0, description="Exclusive end offset into the full transcript text.")

    @model_validator(mode="after")
    def _validate_offsets(self) -> Quote:
        """Ensure offsets are ordered and consistent with the quoted text length."""
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        if self.end_char - self.start_char != len(self.text):
            raise ValueError("end_char - start_char must equal len(text)")
        return self


class Claim(BaseModel):
    """Base class for any assertion an agent extracts from a call.

    Every subclass inherits the mandatory, non-empty `evidence` quote. This
    is enforced at the schema level: a claim whose evidence is missing or
    blank fails validation and can never be constructed, let alone flow
    downstream as "low confidence".
    """

    evidence: Quote = Field(..., description="Verbatim quote supporting this claim. Mandatory, non-empty.")

    @field_validator("evidence")
    @classmethod
    def _evidence_must_be_grounded(cls, value: Quote) -> Quote:
        """Reject claims whose evidence quote is blank or whitespace-only."""
        if not value.text.strip():
            raise ValueError("Claim.evidence.text must be non-empty — an ungrounded claim is invalid.")
        return value


class Commitment(Claim):
    """A promise made during the call: what was promised, by whom, and by when."""

    promised_by: str = Field(..., min_length=1, description="Speaker label of who made the commitment.")
    description: str = Field(..., min_length=1, description="What was promised, in normalized form.")
    deadline: datetime | None = Field(default=None, description="Stated deadline, if any was given.")


class SentimentPoint(BaseModel):
    """A single sentiment sample anchored to a point in the call."""

    quote: Quote = Field(..., description="The utterance this sentiment reading is derived from.")
    speaker: str = Field(..., min_length=1)
    score: float = Field(..., ge=-1.0, le=1.0, description="Polarity, -1 (negative) to 1 (positive).")
    timestamp_sec: float = Field(..., ge=0.0, description="Offset in seconds from call start.")


class SentimentTrajectory(BaseModel):
    """Sentiment across the full call, as a chronologically ordered sequence of samples."""

    points: list[SentimentPoint] = Field(default_factory=list)

    @field_validator("points")
    @classmethod
    def _points_must_be_chronological(cls, value: list[SentimentPoint]) -> list[SentimentPoint]:
        """Reject trajectories whose points are not in non-decreasing timestamp order."""
        for earlier, later in pairwise(value):
            if later.timestamp_sec < earlier.timestamp_sec:
                raise ValueError("SentimentTrajectory.points must be chronologically ordered")
        return value


class ComplianceFlag(Claim):
    """A detected violation of a compliance rule, grounded in a transcript quote."""

    rule_id: str = Field(..., min_length=1, description="Identifier of the violated rule.")
    severity: Severity
    description: str = Field(..., min_length=1)


class RubricScore(Claim):
    """A single QA rubric criterion score, grounded in a transcript quote."""

    criterion: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0)
    justification: str = Field(..., min_length=1)


class ExtractionProposal(BaseModel):
    """Raw shape the LLM is asked to produce during extraction.

    NOT the validated agent output — this exists only as the response_model
    for LLMProvider.structured_complete(). It mirrors CallAnalysis's
    extractable fields but omits what extraction doesn't decide yet (id,
    call_id, language, confidence, requires_human_review).

    Commitment/ComplianceFlag/RubricScore all require a valid Quote, which
    only checks internal arithmetic (end_char - start_char == len(text)) —
    not that the offsets actually point at that text in the real transcript.
    A model can satisfy the first while being wrong about the second.
    ground() closes that gap by re-checking every quote against the real
    transcript at those offsets. A model's offsets are a proposal, never a
    fact, until grounding confirms them.
    """

    summary: str = Field(..., min_length=1)
    commitments: list[Commitment] = Field(default_factory=list)
    compliance_flags: list[ComplianceFlag] = Field(default_factory=list)
    rubric_scores: list[RubricScore] = Field(default_factory=list)
    sentiment_trajectory: SentimentTrajectory = Field(default_factory=SentimentTrajectory)


class CallAnalysis(BaseModel):
    """Top-level output of the analysis agent for a single call.

    Aggregates every extracted artifact for the call plus the overall
    confidence and the resulting human-review routing decision.
    """

    id: UUID = Field(default_factory=uuid4)
    call_id: str = Field(..., min_length=1)
    language: Language
    summary: str = Field(..., min_length=1)
    commitments: list[Commitment] = Field(default_factory=list)
    compliance_flags: list[ComplianceFlag] = Field(default_factory=list)
    rubric_scores: list[RubricScore] = Field(default_factory=list)
    sentiment_trajectory: SentimentTrajectory = Field(default_factory=SentimentTrajectory)
    confidence: float = Field(..., ge=0.0, le=1.0)
    requires_human_review: bool = Field(
        ...,
        description=(
            "Must be True whenever `confidence` is below the configured threshold. "
            "Low-confidence results route to human review, never to an automated verdict."
        ),
    )

    @model_validator(mode="after")
    def _enforce_human_review_routing(self) -> CallAnalysis:
        """Force `requires_human_review = True` whenever confidence is below threshold.

        This makes the "low confidence routes to human review" rule a schema
        invariant rather than something a caller can forget to check.
        """
        threshold = get_settings().confidence_threshold
        if self.confidence < threshold and not self.requires_human_review:
            raise ValueError(
                f"confidence ({self.confidence}) is below the configured threshold "
                f"({threshold}) but requires_human_review is False."
            )
        return self


class Correction(BaseModel):
    """A QA reviewer's correction to a `CallAnalysis`, feeding phase 4 memory induction.

    `reviewer_id` and `timestamp` are mandatory — the audit trail this
    system relies on depends on always knowing who corrected what and when.
    """

    id: UUID = Field(default_factory=uuid4)
    call_id: str = Field(..., min_length=1)
    original: CallAnalysis = Field(..., description="The unreviewed agent output.")
    corrected: CallAnalysis = Field(..., description="The reviewer's corrected version.")
    error_location: str = Field(
        ...,
        min_length=1,
        description="Field path or artifact id of what was wrong, e.g. 'rubric_scores[2]'.",
    )
    note: str | None = Field(default=None, description="Free-text explanation from the reviewer.")
    reviewer_id: str = Field(..., min_length=1, description="Identity of the QA reviewer. Mandatory.")
    timestamp: datetime = Field(..., description="When the correction was made. Mandatory.")
