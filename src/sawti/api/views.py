"""Turn stored rows into the API's response shapes — pure functions, no I/O.

Phase 6.2. Kept apart from the routes so the offset arithmetic the dashboard
depends on (`redacted_transcript[start:end] == quote`) is tested directly.
"""

from __future__ import annotations

import re
from typing import Any

from sawti.api.schemas import (
    ClaimKind,
    GroundedClaim,
    RejectedClaim,
    TranscriptTurn,
)
from sawti.data.transcript_parser import parse_transcript
from sawti.schemas import CallAnalysis, Claim, Commitment, ComplianceFlag, Language, RubricScore

#: `CallAnalysis` list name -> claim kind. Order is the order claims are listed.
CLAIM_LISTS: tuple[tuple[str, ClaimKind], ...] = (
    ("commitments", "commitment"),
    ("compliance_flags", "compliance_flag"),
    ("rubric_scores", "rubric_score"),
)

_ARABIC_SCRIPT = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_LATIN = re.compile(r"[A-Za-z]")
# Redaction placeholders are Latin-script and would make every redacted
# Arabic turn look code-switched.
_PLACEHOLDER = re.compile(r"\[[A-Z_]+\]")


def turn_language(text: str) -> Language:
    """Classify one turn by script: Arabic only -> ar, both -> mixed, otherwise en.

    Script, not language identification: Arabizi (Arabic in Latin letters)
    reads as `en`. Good enough for its one job, choosing text direction.
    """
    text = _PLACEHOLDER.sub("", text)
    has_arabic = bool(_ARABIC_SCRIPT.search(text))
    has_latin = bool(_LATIN.search(text))
    if has_arabic and has_latin:
        return Language.MIXED
    return Language.AR if has_arabic else Language.EN


def split_turns(redacted_transcript: str, call_id: str) -> list[TranscriptTurn]:
    """Split the redacted transcript into turns with offsets into it.

    Reuses `sawti.data.transcript_parser.parse_transcript` — the one parser in
    this codebase — and derives each turn's offset from its source line, so
    the parser itself is unchanged.

    Raises:
        UnattributableLineError: A line has no `Agent:`/`Customer:` prefix.
            The API rejects such a transcript at submission rather than guess.
    """
    line_starts: list[int] = []
    position = 0
    for line in redacted_transcript.splitlines(keepends=True):
        line_starts.append(position)
        position += len(line)

    turns: list[TranscriptTurn] = []
    for index, turn in enumerate(parse_transcript(redacted_transcript, call_id)):
        start = redacted_transcript.index(turn.text, line_starts[turn.source_line - 1])
        turns.append(
            TranscriptTurn(
                index=index,
                speaker=turn.speaker,
                language=turn_language(turn.text),
                text=turn.text,
                start_char=start,
                end_char=start + len(turn.text),
            )
        )
    return turns


def _turn_for(offset: int, turns: list[TranscriptTurn]) -> int | None:
    for turn in turns:
        if turn.start_char <= offset < turn.end_char:
            return turn.index
    return None


def _description(claim: Claim) -> str:
    if isinstance(claim, RubricScore):
        return claim.justification
    if isinstance(claim, Commitment | ComplianceFlag):
        return claim.description
    return ""


def grounded_claims(analysis: CallAnalysis, turns: list[TranscriptTurn]) -> list[GroundedClaim]:
    """Every claim in a stored analysis, with its stable id and its turn.

    The offsets are the ones `ground` computed against the redacted
    transcript; they are passed through, not recomputed.
    """
    claims: list[GroundedClaim] = []
    for list_name, kind in CLAIM_LISTS:
        for index, claim in enumerate(getattr(analysis, list_name)):
            evidence = claim.evidence
            view = GroundedClaim(
                claim_id=f"{list_name}[{index}]",
                kind=kind,
                quote=evidence.text,
                speaker=evidence.speaker,
                start_char=evidence.start_char,
                end_char=evidence.end_char,
                turn_index=_turn_for(evidence.start_char, turns),
                description=_description(claim),
            )
            if isinstance(claim, Commitment):
                view.promised_by, view.deadline = claim.promised_by, claim.deadline
            elif isinstance(claim, ComplianceFlag):
                view.rule_id, view.severity = claim.rule_id, claim.severity
            elif isinstance(claim, RubricScore):
                view.criterion, view.score = claim.criterion, claim.score
            claims.append(view)
    return claims


def claim_kind(claim: Claim) -> ClaimKind:
    """The `ClaimKind` of a claim instance."""
    if isinstance(claim, Commitment):
        return "commitment"
    if isinstance(claim, ComplianceFlag):
        return "compliance_flag"
    return "rubric_score"


def rejected_claims_payload(rejected: list[Claim]) -> list[dict[str, Any]]:
    """Serialize `ground`'s rejects for `call_analyses.rejected_claims`."""
    return [{"kind": claim_kind(claim), "claim": claim.model_dump(mode="json")} for claim in rejected]


def rejected_claims_view(stored: list[dict[str, Any]]) -> list[RejectedClaim]:
    """Read `call_analyses.rejected_claims` back as `RejectedClaim`s."""
    views = []
    for item in stored:
        claim = item["claim"]
        views.append(
            RejectedClaim(
                kind=item["kind"],
                attempted_quote=claim["evidence"]["text"],
                speaker=claim["evidence"]["speaker"],
                description=claim.get("justification") or claim.get("description") or "",
            )
        )
    return views
