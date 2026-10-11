"""`GET /stats`: the dashboard's per-language counters.

Phase 6.4. Every number is per language category, never blended; every rate
carries its n, and n = 0 is an answer (`value: null`), not a zero.
"""

from __future__ import annotations

import sqlalchemy as sa
from fastapi import APIRouter

from sawti.api.deps import CurrentReviewer, DbSession
from sawti.api.queries import active_rules, correction_languages, rule_languages
from sawti.api.schemas import LanguageStats, Rate, StatsResponse
from sawti.db.models import Call, CallAnalysisRecord, MemoryConflict, ReviewSubmission
from sawti.schemas import CallStatus, Language

router = APIRouter(prefix="/stats", tags=["stats"])


def _rate(numerator: int, n: int) -> Rate:
    return Rate(value=numerator / n if n else None, numerator=numerator, n=n)


@router.get("", response_model=StatsResponse)
def stats(session: DbSession, _: CurrentReviewer) -> StatsResponse:
    """Pending reviews, reviewer agreement, active rules and conflict false positives, per language."""
    pending_reviews: dict[str, int] = {
        language: count
        for language, count in session.query(Call.language, sa.func.count())
        .filter(Call.status == CallStatus.AWAITING_REVIEW)
        .group_by(Call.language)
    }

    submissions = (
        session.query(Call.language, ReviewSubmission.verdicts)
        .join(CallAnalysisRecord, CallAnalysisRecord.id == ReviewSubmission.call_analysis_id)
        .join(Call, Call.id == CallAnalysisRecord.call_id)
        .all()
    )
    # Claim-level agreement, the definition of `claim_agreement` in
    # `sawti.eval.metrics.reviewer_agreement_by_category` (a test holds them
    # equal). Counted here because that module imports the agent, which the
    # API process never does.
    verdicts_by_language: dict[str, list[str]] = {language.value: [] for language in Language}
    for language, submission in submissions:
        verdicts_by_language[language].extend(v["verdict"] for v in submission["verdicts"])

    rules = active_rules(session)
    rule_langs = rule_languages(session, rules)

    conflicts = session.query(MemoryConflict.reviewer_action_id, MemoryConflict.resolution).all()
    conflict_langs = correction_languages(session, (action_id for action_id, _ in conflicts))

    blocks = {}
    for language in Language:
        verdicts = verdicts_by_language[language.value]
        resolved = [r for a, r in conflicts if conflict_langs.get(a) == language and r is not None]
        blocks[language.value] = LanguageStats(
            pending_reviews=pending_reviews.get(language.value, 0),
            reviewer_agreement=_rate(verdicts.count("confirm"), len(verdicts)),
            active_rules=sum(language in langs for langs in rule_langs.values()),
            pending_conflicts=sum(conflict_langs.get(a) == language and r is None for a, r in conflicts),
            conflict_false_positive_rate=_rate(resolved.count("not_a_conflict"), len(resolved)),
        )
    return StatsResponse(**blocks, active_rules_distinct=len(rules))
