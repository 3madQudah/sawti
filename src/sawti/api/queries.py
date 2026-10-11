"""Read-side queries the dashboard routes share (`/memory/*`, `/stats`).

Phase 6.4. Read-only. Every write to the memory store goes through
`sawti.memory.persistence`.
"""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Iterable

import sqlalchemy as sa
from sqlalchemy.orm import Session

from sawti.db.models import MemoryRuleRecord, ReviewerAction
from sawti.schemas import Language

_LANGUAGE_ORDER = {language: index for index, language in enumerate(Language)}


def correction_languages(session: Session, correction_ids: Iterable[uuid.UUID]) -> dict[uuid.UUID, Language]:
    """The language category of each correction's call, from `Correction.original.language`."""
    ids = list(set(correction_ids))
    if not ids:
        return {}
    rows = session.execute(
        sa.select(ReviewerAction.id, ReviewerAction.payload["original"]["language"].astext).where(
            ReviewerAction.id.in_(ids)
        )
    )
    return {row_id: Language(language) for row_id, language in rows if language is not None}


def rule_languages(session: Session, rules: list[MemoryRuleRecord]) -> dict[uuid.UUID, list[Language]]:
    """Per rule, the categories its source corrections came from, in `Language` order."""
    sources = {rule.id: [uuid.UUID(str(i)) for i in rule.source_correction_ids] for rule in rules}
    by_correction = correction_languages(session, (c for ids in sources.values() for c in ids))
    found = {
        rule_id: {by_correction[c] for c in ids if c in by_correction} for rule_id, ids in sources.items()
    }
    return {rule_id: sorted(langs, key=_LANGUAGE_ORDER.__getitem__) for rule_id, langs in found.items()}


def analyses_used(session: Session) -> Counter[uuid.UUID]:
    """How many stored analyses retrieved each rule into their prompt (`call_analyses.retrieved_rule_ids`)."""
    rows = session.execute(
        sa.text(
            "SELECT rule_id, count(*) FROM call_analyses, "
            "jsonb_array_elements_text(retrieved_rule_ids) AS rule_id GROUP BY rule_id"
        )
    )
    return Counter({uuid.UUID(rule_id): count for rule_id, count in rows})


def active_rules(session: Session) -> list[MemoryRuleRecord]:
    """Active (non-retired) rules, newest first."""
    return (
        session.query(MemoryRuleRecord)
        .filter_by(retired=False)
        .order_by(MemoryRuleRecord.created_at.desc())
        .all()
    )
