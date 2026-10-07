"""Reviewer agreement with the agent, per language category, from the review submissions in the database.

Phase 6.2. See `sawti.eval.metrics.reviewer_agreement_by_category` for the
two rates and what population they describe (escalated calls only).

Usage:
    uv run python scripts/reviewer_agreement.py
"""

from __future__ import annotations

import json

from sawti.db.models import Call, CallAnalysisRecord, ReviewSubmission
from sawti.db.session import get_session
from sawti.eval.metrics import reviewer_agreement_by_category
from sawti.schemas import Language


def load_reviews() -> list[tuple[Language, list[str]]]:
    with get_session() as session:
        rows = (
            session.query(Call.language, ReviewSubmission.verdicts)
            .join(CallAnalysisRecord, CallAnalysisRecord.call_id == Call.id)
            .join(ReviewSubmission, ReviewSubmission.call_analysis_id == CallAnalysisRecord.id)
            .all()
        )
    return [(Language(lang), [v["verdict"] for v in verdicts["verdicts"]]) for lang, verdicts in rows]


def main() -> int:
    reviews = load_reviews()
    result = {lang.value: stats for lang, stats in reviewer_agreement_by_category(reviews).items()}
    print(json.dumps({"submissions": len(reviews), "per_language": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
