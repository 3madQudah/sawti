"""Re-score the phase 4 analyses stored in the database: grounding metrics before and after the fix.

Phase 6.2 follow-up, zero LLM calls. The 2026-10-07 metric fix makes
`grounding_precision_by_category` and `unsupported_claim_rate_by_category`
check quotes against the *redacted* transcript (what the model saw, what
`ground` verifies) instead of the raw file. Phase 4's full prediction sets
were never saved; the analyses that *were* saved are the `original` side of
every stored correction — one per `call_analyses` row. This scores exactly
those, both ways, so the two numbers differ only by the metric.

Also reported, from the same rows:

  * sentiment points the 2026-10-07 `ground` rule would drop (quote not
    verbatim in the redacted transcript), per language — these analyses
    predate that rule;
  * how many of the 150 phase 1 transcripts per language contain redactable
    PII at all — the only calls where the old metric could be biased, which
    bounds the bias in the phase 2 numbers (whose predictions were not saved).

Usage:
    uv run python scripts/rescore_stored_analyses.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sawti.db.models import CallAnalysisRecord, ReviewerAction
from sawti.db.session import get_session
from sawti.eval.metrics import grounding_precision_by_category, unsupported_claim_rate_by_category
from sawti.privacy.redaction import redact
from sawti.schemas import CallAnalysis, Language

SYNTHETIC_DIR = Path("data/synthetic")


def stored_analyses() -> dict[str, list[CallAnalysis]]:
    """Stored analyses grouped by the reviewer id of their correction (= which phase 4 process wrote them)."""
    groups: dict[str, list[CallAnalysis]] = defaultdict(list)
    with get_session() as session:
        rows = session.query(CallAnalysisRecord, ReviewerAction.reviewer_id).join(
            ReviewerAction, ReviewerAction.call_analysis_id == CallAnalysisRecord.id
        )
        for record, reviewer_id in rows:
            groups[reviewer_id].append(CallAnalysis.model_validate(record.payload))
    return groups


def score(analyses: list[CallAnalysis]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for view in ("raw", "redacted"):
        precision = grounding_precision_by_category(
            analyses, transcript_dir=SYNTHETIC_DIR, transcript_view=view
        )
        unsupported = unsupported_claim_rate_by_category(
            analyses, transcript_dir=SYNTHETIC_DIR, transcript_view=view
        )
        for language in Language:
            if language in precision:
                cell = out.setdefault(language.value, {"n": sum(a.language == language for a in analyses)})
                cell[f"grounding_precision_{view}"] = precision[language]
                cell[f"unsupported_claim_rate_{view}"] = unsupported[language]
    return out


def sentiment_drops(analyses: list[CallAnalysis]) -> dict[str, dict[str, int]]:
    """Per language: sentiment points total, and how many the new `ground` rule would drop."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for analysis in analyses:
        path = SYNTHETIC_DIR / f"{analysis.call_id}.txt"
        if not path.is_file():
            continue
        transcript = redact(path.read_text(encoding="utf-8")).redacted_text
        for point in analysis.sentiment_trajectory.points:
            counts[analysis.language.value]["points"] += 1
            if point.quote.text not in transcript:
                counts[analysis.language.value]["dropped"] += 1
    return {language: dict(c) for language, c in sorted(counts.items())}


def pii_transcripts() -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for path in sorted(SYNTHETIC_DIR.glob("call_*.txt")):
        language = path.stem.rsplit("_", 1)[1]
        counts[language]["transcripts"] += 1
        counts[language]["with_pii"] += int(redact(path.read_text(encoding="utf-8")).redaction_count > 0)
    return {language: dict(c) for language, c in sorted(counts.items())}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    groups = stored_analyses()
    everything = [a for group in groups.values() for a in group]
    result = {
        "by_source": {source: score(group) for source, group in sorted(groups.items())},
        "all_stored": score(everything),
        "sentiment_drops": sentiment_drops(everything),
        "phase1_transcripts_with_pii": pii_transcripts(),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
