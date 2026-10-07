"""Backfill: run the service's memory loop over every correction not yet induced.

Phase 6.2. The 103 phase 4/5 corrections in the dev database predate the
service; this pushes them through exactly the code the service runs after a
review (`sawti.memory.persistence.induce_and_store`: induce, conflict-check,
store, consolidate) so `memory_rules` starts from what they teach.

Cost and pacing, on the Gemini free tier (15 requests/min, 500/day): about
one induction call plus up to five contradiction checks per correction, so
~600 requests for 103 corrections. Every LLM call here is throttled to one
per `--min-interval` seconds. An ordinary 429 waits and retries; the *daily*
cap (`...PerDay...`) cannot be waited out, so the script stops cleanly. Each
correction commits as it finishes (`reviewer_actions.induced_at`), so
re-running the same command resumes where it stopped.

The report at the end is read from the database, not from this run's
counters, so it is complete across resumed runs: corrections induced and
their outcome per language category of the corrected call, and the active
rules whose source corrections include each language (a rule merged from
corrections in two languages counts in both).

Usage:
    uv run python scripts/induce_rules_from_corrections.py
    uv run python scripts/induce_rules_from_corrections.py --report-only
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter, defaultdict
from typing import Any

from sawti.db.models import MemoryRuleRecord, ReviewerAction
from sawti.db.session import get_session
from sawti.llm.provider import LLMProvider, get_llm_provider, is_transient_provider_error
from sawti.memory import conflict, consolidate, induction
from sawti.memory.persistence import induce_and_store


class _Throttled(LLMProvider):
    """Wraps a provider so calls are at least `min_interval` seconds apart."""

    _lock = threading.Lock()
    _last = 0.0

    def __init__(self, inner: LLMProvider, min_interval: float) -> None:
        self._inner = inner
        self._min_interval = min_interval

    def _wait(self) -> None:
        with self._lock:
            delay = _Throttled._last + self._min_interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            _Throttled._last = time.monotonic()

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        self._wait()
        return await self._inner.complete(prompt, **kwargs)

    async def structured_complete(self, prompt: str, **kwargs: Any) -> Any:
        self._wait()
        return await self._inner.structured_complete(prompt, **kwargs)


def report() -> dict[str, Any]:
    """Outcomes and rules per language, from the database."""
    with get_session() as session:
        actions = session.query(ReviewerAction).all()
        language_of = {str(a.id): a.payload["original"]["language"] for a in actions}
        outcomes: dict[str, Counter[str]] = defaultdict(Counter)
        for action in actions:
            outcomes[language_of[str(action.id)]][action.induction_outcome or "pending"] += 1
        rules = session.query(MemoryRuleRecord).filter_by(retired=False).all()
        retired = session.query(MemoryRuleRecord).filter_by(retired=True).count()
        rules_per_language: Counter[str] = Counter()
        for rule in rules:
            for language in {language_of.get(str(i)) for i in rule.source_correction_ids} - {None}:
                rules_per_language[language] += 1
    return {
        "corrections": {lang: dict(c) for lang, c in sorted(outcomes.items())},
        "active_rules": len(rules),
        "active_rules_per_language": dict(sorted(rules_per_language.items())),
        "retired_rules": retired,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--min-interval", type=float, default=4.5, help="seconds between LLM calls")
    parser.add_argument("--retry-wait", type=float, default=65.0, help="seconds to wait after a 429")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    if not args.report_only:
        throttled = _Throttled(get_llm_provider(), args.min_interval)
        for module in (induction, conflict, consolidate):
            module.get_llm_provider = lambda: throttled  # type: ignore[attr-defined]

        def _progress(correction: Any, outcome: str) -> None:
            print(
                f"{correction.id} {correction.original.language.value} {correction.error_location}: {outcome}"
            )

        retries = 0
        while True:
            try:
                run = induce_and_store(on_correction=_progress)
                print(f"done: merged_away={run.merged_away}, active rules={run.active_rules_after}")
                break
            except Exception as exc:
                if "PerDay" in str(exc):
                    print(
                        "daily quota reached; re-run this command after it resets to resume.", file=sys.stderr
                    )
                    print(json.dumps(report(), indent=2))
                    return 2
                if is_transient_provider_error(exc) and retries < args.max_retries:
                    retries += 1
                    print(f"transient error ({exc!r}); waiting {args.retry_wait:.0f}s", file=sys.stderr)
                    time.sleep(args.retry_wait)
                    continue
                raise

    print(json.dumps(report(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
