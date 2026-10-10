"""Phase 6.3 A5: memory off vs on, with the 35 complete-linkage rules, leave-one-out.

Tests the Phase 4 hypothesis that memory's effect was understated by
over-merged rules (docs/09-DECISIONS.md, 2026-10-07). Decided with the user:

  * **all 150 calls** (ar 60 / en 30 / mixed 60), so no selection bias;
  * **leave-one-out retrieval**: for each call, any rule with a source
    correction from that same call is excluded before taking the top 5 —
    69 of the 150 calls sourced corrections, so without this a call could be
    "helped" by a rule learned from its own answer;
  * **two repeats** of both arms, to see Gemini's run-to-run noise next to the
    effect (~600 requests, ~1.3 days of one key's free quota).

The rules are frozen into `rules_snapshot.json` first, so reviews arriving in
the service mid-run cannot change what is measured. Retrieval is the
service's: the same embedding model and cosine ranking as
`sawti.memory.store.MemoryStore`, against the redacted transcript.

Usage:
    uv run python scripts/run_memory_comparison.py snapshot
    uv run python scripts/run_memory_comparison.py run [--provider fake]
    uv run python scripts/run_memory_comparison.py report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from run_cloud_arm import provenance

from sawti.data.ground_truth import load_ground_truth
from sawti.eval.serving_benchmark import (
    CallInput,
    QuotaExhaustedError,
    load_call_inputs,
    load_checkpoint,
    run_arm,
    score_records,
)
from sawti.llm.fake_provider import FakeProvider
from sawti.llm.provider import LLMProvider, get_llm_provider

OUT_DIR = Path("data/memory_comparison")
SNAPSHOT = OUT_DIR / "rules_snapshot.json"
CHECKPOINT = OUT_DIR / "runs.jsonl"
SYNTHETIC_DIR = Path("data/synthetic")
GROUND_TRUTH_DIR = Path("data/ground_truth")
TOP_K = 5
ARMS = ("memory_off", "memory_on_35_loo")


def snapshot_rules() -> dict[str, Any]:
    """Freeze the active rules with the calls their source corrections came from."""
    from sawti.db.models import MemoryRuleRecord, ReviewerAction
    from sawti.db.session import get_session

    with get_session() as session:
        call_of = {str(a.id): a.payload["call_id"] for a in session.query(ReviewerAction)}
        rules = [
            {
                "id": str(rule.id),
                "text": rule.rule_text,
                "source_call_ids": sorted(
                    {call_of[str(i)] for i in rule.source_correction_ids if str(i) in call_of}
                ),
            }
            for rule in session.query(MemoryRuleRecord)
            .filter_by(retired=False)
            .order_by(MemoryRuleRecord.created_at)
        ]
    return {"active_rules": len(rules), "consolidation": "complete-linkage, cosine >= 0.7", "rules": rules}


def leave_one_out(ranked: list[dict[str, Any]], call_id: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """The best `top_k` of `ranked` (best first) that were not learned from `call_id` itself."""
    return [rule for rule in ranked if call_id not in rule["source_call_ids"]][:top_k]


def make_rules_for(
    rules: list[dict[str, Any]], embed: Callable[[str], Any] | None = None
) -> Callable[[CallInput], tuple[list[str], list[str]]]:
    """Retrieval as the service does it (same model, cosine), then the leave-one-out filter."""
    import numpy as np

    from sawti.memory.store import _default_embed

    embed_fn = embed or _default_embed
    vectors = [np.asarray(embed_fn(rule["text"])) for rule in rules]

    def rules_for(call: CallInput) -> tuple[list[str], list[str]]:
        query = np.asarray(embed_fn(call.redacted_transcript))
        scores = [float(query @ v / ((np.linalg.norm(query) * np.linalg.norm(v)) or 1.0)) for v in vectors]
        ranked = [rule for _, rule in sorted(zip(scores, rules, strict=True), key=lambda pair: -pair[0])]
        chosen = leave_one_out(ranked, call.call_id)
        return [r["text"] for r in chosen], [r["id"] for r in chosen]

    return rules_for


def all_calls() -> list[CallInput]:
    ids: dict[str, list[str]] = {}
    for path in sorted(SYNTHETIC_DIR.glob("call_*.txt")):
        ids.setdefault(path.stem.rsplit("_", 1)[1], []).append(path.stem)
    return load_call_inputs(ids, SYNTHETIC_DIR)


async def run(
    provider: LLMProvider,
    *,
    repeats: int,
    min_interval: float,
    note: dict[str, str],
    checkpoint: Path = CHECKPOINT,
    concurrency: int = 4,
) -> None:
    rules = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["rules"]
    rules_for = make_rules_for(rules)
    calls = all_calls()
    for repeat in range(repeats):
        for arm in ARMS:
            await run_arm(
                provider,
                calls,
                arm=arm,
                repeat=repeat,
                checkpoint=checkpoint,
                concurrency=concurrency,
                rules_for=rules_for if arm != "memory_off" else None,
                min_interval_s=min_interval,
                note=note,
                # A hung request (or one stretched by host sleep) is retried, not waited on;
                # no successful Gemini call so far took longer than 135 s.
                call_timeout_s=600.0,
            )


#: Phase 4's memory - control accuracy delta per language: the mean of its five per-batch deltas
#: (eval_results.md, 2026-09-25 round): ar (+0.015, +0.070, +0.007, +0.060, +0.129),
#: en (-0.001, -0.075, -0.001, +0.029, +0.072), mixed (+0.013, +0.022, -0.040, +0.010, -0.005).
PHASE4_DELTA = {"ar": 0.0562, "en": 0.0048, "mixed": 0.0000}


def paired_verdict(
    records: list[Any], truth: list[Any], phase4_delta: dict[str, float] = PHASE4_DELTA
) -> dict[str, Any]:
    """The pre-registered test of the Phase 4 hypothesis (docs/09-DECISIONS.md, 2026-10-10).

    Per language and repeat, accuracy of memory on vs off over the calls valid
    in *both* arms of that repeat (paired); effect = mean delta over repeats;
    noise = the larger of the off arm's run-to-run change and the delta's
    change between repeats. Verdict vs Phase 4's mean per-batch delta:
    "confirms" if effect > phase4 + noise, "contradicts" if effect <
    phase4 - noise, otherwise "inconclusive (within noise)".
    """
    from sawti.eval.metrics import accuracy_by_category
    from sawti.schemas import CallAnalysis, Language

    references = {r.call_id: r for r in truth}
    repeats = sorted({r.repeat for r in records})
    out: dict[str, Any] = {}
    for language in Language:
        deltas, off_scores = [], []
        n_paired = []
        for repeat in repeats:
            arms = {
                arm: {
                    r.call_id: r
                    for r in records
                    if r.arm == arm and r.repeat == repeat and r.language == language and r.prediction
                }
                for arm in ARMS
            }
            common = sorted(set(arms["memory_off"]) & set(arms["memory_on_35_loo"]))
            if not common:
                continue
            refs = [references[c] for c in common]
            acc = {
                arm: accuracy_by_category(
                    [CallAnalysis.model_validate(arms[arm][c].prediction) for c in common], refs
                )[language]
                for arm in ARMS
            }
            deltas.append(acc["memory_on_35_loo"] - acc["memory_off"])
            off_scores.append(acc["memory_off"])
            n_paired.append(len(common))
        if not deltas:
            continue
        effect = sum(deltas) / len(deltas)
        noise = max(
            (max(off_scores) - min(off_scores)) if len(off_scores) > 1 else 0.0,
            (max(deltas) - min(deltas)) if len(deltas) > 1 else 0.0,
        )
        baseline = phase4_delta[language.value]
        if len(deltas) < 2:
            verdict = "incomplete (needs both repeats to estimate noise)"
        elif effect > baseline + noise:
            verdict = "confirms"
        elif effect < baseline - noise:
            verdict = "contradicts"
        else:
            verdict = "inconclusive (within noise)"
        out[language.value] = {
            "n_paired_per_repeat": n_paired,
            "delta_per_repeat": [round(d, 4) for d in deltas],
            "effect": round(effect, 4),
            "noise": round(noise, 4),
            "phase4_delta": baseline,
            "verdict": verdict,
        }
    return out


def report(checkpoint: Path = CHECKPOINT) -> dict[str, Any]:
    records, _ = load_checkpoint(checkpoint)
    truth = load_ground_truth(GROUND_TRUTH_DIR)
    out: dict[str, Any] = {}
    for repeat in sorted({r.repeat for r in records}):
        scored = {
            arm: score_records(
                [r for r in records if r.arm == arm and r.repeat == repeat],
                truth,
                transcript_dir=SYNTHETIC_DIR,
            )
            for arm in ARMS
        }
        out[f"repeat_{repeat}"] = scored
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["snapshot", "run", "report"])
    parser.add_argument("--provider", choices=["gemini", "fake"], default="gemini")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--min-interval", type=float, default=4.5)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    # Accuracy does not depend on concurrency, and free-tier Gemini answers in
    # ~1-2 min per call, so a few in flight (still paced by --min-interval)
    # keeps the run to about a day. Latency is not a result of this script.
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.command == "snapshot":
        if SNAPSHOT.exists():
            raise SystemExit(
                f"{SNAPSHOT} exists; the snapshot is frozen once — delete it deliberately to redo"
            )
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(
            json.dumps(snapshot_rules(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"froze {json.loads(SNAPSHOT.read_text())['active_rules']} rules to {SNAPSHOT}")
        return 0
    if args.command == "report":
        records, _ = load_checkpoint(args.checkpoint)
        out = {
            "by_repeat": report(args.checkpoint),
            "verdict": paired_verdict(records, load_ground_truth(GROUND_TRUTH_DIR)),
        }
        print(json.dumps(out, indent=2))
        return 0

    provider: LLMProvider = FakeProvider() if args.provider == "fake" else get_llm_provider()
    try:
        asyncio.run(
            run(
                provider,
                repeats=args.repeats,
                min_interval=args.min_interval if args.provider == "gemini" else 0.0,
                note=provenance(args.provider),
                checkpoint=args.checkpoint,
                concurrency=args.concurrency,
            )
        )
    except QuotaExhaustedError:
        print("daily quota reached; re-run after it resets to resume.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
