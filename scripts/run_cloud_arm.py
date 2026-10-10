"""Cloud arm of the phase 6.3 comparison: the 45 comparison calls through Gemini.

The same calls (`data/serving_benchmark/call_set.json`), the same pipeline
(`sawti.eval.serving_benchmark.run_arm` → the service graph) and the same
recorded fields as the Kaggle arms, so the assembler scores them with one
function. Free tier: one request in flight, at most one start per
`--min-interval` seconds (15 requests/min), resumable across days — the
daily cap stops it cleanly and re-running continues.

Each segment records provenance: provider, configured model, and a SHA-256
fingerprint of the API key (never the key), so a run that spans two keys or
projects is visible in the data.

Usage:
    uv run python scripts/run_cloud_arm.py
    uv run python scripts/run_cloud_arm.py --provider fake   # dry run, no quota
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from pathlib import Path

from sawti.config import get_settings
from sawti.eval.serving_benchmark import QuotaExhaustedError, load_call_inputs, load_checkpoint, run_arm
from sawti.llm.fake_provider import FakeProvider
from sawti.llm.provider import LLMProvider, get_llm_provider

CALL_SET = Path("data/serving_benchmark/call_set.json")
CHECKPOINT = Path("data/serving_benchmark/gemini.jsonl")
SYNTHETIC_DIR = Path("data/synthetic")


def provenance(provider_name: str) -> dict[str, str]:
    """Which provider, model and key produced a segment — the key only as a short fingerprint."""
    settings = get_settings()
    note = {"provider": provider_name}
    if provider_name == "gemini":
        note["configured_model"] = settings.gemini_model
        key = settings.gemini_api_key or ""
        note["api_key_sha256_12"] = hashlib.sha256(key.encode()).hexdigest()[:12] if key else "none"
    return note


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--provider", choices=["gemini", "fake"], default="gemini")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--min-interval", type=float, default=4.5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    call_set = json.loads(CALL_SET.read_text(encoding="utf-8"))
    calls = load_call_inputs({k: v for k, v in call_set.items() if k != "_meta"}, SYNTHETIC_DIR)
    provider: LLMProvider = FakeProvider() if args.provider == "fake" else get_llm_provider()
    if args.provider == "gemini" and get_settings().llm_provider != "gemini":
        raise SystemExit("set SAWTI_LLM_PROVIDER=gemini (the cloud arm is Gemini)")
    try:
        asyncio.run(
            run_arm(
                provider,
                calls,
                arm=args.provider,
                checkpoint=args.checkpoint,
                concurrency=1,
                min_interval_s=args.min_interval if args.provider == "gemini" else 0.0,
                note=provenance(args.provider),
                call_timeout_s=600.0,  # see run_memory_comparison.py
            )
        )
    except QuotaExhaustedError:
        print("daily quota reached; re-run after it resets to resume.", file=sys.stderr)
        return 2
    records, _ = load_checkpoint(args.checkpoint)
    print(f"{len({r.call_id for r in records})} of {len(calls)} calls recorded in {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
