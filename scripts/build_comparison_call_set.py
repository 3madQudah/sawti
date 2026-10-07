"""Fix the call set for the phase 6.3 cloud vs self-hosted comparison, once, for every arm.

Decided with the user (docs/09-DECISIONS.md, 2026-10-07): 15 calls per
language category —

  * ar: exactly the Phase 5 held-out set (`data/finetune/held_out_call_ids.json`),
    never trained or validated on;
  * en, mixed: 15 each, a seeded sample of that category's calls. The Phase 5
    adapter was trained on ar only, so every en/mixed call is unseen by it —
    and also outside its training language, which the report states next to
    every adapter number.

Also records, per language, how many of the selected calls contain a
redaction placeholder after redaction (`[PHONE]`, ...) — the Phase 5 adapter
was trained on raw text and has never seen one (A6).

Writes `data/serving_benchmark/call_set.json`; every runner reads that file,
so the arms cannot drift apart.

Usage:
    uv run python scripts/build_comparison_call_set.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from sawti.privacy.redaction import redact

SYNTHETIC_DIR = Path("data/synthetic")
GROUND_TRUTH_DIR = Path("data/ground_truth")
HELD_OUT_PATH = Path("data/finetune/held_out_call_ids.json")
OUT_PATH = Path("data/serving_benchmark/call_set.json")
PER_LANGUAGE = 15
SEED = 20261007
PLACEHOLDERS = ("[PHONE]", "[NATIONAL_ID]", "[EMAIL]", "[CARD]", "[IBAN]")


def has_placeholder(call_id: str, synthetic_dir: Path = SYNTHETIC_DIR) -> bool:
    text = redact((synthetic_dir / f"{call_id}.txt").read_text(encoding="utf-8")).redacted_text
    return any(p in text for p in PLACEHOLDERS)


def select(
    *,
    held_out_path: Path = HELD_OUT_PATH,
    ground_truth_dir: Path = GROUND_TRUTH_DIR,
    synthetic_dir: Path = SYNTHETIC_DIR,
    per_language: int = PER_LANGUAGE,
    seed: int = SEED,
) -> dict[str, Any]:
    held_out = json.loads(held_out_path.read_text(encoding="utf-8"))["ar"]
    labelled = {p.stem for p in ground_truth_dir.glob("call_*.json")}
    rng = random.Random(seed)
    calls: dict[str, list[str]] = {"ar": sorted(held_out)}
    for language in ("en", "mixed"):
        pool = sorted(p.stem for p in synthetic_dir.glob(f"call_*_{language}.txt") if p.stem in labelled)
        calls[language] = sorted(rng.sample(pool, per_language))
    return {
        "_meta": {
            "per_language": per_language,
            "seed": seed,
            "ar": "Phase 5 held-out set (data/finetune/held_out_call_ids.json)",
            "en_mixed": (
                "seeded sample; unseen by the ar-only Phase 5 adapter, and outside its training language"
            ),
            "calls_with_redaction_placeholders": {
                language: sum(has_placeholder(c, synthetic_dir) for c in ids)
                for language, ids in calls.items()
            },
        },
        **calls,
    }


def main() -> int:
    call_set = select()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(call_set, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(call_set["_meta"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
