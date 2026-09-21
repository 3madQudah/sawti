"""Generate ground-truth authoring templates, one per synthetic transcript.

Phase 1 item 3: manual review pass to establish ground truth. For every
`data/synthetic/call_<idx>_<lang>.txt` transcript, writes a corresponding
`data/ground_truth/call_<idx>_<lang>.json` template shaped like
`sawti.schemas.CallAnalysis`, with every value left empty/null/placeholder
for a human reviewer to fill in by hand.

This script never calls an LLM and never invents, guesses, or infers any
call content — every field in the generated template is a placeholder. The
raw template does NOT itself pass `CallAnalysis.model_validate()` (e.g.
`summary` is `""`, `confidence` is `null`) — that is expected. Validation
happens later, once a human has filled a file in, via
`sawti.data.ground_truth.load_ground_truth`.

Usage:
    uv run python scripts/generate_ground_truth_templates.py
    uv run python scripts/generate_ground_truth_templates.py --overwrite

Existing files under `data/ground_truth/` are left untouched by default —
this must never silently clobber a reviewer's in-progress work. Pass
`--overwrite` to regenerate blank templates anyway.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from sawti.data.ground_truth import RUBRIC_CRITERIA, infer_call_id_and_language
from sawti.schemas import Language

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("generate_ground_truth_templates")

# Re-exported from `sawti.data.ground_truth`, which owns the call_<idx>_<lang>.txt
# naming convention, so this script and the reference-label generator can never
# drift apart on what a transcript filename means.
__all__ = ["build_template", "generate_templates", "infer_call_id_and_language", "main"]

DEFAULT_SYNTHETIC_DIR = Path("data/synthetic")
DEFAULT_OUTPUT_DIR = Path("data/ground_truth")

_INSTRUCTIONS: dict[str, str] = {
    "_meaning": (
        "This is a ground-truth authoring template, not part of the real "
        "CallAnalysis schema — this whole key is ignored by "
        "sawti.data.ground_truth.load_ground_truth(). Delete it or leave it; "
        "either is fine once the file is filled in."
    ),
    "call_id": "Do not edit — identifies the transcript this reviews (data/synthetic/<call_id>.txt).",
    "language": "Do not edit — inferred from the transcript filename's _ar/_en/_mixed suffix.",
    "summary": "Fill in: a short human-written summary of the call. Required, non-empty.",
    "commitments": (
        "Open-ended list — append one entry per promise/commitment you find in the "
        "call (evidence, promised_by, description, deadline). Leave as [] if there are none."
    ),
    "compliance_flags": (
        "Open-ended list — append one entry per compliance violation you find in the "
        "call (evidence, rule_id, severity, description). Leave as [] if there are none."
    ),
    "rubric_scores": (
        "Fill in every one of the 7 entries below (do not add, remove, or rename "
        "criteria) — set 'score' (0.0-1.0), 'justification', and a grounding "
        "'evidence' quote for each."
    ),
    "evidence": (
        "A verbatim quote grounding the claim/score: {text, speaker, start_char, "
        "end_char}. Use `uv run python scripts/find_quote.py <transcript> "
        "\"<exact substring>\"` to get exact offsets — do not hand-count them."
    ),
    "sentiment_trajectory": (
        "Optional — append chronologically ordered {quote, speaker, score, "
        "timestamp_sec} points if you want to track sentiment across the call. "
        "Leave 'points' as [] if skipping."
    ),
    "confidence": "Fill in: your overall confidence in this review, 0.0-1.0. Required.",
    "requires_human_review": (
        "Fill in: true or false. Must be true if confidence is below the configured "
        "threshold (sawti.config.Settings.confidence_threshold)."
    ),
}


def _empty_evidence() -> dict[str, Any]:
    """A placeholder `Quote`-shaped dict with every value empty/null."""
    return {"text": "", "speaker": "", "start_char": None, "end_char": None}


def build_template(call_id: str, language: Language) -> dict[str, Any]:
    """Build an empty ground-truth authoring template for one call.

    Every value is a placeholder for a human reviewer to fill in by hand —
    this never invents, guesses, or infers any call content. The result
    mirrors `sawti.schemas.CallAnalysis`'s shape but will NOT itself pass
    `CallAnalysis.model_validate()` until filled in (e.g. `summary` starts as
    `""`, `confidence` starts as `null`) — that is expected, see module
    docstring.

    Args:
        call_id: The call's id (the transcript filename stem).
        language: The call's language category.

    Returns:
        A JSON-serializable template dict, including a top-level
        `_instructions` key that is not part of the real schema.
    """
    return {
        "_instructions": _INSTRUCTIONS,
        "call_id": call_id,
        "language": language.value,
        "summary": "",
        "commitments": [],
        "compliance_flags": [],
        "rubric_scores": [
            {
                "criterion": criterion,
                "score": None,
                "justification": "",
                "evidence": _empty_evidence(),
            }
            for criterion in RUBRIC_CRITERIA
        ],
        "sentiment_trajectory": {"points": []},
        "confidence": None,
        "requires_human_review": None,
    }


def generate_templates(synthetic_dir: Path, output_dir: Path, *, overwrite: bool = False) -> list[Path]:
    """Write one ground-truth template per transcript in `synthetic_dir`.

    Args:
        synthetic_dir: Directory of `call_<idx>_<lang>.txt` transcripts.
        output_dir: Directory to write `<call_id>.json` templates into
            (created if missing).
        overwrite: If False (default), an existing `<call_id>.json` is left
            untouched — this must never silently clobber a reviewer's
            in-progress work. If True, existing files are replaced with a
            fresh blank template.

    Returns:
        Paths actually written. Files skipped because they already exist
        (and `overwrite` is False) are not included.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for transcript_path in sorted(synthetic_dir.glob("call_*.txt")):
        call_id, language = infer_call_id_and_language(transcript_path)
        out_path = output_dir / f"{call_id}.json"
        if out_path.exists() and not overwrite:
            logger.info("skipping %s (already exists; pass --overwrite to replace)", out_path.name)
            continue
        template = build_template(call_id, language)
        out_path.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(out_path)
        logger.info("wrote %s", out_path.name)
    return written


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-dir", type=Path, default=DEFAULT_SYNTHETIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing template files instead of skipping them.",
    )
    args = parser.parse_args()

    written = generate_templates(args.synthetic_dir, args.output_dir, overwrite=args.overwrite)

    total = len(list(args.synthetic_dir.glob("call_*.txt")))
    logger.info("wrote %d/%d template(s) to %s", len(written), total, args.output_dir)


if __name__ == "__main__":
    main()
