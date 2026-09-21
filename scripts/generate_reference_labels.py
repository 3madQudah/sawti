"""Generate LLM reference labels for every synthetic transcript.

This REPLACES the manual ground-truth review pass that
`scripts/generate_ground_truth_templates.py` was written to support. Instead
of a human filling in blank templates, `sawti.data.ground_truth.
generate_reference_labels` prompts an LLM for the analysis and verifies its
quotes against the transcript programmatically.

What that does and does not buy us:
  * Verified: every quote in every written file provably appears in the
    transcript, at the offsets recorded — offsets are computed here, never
    reported by the model.
  * NOT verified: whether the analysis is *correct*. No human read these.

Because that distinction is easy to lose track of months later, every file
this script writes carries a top-level `_provenance` block recording the
method, the model, when it ran, and `"human_reviewed": false`. That block is
written unconditionally, from a single code path — see `build_provenance`.

By default this OVERWRITES existing files in `data/ground_truth/`: the files
it replaces are blank authoring templates, which hold no reviewed work. Pass
`--skip-existing` to leave already-written files alone (e.g. to resume an
interrupted run).

Usage:
    uv run python scripts/generate_reference_labels.py
    uv run python scripts/generate_reference_labels.py --limit 3
    uv run python scripts/generate_reference_labels.py --skip-existing
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sawti.config import get_settings
from sawti.data.ground_truth import generate_reference_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("generate_reference_labels")

DEFAULT_SYNTHETIC_DIR = Path("data/synthetic")
DEFAULT_OUTPUT_DIR = Path("data/ground_truth")

PROVENANCE_KEY = "_provenance"


def build_provenance() -> dict[str, Any]:
    """Build the `_provenance` block stamped onto every written file.

    Single source of this block, so a future edit cannot accidentally write
    reference labels that don't disclose how they were made. `human_reviewed`
    is hard-coded False here; if any of these files is ever actually reviewed
    by a person, that flag should be flipped by whoever did the reviewing,
    per file.

    Returns:
        A JSON-serializable provenance dict.
    """
    settings = get_settings()
    return {
        "method": "llm_generated",
        "model": settings.gemini_model,
        "generated_at": datetime.now(UTC).isoformat(),
        "human_reviewed": False,
    }


def write_reference_label(transcript_path: Path, output_dir: Path) -> Path:
    """Generate and write the reference label for one transcript.

    Args:
        transcript_path: Path to a `call_<idx>_<lang>.txt` transcript.
        output_dir: Directory to write `<call_id>.json` into.

    Returns:
        The path written.
    """
    analysis = generate_reference_labels(transcript_path)
    payload: dict[str, Any] = {PROVENANCE_KEY: build_provenance()}
    payload.update(json.loads(analysis.model_dump_json()))

    out_path = output_dir / f"{analysis.call_id}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_path


def generate_all(
    synthetic_dir: Path,
    output_dir: Path,
    *,
    skip_existing: bool = False,
    limit: int | None = None,
    sleep_seconds: float = 4.0,
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Generate reference labels for every transcript in `synthetic_dir`.

    Args:
        synthetic_dir: Directory of `call_<idx>_<lang>.txt` transcripts.
        output_dir: Directory to write `<call_id>.json` files into.
        skip_existing: Leave already-written output files untouched.
        limit: Process at most this many transcripts (for smoke tests).
        sleep_seconds: Delay between calls, to respect free-tier rate limits.

    Returns:
        `(written_paths, failures)` where each failure is `(transcript_path, reason)`.
        A call that fails is reported and skipped; one bad call never aborts
        the batch, since re-running the whole set is expensive.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    transcripts = sorted(synthetic_dir.glob("call_*.txt"))
    if limit is not None:
        transcripts = transcripts[:limit]

    written: list[Path] = []
    failures: list[tuple[Path, str]] = []

    for index, transcript_path in enumerate(transcripts, start=1):
        out_path = output_dir / f"{transcript_path.stem}.json"
        if skip_existing and out_path.exists():
            logger.info("[%d/%d] skipping %s (exists)", index, len(transcripts), out_path.name)
            continue
        try:
            written.append(write_reference_label(transcript_path, output_dir))
            logger.info("[%d/%d] wrote %s", index, len(transcripts), out_path.name)
        except Exception as exc:
            failures.append((transcript_path, repr(exc)))
            logger.error("[%d/%d] FAILED %s: %r", index, len(transcripts), transcript_path.name, exc)
        if index < len(transcripts):
            time.sleep(sleep_seconds)

    return written, failures


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-dir", type=Path, default=DEFAULT_SYNTHETIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Leave already-written output files alone (resume an interrupted run).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N transcripts.")
    parser.add_argument("--sleep-seconds", type=float, default=4.0)
    args = parser.parse_args()

    written, failures = generate_all(
        args.synthetic_dir,
        args.output_dir,
        skip_existing=args.skip_existing,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
    )

    logger.info("wrote %d file(s) to %s", len(written), args.output_dir)
    if failures:
        logger.error("%d transcript(s) failed:", len(failures))
        for path, reason in failures:
            logger.error("  %s: %s", path.name, reason)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
