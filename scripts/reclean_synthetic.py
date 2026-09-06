"""One-off cleanup pass over `data/synthetic/`: re-scan every existing call
transcript for the dialect/script violations `_find_dialect_violations`
catches (Egyptian-dialect leakage, Arabizi, mixed-script glitch words), and
regenerate ONLY the files that have them — in place, same filename/index/seed
— via the now-fixed prompts in `sawti.data.generate_calls`.

Clean files are left untouched; this never regenerates the whole dataset.

Usage:
    uv run python scripts/reclean_synthetic.py [--data-dir DIR] [--dry-run]

Like the rest of the codebase, this never calls the Gemini SDK (or any LLM
SDK) directly — regeneration goes through `generate_call`/`_generate_with_retry`,
which go through `get_llm_provider()`.
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

from sawti.data.generate_calls import _find_dialect_violations, _generate_with_retry
from sawti.schemas import Language

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("reclean_synthetic")

_FILENAME_RE = re.compile(r"^call_(\d+)_(ar|en|mixed)\.txt$")

DEFAULT_DATA_DIR = Path("data/synthetic")
SLEEP_SECONDS = 4.0
MAX_RETRIES = 6


def _iter_call_files(data_dir: Path) -> list[tuple[Path, int, Language]]:
    """Return (path, seed, language) for every `call_<idx>_<lang>.txt` file in `data_dir`.

    The seed is re-derived from the filename index (e.g. `call_0017_ar.txt` ->
    seed=17), matching how `generate_batch` originally derived seeds from
    each call's position in its (seeded-shuffle) sequence.
    """
    entries: list[tuple[Path, int, Language]] = []
    for path in sorted(data_dir.glob("call_*.txt")):
        match = _FILENAME_RE.match(path.name)
        if not match:
            logger.warning("skipping file with unexpected name: %s", path.name)
            continue
        seed = int(match.group(1))
        language = Language(match.group(2))
        entries.append((path, seed, language))
    return entries


def reclean(data_dir: Path, *, dry_run: bool = False) -> dict[str, list[str]]:
    """Scan `data_dir` and regenerate any file with dialect/script violations.

    A file whose regeneration keeps violating even after `_generate_with_retry`
    exhausts its own retries is logged and skipped, not fatal — one stubborn
    seed (e.g. the model fixating on a single banned word) must not abort the
    rest of the batch. Failures are reported in the returned mapping (value
    `["FAILED TO REGENERATE: ..."]`) and in the final summary.

    Returns:
        Mapping of filename -> violations found (pre-regeneration), for
        every file that had violations. An empty dict means nothing needed
        regenerating.
    """
    entries = _iter_call_files(data_dir)
    logger.info("scanning %d files in %s", len(entries), data_dir)

    regenerated: dict[str, list[str]] = {}
    for path, seed, language in entries:
        text = path.read_text(encoding="utf-8")
        violations = _find_dialect_violations(text, language)
        if not violations:
            continue

        regenerated[path.name] = violations
        logger.info("%s: %d violation(s):", path.name, len(violations))
        for violation in violations:
            logger.info("  - %s", violation)

        if dry_run:
            continue

        try:
            new_transcript = _generate_with_retry(
                language, seed=seed, max_retries=MAX_RETRIES, initial_delay=SLEEP_SECONDS
            )
        except Exception as exc:
            logger.error("  -> FAILED to regenerate %s, skipping: %s", path.name, exc)
            regenerated[path.name] = [f"FAILED TO REGENERATE: {exc}"]
            time.sleep(SLEEP_SECONDS)
            continue

        path.write_text(new_transcript, encoding="utf-8")
        logger.info("  -> regenerated %s (clean)", path.name)
        time.sleep(SLEEP_SECONDS)

    return regenerated


def rescan(data_dir: Path) -> dict[str, list[str]]:
    """Re-scan `data_dir` after cleanup and return any files still violating."""
    remaining: dict[str, list[str]] = {}
    for path, _seed, language in _iter_call_files(data_dir):
        violations = _find_dialect_violations(path.read_text(encoding="utf-8"), language)
        if violations:
            remaining[path.name] = violations
    return remaining


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report violations found; do not regenerate anything.",
    )
    args = parser.parse_args()

    regenerated = reclean(args.data_dir, dry_run=args.dry_run)

    total = len(list(args.data_dir.glob("call_*.txt")))
    logger.info("=" * 60)
    logger.info(
        "%d/%d files had violations%s.",
        len(regenerated),
        total,
        " (dry run — not regenerated)" if args.dry_run else " and were regenerated",
    )

    if args.dry_run:
        return

    remaining = rescan(args.data_dir)
    if remaining:
        logger.error("%d file(s) STILL have violations after regeneration:", len(remaining))
        for name, violations in remaining.items():
            logger.error("  %s: %s", name, violations)
    else:
        logger.info("re-scan of all %d files: zero violations.", total)


if __name__ == "__main__":
    main()
