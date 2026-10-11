"""Redaction alone, before vs after a fix, per language category: no ASR, no LLM.

Runs two versions of `sawti.privacy.redaction.redact` over a directory of
`call_<id>_<lang>.txt` transcripts: the one at a git revision (`--before-rev`,
loaded from `git show`) and the one in the working tree. Reports, per language:
PII spans caught by each (by placeholder), and digit runs of 7+ digits left
unredacted by each.

A **digit run** is a maximal sequence of digits (ASCII or Arabic-Indic) joined
by at most one space, dash or dot between any two digits — the way phone and
account numbers are written or dictated. Commas and slashes break a run, so
amounts and dates do not join up. Runs are counted on the text each redactor
returns, so a placeholder ends a run.

**The output carries no PII**: counts, call ids, and leftover runs as shapes
with every digit replaced by `d`.

Written for the 2026-10-11 production bug (an uploaded call kept
"555-019-8234" in its redacted transcript); see eval_results.md.

Usage:
    uv run python scripts/measure_redaction.py --before-rev 610ddc4 \\
        [--transcripts data/asr_transcripts] [--out data/redaction_audit/<name>.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sawti.privacy.redaction import redact as redact_now

LANGUAGES = ("ar", "en", "mixed")
MIN_RUN_DIGITS = 7
_RUN = re.compile(r"\d(?:[ .\-]?\d)*")
_PLACEHOLDER = re.compile(r"\[[A-Z_]+\]")
_FILE = re.compile(r"^call_(\d+)_(ar|en|mixed)\.txt$")
REDACTION_MODULE = "src/sawti/privacy/redaction.py"

#: (text) -> (redacted text, number of spans replaced)
Redactor = Callable[[str], tuple[str, int]]


def digit_runs(text: str, min_digits: int = MIN_RUN_DIGITS) -> list[str]:
    """Every digit run in `text` with at least `min_digits` digits, in order."""
    return [m.group() for m in _RUN.finditer(text) if sum(ch.isdigit() for ch in m.group()) >= min_digits]


def shape(run: str) -> str:
    """The run with every digit replaced by `d`: its format, without the number."""
    return re.sub(r"\d", "d", run)


def placeholders(redacted: str) -> Counter[str]:
    """How many of each placeholder a redacted text holds."""
    return Counter(_PLACEHOLDER.findall(redacted))


def load_transcripts(directory: Path) -> list[tuple[str, str, str]]:
    """`(call_id, language, text)` for every `call_<id>_<lang>.txt` in `directory`, by call id."""
    rows = []
    for path in sorted(directory.iterdir()):
        match = _FILE.match(path.name)
        if match:
            rows.append((path.stem, match.group(2), path.read_text(encoding="utf-8")))
    return rows


def redactor_from_source(source: str) -> Redactor:
    """Import a redaction module from its source text and wrap its `redact`."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "redaction_before.py"
        path.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("redaction_before", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    def _redact(text: str) -> tuple[str, int]:
        result = module.redact(text)
        return result.redacted_text, result.redaction_count

    return _redact


def redactor_at_rev(rev: str) -> Redactor:
    """The redactor as it was at git revision `rev`."""
    source = subprocess.run(
        ["git", "show", f"{rev}:{REDACTION_MODULE}"], check=True, capture_output=True, text=True
    ).stdout
    return redactor_from_source(source)


def current_redactor(text: str) -> tuple[str, int]:
    """The working tree's redactor."""
    result = redact_now(text)
    return result.redacted_text, result.redaction_count


def _side(rows: list[tuple[str, str, str]], redactor: Redactor) -> dict[str, Any]:
    spans = 0
    by_placeholder: Counter[str] = Counter()
    left: Counter[str] = Counter()
    left_calls: list[str] = []
    for call_id, _, text in rows:
        redacted, count = redactor(text)
        spans += count
        by_placeholder += placeholders(redacted)
        runs = digit_runs(redacted)
        left.update(shape(run) for run in runs)
        if runs:
            left_calls.append(call_id)
    return {
        "spans": spans,
        "by_placeholder": dict(sorted(by_placeholder.items())),
        "runs_left": sum(left.values()),
        "runs_left_by_shape": dict(left.most_common()),
        "calls_with_runs_left": left_calls,
    }


def compare(rows: list[tuple[str, str, str]], before: Redactor, after: Redactor) -> dict[str, Any]:
    """Per-language before/after counts. Never a blended total."""
    report: dict[str, Any] = {}
    for language in LANGUAGES:
        subset = [row for row in rows if row[1] == language]
        report[language] = {
            "calls": len(subset),
            "runs_in_raw_text": sum(len(digit_runs(text)) for _, _, text in subset),
            "before": _side(subset, before),
            "after": _side(subset, after),
            "calls_changed": [cid for cid, _, text in subset if before(text)[0] != after(text)[0]],
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--before-rev", required=True, help="git revision of the redactor to compare against")
    parser.add_argument("--transcripts", type=Path, default=Path("data/asr_transcripts"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = load_transcripts(args.transcripts)
    if not rows:
        print(f"no call_<id>_<lang>.txt files in {args.transcripts}", file=sys.stderr)
        return 1
    report = {
        "transcripts": str(args.transcripts),
        "before_rev": args.before_rev,
        "min_run_digits": MIN_RUN_DIGITS,
        "by_language": compare(rows, redactor_at_rev(args.before_rev), current_redactor),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    for language, row in report["by_language"].items():
        before, after = row["before"], row["after"]
        print(
            f"{language:5} calls={row['calls']:3} spans {before['spans']:3} -> {after['spans']:3}"
            f"   7+ digit runs left {before['runs_left']:3} -> {after['runs_left']:3}"
            f"   (raw {row['runs_in_raw_text']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
