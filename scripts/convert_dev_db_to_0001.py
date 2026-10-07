"""One-off: convert the pre-Alembic dev database to revision 0001 in place, keeping its data.

Phase 6.1 follow-up. The dev database was built by `Base.metadata.create_all()`
under the phase 1 models (raw `calls.transcript`, no lifecycle columns) and
holds the phase 4 calls, analyses and corrections. Revision 0001 cannot be
applied to it (its tables exist), and `alembic stamp` alone would be false
(the columns differ). This script reshapes it to exactly 0001, then stamps.

Everything runs in **one transaction**, in this order:

  1. Convert: redact every `calls.transcript` into `redacted_transcript`
     (with `pii_redacted_count`), drop the raw column, add the 0001 columns
     and constraints. Existing rows get status `completed` — they were
     persisted directly, never queued — and no lifecycle timestamps.
  2. Redact quote text inside the stored payloads (`call_analyses`,
     `reviewer_actions`) and re-anchor each changed quote's offsets to the
     redacted transcript. Ground-truth-derived corrections quote the raw text,
     so without this raw PII would survive inside the JSON.
  3. Drift check: Alembic's autogenerate comparison against the schema
     revision 0001 itself creates (built in a scratch database and
     reflected — not today's models, which have moved past 0001) must
     report zero differences.
  4. Quote check: every quote must appear verbatim in its call's new
     `redacted_transcript`, and no quote text may still contain PII-shaped
     spans. A quote matching *neither* the raw nor the redacted transcript
     cannot have been broken by this conversion; it is reported as a
     pre-existing exception and does not block.

Only if every check is clean *and* `--commit` is given is `alembic_version`
stamped `0001` and the transaction committed. Otherwise it rolls back and the
database is exactly as it was. The default is a dry run.

Take a `pg_dump` first (see docs/09-DECISIONS.md, 2026-10-06); the raw
transcripts this drops exist nowhere else.

Usage:
    uv run python scripts/convert_dev_db_to_0001.py            # dry run, report only
    uv run python scripts/convert_dev_db_to_0001.py --commit   # convert + stamp if clean
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy.dialects.postgresql import JSONB

from sawti.config import get_settings
from sawti.db.models import alembic_include_object
from sawti.privacy.redaction import redact

TARGET_REVISION = "0001"

_CONVERT_DDL = (
    # calls: the lifecycle and redaction columns. `status` is added with the
    # same default/CHECK 0001 creates, then existing rows are set explicitly.
    """ALTER TABLE calls
        ADD COLUMN status VARCHAR(32) NOT NULL DEFAULT 'queued',
        ADD COLUMN queued_at TIMESTAMP WITH TIME ZONE,
        ADD COLUMN started_at TIMESTAMP WITH TIME ZONE,
        ADD COLUMN finished_at TIMESTAMP WITH TIME ZONE,
        ADD COLUMN pii_redacted_count INTEGER NOT NULL DEFAULT 0,
        ADD CONSTRAINT call_status CHECK (status IN
            ('queued', 'processing', 'awaiting_review', 'reviewed', 'completed', 'failed'))""",
    "UPDATE calls SET status = 'completed'",
    # call_analyses: one analysis per call, plus the review-screen columns.
    """ALTER TABLE call_analyses
        ADD COLUMN grounding_coverage DOUBLE PRECISION,
        ADD COLUMN confidence_threshold DOUBLE PRECISION,
        ADD COLUMN escalation_reason TEXT,
        ADD COLUMN rejected_claims JSONB NOT NULL DEFAULT '[]'::jsonb,
        ADD COLUMN retrieved_rule_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
        ADD CONSTRAINT uq_call_analyses_call_id UNIQUE (call_id)""",
)


@dataclass
class QuoteCheck:
    """Outcome of checking stored quotes against the converted transcripts."""

    total: int = 0
    #: Blocking: (source, quote text, True) — verbatim in the raw transcript
    #: but not in the redacted one, i.e. broken by this conversion.
    mismatches: list[tuple[str, str, bool]] = field(default_factory=list)
    #: Non-blocking: (source, quote text) — verbatim in neither transcript, so
    #: already broken before this conversion ran.
    preexisting: list[tuple[str, str]] = field(default_factory=list)
    #: Blocking: (source, quote text) still containing a PII-shaped span.
    residual_pii: list[tuple[str, str]] = field(default_factory=list)
    #: Quotes found verbatim whose stored offsets no longer point at them —
    #: informational (redaction shifts offsets; `ground` recomputes them).
    offset_drift: int = 0


@dataclass
class Report:
    """Everything the conversion found, for the operator to read before committing."""

    calls: int = 0
    redactions: int = 0
    calls_with_pii: int = 0
    payload_quotes_redacted: int = 0
    drift: list[Any] = field(default_factory=list)
    quotes: QuoteCheck = field(default_factory=QuoteCheck)
    committed: bool = False

    @property
    def clean(self) -> bool:
        return not (self.drift or self.quotes.mismatches or self.quotes.residual_pii)


def iter_quotes(payload: Any, path: str = "") -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield `(json path, quote dict)` for every `Quote`-shaped object in a stored payload.

    Walks generically rather than through `CallAnalysis`/`Correction` models:
    validation is not what is being tested here, and a row that no longer
    validates against today's schema must still have its quotes checked.
    """
    if isinstance(payload, dict):
        if {"text", "speaker", "start_char", "end_char"} <= payload.keys():
            yield path, payload
            return
        for key, value in payload.items():
            yield from iter_quotes(value, f"{path}.{key}" if path else key)
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            yield from iter_quotes(item, f"{path}[{index}]")


def _check_quotes(conn: sa.Connection, raw_by_call: dict[Any, str]) -> QuoteCheck:
    rows = conn.execute(
        sa.text(
            """SELECT 'call_analyses:' || ca.id::text AS source, ca.call_id, ca.payload, c.redacted_transcript
                 FROM call_analyses ca JOIN calls c ON c.id = ca.call_id
               UNION ALL
               SELECT 'reviewer_actions:' || ra.id::text, ca.call_id, ra.payload, c.redacted_transcript
                 FROM reviewer_actions ra
                 JOIN call_analyses ca ON ca.id = ra.call_analysis_id
                 JOIN calls c ON c.id = ca.call_id"""
        )
    ).all()

    check = QuoteCheck()
    for source, call_id, payload, transcript in rows:
        transcript = transcript or ""
        for path, quote in iter_quotes(payload):
            check.total += 1
            text = quote["text"]
            where = f"{source}/{path}"
            if redact(text).redaction_count:
                check.residual_pii.append((where, text))
            if text not in transcript:
                if text in raw_by_call.get(call_id, ""):
                    check.mismatches.append((where, text, True))
                else:
                    check.preexisting.append((where, text))
            elif transcript[quote["start_char"] : quote["end_char"]] != text:
                check.offset_drift += 1
    return check


def _redact_payload_quotes(conn: sa.Connection) -> int:
    """Redact PII inside every stored quote's text, re-anchoring changed quotes.

    A changed quote is located in its call's redacted transcript (first
    occurrence, the same rule `ground` uses); if it is not found there its
    start is kept and only `end_char` is recomputed, so the `Quote` offset
    arithmetic stays valid and the quote check reports it.

    Returns:
        How many quotes were changed.
    """
    changed = 0
    for table, join in (
        ("call_analyses", "JOIN calls c ON c.id = t.call_id"),
        (
            "reviewer_actions",
            "JOIN call_analyses ca ON ca.id = t.call_analysis_id JOIN calls c ON c.id = ca.call_id",
        ),
    ):
        rows = conn.execute(
            sa.text(f"SELECT t.id, t.payload, c.redacted_transcript FROM {table} t {join}")
        ).all()
        for row_id, payload, transcript in rows:
            row_changed = False
            for _, quote in iter_quotes(payload):
                result = redact(quote["text"])
                if not result.redaction_count:
                    continue
                text = result.redacted_text
                start = (transcript or "").find(text)
                quote["text"] = text
                quote["start_char"] = start if start >= 0 else quote["start_char"]
                quote["end_char"] = quote["start_char"] + len(text)
                changed += 1
                row_changed = True
            if row_changed:
                conn.execute(
                    sa.text(f"UPDATE {table} SET payload = :p WHERE id = :id").bindparams(
                        sa.bindparam("p", type_=JSONB)
                    ),
                    {"p": payload, "id": row_id},
                )
    return changed


def metadata_at_revision(engine: sa.Engine, revision: str = TARGET_REVISION) -> sa.MetaData:
    """The schema `revision` creates, reflected from a scratch database on the same server.

    The scratch database is created, upgraded to `revision`, reflected and
    dropped again. Comparing against this rather than `Base.metadata` keeps
    the drift check true to "exactly 0001" after the models move on.
    """
    from alembic import command
    from alembic.config import Config

    scratch = f"{engine.url.database}_convert_ref"
    admin = sa.create_engine(engine.url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    url = engine.url.set(database=scratch)
    try:
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE IF EXISTS "{scratch}" WITH (FORCE)'))
            conn.execute(sa.text(f'CREATE DATABASE "{scratch}"'))
        config = Config(str(Path(__file__).resolve().parent.parent / "alembic.ini"))
        config.set_main_option("sqlalchemy.url", url.render_as_string(hide_password=False).replace("%", "%%"))
        command.upgrade(config, revision)
        reference = sa.MetaData()
        ref_engine = sa.create_engine(url)
        try:
            reference.reflect(ref_engine)
        finally:
            ref_engine.dispose()
        reference.remove(reference.tables["alembic_version"])
        return reference
    finally:
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE IF EXISTS "{scratch}" WITH (FORCE)'))
        admin.dispose()


def convert(conn: sa.Connection, reference: sa.MetaData) -> Report:
    """Reshape the old schema to 0001 on `conn` and run the checks. Does not commit or stamp."""
    report = Report()
    raw_rows = conn.execute(sa.text("SELECT id, transcript FROM calls")).all()
    raw_by_call = {row.id: row.transcript or "" for row in raw_rows}

    for statement in _CONVERT_DDL:
        conn.execute(sa.text(statement))

    conn.execute(sa.text("ALTER TABLE calls ADD COLUMN redacted_transcript TEXT"))
    for call_id, raw in raw_by_call.items():
        result = redact(raw)
        report.calls += 1
        report.redactions += result.redaction_count
        report.calls_with_pii += int(result.redaction_count > 0)
        conn.execute(
            sa.text("UPDATE calls SET redacted_transcript = :text, pii_redacted_count = :n WHERE id = :id"),
            {"text": result.redacted_text, "n": result.redaction_count, "id": call_id},
        )
    conn.execute(sa.text("ALTER TABLE calls DROP COLUMN transcript"))
    report.payload_quotes_redacted = _redact_payload_quotes(conn)

    context = MigrationContext.configure(conn, opts={"include_object": alembic_include_object})
    report.drift = compare_metadata(context, reference)
    report.quotes = _check_quotes(conn, raw_by_call)
    return report


def stamp(conn: sa.Connection) -> None:
    """Create `alembic_version` exactly as Alembic does and record 0001."""
    conn.execute(
        sa.text(
            "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL, "
            "CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num))"
        )
    )
    conn.execute(sa.text("INSERT INTO alembic_version (version_num) VALUES (:rev)"), {"rev": TARGET_REVISION})


def run(engine: sa.Engine, *, commit: bool) -> Report:
    """Convert inside one transaction; stamp and commit only when clean and asked to.

    Raises:
        RuntimeError: If the database is already under Alembic or lacks the
            pre-0001 `calls.transcript` column — i.e. it is not the shape this
            one-off converts.
    """
    reference = metadata_at_revision(engine)
    with engine.connect() as conn, conn.begin() as transaction:
        inspector = sa.inspect(conn)
        if inspector.has_table("alembic_version"):
            raise RuntimeError("database already has alembic_version — nothing to convert")
        if "transcript" not in {col["name"] for col in inspector.get_columns("calls")}:
            raise RuntimeError("calls.transcript not found — not a pre-0001 database")

        report = convert(conn, reference)
        if commit and report.clean:
            stamp(conn)
            report.committed = True
        else:
            # Leaving the block after an explicit rollback commits nothing;
            # an exception anywhere above rolls back too.
            transaction.rollback()
    return report


def _print(report: Report) -> None:
    print(f"calls converted:        {report.calls}")
    print(f"calls containing PII:   {report.calls_with_pii} ({report.redactions} spans redacted)")
    print(f"schema drift vs models: {len(report.drift)} difference(s)")
    for diff in report.drift:
        print(f"    {diff}")
    print(f"payload quotes redacted: {report.payload_quotes_redacted}")
    print(f"quotes checked:         {report.quotes.total}")
    print(f"BLOCKING — broken by conversion: {len(report.quotes.mismatches)}")
    for source, text, _ in report.quotes.mismatches:
        print(f"    {source}: {text!r}")
    print(f"BLOCKING — PII left in quote text: {len(report.quotes.residual_pii)}")
    for source, text in report.quotes.residual_pii:
        print(f"    {source}: {text!r}")
    print(f"pre-existing (matches neither transcript, non-blocking): {len(report.quotes.preexisting)}")
    for source, text in report.quotes.preexisting:
        print(f"    {source}: {text!r}")
    print(f"quotes w/ stale offsets (informational): {report.quotes.offset_drift}")
    print("RESULT:", "committed and stamped 0001" if report.committed else "rolled back — database unchanged")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--commit", action="store_true", help="stamp 0001 and commit if both checks are clean"
    )
    args = parser.parse_args()

    engine = sa.create_engine(get_settings().database_url)
    try:
        report = run(engine, commit=args.commit)
    finally:
        engine.dispose()
    _print(report)
    return 0 if report.clean else 1


if __name__ == "__main__":
    sys.exit(main())
