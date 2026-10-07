"""Tests for `scripts/convert_dev_db_to_0001.py`.

Phase 6.1 follow-up. Each test builds its own database in the exact
pre-Alembic shape — `_OLD_SCHEMA` is the dev database's schema as `pg_dump`
recorded it before conversion — seeds it, and converts it.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime

import pytest
import sqlalchemy as sa
from convert_dev_db_to_0001 import iter_quotes, run

_OLD_SCHEMA = """
CREATE TABLE agents (id uuid NOT NULL PRIMARY KEY, name varchar(255) NOT NULL,
    external_id varchar(255) UNIQUE, created_at timestamptz NOT NULL);
CREATE TABLE calls (id uuid NOT NULL PRIMARY KEY, agent_id uuid NOT NULL REFERENCES agents(id),
    audio_path varchar(1024), transcript text, language varchar(16) NOT NULL,
    occurred_at timestamptz NOT NULL);
CREATE TABLE call_analyses (id uuid NOT NULL PRIMARY KEY, call_id uuid NOT NULL REFERENCES calls(id),
    payload jsonb NOT NULL, confidence double precision NOT NULL,
    requires_human_review boolean NOT NULL, created_at timestamptz NOT NULL);
CREATE TABLE reviewer_actions (id uuid NOT NULL PRIMARY KEY,
    call_analysis_id uuid NOT NULL REFERENCES call_analyses(id),
    reviewer_id varchar(255) NOT NULL, payload jsonb NOT NULL, created_at timestamptz NOT NULL);
"""

RAW = "Agent: I will call you back on 0791234567 tomorrow.\nCustomer: Thanks, bye.\n"


def _quote(transcript: str, text: str) -> dict[str, object]:
    start = transcript.index(text)
    return {"text": text, "speaker": "Agent", "start_char": start, "end_char": start + len(text)}


def _analysis(quote: dict[str, object]) -> dict[str, object]:
    return {
        "call_id": "call_x",
        "commitments": [{"evidence": quote, "promised_by": "Agent", "description": "Callback."}],
        "sentiment_trajectory": {"points": []},
    }


@pytest.fixture
def old_db(fresh_database: Callable[[str], str]) -> Iterator[sa.Engine]:
    engine = sa.create_engine(fresh_database("sawti_convert_test"))
    with engine.begin() as conn:
        for statement in filter(str.strip, _OLD_SCHEMA.split(";")):
            conn.execute(sa.text(statement))
    yield engine
    engine.dispose()


def _seed(engine: sa.Engine, *, analysis_quote: str, correction_quote: str) -> uuid.UUID:
    agent_id, call_id, analysis_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    now = datetime.now(UTC)
    redacted = RAW.replace("0791234567", "[NATIONAL_ID]")
    with engine.begin() as conn:
        conn.execute(
            sa.text("INSERT INTO agents VALUES (:id, 'a', NULL, :now)"), {"id": agent_id, "now": now}
        )
        conn.execute(
            sa.text("INSERT INTO calls VALUES (:id, :agent, NULL, :t, 'en', :now)"),
            {"id": call_id, "agent": agent_id, "t": RAW, "now": now},
        )
        # Phase 4 analyses were grounded against the *redacted* text `extract` saw.
        conn.execute(
            sa.text("INSERT INTO call_analyses VALUES (:id, :call, :p, 0.5, true, :now)"),
            {
                "id": analysis_id,
                "call": call_id,
                "now": now,
                "p": json.dumps(_analysis(_quote(redacted, analysis_quote))),
            },
        )
        source = RAW if correction_quote in RAW else redacted
        conn.execute(
            sa.text("INSERT INTO reviewer_actions VALUES (:id, :ca, 'qa', :p, :now)"),
            {
                "id": uuid.uuid4(),
                "ca": analysis_id,
                "now": now,
                "p": json.dumps({"corrected": _analysis(_quote(source, correction_quote))}),
            },
        )
    return call_id


def test_iter_quotes_finds_every_quote_shaped_object() -> None:
    quote = {"text": "x", "speaker": "A", "start_char": 0, "end_char": 1}
    payload = {"a": [quote, {"evidence": quote}], "b": {"points": [{"quote": quote}]}}
    assert [path for path, _ in iter_quotes(payload)] == ["a[0]", "a[1].evidence", "b.points[0].quote"]


def test_dry_run_reports_and_leaves_the_database_unchanged(old_db: sa.Engine) -> None:
    _seed(old_db, analysis_quote="I will call you back", correction_quote="Thanks, bye.")
    report = run(old_db, commit=False)

    assert report.clean and not report.committed
    assert report.redactions == 1
    columns = {c["name"] for c in sa.inspect(old_db).get_columns("calls")}
    assert "transcript" in columns and "redacted_transcript" not in columns
    assert not sa.inspect(old_db).has_table("alembic_version")


def test_commit_converts_to_exactly_0001_and_drops_the_raw_text(old_db: sa.Engine) -> None:
    call_id = _seed(old_db, analysis_quote="I will call you back", correction_quote="Thanks, bye.")
    report = run(old_db, commit=True)

    assert report.committed
    assert report.drift == []
    assert report.quotes.total == 2 and report.quotes.mismatches == []
    with old_db.connect() as conn:
        row = conn.execute(
            sa.text("SELECT redacted_transcript, pii_redacted_count, status FROM calls WHERE id = :id"),
            {"id": call_id},
        ).one()
        assert conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar_one() == "0001"
    assert "0791234567" not in row.redacted_transcript
    assert (row.pii_redacted_count, row.status) == (1, "completed")
    assert "transcript" not in {c["name"] for c in sa.inspect(old_db).get_columns("calls")}


def test_pii_inside_a_correction_quote_is_redacted_and_re_anchored(old_db: sa.Engine) -> None:
    """A ground-truth correction quoting raw PII is redacted in the payload and found again."""
    _seed(old_db, analysis_quote="I will call you back", correction_quote="back on 0791234567")
    report = run(old_db, commit=True)

    assert report.committed
    assert report.payload_quotes_redacted == 1
    with old_db.connect() as conn:
        payload, transcript = conn.execute(
            sa.text(
                "SELECT ra.payload, c.redacted_transcript FROM reviewer_actions ra "
                "JOIN call_analyses ca ON ca.id = ra.call_analysis_id JOIN calls c ON c.id = ca.call_id"
            )
        ).one()
    quote = payload["corrected"]["commitments"][0]["evidence"]
    assert "0791234567" not in json.dumps(payload)
    assert transcript[quote["start_char"] : quote["end_char"]] == quote["text"]


def test_a_quote_broken_by_redaction_blocks_the_commit(old_db: sa.Engine) -> None:
    """A quote verbatim in the raw text but not the redacted one, with no PII of its own: roll back."""
    # A fragment of the number: too short to redact on its own, gone once the number is redacted.
    _seed(old_db, analysis_quote="I will call you back", correction_quote="1234567 tomorrow")
    report = run(old_db, commit=True)

    assert not report.committed
    assert [text for _, text, _ in report.quotes.mismatches] == ["1234567 tomorrow"]
    assert "transcript" in {c["name"] for c in sa.inspect(old_db).get_columns("calls")}
    assert not sa.inspect(old_db).has_table("alembic_version")


def test_a_quote_matching_neither_transcript_is_reported_but_does_not_block(old_db: sa.Engine) -> None:
    """Already broken before the conversion (a model paraphrase), so it cannot block it."""
    call_id = _seed(old_db, analysis_quote="I will call you back", correction_quote="Thanks, bye.")
    with old_db.begin() as conn:
        conn.execute(
            sa.text(
                "UPDATE reviewer_actions SET payload = jsonb_set(payload, "
                "'{corrected,commitments,0,evidence,text}', '\"Thanks bye\"')"
            )
        )
        conn.execute(
            sa.text(
                "UPDATE reviewer_actions SET payload = jsonb_set(payload, "
                "'{corrected,commitments,0,evidence,end_char}', "
                "to_jsonb((payload #>> '{corrected,commitments,0,evidence,start_char}')::int + 10))"
            )
        )
    report = run(old_db, commit=True)

    assert report.committed, call_id
    assert [text for _, text in report.quotes.preexisting] == ["Thanks bye"]
    assert report.quotes.mismatches == []


def test_refuses_a_database_already_under_alembic(old_db: sa.Engine) -> None:
    _seed(old_db, analysis_quote="I will call you back", correction_quote="Thanks, bye.")
    run(old_db, commit=True)
    with pytest.raises(RuntimeError, match="alembic_version"):
        run(old_db, commit=True)
