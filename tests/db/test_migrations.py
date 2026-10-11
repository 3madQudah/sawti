"""Tests for `sawti/db/migrations`: the Alembic schema, against a real Postgres.

Phase 6.1. Each test owns a dedicated database (`fresh_database`), because
downgrading to `base` drops every application table — doing that on the
suite's shared test database would break every other DB-backed test.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy.exc import IntegrityError

from sawti.db.models import CHECKPOINT_TABLES, Base, alembic_include_object

APP_TABLES = {
    "agents",
    "calls",
    "call_analyses",
    "reviewer_actions",
    "review_submissions",
    "memory_rules",
    "memory_conflicts",
}


@pytest.fixture
def migration_db(fresh_database: Callable[[str], str]) -> Iterator[sa.Engine]:
    """An empty database of its own, as an engine; the caller runs migrations on it."""
    engine = sa.create_engine(fresh_database("sawti_migration_test"))
    yield engine
    engine.dispose()


@pytest.fixture
def config(migration_db: sa.Engine, alembic_config_for: Callable[[str], Config]) -> Config:
    """Alembic config pointed at `migration_db`."""
    return alembic_config_for(migration_db.url.render_as_string(hide_password=False))


def _tables(engine: sa.Engine) -> set[str]:
    return set(sa.inspect(engine).get_table_names()) - {"alembic_version"}


def test_upgrade_creates_the_application_tables(migration_db: sa.Engine, config: Config) -> None:
    """upgrade head creates exactly the application tables — no checkpoint tables."""
    command.upgrade(config, "head")
    assert _tables(migration_db) == APP_TABLES
    assert not _tables(migration_db) & CHECKPOINT_TABLES


def test_migrated_schema_matches_the_orm_models(migration_db: sa.Engine, config: Config) -> None:
    """No drift: autogenerate finds nothing to change between head and `Base.metadata`.

    Uses the same `include_object` hook as `env.py`, so this is exactly what a
    developer's `alembic revision --autogenerate` would see.
    """
    command.upgrade(config, "head")
    with migration_db.connect() as conn:
        context = MigrationContext.configure(conn, opts={"include_object": alembic_include_object})
        diff = compare_metadata(context, Base.metadata)
    assert diff == []


def test_calls_stores_only_the_redacted_transcript(migration_db: sa.Engine, config: Config) -> None:
    """The raw-transcript column is gone; the phase 6 lifecycle columns exist and are tz-aware."""
    command.upgrade(config, "head")
    columns = {col["name"]: col for col in sa.inspect(migration_db).get_columns("calls")}

    assert "transcript" not in columns
    assert "redacted_transcript" in columns
    for name in ("queued_at", "started_at", "finished_at"):
        assert columns[name]["nullable"]
        assert columns[name]["type"].timezone
    assert not columns["pii_redacted_count"]["nullable"]
    assert not columns["status"]["nullable"]


def test_call_status_rejects_values_outside_the_lifecycle(migration_db: sa.Engine, config: Config) -> None:
    """The CHECK constraint holds at the database level, not just in the ORM."""
    command.upgrade(config, "head")
    with migration_db.begin() as conn:
        agent_id = conn.execute(
            sa.text(
                "INSERT INTO agents (id, name, created_at) VALUES (gen_random_uuid(), 'a', :now) RETURNING id"
            ),
            {"now": datetime.now(UTC)},
        ).scalar_one()
        status = conn.execute(
            sa.text(
                "INSERT INTO calls (id, agent_id, language, occurred_at) "
                "VALUES (gen_random_uuid(), :agent, 'ar', :now) RETURNING status"
            ),
            {"agent": agent_id, "now": datetime.now(UTC)},
        ).scalar_one()
    assert status == "queued"

    with pytest.raises(IntegrityError), migration_db.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO calls (id, agent_id, language, occurred_at, status) "
                "VALUES (gen_random_uuid(), :agent, 'ar', :now, 'escalated')"
            ),
            {"agent": agent_id, "now": datetime.now(UTC)},
        )


def test_call_analyses_allows_one_analysis_per_call(migration_db: sa.Engine, config: Config) -> None:
    """`call_analyses.call_id` is unique."""
    command.upgrade(config, "head")
    constraints = sa.inspect(migration_db).get_unique_constraints("call_analyses")
    assert {"name": "uq_call_analyses_call_id", "column_names": ["call_id"]} in [
        {"name": c["name"], "column_names": c["column_names"]} for c in constraints
    ]


def test_downgrade_then_upgrade_round_trips(migration_db: sa.Engine, config: Config) -> None:
    """downgrade base removes every application table, and upgrade head restores them cleanly."""
    command.upgrade(config, "head")
    command.downgrade(config, "base")
    assert _tables(migration_db) == set()

    command.upgrade(config, "head")
    assert _tables(migration_db) == APP_TABLES


def _insert_agent(conn: sa.Connection) -> object:
    return conn.execute(
        sa.text(
            "INSERT INTO agents (id, name, created_at) VALUES (gen_random_uuid(), 'a', :now) RETURNING id"
        ),
        {"now": datetime.now(UTC)},
    ).scalar_one()


def _insert_call(conn: sa.Connection, agent_id: object, status: str) -> None:
    conn.execute(
        sa.text(
            "INSERT INTO calls (id, agent_id, language, occurred_at, status) "
            "VALUES (gen_random_uuid(), :agent, 'ar', :now, :status)"
        ),
        {"agent": agent_id, "now": datetime.now(UTC), "status": status},
    )


def test_0005_allows_transcribing_and_keeps_existing_calls(migration_db: sa.Engine, config: Config) -> None:
    """0004 rejects `transcribing`; after 0005 it is valid, and rows written under 0004 survive."""
    command.upgrade(config, "0004")
    with migration_db.begin() as conn:
        agent_id = _insert_agent(conn)
        _insert_call(conn, agent_id, "completed")
    with pytest.raises(IntegrityError), migration_db.begin() as conn:
        _insert_call(conn, agent_id, "transcribing")

    command.upgrade(config, "0005")
    with migration_db.begin() as conn:
        _insert_call(conn, agent_id, "transcribing")
        statuses = sorted(conn.execute(sa.text("SELECT status FROM calls")).scalars())
        reasons = set(conn.execute(sa.text("SELECT failure_reason FROM calls")).scalars())
    assert statuses == ["completed", "transcribing"]
    assert reasons == {None}


def test_0005_downgrade_restores_the_0004_status_check(migration_db: sa.Engine, config: Config) -> None:
    """Downgrading 0005 drops its table and column and rejects `transcribing` again."""
    command.upgrade(config, "0005")
    command.downgrade(config, "0004")
    assert "memory_conflicts" not in _tables(migration_db)
    assert "failure_reason" not in {c["name"] for c in sa.inspect(migration_db).get_columns("calls")}
    with migration_db.begin() as conn:
        agent_id = _insert_agent(conn)
    with pytest.raises(IntegrityError), migration_db.begin() as conn:
        _insert_call(conn, agent_id, "transcribing")


def _conflict_row(conn: sa.Connection, *, resolution: str | None, resolved_by: str | None) -> None:
    agent_id = _insert_agent(conn)
    now = datetime.now(UTC)
    call_id = conn.execute(
        sa.text(
            "INSERT INTO calls (id, agent_id, language, occurred_at) "
            "VALUES (gen_random_uuid(), :agent, 'ar', :now) RETURNING id"
        ),
        {"agent": agent_id, "now": now},
    ).scalar_one()
    analysis_id = conn.execute(
        sa.text(
            "INSERT INTO call_analyses (id, call_id, payload, confidence, requires_human_review, created_at) "
            "VALUES (gen_random_uuid(), :call, '{}', 0.5, true, :now) RETURNING id"
        ),
        {"call": call_id, "now": now},
    ).scalar_one()
    action_id = conn.execute(
        sa.text(
            "INSERT INTO reviewer_actions (id, call_analysis_id, reviewer_id, payload, created_at) "
            "VALUES (gen_random_uuid(), :analysis, 'r', '{}', :now) RETURNING id"
        ),
        {"analysis": analysis_id, "now": now},
    ).scalar_one()
    conn.execute(
        sa.text(
            "INSERT INTO memory_conflicts (id, reviewer_action_id, candidate_rule_id, candidate_text, "
            "conflicting, detected_at, resolution, resolved_by, resolved_at) VALUES (gen_random_uuid(), "
            ":action, gen_random_uuid(), 'rule', '[]', :now, :resolution, :by, :at)"
        ),
        {
            "action": action_id,
            "now": now,
            "resolution": resolution,
            "by": resolved_by,
            "at": now if resolution else None,
        },
    )


@pytest.mark.parametrize(
    ("resolution", "resolved_by", "valid"),
    [
        (None, None, True),
        ("not_a_conflict", "qa-1", True),
        ("keep_both", "qa-1", False),
        ("keep_new", None, False),
    ],
)
def test_memory_conflicts_constraints(
    migration_db: sa.Engine, config: Config, resolution: str | None, resolved_by: str | None, valid: bool
) -> None:
    """Only the four resolutions are storable, and a resolution always carries who and when."""
    command.upgrade(config, "head")
    if valid:
        with migration_db.begin() as conn:
            _conflict_row(conn, resolution=resolution, resolved_by=resolved_by)
    else:
        with pytest.raises(IntegrityError), migration_db.begin() as conn:
            _conflict_row(conn, resolution=resolution, resolved_by=resolved_by)
