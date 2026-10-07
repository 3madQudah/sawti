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

APP_TABLES = {"agents", "calls", "call_analyses", "reviewer_actions", "review_submissions", "memory_rules"}


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
