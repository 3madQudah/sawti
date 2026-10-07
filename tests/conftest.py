"""Suite-wide fixtures: every test talks to a disposable, migrated test database.

Phase 6.1. Before this, DB-backed tests wrote into whatever `DATABASE_URL`
pointed at — the dev database, which also holds the phase 4/5 experiment data
— and built the schema with `Base.metadata.create_all()`. Now the schema is
owned by Alembic, so the suite gets its own database on the same server,
`<dev db name>_test`, dropped, recreated and upgraded to `head` once per run.

`DATABASE_URL` is overridden here, at conftest import, before any test module
is collected: some modules (e.g. `sawti.api.tasks`) read settings at import.

If Postgres is not reachable, the database setup is skipped with a warning so
the unit tests still run; DB-backed tests then fail with a connection error,
exactly as they did before (`docker compose up -d` first).
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import make_url

from sawti.config import Settings, get_settings
from sawti.db.checkpointer import checkpointer_conninfo
from sawti.db.session import get_engine, get_session_factory

_ALEMBIC_INI = Path(__file__).resolve().parent.parent / "alembic.ini"

_DEV_URL = make_url(Settings().database_url)
TEST_DATABASE_NAME = f"{_DEV_URL.database}_test"
TEST_DATABASE_URL = _DEV_URL.set(database=TEST_DATABASE_NAME).render_as_string(hide_password=False)

os.environ["DATABASE_URL"] = TEST_DATABASE_URL
for _cached in (get_settings, get_engine, get_session_factory):
    _cached.cache_clear()


def database_url(name: str) -> str:
    """A SQLAlchemy URL for database `name` on the test server."""
    return _DEV_URL.set(database=name).render_as_string(hide_password=False)


def recreate_database(name: str) -> None:
    """Drop (if present) and create an empty database `name` on the test server.

    Runs on the server's `postgres` maintenance database, in autocommit mode —
    `CREATE/DROP DATABASE` cannot run inside a transaction.
    """
    admin = checkpointer_conninfo(database_url("postgres"))
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        conn.execute(f'CREATE DATABASE "{name}"')


def alembic_config(url: str) -> Config:
    """An Alembic `Config` for this repo's migrations, targeting `url`."""
    config = Config(str(_ALEMBIC_INI))
    # `%` is configparser's interpolation character.
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return config


@pytest.fixture(scope="session", autouse=True)
def _test_database() -> None:
    """Create the suite's database and migrate it to head, once per run."""
    try:
        recreate_database(TEST_DATABASE_NAME)
    except psycopg.OperationalError as exc:
        warnings.warn(f"Postgres unreachable, DB-backed tests will fail: {exc}", stacklevel=1)
        return
    command.upgrade(alembic_config(TEST_DATABASE_URL), "head")


@pytest.fixture
def fresh_database() -> Callable[[str], str]:
    """Factory: recreate an empty database by name and return its SQLAlchemy URL.

    For tests that must own a whole database — e.g. a migration test that
    downgrades to an empty schema, which would wipe the shared test database
    out from under every other test.
    """

    def _make(name: str) -> str:
        recreate_database(name)
        return database_url(name)

    return _make


@pytest.fixture
def alembic_config_for() -> Callable[[str], Config]:
    """Factory: an Alembic `Config` for this repo's migrations, targeting a URL."""
    return alembic_config
