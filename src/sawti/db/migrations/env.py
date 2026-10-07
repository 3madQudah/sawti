"""Alembic environment: runs migrations against the application database.

Phase 6.1. The URL comes from `sawti.config` (one source of truth with the
app) unless the caller has set `sqlalchemy.url` on the Alembic `Config`, which
is how the migration tests point at a disposable database.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine, pool

from sawti.config import get_settings
from sawti.db.models import Base, alembic_include_object

config = context.config
target_metadata = Base.metadata


def _database_url() -> str:
    """The URL to migrate: an explicit `sqlalchemy.url` wins, else settings."""
    return config.get_main_option("sqlalchemy.url") or get_settings().database_url


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of executing it (`alembic upgrade --sql`)."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        include_object=alembic_include_object,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations on a live connection."""
    connectable = create_engine(_database_url(), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata, include_object=alembic_include_object
        )
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
