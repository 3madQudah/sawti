"""Database session/engine management.

Phase 1: persistence layer plumbing.

Requires a running Postgres instance — `docker compose up -d` starts one
matching `docker-compose.yml`'s `postgres` service, whose defaults line up
with `Settings.database_url`'s default. Nothing here hardcodes that
assumption: `get_engine()` reads `Settings.database_url` alone, so pointing
at a different Postgres (or a CI-provisioned one) is a config change, not a
code change. See `docs/09-DECISIONS.md`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from sawti.config import get_settings


@lru_cache
def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy engine, created on first call.

    Memoized: repeated calls return the same `Engine`, so its connection
    pool is shared rather than rebuilt per call.

    Returns:
        A configured SQLAlchemy `Engine` bound to `sawti.config.Settings.database_url`.
    """
    return create_engine(get_settings().database_url, pool_pre_ping=True)


@contextlib.contextmanager
def get_session() -> Iterator[Session]:
    """Yield a `Session`, committing on success and rolling back on error.

    Use as a context manager:

        with get_session() as session:
            session.add(some_row)

    Yields:
        An active SQLAlchemy `Session`, bound to `get_engine()`. Committed
        when the block exits cleanly, rolled back if it raises, and always
        closed.
    """
    with Session(get_engine()) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
