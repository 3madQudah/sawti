"""Database session/engine management.

Phase 1: persistence layer plumbing.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import Engine
from sqlalchemy.orm import Session


def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy engine, created on first call.

    Returns:
        A configured SQLAlchemy `Engine` bound to `sawti.config.Settings.database_url`.

    Raises:
        NotImplementedError: Until engine construction is implemented.
    """
    # TODO(phase-1): create_engine(get_settings().database_url, pool_pre_ping=True), memoized.
    raise NotImplementedError


def get_session() -> Iterator[Session]:
    """Yield a `Session`, committing on success and rolling back on error.

    Intended for use as a context manager once implemented (decorate with
    `@contextlib.contextmanager`).

    Yields:
        An active SQLAlchemy `Session`.

    Raises:
        NotImplementedError: Until session management is implemented.
    """
    # TODO(phase-1): open a Session from a sessionmaker bound to get_engine(); commit on
    # success, rollback on error, always close.
    raise NotImplementedError
