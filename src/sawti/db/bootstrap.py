"""Create the extra databases the stack needs on the shared Postgres server, if missing.

Phase 6.3: the self-hosted Langfuse v2 server keeps its own schema in its own
database (`langfuse`) on the same Postgres as the application. The existing
Postgres volume never re-runs init scripts, so the one-shot `migrate` service
calls this before anything that needs the database:

    python -m sawti.db.bootstrap langfuse

Idempotent: an existing database is left alone.
"""

from __future__ import annotations

import re
import sys

import psycopg
from sqlalchemy.engine import make_url

from sawti.config import get_settings
from sawti.db.checkpointer import checkpointer_conninfo

_VALID_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def ensure_database(name: str, *, database_url: str | None = None) -> bool:
    """Create database `name` on the application's Postgres server unless it exists.

    Args:
        name: Database to ensure. Lower-case identifier only — it is
            interpolated into DDL, which cannot take a bound parameter.
        database_url: Any database URL on the target server; defaults to
            `Settings.database_url`.

    Returns:
        True if it was created, False if it already existed.

    Raises:
        ValueError: `name` is not a plain lower-case identifier.
    """
    if not _VALID_NAME.match(name):
        raise ValueError(f"not a plain database name: {name!r}")
    url = make_url(database_url or get_settings().database_url).set(database="postgres")
    with psycopg.connect(
        checkpointer_conninfo(url.render_as_string(hide_password=False)), autocommit=True
    ) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
        if exists:
            return False
        conn.execute(f'CREATE DATABASE "{name}"')
        return True


def main(argv: list[str]) -> int:
    for name in argv:
        print(f"database {name}: {'created' if ensure_database(name) else 'exists'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
