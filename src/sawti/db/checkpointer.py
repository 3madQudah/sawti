"""Postgres-backed LangGraph checkpointer: durable storage for interrupted runs.

Phase 6.1. Clears the phase 2 `MemorySaver` debt (see `docs/09-DECISIONS.md`,
2026-09-22 and 2026-10-06): a run suspended in `escalate` is written to
Postgres, so it can be resumed by a different process after a restart.

`build_graph()` still defaults to `MemorySaver` for tests and the eval
harness. The service opens a saver here and passes it in explicitly:

    async with postgres_checkpointer() as saver:
        graph = build_graph(checkpointer=saver)
        await graph.ainvoke(state, {"configurable": {"thread_id": call_id}})

Table ownership: the four checkpoint tables (`checkpoints`,
`checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`) belong to
`langgraph-checkpoint-postgres`, which versions them itself through
`setup()`. Alembic owns the application tables only and is told to ignore
these (`sawti.db.models.alembic_include_object`).

Async, not the sync `PostgresSaver`: every graph node is `async` and the
graph is driven with `ainvoke`, which needs the saver's async methods.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy.engine import make_url

from sawti.config import get_settings


def checkpointer_conninfo(database_url: str) -> str:
    """Convert a SQLAlchemy URL into a libpq connection string for psycopg.

    `Settings.database_url` names a SQLAlchemy driver
    (`postgresql+psycopg://...`), which psycopg itself does not accept. Both
    connections must point at the same database, so this derives one from the
    other rather than adding a second setting that could drift.

    Args:
        database_url: A SQLAlchemy Postgres URL.

    Returns:
        The same URL with the plain `postgresql://` scheme.
    """
    return make_url(database_url).set(drivername="postgresql").render_as_string(hide_password=False)


@contextlib.asynccontextmanager
async def postgres_checkpointer(database_url: str | None = None) -> AsyncIterator[AsyncPostgresSaver]:
    """Open a Postgres checkpointer, ensure its tables exist, and close it on exit.

    `setup()` runs on every open. It is idempotent — it checks
    `checkpoint_migrations` and applies only what is missing — and it is
    cheap next to a graph run that makes an LLM call.

    Args:
        database_url: SQLAlchemy URL of the target database. Defaults to
            `Settings.database_url`, so the checkpoint tables live beside the
            application tables.

    Yields:
        A ready `AsyncPostgresSaver` on its own connection, closed when the
        block exits.
    """
    conninfo = checkpointer_conninfo(database_url or get_settings().database_url)
    async with AsyncPostgresSaver.from_conn_string(conninfo) as saver:
        await saver.setup()
        yield saver


async def _setup_once() -> None:
    async with postgres_checkpointer():
        pass


def main() -> None:
    """Create/upgrade the checkpoint tables once: `python -m sawti.db.checkpointer`.

    Run by the one-shot `migrate` service before any worker starts.
    `setup()` is not safe against two processes running it for the *first*
    time concurrently (both insert the same `checkpoint_migrations` version);
    once it has run, the workers' own idempotent `setup()` calls are no-ops.
    """
    asyncio.run(_setup_once())


if __name__ == "__main__":
    main()
