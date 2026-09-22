"""Tests for `sawti.db.session`.

Phase 1: mirrors `src/sawti/db/session.py`.

Integration tests against a real Postgres instance — the ORM models
(`sawti.db.models`) use Postgres-only column types (`JSONB`, native `UUID`),
so a lighter substitute like sqlite cannot stand in. Run `docker compose up
-d` first; see `docs/09-DECISIONS.md`.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from sawti.db.models import Agent, Base
from sawti.db.session import get_engine, get_session


@pytest.fixture(scope="module", autouse=True)
def _schema() -> None:
    """Create the ORM schema once for this test module, against the real database."""
    Base.metadata.create_all(get_engine())


def test_get_engine_returns_memoized_singleton() -> None:
    """get_engine() returns the same Engine instance on repeated calls."""
    assert get_engine() is get_engine()


def test_get_session_commits_on_success() -> None:
    """get_session() commits the transaction when the block completes without error."""
    agent_id = uuid.uuid4()
    with get_session() as session:
        session.add(Agent(id=agent_id, name="commit-test", created_at=datetime.now(UTC)))

    with get_session() as session:
        found = session.get(Agent, agent_id)
        assert found is not None
        assert found.name == "commit-test"
        session.delete(found)


def test_get_session_rolls_back_on_exception() -> None:
    """get_session() rolls back the transaction when an exception is raised inside the block."""
    agent_id = uuid.uuid4()
    with pytest.raises(RuntimeError, match="boom"):
        with get_session() as session:
            session.add(Agent(id=agent_id, name="rollback-test", created_at=datetime.now(UTC)))
            raise RuntimeError("boom")

    with get_session() as session:
        assert session.get(Agent, agent_id) is None
