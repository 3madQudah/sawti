"""Tests for `sawti.db.bootstrap` (phase 6.3)."""

from __future__ import annotations

import psycopg
import pytest

from sawti.db.bootstrap import ensure_database
from sawti.db.checkpointer import checkpointer_conninfo


def test_creates_once_then_leaves_it_alone(fresh_database) -> None:  # type: ignore[no-untyped-def]
    admin_url = fresh_database("sawti_bootstrap_probe")  # exists now; drop it to start clean
    with psycopg.connect(
        checkpointer_conninfo(admin_url.rsplit("/", 1)[0] + "/postgres"), autocommit=True
    ) as c:
        c.execute('DROP DATABASE "sawti_bootstrap_probe" WITH (FORCE)')
    try:
        assert ensure_database("sawti_bootstrap_probe", database_url=admin_url) is True
        assert ensure_database("sawti_bootstrap_probe", database_url=admin_url) is False
    finally:
        with psycopg.connect(
            checkpointer_conninfo(admin_url.rsplit("/", 1)[0] + "/postgres"), autocommit=True
        ) as c:
            c.execute('DROP DATABASE IF EXISTS "sawti_bootstrap_probe" WITH (FORCE)')


@pytest.mark.parametrize("name", ["Langfuse", 'x"; DROP DATABASE sawti; --', "", "a" * 64])
def test_refuses_anything_but_a_plain_identifier(name: str) -> None:
    with pytest.raises(ValueError, match="plain database name"):
        ensure_database(name)
