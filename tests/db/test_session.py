"""Tests for `sawti.db.session`.

Phase 1: mirrors `src/sawti/db/session.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 1")
def test_get_engine_returns_memoized_singleton() -> None:
    """get_engine() returns the same Engine instance on repeated calls."""


@pytest.mark.skip(reason="phase 1")
def test_get_session_commits_on_success() -> None:
    """get_session() commits the transaction when the block completes without error."""


@pytest.mark.skip(reason="phase 1")
def test_get_session_rolls_back_on_exception() -> None:
    """get_session() rolls back the transaction when an exception is raised inside the block."""
