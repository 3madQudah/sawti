"""Tests for `sawti.db.models`.

Phase 1: mirrors `src/sawti/db/models.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 1")
def test_agent_table_has_no_credential_columns() -> None:
    """Agent has no password/credential/permission columns — it is a record, not a login."""


@pytest.mark.skip(reason="phase 1")
def test_call_requires_an_agent_foreign_key() -> None:
    """Call cannot be persisted without a valid agent_id."""


@pytest.mark.skip(reason="phase 1")
def test_reviewer_action_is_the_only_authenticated_identity() -> None:
    """ReviewerAction.reviewer_id is the sole authenticated identity referenced by the schema."""
