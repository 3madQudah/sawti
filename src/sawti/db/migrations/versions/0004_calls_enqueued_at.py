"""`calls.enqueued_at`: when a message for the call was last queued.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-07

Phase 6.2 reaper: it re-enqueues calls stuck `queued` with no message for
longer than a timeout. Keying on the last enqueue — not `queued_at` — keeps a
healthy backlog from being re-enqueued every interval. Existing rows get
`queued_at` (their only enqueue).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the column and backfill it from `queued_at`."""
    op.add_column("calls", sa.Column("enqueued_at", sa.DateTime(timezone=True), nullable=True))
    op.execute("UPDATE calls SET enqueued_at = queued_at")


def downgrade() -> None:
    """Drop it."""
    op.drop_column("calls", "enqueued_at")
