"""Per-correction induction state on `reviewer_actions`.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-07

Phase 6.2 memory loop: `induced_at` / `induction_outcome` record that a
correction has been through rule induction, so the service task runs it at
most once per correction and the backfill over existing corrections can
resume after the provider's daily quota stops it. See docs/09-DECISIONS.md.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the two nullable columns."""
    op.add_column("reviewer_actions", sa.Column("induced_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("reviewer_actions", sa.Column("induction_outcome", sa.String(length=16), nullable=True))


def downgrade() -> None:
    """Drop them."""
    op.drop_column("reviewer_actions", "induction_outcome")
    op.drop_column("reviewer_actions", "induced_at")
