"""Audio ingestion (`transcribing`, `failure_reason`) and recorded memory conflicts.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-10

Phase 6.4 stage A (docs/09-DECISIONS.md, 2026-10-10):

  * `calls.status` gains `transcribing`: uploaded audio waiting for the native
    ASR worker. The status is a VARCHAR + CHECK (0001), so this is a
    constraint swap. `calls.audio_path` already exists (0001).
  * `calls.failure_reason`: why a call is `failed`, when known.
  * `memory_conflicts`: candidate rules `insert_rule` refused, with what they
    conflicted with and why. Before this they were only logged.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OLD_STATUSES = ("queued", "processing", "awaiting_review", "reviewed", "completed", "failed")
_NEW_STATUSES = ("transcribing", *_OLD_STATUSES)


def _status_check(statuses: tuple[str, ...]) -> str:
    return "status IN (" + ", ".join(f"'{s}'" for s in statuses) + ")"


def upgrade() -> None:
    """Widen the status CHECK, add `failure_reason`, create `memory_conflicts`."""
    op.drop_constraint("call_status", "calls", type_="check")
    op.create_check_constraint("call_status", "calls", _status_check(_NEW_STATUSES))
    op.add_column("calls", sa.Column("failure_reason", sa.Text(), nullable=True))

    op.create_table(
        "memory_conflicts",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("reviewer_action_id", sa.UUID(), nullable=False),
        sa.Column("candidate_rule_id", sa.UUID(), nullable=False),
        sa.Column("candidate_text", sa.Text(), nullable=False),
        sa.Column("conflicting", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolution", sa.String(length=32), nullable=True),
        sa.Column("resolved_by", sa.String(length=255), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "resolution IS NULL OR resolution IN ('keep_existing', 'keep_new', 'discard', 'not_a_conflict')",
            name="ck_memory_conflicts_resolution",
        ),
        sa.CheckConstraint(
            "(resolution IS NULL) = (resolved_by IS NULL) AND (resolution IS NULL) = (resolved_at IS NULL)",
            name="ck_memory_conflicts_resolved_together",
        ),
        sa.ForeignKeyConstraint(["reviewer_action_id"], ["reviewer_actions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("reviewer_action_id"),
    )


def downgrade() -> None:
    """Reverse it. Fails if any call is `transcribing` — finish or fail those first."""
    op.drop_table("memory_conflicts")
    op.drop_column("calls", "failure_reason")
    op.drop_constraint("call_status", "calls", type_="check")
    op.create_check_constraint("call_status", "calls", _status_check(_OLD_STATUSES))
