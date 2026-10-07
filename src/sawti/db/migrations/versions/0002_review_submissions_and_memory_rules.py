"""Review submissions, persisted memory rules, and task attempt counts.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-07

Phase 6.2:

  * `review_submissions` — a reviewer's full per-claim verdict set,
    `confirm` included, which `sawti.schemas.Correction` cannot represent.
    One per analysis. `reviewer_actions.review_submission_id` links each
    reject/correct `Correction` back to it (nullable: phase 4/5 rows predate it).
  * `memory_rules` — `MemoryRule` records, previously never persisted.
  * `calls.attempts` — how many times the analysis task claimed the call;
    the source of the retry-rate number.

See docs/09-DECISIONS.md, 2026-10-07.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the two tables and two columns."""
    op.create_table(
        "memory_rules",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("rule_text", sa.Text(), nullable=False),
        sa.Column(
            "source_correction_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("success_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("failure_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("retired", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "review_submissions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("call_analysis_id", sa.UUID(), nullable=False),
        sa.Column("reviewer_id", sa.String(length=255), nullable=False),
        sa.Column("verdicts", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["call_analysis_id"], ["call_analyses.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("call_analysis_id"),
    )
    op.add_column("calls", sa.Column("attempts", sa.Integer(), server_default="0", nullable=False))
    op.add_column("reviewer_actions", sa.Column("review_submission_id", sa.UUID(), nullable=True))
    op.create_foreign_key(
        "fk_reviewer_actions_review_submission_id",
        "reviewer_actions",
        "review_submissions",
        ["review_submission_id"],
        ["id"],
    )


def downgrade() -> None:
    """Remove them again, dependents first."""
    op.drop_constraint("fk_reviewer_actions_review_submission_id", "reviewer_actions", type_="foreignkey")
    op.drop_column("reviewer_actions", "review_submission_id")
    op.drop_column("calls", "attempts")
    op.drop_table("review_submissions")
    op.drop_table("memory_rules")
