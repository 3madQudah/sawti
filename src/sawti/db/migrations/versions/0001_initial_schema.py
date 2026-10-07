"""Initial schema: the phase 1 models, reshaped for the phase 6 service.

Revision ID: 0001
Revises:
Create Date: 2026-10-06

Before this revision the schema existed only as `Base.metadata.create_all()`.
Differences from that phase 1 shape:

  * `calls.transcript` is now `calls.redacted_transcript` — only redacted text
    is stored (docs/09-DECISIONS.md, 2026-10-06, redaction at the API boundary).
  * `calls.status` (VARCHAR + CHECK over `sawti.schemas.CallStatus`),
    `queued_at` / `started_at` / `finished_at` (timestamptz, nullable) and
    `pii_redacted_count` (count only, never the values).
  * `call_analyses.call_id` is unique: one analysis per call.
  * `call_analyses` gains the review-screen context columns:
    `grounding_coverage`, `confidence_threshold`, `escalation_reason`,
    `rejected_claims`, `retrieved_rule_ids`.

LangGraph's checkpoint tables are not created here; `sawti.db.checkpointer`
creates them via the library's own `setup()`.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CALL_STATUSES = ("queued", "processing", "awaiting_review", "reviewed", "completed", "failed")


def upgrade() -> None:
    """Create the four application tables."""
    op.create_table(
        "agents",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("external_id"),
    )
    op.create_table(
        "calls",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("agent_id", sa.UUID(), nullable=False),
        sa.Column("audio_path", sa.String(length=1024), nullable=True),
        sa.Column("redacted_transcript", sa.Text(), nullable=True),
        sa.Column("language", sa.String(length=16), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                *_CALL_STATUSES, name="call_status", native_enum=False, create_constraint=True, length=32
            ),
            server_default="queued",
            nullable=False,
        ),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("pii_redacted_count", sa.Integer(), server_default="0", nullable=False),
        sa.ForeignKeyConstraint(["agent_id"], ["agents.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "call_analyses",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("call_id", sa.UUID(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("requires_human_review", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("grounding_coverage", sa.Float(), nullable=True),
        sa.Column("confidence_threshold", sa.Float(), nullable=True),
        sa.Column("escalation_reason", sa.Text(), nullable=True),
        sa.Column(
            "rejected_claims",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "retrieved_rule_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["call_id"], ["calls.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("call_id", name="uq_call_analyses_call_id"),
    )
    op.create_table(
        "reviewer_actions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("call_analysis_id", sa.UUID(), nullable=False),
        sa.Column("reviewer_id", sa.String(length=255), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["call_analysis_id"], ["call_analyses.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Drop the application tables, children first. Checkpoint tables are untouched."""
    op.drop_table("reviewer_actions")
    op.drop_table("call_analyses")
    op.drop_table("calls")
    op.drop_table("agents")
