"""SQLAlchemy ORM models.

Phase 1: persistence layer structure.

Note on identity: `Agent` here is a contact-center-agent record that calls
and rubric scores attach to for reporting — it is not a login/account and
has no credentials. The only authenticating identity anywhere in this
system is the QA reviewer, captured on `ReviewerAction` /
`sawti.schemas.Correction`. Do not add auth/permissions fields to `Agent`.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all Sawti ORM models."""


class Agent(Base):
    """A contact-center agent record. Not a login — has no credentials.

    Calls and rubric scores attach to this record for reporting purposes only.
    """

    __tablename__ = "agents"

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    external_id: Mapped[str | None] = mapped_column(String(255), nullable=True, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class Call(Base):
    """A single contact-center call and its metadata."""

    __tablename__ = "calls"

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("agents.id"), nullable=False)
    audio_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str] = mapped_column(String(16), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    agent: Mapped[Agent] = relationship()


class CallAnalysisRecord(Base):
    """Persisted `sawti.schemas.CallAnalysis` output for a call."""

    __tablename__ = "call_analyses"

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("calls.id"), nullable=False)
    payload: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    confidence: Mapped[float] = mapped_column(nullable=False)
    requires_human_review: Mapped[bool] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    call: Mapped[Call] = relationship()


class ReviewerAction(Base):
    """A persisted `sawti.schemas.Correction` — the only authenticated identity in this system."""

    __tablename__ = "reviewer_actions"

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_analysis_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("call_analyses.id"), nullable=False
    )
    reviewer_id: Mapped[str] = mapped_column(String(255), nullable=False)
    payload: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    call_analysis: Mapped[CallAnalysisRecord] = relationship()
