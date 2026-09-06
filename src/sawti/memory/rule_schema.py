"""Schema for induced memory rules — general corrections derived from reviewer feedback.

Phase 4: agent memory.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MemoryRule(BaseModel):
    """A general rule induced from one or more supervisor corrections.

    Retrieved at inference time (top-5, see `sawti.memory.store`) and
    injected into node prompts to prevent repeat mistakes.
    """

    id: UUID = Field(default_factory=uuid4)
    rule_text: str = Field(..., min_length=1, description="The general rule, in natural language.")
    source_correction_ids: list[UUID] = Field(default_factory=list)
    created_at: datetime
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    retired: bool = Field(default=False)
