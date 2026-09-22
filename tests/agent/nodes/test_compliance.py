"""Tests for `sawti.agent.nodes.compliance`.

Phase 2, stage 2: mirrors `src/sawti/agent/nodes/compliance.py`.

`compliance` is a deliberate passthrough — there is no canonical rule registry
to validate rule_ids or severities against. These tests pin that the guarantees
its stub asked for are genuinely met upstream, so the passthrough is safe rather
than merely unfinished.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sawti.agent.nodes.compliance import compliance
from sawti.schemas import ComplianceFlag, Quote, Severity

TRANSCRIPT = "Agent: I can't help you with that.\n"


def _flag(rule_id: str = "dismissive_tone", severity: Severity = Severity.MEDIUM) -> ComplianceFlag:
    """A grounded compliance flag."""
    text = "I can't help you with that."
    start = TRANSCRIPT.index(text)
    return ComplianceFlag(
        evidence=Quote(text=text, speaker="Agent", start_char=start, end_char=start + len(text)),
        rule_id=rule_id,
        severity=severity,
        description="Agent dismissed the customer.",
    )


async def test_compliance_passes_grounded_flags_through_untouched() -> None:
    """The node returns an empty update: flags survive exactly as ground() left them."""
    update = await compliance({"compliance_flags": [_flag()]})

    assert update == {}


async def test_compliance_preserves_the_extracted_severity() -> None:
    """With no rule registry, the model's severity stands rather than being overwritten."""
    flag = _flag(severity=Severity.CRITICAL)
    state = {"compliance_flags": [flag]}

    await compliance(state)

    assert state["compliance_flags"][0].severity is Severity.CRITICAL


def test_compliance_flag_cannot_exist_without_evidence() -> None:
    """compliance() never emits a ComplianceFlag without a grounded evidence quote.

    This is enforced by the schema, not by the node — a flag with no evidence
    cannot be constructed at all, so there is nothing for the node to strip.
    """
    with pytest.raises(ValidationError):
        ComplianceFlag(rule_id="x", severity=Severity.LOW, description="y")  # type: ignore[call-arg]
