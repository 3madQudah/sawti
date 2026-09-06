"""Tests for `sawti.agent.nodes.compliance`.

Phase 2: mirrors `src/sawti/agent/nodes/compliance.py`.
"""

import pytest


@pytest.mark.skip(reason="phase 2")
def test_compliance_flags_only_rules_with_grounded_evidence() -> None:
    """compliance() never emits a ComplianceFlag without a grounded evidence quote."""


@pytest.mark.skip(reason="phase 2")
def test_compliance_assigns_severity_per_rule_definition() -> None:
    """compliance() assigns the severity defined for each violated rule_id."""
