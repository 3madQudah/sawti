"""`escalation`: the structured reason a call is escalated, beside the prose reason (phase 6.4)."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from fastapi.testclient import TestClient

from sawti.api.records import NO_ANALYSIS_SUMMARY, escalation_info
from sawti.db.models import CallAnalysisRecord


def test_queue_and_detail_carry_the_numbers_behind_the_reason(
    client: TestClient, headers: dict[str, str], submit: Callable[..., str], llm: Any
) -> None:
    call_id = submit()  # the fake LLM proposes one ungrounded claim, so the call escalates

    detail = client.get(f"/calls/{call_id}", headers=headers).json()
    [item] = client.get("/reviews", headers=headers).json()["items"]

    escalation = detail["escalation"]
    assert escalation == item["escalation"]
    assert escalation["kind"] == "low_grounding"
    assert escalation["rejected_claims"] == len(detail["rejected_claims"]) == 1
    assert escalation["total_claims"] == len(detail["claims"]) + len(detail["rejected_claims"])
    assert escalation["grounding_coverage"] == detail["grounding_coverage"]
    assert escalation["threshold"] == detail["confidence_threshold"]
    # Same facts as the prose, which stays for compatibility.
    assert (
        f"{escalation['rejected_claims']} of {escalation['total_claims']} claims"
        in detail["escalation_reason"]
    )


def _record(**values: Any) -> CallAnalysisRecord:
    defaults: dict[str, Any] = {
        "id": uuid.uuid4(),
        "call_id": uuid.uuid4(),
        "payload": {"summary": "s", "commitments": [{}, {}], "compliance_flags": [], "rubric_scores": [{}]},
        "confidence": 0.4,
        "requires_human_review": True,
        "created_at": datetime.now(UTC),
        "grounding_coverage": 0.6,
        "confidence_threshold": 0.7,
        "escalation_reason": "1 of 4 claims had no verbatim evidence",
        "rejected_claims": [{}],
    }
    return CallAnalysisRecord(**{**defaults, **values})


def test_counts_grounded_plus_rejected() -> None:
    info = escalation_info(_record())
    assert info is not None
    assert (info.kind, info.rejected_claims, info.total_claims) == ("low_grounding", 1, 4)
    assert (info.grounding_coverage, info.threshold) == (0.6, 0.7)


def test_a_failed_extraction_is_an_analysis_error() -> None:
    record = _record(payload={"summary": NO_ANALYSIS_SUMMARY}, rejected_claims=[], grounding_coverage=0.0)
    info = escalation_info(record)
    assert info is not None and info.kind == "analysis_error" and info.total_claims == 0


def test_auto_passed_or_missing_has_no_escalation() -> None:
    assert escalation_info(_record(escalation_reason=None)) is None
    assert escalation_info(None) is None
