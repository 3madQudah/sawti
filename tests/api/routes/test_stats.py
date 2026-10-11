"""`GET /stats`: per-language counters, every rate with its n (phase 6.4)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sawti.api import tasks
from sawti.eval.metrics import reviewer_agreement_by_category
from sawti.schemas import Language

EMPTY_RATE = {"value": None, "numerator": 0, "n": 0}


def test_an_empty_system_reports_zero_counts_and_rates_with_n_zero(
    client: TestClient, headers: dict[str, str]
) -> None:
    response = client.get("/stats", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["active_rules_distinct"] == 0
    for language in ("ar", "en", "mixed"):
        assert body[language] == {
            "pending_reviews": 0,
            "reviewer_agreement": EMPTY_RATE,
            "active_rules": 0,
            "pending_conflicts": 0,
            "conflict_false_positive_rate": EMPTY_RATE,
        }


def test_counts_are_per_language_and_never_blended(
    client: TestClient,
    headers: dict[str, str],
    submit: Callable[..., str],
    review_call: Callable[[str], None],
    transcripts: dict[str, str],
    make_conflicts: Callable[..., list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tasks.consolidate_rules_task, "delay", lambda: None)
    # en: one review (stores the rule); mixed, ar: one review each (both conflict with it).
    make_conflicts("mixed", "ar", first="en")
    # Two more escalated calls wait for review: one ar, one mixed.
    submit(transcripts["ar"], "ar")
    submit(transcripts["mixed"], "mixed")
    conflicts = client.get("/memory/conflicts", headers=headers).json()["conflicts"]
    by_language = {c["source"]["language"]: c["id"] for c in conflicts}
    client.post(
        f"/memory/conflicts/{by_language['mixed']}/resolve",
        json={"action": "not_a_conflict"},
        headers=headers,
    )

    body = client.get("/stats", headers=headers).json()

    assert {lang: body[lang]["pending_reviews"] for lang in ("ar", "en", "mixed")} == {
        "ar": 1,
        "en": 0,
        "mixed": 1,
    }
    # Each review rejected one claim and confirmed the rest.
    for language in ("ar", "en", "mixed"):
        rate = body[language]["reviewer_agreement"]
        assert rate["n"] >= 1 and rate["numerator"] == rate["n"] - 1
        assert rate["value"] == pytest.approx(rate["numerator"] / rate["n"])
    assert body["ar"]["pending_conflicts"] == 1 and body["mixed"]["pending_conflicts"] == 0
    assert body["mixed"]["conflict_false_positive_rate"] == {"value": 1.0, "numerator": 1, "n": 1}
    assert body["ar"]["conflict_false_positive_rate"] == EMPTY_RATE
    # The en rule plus the mixed candidate made active by not_a_conflict.
    assert body["en"]["active_rules"] == 1 and body["mixed"]["active_rules"] == 1
    assert body["ar"]["active_rules"] == 0
    assert body["active_rules_distinct"] == 2


def test_agreement_matches_the_metric_definition(
    client: TestClient,
    headers: dict[str, str],
    submit: Callable[..., str],
    review_call: Callable[[str], None],
    transcripts: dict[str, str],
    llm: Any,
) -> None:
    """`/stats` counts agreement itself (the metrics module imports the agent); it must agree."""
    verdicts: dict[Language, list[str]] = {}
    for language in ("en", "mixed"):
        call_id = submit(transcripts[language], language)
        review_call(call_id)
        detail = client.get(f"/calls/{call_id}", headers=headers).json()
        verdicts[Language(language)] = [v["verdict"] for v in detail["review"]["verdicts"]]

    expected = reviewer_agreement_by_category(list(verdicts.items()))
    body = client.get("/stats", headers=headers).json()
    for language, metric in expected.items():
        assert body[language.value]["reviewer_agreement"]["value"] == pytest.approx(metric["claim_agreement"])
        assert body[language.value]["reviewer_agreement"]["n"] == metric["claims"]


@pytest.mark.parametrize(
    "sent",
    [
        {},
        {"X-API-Key": "", "X-Reviewer-Id": "qa"},
        {"X-API-Key": "wrong", "X-Reviewer-Id": "qa"},
        {"X-API-Key": "x"},
    ],
)
def test_stats_needs_a_key_and_a_reviewer(client: TestClient, sent: dict[str, str]) -> None:
    assert client.get("/stats", headers=sent).status_code == 401
