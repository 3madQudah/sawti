"""`GET /results`: the 6.3 numbers, served exactly as eval_results.md records them (phase 6.4).

The tie-out tests parse the 2026-10-10 tables in eval_results.md and check
every served number against them, formatted at the table's own precision. If
either file changes without the other, these fail.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sawti.api import results as results_module
from sawti.api.results import load_results

ROOT = Path(__file__).resolve().parents[3]
EVAL_RESULTS = (ROOT / "eval_results.md").read_text(encoding="utf-8")
SERVING_ROUND = "## Round: 2026-10-10 — phase 6.3: cloud vs self-hosted serving, per language category"
MEMORY_ROUND = (
    "## Round: 2026-10-10 — memory comparison, corrected (35 complete-linkage rules) — FINAL, 600 / 600"
)


def _section(heading: str, until: str = "\n## ") -> str:
    start = EVAL_RESULTS.index(heading)
    end = EVAL_RESULTS.find(until, start + len(heading))
    return EVAL_RESULTS[start : end if end != -1 else None]


def _table_rows(text: str) -> list[list[str]]:
    """Body rows of the first markdown table in `text`."""
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("|"):
            lines.append(line)
        elif lines:
            break
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in lines[2:]]


def _matches(value: float, recorded: str) -> bool:
    """`value` printed at `recorded`'s precision is `recorded` (sign included)."""
    recorded = recorded.strip("*")
    decimals = len(recorded.split(".")[1]) if "." in recorded else 0
    formatted = f"{value:+.{decimals}f}" if recorded.startswith("+") else f"{value:.{decimals}f}"
    return formatted == recorded


@pytest.fixture(scope="module")
def served() -> dict[str, object]:
    return load_results().model_dump()


@pytest.mark.parametrize("language", ["ar", "en", "mixed"])
def test_every_serving_number_is_the_one_in_eval_results(served: dict, language: str) -> None:  # type: ignore[type-arg]
    rows = _table_rows(_section(f"### {language}\n", "\n### ").split("\n", 1)[1])
    arms = {arm["label"]: arm for arm in served["serving"]["arms"]}
    assert [row[0] for row in rows] == list(arms), "same arms, same order"
    for label, hardware, n, acc, rubric, grounding, unsupported, valid, latency, tput, cost, sent in rows:
        arm, cell = arms[label], arms[label][language]
        assert arm["hardware"] == hardware
        assert cell["n"] == int(n)
        for value, recorded in [
            (cell["accuracy"], acc),
            (cell["rubric_agreement"], rubric),
            (cell["grounding_precision"], grounding),
            (cell["unsupported_claim_rate"], unsupported),
            (cell["schema_valid_rate"], valid),
            (arm["throughput"]["calls_per_min"], tput),
            (cell["cost_per_1000_usd"], cost),
            (cell["third_party_transcript_bytes_per_call"], sent),
        ]:
            assert _matches(value, recorded), (label, value, recorded)
        p50, p95, latency_n = re.fullmatch(r"([\d.]+) / ([\d.]+)(?: \(n=(\d+)\))?", latency).groups()  # type: ignore[union-attr]
        assert _matches(cell["latency_p50_s"], p50) and _matches(cell["latency_p95_s"], p95)
        assert cell["latency_n"] == int(latency_n or n)


def test_every_throughput_row_is_the_one_in_eval_results(served: dict) -> None:  # type: ignore[type-arg]
    text = _section("### Throughput per arm", "\n### ")
    arms = {arm["label"]: arm for arm in served["serving"]["arms"]}
    rows = _table_rows(text)
    assert len(rows) == len(arms)
    for label, _, calls, in_flight, tput, tokens, method in rows:
        throughput = arms[label]["throughput"]
        assert throughput["calls"] == int(calls) and throughput["concurrency"] == int(in_flight)
        assert _matches(throughput["calls_per_min"], tput) and _matches(
            throughput["output_tokens_per_s"], tokens
        )
        assert throughput["method"] == method


def test_every_memory_number_is_the_one_in_eval_results(served: dict) -> None:  # type: ignore[type-arg]
    memory = served["memory"]
    section = _section(MEMORY_ROUND)
    per_repeat = _table_rows(section.split("### Accuracy per repeat", 1)[1])
    assert len(per_repeat) == 6
    for language, repeat, n, off, on, delta in per_repeat:
        result, r = memory[language.strip()], int(repeat)
        assert result["n_paired_per_repeat"][r] == int(n)
        assert _matches(result["accuracy_off_per_repeat"][r], off)
        assert _matches(result["accuracy_on_per_repeat"][r], on)
        assert _matches(result["delta_per_repeat"][r], delta)

    verdicts = _table_rows(section.split("### Verdict against Phase 4", 1)[1])
    expected = {"ar": "contradicts", "en": "inconclusive", "mixed": "confirms"}
    for language, effect, noise, phase4, verdict in verdicts:
        result = memory[language.strip()]
        assert _matches(result["delta"], effect)
        assert _matches(result["noise"], noise)
        assert _matches(result["phase4_delta"], phase4)
        assert result["verdict"] == expected[language.strip()] and expected[language.strip()] in verdict
    assert memory["calls"] == 600


def test_every_number_carries_its_method_n_and_hardware(served: dict) -> None:  # type: ignore[type-arg]
    serving = served["serving"]
    assert serving["accuracy_comparable_across_providers"] is False
    assert serving["actual_spend_usd"] == 0
    assert set(serving["methods"]) >= {"accuracy", "latency", "cost_per_1000_usd", "throughput", "n"}
    for arm in serving["arms"]:
        assert arm["hardware"] and arm["throughput"]["method"]
        assert arm["deployment"] == ("cloud" if arm["id"] == "gemini" else "self_hosted")
        for language in ("ar", "en", "mixed"):
            assert arm[language]["n"] > 0 and arm[language]["latency_n"] > 0
    memory = served["memory"]
    assert memory["method"] and memory["hardware"] and memory["provider"]


def test_results_route_serves_the_files(client: TestClient, headers: dict[str, str]) -> None:
    response = client.get("/results", headers=headers)
    assert response.status_code == 200
    assert response.json() == load_results().model_dump(mode="json")


def test_results_route_is_503_when_a_file_is_missing(
    client: TestClient, headers: dict[str, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(results_module, "MEMORY_PATH", tmp_path / "absent.json")
    response = client.get("/results", headers=headers)
    assert response.status_code == 503


@pytest.mark.parametrize(
    "sent", [{}, {"X-API-Key": "", "X-Reviewer-Id": "qa"}, {"X-API-Key": "test-api-key"}]
)
def test_results_needs_a_key_and_a_reviewer(client: TestClient, sent: dict[str, str]) -> None:
    assert client.get("/results", headers=sent).status_code == 401


def test_the_dashboard_mock_is_exactly_what_the_api_serves() -> None:
    """`frontend/src/test/results.json` backs the Results page's tests; it must not drift from `/results`.

    Regenerate after a results change: `load_results().model_dump(mode="json")`, dumped with indent=1.
    """
    import json

    mock = json.loads((ROOT / "frontend/src/test/results.json").read_text(encoding="utf-8"))
    assert mock == load_results().model_dump(mode="json")
