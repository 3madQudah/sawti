"""capture_demo_api: what it fetches, and that it refuses a capture taken at the wrong moment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from capture_demo_api import capture, endpoints, env_value

CALL = "c-1"


def fake_fetch(call: dict[str, Any]):
    seen: list[str] = []

    def fetch(path: str) -> dict[str, Any]:
        seen.append(path)
        return call if path.startswith("/calls/") else {"path": path}

    return fetch, seen


def test_before_reads_results_and_the_call() -> None:
    keys = endpoints("before", CALL)
    assert "/results" in keys and f"/calls/{CALL}" in keys
    assert keys["/reviews"] == "/reviews?limit=500"


def test_after_also_reads_results_so_it_can_stand_alone() -> None:
    assert "/results" in endpoints("after", CALL)


def test_unknown_stage_is_rejected() -> None:
    with pytest.raises(ValueError):
        endpoints("during", CALL)


def test_before_capture_records_every_response() -> None:
    fetch, seen = fake_fetch({"status": "awaiting_review", "claims": []})
    data = capture("before", CALL, fetch, audio_name="call.wav")
    assert data["audio_name"] == "call.wav"
    assert set(data["responses"]) == set(endpoints("before", CALL))
    assert len(seen) == len(endpoints("before", CALL))


def test_before_refuses_a_call_not_awaiting_review() -> None:
    fetch, _ = fake_fetch({"status": "processing"})
    with pytest.raises(SystemExit):
        capture("before", CALL, fetch)


UNREVIEWED = [{"status": "awaiting_review", "review": None}, {"status": "reviewed", "review": None}]


@pytest.mark.parametrize("call", UNREVIEWED)
def test_after_refuses_an_unreviewed_call(call: dict[str, Any]) -> None:
    fetch, _ = fake_fetch(call)
    with pytest.raises(SystemExit):
        capture("after", CALL, fetch)


def test_after_accepts_a_reviewed_call() -> None:
    fetch, _ = fake_fetch({"status": "reviewed", "review": {"verdicts": []}})
    assert capture("after", CALL, fetch)["stage"] == "after"


def test_env_value_prefers_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("SAWTI_API_KEY='from-file'\n", encoding="utf-8")
    monkeypatch.delenv("SAWTI_API_KEY", raising=False)
    assert env_value("SAWTI_API_KEY", env) == "from-file"
    monkeypatch.setenv("SAWTI_API_KEY", "from-env")
    assert env_value("SAWTI_API_KEY", env) == "from-env"


def test_env_value_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAWTI_API_KEY", raising=False)
    assert env_value("SAWTI_API_KEY", tmp_path / "none") is None
