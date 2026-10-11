"""`frontend/openapi.json` is what the app generates, and covers the dashboard's endpoints (phase 6.4)."""

from __future__ import annotations

import json
from pathlib import Path

import export_openapi

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = ROOT / "frontend" / "openapi.json"


def test_the_committed_schema_is_current() -> None:
    assert COMMITTED.read_text(encoding="utf-8") == export_openapi.render(
        export_openapi.openapi_document()
    ), "frontend/openapi.json is stale: run `make openapi`"


def test_the_schema_covers_every_dashboard_endpoint() -> None:
    paths = json.loads(COMMITTED.read_text(encoding="utf-8"))["paths"]
    expected = {
        "/calls": {"post"},
        "/calls/audio": {"post"},
        "/calls/{call_id}": {"get"},
        "/reviews": {"get"},
        "/reviews/{call_id}/corrections": {"post"},
        "/memory/rules": {"get"},
        "/memory/conflicts": {"get"},
        "/memory/conflicts/{conflict_id}/resolve": {"post"},
        "/stats": {"get"},
        "/results": {"get"},
        "/health": {"get"},
    }
    assert {path: set(ops) for path, ops in paths.items()} == expected


def test_main_writes_the_file(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import sys

    out = tmp_path / "nested" / "openapi.json"
    monkeypatch.setattr(sys, "argv", ["export_openapi.py", "--out", str(out)])
    assert export_openapi.main() == 0
    assert json.loads(out.read_text(encoding="utf-8"))["info"]["title"] == "Sawti"
