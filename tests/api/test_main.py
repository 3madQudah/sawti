"""Tests for `sawti.api.main`.

Phase 6.2: the app's routes, CORS, and the rule that the API process never
loads the analysis graph.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sawti.api.main import create_app


def test_create_app_mounts_calls_reviews_and_health() -> None:
    app = create_app()
    assert isinstance(app, FastAPI)
    paths = {route.path for route in app.routes}  # type: ignore[attr-defined]
    assert {
        "/calls",
        "/calls/{call_id}",
        "/reviews",
        "/reviews/{call_id}/corrections",
        "/memory/rules",
        "/health",
    } <= paths


def test_importing_the_api_does_not_load_the_graph_or_torch() -> None:
    """The API only enqueues. A fresh interpreter proves the graph and torch stay out of it."""
    probe = (
        "import sys, sawti.api.main; "
        "heavy = ('sawti.agent.graph', 'torch', 'sentence_transformers', 'langgraph.checkpoint.postgres'); "
        "print([m for m in heavy if m in sys.modules])"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_cors_allows_only_configured_origins(monkeypatch: pytest.MonkeyPatch) -> None:
    from sawti.config import get_settings

    monkeypatch.setenv("SAWTI_CORS_ORIGINS", '["http://localhost:5173"]')
    get_settings.cache_clear()
    client = TestClient(create_app())
    preflight = {"Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "X-API-Key"}
    allowed = client.options("/reviews", headers={"Origin": "http://localhost:5173", **preflight})
    denied = client.options("/reviews", headers={"Origin": "http://evil.example", **preflight})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-origin" not in denied.headers
