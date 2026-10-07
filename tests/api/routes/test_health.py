"""Tests for `sawti.api.routes.health`.

Phase 6.2: the probe checks the database and Redis for real.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sawti.api.routes import health


def test_health_is_ok_when_db_and_redis_answer(client: TestClient) -> None:
    """Against the real test Postgres and the local Redis."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok", "redis": "ok"}


def test_health_is_503_and_names_the_dead_dependency(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CELERY_BROKER_URL", "redis://127.0.0.1:1/0")
    from sawti.config import get_settings

    get_settings.cache_clear()
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "database": "ok", "redis": "error"}


def test_check_database_reports_failure_instead_of_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    def _broken() -> None:
        raise RuntimeError("no engine")

    monkeypatch.setattr(health, "get_engine", _broken)
    assert health.check_database() is False


def test_health_needs_no_credentials(client: TestClient) -> None:
    assert client.get("/health").status_code != 401
