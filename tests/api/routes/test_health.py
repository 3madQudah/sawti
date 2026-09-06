"""Tests for `sawti.api.routes.health`.

Phase 3: mirrors `src/sawti/api/routes/health.py`.
"""

from sawti.api.routes.health import health


async def test_health_returns_ok_status() -> None:
    """health() reports {"status": "ok"} for a live service."""
    result = await health()
    assert result == {"status": "ok"}
