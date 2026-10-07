"""Liveness/readiness: is the database reachable, is Redis (the Celery broker) reachable.

Phase 6.2.
"""

from __future__ import annotations

import redis
import sqlalchemy as sa
from fastapi import APIRouter, Response, status

from sawti.api.schemas import HealthResponse
from sawti.config import get_settings
from sawti.db.session import get_engine

router = APIRouter(prefix="/health", tags=["health"])


def check_database() -> bool:
    """`SELECT 1` on the shared engine."""
    try:
        with get_engine().connect() as conn:
            conn.execute(sa.text("SELECT 1"))
    except Exception:
        return False
    return True


def check_redis() -> bool:
    """`PING` the broker, with a short timeout so a dead Redis cannot hang the probe."""
    try:
        client = redis.Redis.from_url(
            get_settings().celery_broker_url, socket_timeout=2, socket_connect_timeout=2
        )
        return bool(client.ping())
    except Exception:
        return False


@router.get("", response_model=HealthResponse)
def health(response: Response) -> HealthResponse:
    """200 when both dependencies answer, 503 (with which one failed) otherwise."""
    database, broker = check_database(), check_redis()
    if not (database and broker):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if database and broker else "degraded",
        database="ok" if database else "error",
        redis="ok" if broker else "error",
    )
