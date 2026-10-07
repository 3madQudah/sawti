"""FastAPI application entry point.

Phase 6.2. The API reads and writes the database and enqueues Celery tasks;
it never runs the analysis graph (see `sawti.api.tasks`).
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sawti.api.routes import calls, health, memory, reviews
from sawti.config import get_settings


def create_app() -> FastAPI:
    """Construct the FastAPI application with its routers and CORS policy.

    CORS origins come from `SAWTI_CORS_ORIGINS` (empty by default: no
    browser origin may call the API until the 6.4 dashboard's is listed).

    Returns:
        The configured app.
    """
    settings = get_settings()
    app = FastAPI(title="Sawti", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "X-Reviewer-Id"],
    )
    for router in (calls.router, reviews.router, memory.router, health.router):
        app.include_router(router)
    return app


def main() -> None:
    """Run the API with uvicorn. Entry point for the `sawti` console script."""
    settings = get_settings()
    uvicorn.run(create_app(), host=settings.api_host, port=settings.api_port)
