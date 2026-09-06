"""FastAPI application entry point.

Phase 3: API layer.
"""

from __future__ import annotations

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application.

    Returns:
        The configured `FastAPI` app, with routers mounted.

    Raises:
        NotImplementedError: Until routers and middleware are wired up.
    """
    # TODO(phase-3): include_router for calls/reviews/health, add exception handlers,
    # and Langfuse tracing middleware.
    raise NotImplementedError


def main() -> None:
    """Run the API server via uvicorn. Entry point for the `sawti` console script.

    Raises:
        NotImplementedError: Until the server bootstrap is implemented.
    """
    # TODO(phase-3): uvicorn.run(create_app(), host=..., port=...) using sawti.config.get_settings().
    raise NotImplementedError
