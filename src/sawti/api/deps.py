"""FastAPI dependencies: database sessions, the shared API key, and reviewer identity.

Phase 6.2. **`get_current_reviewer()` is the only place reviewer identity is
read.** No route touches the `X-Reviewer-Id` / `X-API-Key` headers itself, so
replacing this with OIDC in 6.4 means changing this module and nothing else.

Stated plainly (docs/09-DECISIONS.md, 2026-10-06, decision 3): this is **not
authentication**. The reviewer id is self-asserted, and the shared key will be
visible in the dashboard's browser bundle once one exists. It keeps casual
callers out of a local deployment and gives `ReviewerAction.reviewer_id` a
value. Nothing more.
"""

from __future__ import annotations

import hmac
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from sawti.config import get_settings
from sawti.db.session import get_session_factory


class Reviewer(BaseModel):
    """Who is acting, as far as this placeholder scheme can tell."""

    reviewer_id: str


def get_db() -> Iterator[Session]:
    """A session per request. Routes commit explicitly; anything uncommitted is rolled back on close."""
    with get_session_factory()() as session:
        yield session


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Check the shared `X-API-Key` against `SAWTI_API_KEY`.

    Fails closed: if no key is configured (unset or empty), every protected
    route answers 503 rather than becoming open.
    """
    configured = get_settings().api_key
    # An empty key counts as unset: otherwise an empty header would match it.
    expected = configured.get_secret_value() if configured is not None else ""
    if not expected:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "SAWTI_API_KEY is not configured")
    if x_api_key is None or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or invalid API key")


def get_current_reviewer(
    _: Annotated[None, Depends(require_api_key)],
    x_reviewer_id: Annotated[str | None, Header()] = None,
) -> Reviewer:
    """The acting reviewer: a valid API key plus a non-blank `X-Reviewer-Id`."""
    reviewer_id = (x_reviewer_id or "").strip()
    if not reviewer_id or len(reviewer_id) > 255:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or invalid X-Reviewer-Id")
    return Reviewer(reviewer_id=reviewer_id)


DbSession = Annotated[Session, Depends(get_db)]
CurrentReviewer = Annotated[Reviewer, Depends(get_current_reviewer)]
ApiKey = Annotated[None, Depends(require_api_key)]
