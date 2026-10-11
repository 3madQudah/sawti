"""`GET /results`: the 6.3 numbers for the dashboard's results page, served as recorded.

Phase 6.4. See `sawti.api.results`: nothing is recomputed, and every number
comes with its method, n and hardware.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from sawti.api.deps import CurrentReviewer
from sawti.api.results import ResultsUnavailableError, load_results
from sawti.api.schemas import ResultsResponse

router = APIRouter(prefix="/results", tags=["results"])


@router.get("", response_model=ResultsResponse)
def results(_: CurrentReviewer) -> ResultsResponse:
    """The cloud vs self-hosted table and the memory comparison. 503 if a results file is missing."""
    try:
        return load_results()
    except ResultsUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
