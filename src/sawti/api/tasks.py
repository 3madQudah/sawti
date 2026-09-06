"""Celery task definitions for async call analysis.

Phase 3: async processing layer.
"""

from __future__ import annotations

from celery import Celery

from sawti.config import get_settings

_settings = get_settings()

celery_app = Celery(
    "sawti",
    broker=_settings.celery_broker_url,
    backend=_settings.celery_result_backend,
)


@celery_app.task(name="sawti.analyze_call")
def analyze_call_task(call_id: str) -> None:
    """Run the full analysis graph for `call_id` asynchronously.

    Args:
        call_id: Identifier of the call to analyze.

    Raises:
        NotImplementedError: Until the task body is implemented.
    """
    # TODO(phase-3): load the call, redact PII, run sawti.agent.graph, persist CallAnalysisRecord.
    raise NotImplementedError
