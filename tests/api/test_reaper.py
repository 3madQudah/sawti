"""Tests for `sawti.api.tasks.reap_stuck_calls` (phase 6.2).

Calls are inserted directly in the state under test, with timestamps in the
past; eager Celery means a re-enqueued call is analyzed inside the reaper's
own `.delay()`, so its end state shows the recovery worked.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from sawti.api import tasks
from sawti.db.models import Agent, Call
from sawti.db.session import get_session
from sawti.schemas import CallStatus

TRANSCRIPT = "Agent: I will send a technician tomorrow.\nCustomer: Thanks.\n"


@pytest.fixture
def stuck_call() -> Callable[..., str]:
    def _make(status: CallStatus, *, minutes_ago: float, attempts: int = 0) -> str:
        then = datetime.now(UTC) - timedelta(minutes=minutes_ago)
        call_id = uuid.uuid4()
        with get_session() as session:
            agent = Agent(id=uuid.uuid4(), name="a", created_at=then)
            session.add(agent)
            session.flush()
            session.add(
                Call(
                    id=call_id,
                    agent_id=agent.id,
                    redacted_transcript=TRANSCRIPT,
                    language="en",
                    occurred_at=then,
                    status=status,
                    queued_at=then,
                    enqueued_at=then,
                    started_at=then if status == CallStatus.PROCESSING else None,
                    attempts=attempts,
                )
            )
        return str(call_id)

    return _make


def _status(call_id: str) -> CallStatus:
    with get_session() as session:
        call = session.get(Call, uuid.UUID(call_id))
        assert call is not None
        return call.status


def test_a_call_whose_message_was_lost_is_re_enqueued_and_analyzed(
    stuck_call: Callable[..., str], llm: Any
) -> None:
    call_id = stuck_call(CallStatus.QUEUED, minutes_ago=30)
    assert tasks.reap_stuck_calls.apply().get() == {"requeued": 1, "recovered": 0, "failed": 0}
    assert _status(call_id) == CallStatus.AWAITING_REVIEW


def test_a_recent_backlog_is_left_alone(stuck_call: Callable[..., str], llm: Any) -> None:
    """Queued for 5 minutes is a backlog, not a lost message: no duplicate enqueue."""
    call_id = stuck_call(CallStatus.QUEUED, minutes_ago=5)
    assert tasks.reap_stuck_calls.apply().get() == {"requeued": 0, "recovered": 0, "failed": 0}
    assert _status(call_id) == CallStatus.QUEUED
    assert llm.calls == 0


def test_a_call_stuck_processing_is_recovered(stuck_call: Callable[..., str], llm: Any) -> None:
    """The worker died mid-task: back to queued, re-enqueued, finished by the next claim."""
    call_id = stuck_call(CallStatus.PROCESSING, minutes_ago=30, attempts=1)
    assert tasks.reap_stuck_calls.apply().get() == {"requeued": 0, "recovered": 1, "failed": 0}
    assert _status(call_id) == CallStatus.AWAITING_REVIEW
    with get_session() as session:
        assert session.get(Call, uuid.UUID(call_id)).attempts == 2  # type: ignore[union-attr]


def test_a_call_that_keeps_dying_is_failed_not_retried_forever(
    stuck_call: Callable[..., str], llm: Any
) -> None:
    """max_retries=2 in the fixture: the third stalled attempt is the last."""
    call_id = stuck_call(CallStatus.PROCESSING, minutes_ago=30, attempts=3)
    assert tasks.reap_stuck_calls.apply().get() == {"requeued": 0, "recovered": 0, "failed": 1}
    assert _status(call_id) == CallStatus.FAILED
    assert llm.calls == 0


def test_a_slow_but_alive_call_is_not_touched(stuck_call: Callable[..., str], llm: Any) -> None:
    call_id = stuck_call(CallStatus.PROCESSING, minutes_ago=2, attempts=1)
    assert tasks.reap_stuck_calls.apply().get() == {"requeued": 0, "recovered": 0, "failed": 0}
    assert _status(call_id) == CallStatus.PROCESSING


def test_a_re_enqueued_duplicate_does_the_work_once(stuck_call: Callable[..., str], llm: Any) -> None:
    """If the original message was only late, the reaper's copy and it cannot both analyze."""
    call_id = stuck_call(CallStatus.QUEUED, minutes_ago=30)
    tasks.reap_stuck_calls.apply().get()
    assert tasks.analyze_call_task.apply(args=[call_id]).get() == "skipped"
    assert llm.calls == 1


def test_beat_schedules_the_reaper() -> None:
    entry = tasks.celery_app.conf.beat_schedule["reap-stuck-calls"]
    assert entry["task"] == "sawti.reap_stuck_calls"
