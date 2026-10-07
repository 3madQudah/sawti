"""Tests for `sawti.db.checkpointer`: durable human-in-the-loop across a process restart.

Phase 6.1. The restart test is the stage's done-when: a run suspended for
human review in one process is resumed, and finished, by another.
"""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from operator import add
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Send

from sawti.agent.graph import NODE_ESCALATE, build_graph
from sawti.db.checkpointer import checkpointer_conninfo, postgres_checkpointer

FIRST_HALF = Path(__file__).with_name("_restart_first_half.py")


def test_checkpointer_conninfo_strips_the_sqlalchemy_driver() -> None:
    """psycopg gets a plain libpq URL for the same database, password intact."""
    url = "postgresql+psycopg://sawti:s3cret@db.internal:5433/sawti"
    assert checkpointer_conninfo(url) == "postgresql://sawti:s3cret@db.internal:5433/sawti"


async def test_setup_is_idempotent() -> None:
    """Opening the saver twice runs `setup()` twice without error."""
    for _ in range(2):
        async with postgres_checkpointer() as saver:
            assert await saver.aget_tuple({"configurable": {"thread_id": "never-used"}}) is None


async def test_interrupted_run_resumes_after_a_process_restart() -> None:
    """Interrupt in one process, resume in another: the run finishes as human-reviewed.

    First half (subprocess): a low-confidence call runs until `escalate`
    interrupts it, then the process exits — graph, saver and connection gone.
    Second half (here): a fresh saver on a new connection, a fresh graph, the
    same `thread_id`, and `Command(resume=...)`.
    """
    thread_id = f"restart-{uuid.uuid4()}"
    first = subprocess.run(
        [sys.executable, str(FIRST_HALF), thread_id], capture_output=True, text=True, timeout=120, check=False
    )
    assert first.returncode == 0, first.stderr
    suspended = json.loads(first.stdout.strip().splitlines()[-1])
    assert suspended["next"] == [NODE_ESCALATE]
    assert suspended["review_status"] == "awaiting_human"
    assert suspended["interrupts"][0]["call_id"] == thread_id

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    verdict = {"verdict": "the refund promise is real", "reviewer_id": "qa-1"}
    async with postgres_checkpointer() as saver:
        graph = build_graph(checkpointer=saver)
        restored = await graph.aget_state(config)
        assert restored.next == (NODE_ESCALATE,)

        resumed = await graph.ainvoke(Command(resume=verdict), config)
        finished = await graph.aget_state(config)

    assert resumed["review_status"] == "human_reviewed"
    assert resumed["reviewer_input"] == verdict
    assert finished.next == ()
    # Grounding state written before the restart survived it intact.
    assert resumed["grounding_coverage"] == 0.0
    assert len(resumed["rejected_claims"]) == 1


async def test_unknown_thread_has_nothing_to_resume() -> None:
    """Failure path: a thread id the checkpointer never saw has no suspended run."""
    async with postgres_checkpointer() as saver:
        snapshot = await build_graph(checkpointer=saver).aget_state(
            {"configurable": {"thread_id": f"missing-{uuid.uuid4()}"}}
        )
    assert snapshot.next == ()
    assert snapshot.values == {}


class _FanOut(TypedDict, total=False):
    items: list[str]
    done: Annotated[list[str], add]


async def _work(state: dict[str, str]) -> dict[str, list[str]]:
    return {"done": [state["item"]]}


def _fan_out_graph(saver: BaseCheckpointSaver[Any]) -> CompiledStateGraph:
    """START fans out one `Send` per item, pausing before the workers run."""
    graph = StateGraph(_FanOut)
    graph.add_node("work", _work)
    graph.add_conditional_edges(START, lambda s: [Send("work", {"item": i}) for i in s["items"]], ["work"])
    graph.add_edge("work", END)
    return graph.compile(checkpointer=saver, interrupt_before=["work"])


async def test_pending_sends_survive_a_postgres_round_trip() -> None:
    """Version-compatibility guard for the pinned `langgraph-checkpoint-postgres`.

    langgraph 0.2.x keeps `Send` packets in the checkpoint's `pending_sends`;
    checkpoint-postgres >= 2.0.22 targets langgraph >= 0.5, which moved them
    into a channel, and no longer returns `pending_sends` at all. On that
    version a run paused between a fan-out and its workers resumes with the
    work silently gone. The analysis graph does not use `Send` today; this
    test is what fails if a dependency bump reintroduces the mismatch.
    See docs/09-DECISIONS.md (2026-10-06, checkpointer version).
    """
    config: RunnableConfig = {"configurable": {"thread_id": f"sends-{uuid.uuid4()}"}}
    async with postgres_checkpointer() as saver:
        await _fan_out_graph(saver).ainvoke({"items": ["a", "b"]}, config)

    async with postgres_checkpointer() as saver:
        resumed = await _fan_out_graph(saver).ainvoke(None, config)

    assert sorted(resumed["done"]) == ["a", "b"]
