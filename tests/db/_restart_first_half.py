"""First half of the restart test: run a low-confidence call until it interrupts, then exit.

Run as its own process by `tests/db/test_checkpointer.py`, so the graph, its
saver and its connection all die with the process — a real process boundary,
not just a fresh object. Prints one JSON line describing the suspended run.

Usage: python _restart_first_half.py <thread_id>
(`DATABASE_URL` comes from the environment, which the test sets.)
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from langchain_core.runnables import RunnableConfig

from sawti.agent.graph import build_graph
from sawti.agent.nodes import extract as extract_module
from sawti.db.checkpointer import postgres_checkpointer
from sawti.schemas import Commitment, ExtractionProposal, Language, Quote

TRANSCRIPT = "Agent: I will refund you by Sunday.\nCustomer: Thank you.\n"


class _UngroundedProvider:
    """Proposes one paraphrased (unsupported) claim, so coverage is 0.0 and the call escalates."""

    async def structured_complete(self, prompt: str, **kwargs: object) -> ExtractionProposal:
        text = "I will refund you today."
        quote = Quote(text=text, speaker="Agent", start_char=0, end_char=len(text))
        return ExtractionProposal(
            summary="Refund promised.",
            commitments=[Commitment(evidence=quote, promised_by="Agent", description="Refund.")],
        )


async def _run(thread_id: str) -> dict[str, Any]:
    extract_module.get_llm_provider = lambda: _UngroundedProvider()  # type: ignore[assignment]
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    async with postgres_checkpointer() as saver:
        graph = build_graph(checkpointer=saver)
        await graph.ainvoke(
            {"call_id": thread_id, "redacted_transcript": TRANSCRIPT, "language": Language.EN}, config
        )
        snapshot = await graph.aget_state(config)
    return {
        "next": list(snapshot.next),
        "review_status": snapshot.values.get("review_status"),
        "interrupts": [i.value for task in snapshot.tasks for i in task.interrupts],
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(_run(sys.argv[1]))))
