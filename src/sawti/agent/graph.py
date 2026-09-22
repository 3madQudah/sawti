"""Assembly of the LangGraph analysis graph: node wiring and conditional edges.

Phase 2, stage 1: the skeleton. Every node is wired in its final position, but
each one is a no-op placeholder, so the graph runs end to end *before* any real
logic exists. The point is to get the topology — and the one conditional edge
that carries the project's human-review rule — under test first, then replace
placeholders one at a time in later stages without touching this file's shape.

Why a fixed, deterministic pipeline rather than a ReAct-style agent choosing its
own tools: the grounding check must run on every claim, every time. A model that
can decide whether to call a verification step can also decide to skip it, which
turns a hard invariant ("an unsupported claim is rejected") into a suggestion.
The ordering here is a control, not a performance optimisation.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from itertools import pairwise
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from sawti.agent.nodes.compliance import compliance
from sawti.agent.nodes.confidence import confidence
from sawti.agent.nodes.escalate import escalate
from sawti.agent.nodes.extract import extract
from sawti.agent.nodes.ground import ground
from sawti.agent.nodes.score import score
from sawti.agent.state import AgentState

# Node names are constants because two places have to agree on them: the
# `add_node` calls and the conditional-edge mapping. A typo in a string literal
# in the mapping is a silent routing bug, not an import error.
NODE_EXTRACT = "extract"
NODE_GROUND = "ground"
NODE_SCORE = "score"
NODE_COMPLIANCE = "compliance"
# Named "assess_confidence", not "confidence": LangGraph forbids a node whose
# name collides with a state key, and `confidence` is the key this node writes.
# The node got renamed rather than the key, so the key keeps matching the
# `CallAnalysis.confidence` field it ends up in.
NODE_CONFIDENCE = "assess_confidence"
NODE_ESCALATE = "escalate"

# The linear spine of the graph, in execution order. Ordering is load-bearing:
#   extract    — proposes claims (may hallucinate quotes).
#   ground     — deletes any claim not verbatim in the transcript. Must come
#                *before* anything reads the claims, so no downstream node can
#                ever see an unsupported claim.
#   score      — QA rubric, over grounded claims only.
#   compliance — rule checks, over grounded claims only.
#   assess_confidence — aggregates the above into the routing decision.
PIPELINE: tuple[str, ...] = (
    NODE_EXTRACT,
    NODE_GROUND,
    NODE_SCORE,
    NODE_COMPLIANCE,
    NODE_CONFIDENCE,
)

# A node is an async callable returning a *partial* state update. Returning an
# empty dict leaves state untouched, which is exactly what a placeholder wants.
NodeFn = Callable[[AgentState], Awaitable[dict[str, Any]]]

# Real implementations, keyed by node name. A node absent from this mapping is
# still wired into the topology but runs as a `_passthrough` — which is how
# stages land one at a time without the graph's shape ever changing.
NODE_IMPLEMENTATIONS: dict[str, NodeFn] = {
    NODE_EXTRACT: extract,
    NODE_GROUND: ground,
    NODE_SCORE: score,
    NODE_COMPLIANCE: compliance,
    NODE_CONFIDENCE: confidence,
    NODE_ESCALATE: escalate,
}


def _passthrough(node_name: str) -> NodeFn:
    """Build a no-op stand-in for a node whose real implementation lands later.

    Role in the graph: occupies a real position in the topology so the pipeline
    is executable and testable now. It returns an empty state update, so running
    the graph is observably equivalent to not running it — a placeholder that
    quietly wrote defaults would make stage 1 tests pass for the wrong reason.

    Args:
        node_name: The graph node this placeholder stands in for, used only to
            name the function so traces and graph dumps stay readable.

    Returns:
        An async node function that accepts state and changes nothing.
    """

    async def _node(state: AgentState) -> dict[str, Any]:
        """Placeholder node body: accept the state, change nothing, move on."""
        return {}

    _node.__name__ = f"{node_name}_placeholder"
    return _node


def route_after_confidence(state: AgentState) -> str:
    """Conditional edge: end the graph with a verdict, or escalate to a human.

    Role in the graph: the single branch point, and the place where "low
    confidence routes to human review, never an automated verdict" is enforced
    as topology rather than as a check some caller might forget.

    The default is deliberately the *safe* one. `requires_human_review` is read
    with a default of True, so a run where the confidence node never executed,
    or crashed before writing its keys, escalates instead of silently taking the
    auto-pass branch. Missing information is treated as low confidence.

    Args:
        state: Current agent state, read after the confidence node has run.

    Returns:
        `NODE_ESCALATE` when the call needs a human, otherwise `END`.
    """
    if state.get("error") is not None:
        return NODE_ESCALATE
    if state.get("requires_human_review", True):
        return NODE_ESCALATE
    return END


def build_graph(checkpointer: BaseCheckpointSaver[Any] | None = None) -> CompiledStateGraph:
    """Construct and compile the LangGraph analysis graph.

    Role in the graph: this *is* the graph — the only place node wiring lives.

    Wires START -> extract -> ground -> score -> compliance -> assess_confidence,
    then a conditional edge from there to either END (auto-pass) or escalate
    (human review). Escalate terminates at END too: after a reviewer resumes the
    run in stage 4, the graph finishes rather than looping back through scoring,
    because a human's verdict is final and must not be re-scored by the model.

    Nodes present in `NODE_IMPLEMENTATIONS` run their real bodies; any node not
    yet implemented runs as a `_passthrough`, so the graph stays executable
    end to end at every stage and the topology below never changes.

    A checkpointer is not optional in practice. `escalate` suspends the run with
    `interrupt()`, and without a checkpointer LangGraph has nowhere to persist the
    paused state — the interrupt silently does nothing and the run continues past
    the point a human was supposed to take over. Defaulting to `MemorySaver()`
    makes the escalation branch real out of the box.

    `MemorySaver` is process-local: a resume only works inside the same process,
    and everything is lost on restart. That is the correct trade for phase 2,
    where the graph is driven by tests and the eval harness. Durable review
    queues need a Postgres-backed saver, which depends on `sawti.db.session` —
    still an unimplemented phase 1 stub — and is deferred to phase 5 deployment.
    See `docs/09-DECISIONS.md`.

    Args:
        checkpointer: Where interrupted runs are persisted. Defaults to an
            in-process `MemorySaver`. Pass an explicit saver to share state
            across processes, or to isolate threads between tests.

    Returns:
        The compiled graph, ready to `.ainvoke()` or `.astream()`. Because it is
        checkpointed, every call must carry a `configurable.thread_id`.
    """
    graph: StateGraph = StateGraph(AgentState)

    for node_name in (*PIPELINE, NODE_ESCALATE):
        graph.add_node(node_name, NODE_IMPLEMENTATIONS.get(node_name) or _passthrough(node_name))

    # The linear spine: START into the first node, then each node to the next.
    graph.add_edge(START, PIPELINE[0])
    for source, target in pairwise(PIPELINE):
        graph.add_edge(source, target)

    # The one branch. The explicit mapping (rather than relying on the router's
    # return value naming a node directly) keeps every reachable destination
    # visible here, so the set of possible exits is readable in one place.
    graph.add_conditional_edges(
        NODE_CONFIDENCE,
        route_after_confidence,
        {NODE_ESCALATE: NODE_ESCALATE, END: END},
    )
    graph.add_edge(NODE_ESCALATE, END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
