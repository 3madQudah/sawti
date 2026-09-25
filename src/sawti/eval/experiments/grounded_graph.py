"""Phase 2 experiment: the grounded LangGraph agent, scored against the phase 1 set.

Phase 2: grounding experiment.

Same 150 reference records, same metrics, same runner as
`sawti.eval.experiments.baseline`. The *only* difference is the extractor: this
one drives `sawti.agent.graph`, so every claim passes the grounding gate before
it is emitted, instead of being returned exactly as the model produced it.

One honest caveat about how to read the resulting numbers. The unsupported-claim
rate for this experiment is near-zero *by construction* — `ground` drops claims
whose quotes do not verify, so unsupported claims cannot reach the output. That
is the intended behaviour and it is a real improvement in what downstream
consumers receive, but it is not evidence that the model hallucinates less.
`GateStats` exists so the two things can be reported separately: how often the
model proposed something unsupported, and how much of that reached the output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from langchain_core.runnables import RunnableConfig

from sawti.agent.graph import build_graph
from sawti.agent.state import AgentState
from sawti.data.ground_truth import infer_call_id_and_language
from sawti.eval.metrics import DEFAULT_SYNTHETIC_DIR
from sawti.eval.runner import run_eval
from sawti.llm.provider import DEFAULT_REQUEST_TIMEOUT_SECONDS, run_with_timeout
from sawti.schemas import CallAnalysis, Language, SentimentTrajectory

logger = logging.getLogger(__name__)

DEFAULT_GROUND_TRUTH_DIR = Path("data/ground_truth")
DEFAULT_OUTPUT_PATH = Path("eval_results.md")
EXPERIMENT_NAME = "phase 2 grounded agent (LangGraph, grounding gate, no memory)"


@dataclass
class GateStats:
    """Running tally of what the grounding gate saw, so its effect can be reported.

    Role: this is the number that says whether the *model* improved, as opposed
    to whether the *pipeline* filtered better. Without it, a near-zero
    unsupported-claim rate could be mistaken for the former.
    """

    proposed: int = 0
    rejected: int = 0
    escalated: int = 0
    calls: int = 0
    rejected_by_language: dict[Language, int] = field(default_factory=dict)
    proposed_by_language: dict[Language, int] = field(default_factory=dict)

    def record(self, language: Language, *, proposed: int, rejected: int, escalated: bool) -> None:
        """Fold one call's gate outcome into the tally."""
        self.calls += 1
        self.proposed += proposed
        self.rejected += rejected
        self.escalated += int(escalated)
        self.proposed_by_language[language] = self.proposed_by_language.get(language, 0) + proposed
        self.rejected_by_language[language] = self.rejected_by_language.get(language, 0) + rejected

    def rejection_rate(self, language: Language) -> float:
        """Fraction of this language's proposed claims that failed grounding."""
        proposed = self.proposed_by_language.get(language, 0)
        return self.rejected_by_language.get(language, 0) / proposed if proposed else 0.0


def _to_call_analysis(state: AgentState, call_id: str, language: Language) -> CallAnalysis:
    """Assemble the graph's final state into the validated output contract.

    Args:
        state: Final (or suspended) graph state.
        call_id: Call identifier, from the transcript filename.
        language: Language category, from the transcript filename.

    Returns:
        A validated `CallAnalysis`.

    Raises:
        RuntimeError: If the run recorded an error or produced no summary. This
            surfaces to `run_eval` as an extraction failure and is excluded from
            the metrics, rather than being padded into a fake result.
    """
    error = state.get("error")
    if error is not None:
        raise RuntimeError(f"graph run failed: {error}")
    summary = state.get("summary")
    if not summary:
        raise RuntimeError("graph run produced no summary")

    return CallAnalysis(
        call_id=call_id,
        language=language,
        summary=summary,
        commitments=list(state.get("commitments", [])),
        compliance_flags=list(state.get("compliance_flags", [])),
        rubric_scores=list(state.get("rubric_scores", [])),
        sentiment_trajectory=state.get("sentiment_trajectory") or SentimentTrajectory(),
        confidence=state.get("confidence", 0.0),
        requires_human_review=state.get("requires_human_review", True),
    )


def extract_via_graph(
    transcript_path: Path,
    *,
    stats: GateStats | None = None,
    retrieved_rules: list[str] | None = None,
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> CallAnalysis:
    """Run the phase 2 agent graph over one transcript and return its analysis.

    Role: the `Extractor` handed to `run_eval`, mirroring `run_plain_extraction`'s
    signature so the two experiments differ by exactly one callable. Also the
    extractor `sawti.eval.experiments.five_batch` uses for both its arms —
    `retrieved_rules` is what tells them apart: omitted (or empty) for the
    no-memory control, populated from `sawti.memory.store.MemoryStore.retrieve()`
    for the memory-on arm. Neither arm needed a second code path here.

    A run that escalates is *not* resumed — there is no human here. The suspended
    state already holds everything the metrics need, and its
    `requires_human_review=True` is itself part of what the eval reports.

    The whole graph run is bounded by `timeout` (`run_with_timeout`, not a
    bare `asyncio.run`) — found necessary the hard way running
    `sawti.eval.experiments.five_batch`'s real experiment overnight: the
    host machine slept mid-run, and the in-flight provider call, with
    nothing bounding it, hung indefinitely on wake rather than failing into
    that module's own retry loop. `extract()`'s own LLM call has no timeout
    of its own; this one wraps the entire graph invocation instead, so a
    hang anywhere in the pipeline — not just in `extract` — surfaces as a
    `TimeoutError` a retrying caller can catch.

    Args:
        transcript_path: Path to a `call_<idx>_<lang>.txt` transcript.
        stats: Optional tally to fold this call's grounding-gate outcome into.
        retrieved_rules: Rule text to fold into `extract`'s prompt via
            `AgentState["retrieved_rules"]` (see that node). Omitted or
            empty runs exactly as phase 2 did — no memory involved.
        timeout: Seconds to allow the whole graph run before raising
            `TimeoutError`. Defaults to the same
            `sawti.llm.provider.DEFAULT_REQUEST_TIMEOUT_SECONDS` every
            provider call in this project is already bounded by.

    Returns:
        The agent's analysis as a validated `CallAnalysis`.
    """
    call_id, language = infer_call_id_and_language(transcript_path)
    transcript = transcript_path.read_text(encoding="utf-8")

    graph = build_graph()
    # Each call is its own thread: runs must not share checkpointed state.
    config: RunnableConfig = {"configurable": {"thread_id": f"eval-{call_id}"}}
    initial_state: dict[str, Any] = {"call_id": call_id, "transcript": transcript, "language": language}
    if retrieved_rules:
        initial_state["retrieved_rules"] = retrieved_rules
    state = cast(AgentState, run_with_timeout(graph.ainvoke(initial_state, config), timeout=timeout))

    if stats is not None:
        rejected = len(state.get("rejected_claims", []))
        emitted = (
            len(state.get("commitments", []))
            + len(state.get("compliance_flags", []))
            + len(state.get("rubric_scores", []))
        )
        stats.record(
            language,
            proposed=emitted + rejected,
            rejected=rejected,
            escalated=bool(state.get("requires_human_review", True)),
        )

    return _to_call_analysis(state, call_id, language)


def run_grounded_graph(
    ground_truth_path: Path = DEFAULT_GROUND_TRUTH_DIR,
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    synthetic_dir: Path = DEFAULT_SYNTHETIC_DIR,
    sleep_seconds: float = 4.0,
    stats: GateStats | None = None,
) -> dict[str, dict[Language, float]]:
    """Run the phase 2 evaluation over the same reference set as the phase 1 baseline.

    Args:
        ground_truth_path: Directory of reference-label records.
        output_path: Where to append the results table.
        synthetic_dir: Directory of `<call_id>.txt` transcripts.
        sleep_seconds: Delay between calls, to respect free-tier rate limits.
        stats: Optional grounding-gate tally, filled in as the run proceeds.

    Returns:
        A mapping from metric name to its per-`Language` category scores.
    """
    return run_eval(
        ground_truth_path,
        output_path=output_path,
        synthetic_dir=synthetic_dir,
        sleep_seconds=sleep_seconds,
        extractor=lambda path: extract_via_graph(path, stats=stats),
        experiment_name=EXPERIMENT_NAME,
    )
