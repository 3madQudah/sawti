"""Compliance rule-checking node — a documented passthrough, and why.

Phase 2, stage 2.

Unlike `score`, this node has no canonical set to reconcile against, so it does
nothing. That is a decision, not an omission, and it is recorded here because a
silent no-op node is otherwise indistinguishable from an unfinished one.

The stub this replaces asked for two things:

  * "never emit a ComplianceFlag without a grounded evidence quote" — already
    guaranteed twice over, by `Claim.evidence` at the schema level and by
    `ground` running upstream of this node. There is nothing left to enforce.
  * "assign the severity defined for each violated rule_id" — this needs a rule
    registry mapping rule_id to severity. No such registry exists. `rule_id` is
    free-form: `sawti.data.ground_truth`'s analyst prompt tells the model to
    "choose a short snake_case rule_id", and the eval layer already made the
    call that follows from that — `sawti.eval.metrics.accuracy_by_category`
    scores compliance as a binary "did both sides find any violation at all",
    explaining that "rule_id vocabularies are model-invented and not comparable,
    so matching individual flags would measure naming, not detection".

Inventing a registry here would contradict a decision already taken in an
implemented, tested module, and would drop every flag whose model-chosen id
missed our spelling — lowering measured accuracy while detecting nothing new.
The honest move is to leave compliance flags as extracted-and-grounded, and to
revisit when a real rule catalogue exists to check against. See
`docs/09-DECISIONS.md`.
"""

from __future__ import annotations

from typing import Any

from sawti.agent.state import AgentState


async def compliance(state: AgentState) -> dict[str, Any]:
    """Check the call against compliance rules, producing grounded `ComplianceFlag` entries.

    Role in the graph: currently a no-op placeholder held open for a future rule
    catalogue. Flags arrive already extracted by `extract` and already
    evidence-verified by `ground`; with no canonical rule set to validate
    `rule_id` or normalize `severity` against, there is no check this node can
    perform that is not already performed upstream.

    It stays in `PIPELINE` rather than being deleted so that adding a rule
    catalogue later is a change to one function body, not a change to the
    graph's topology.

    Args:
        state: Current agent state, containing grounded claims and the transcript.

    Returns:
        An empty state update — compliance flags pass through untouched.
    """
    return {}
