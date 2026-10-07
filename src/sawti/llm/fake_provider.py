"""Deterministic, offline `LLMProvider` — for the service e2e and load measurement only.

Phase 6.2. Selected with `SAWTI_LLM_PROVIDER=fake`. Not a model: it reads the
transcript it is given and proposes claims mechanically, so a containerized
run is reproducible, spends no provider quota, and measures this system's
queueing and processing rather than a free-tier rate limit. Numbers measured
with it are labelled as such in `eval_results.md`.

It also answers the memory loop's prompts: induction / consolidation get a
rule quoting their (flattened) input — so a rule carries the corrected
field's words and similarity retrieval has something to match — and the
conflict check never finds a contradiction.

What it proposes, for an `ExtractionProposal`-shaped `response_model`:

  * up to two commitments quoting `Agent:` lines verbatim (they ground);
  * `ungrounded_claims` commitments whose quotes appear nowhere (they don't);
  * one sentiment point per `Customer:` line, up to three.

With the defaults (two grounded, one ungrounded) coverage is 2/3, below the
0.7 threshold, so every call escalates — the path the e2e exercises.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from sawti.llm.provider import LLMProvider, ResponseModelT
from sawti.llm.usage import record_usage

_LINE = re.compile(r"^(Agent|Customer):[ \t]*(.+?)[ \t]*$", re.MULTILINE)


class FakeProvider(LLMProvider):
    """Deterministic stand-in for a real provider. See the module docstring."""

    def __init__(self, *, latency_seconds: float = 0.0, ungrounded_claims: int = 1) -> None:
        """
        Args:
            latency_seconds: Simulated model latency per call.
            ungrounded_claims: How many unsupported commitments to propose.
        """
        self._latency = latency_seconds
        self._ungrounded = ungrounded_claims

    async def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        """Return a fixed completion after the simulated latency."""
        await asyncio.sleep(self._latency)
        return "fake completion"

    async def structured_complete(
        self,
        prompt: str,
        *,
        response_model: type[ResponseModelT],
        system: str | None = None,
        **kwargs: Any,
    ) -> ResponseModelT:
        """Answer mechanically from `prompt`. See the module docstring."""
        await asyncio.sleep(self._latency)
        fields = response_model.model_fields
        if "contradicts" in fields:  # sawti.memory.conflict: never a contradiction
            self._report(prompt, "false")
            return response_model.model_validate({"contradicts": False, "reasoning": "fake provider"})
        if "rule_text" in fields:  # sawti.memory.induction / consolidate: a rule quoting its input
            flat = " ".join(prompt.split()) or "a correction"
            self._report(prompt, flat[:400])
            return response_model.model_validate({"rule_text": f"Fake rule learned from: {flat[:400]}"})
        commitments: list[dict[str, Any]] = []
        points: list[dict[str, Any]] = []
        for match in _LINE.finditer(prompt):
            speaker, text = match.group(1), match.group(2)
            quote = {"text": text, "speaker": speaker, "start_char": match.start(2), "end_char": match.end(2)}
            if speaker == "Agent" and len(commitments) < 2:
                commitments.append({"evidence": quote, "promised_by": "Agent", "description": "Follow up."})
            elif speaker == "Customer" and len(points) < 3:
                points.append(
                    {"quote": quote, "speaker": speaker, "score": 0.0, "timestamp_sec": 8.0 * len(points)}
                )
        for index in range(self._ungrounded):
            text = f"(fake provider: unsupported claim {index + 1})"
            commitments.append(
                {
                    "evidence": {"text": text, "speaker": "Agent", "start_char": 0, "end_char": len(text)},
                    "promised_by": "Agent",
                    "description": "An unsupported promise.",
                }
            )
        result = response_model.model_validate(
            {
                "summary": "Deterministic fake analysis (no model was called).",
                "commitments": commitments,
                "sentiment_trajectory": {"points": points},
            }
        )
        self._report(prompt, result.model_dump_json())
        return result

    @staticmethod
    def _report(prompt: str, output: str) -> None:
        """Usage for tracing: whitespace-separated *word* counts — there is no model, so no tokens."""
        record_usage(len(prompt.split()), len(output.split()), model="fake")
