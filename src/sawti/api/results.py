"""The 6.3 results, read from the files they were recorded in. Nothing is recomputed.

Phase 6.4, for the dashboard's results page (`GET /results`). Sources:

* `data/serving_benchmark/comparison.json`: the cloud vs self-hosted table,
  written by `scripts/assemble_serving_comparison.py`;
* `data/memory_comparison/report.json`: the memory comparison's report,
  written once by `scripts/run_memory_comparison.py report --out` (zero LLM
  calls; the pre-registered verdict code).

Values are served as stored. `eval_results.md` (2026-10-10) shows the same
numbers rounded for display, and `tests/api/routes/test_results.py` ties
each one to that table. The method text for each column is the one stated
in `eval_results.md`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sawti.api.schemas import (
    MemoryLanguageResult,
    MemoryResults,
    ResultsResponse,
    ServingArmResult,
    ServingLanguageResult,
    ServingResults,
    ServingThroughput,
)
from sawti.schemas import Language

SERVING_PATH = Path("data/serving_benchmark/comparison.json")
MEMORY_PATH = Path("data/memory_comparison/report.json")

#: The arm that ran in the cloud; every other arm in `comparison.json` is self-hosted.
CLOUD_ARM = "gemini"

#: Per-column methods, as eval_results.md (2026-10-10, cloud vs self-hosted) states them.
SERVING_METHODS = {
    "n": (
        "calls in the cell: 15 per language; the vLLM concurrency-1 arm the first 5 per language; "
        "HF 1 per language"
    ),
    "accuracy": (
        "against Gemini-generated reference labels (data/ground_truth), schema-valid predictions "
        "only; not a fair cross-provider comparison"
    ),
    "rubric_agreement": "against Gemini-generated reference labels, schema-valid predictions only",
    "grounding_precision": "claims whose quote is verbatim in the redacted transcript; provider-neutral",
    "unsupported_claim_rate": "claims not supported by the redacted transcript; provider-neutral",
    "schema_valid_rate": (
        "predictions that validated against CallAnalysis; failures count here, not in accuracy"
    ),
    "latency": "end to end per call; concurrency-16 arms are under load (16 calls sharing the GPUs)",
    "throughput": "per arm, on its whole mixed-language run; method per arm",
    "cost_per_1000_usd": (
        "priced as if paid (actual spend $0): Gemini at gemini-3.5-flash-lite's paid price x mean "
        "tokens; self-hosted at $0.845/h for 2xT4 x GPU time per 1000 calls, allocated by output tokens"
    ),
    "third_party_transcript_bytes_per_call": (
        "UTF-8 bytes of transcript text per call; the Kaggle arms uploaded the transcripts to "
        "Kaggle (Google) once, so this run did send them to a third party"
    ),
}

MEMORY_METHOD = (
    "memory off vs on (top-5 of 35 frozen complete-linkage rules, leave-one-out), all 150 calls, "
    "two repeats; accuracy paired over calls valid in both arms; effect = mean delta; noise = the "
    "larger of the off arm's change between repeats and the delta's change between repeats; "
    "verdict vs Phase 4's mean per-batch delta (rule pre-registered 2026-10-10)"
)
MEMORY_PROVIDER = "Gemini API, gemini-flash-lite-latest (resolved to gemini-3.5-flash-lite on all 600 calls)"
MEMORY_HARDWARE = "Google-hosted (Gemini API; 528 calls free tier, 72 paid tier)"

_ARMS = ("memory_off", "memory_on_35_loo")


class ResultsUnavailableError(FileNotFoundError):
    """A results file is missing (e.g. not mounted into the API container)."""


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ResultsUnavailableError(f"{path} is not available")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _serving_cell(arm: dict[str, Any], language: str) -> ServingLanguageResult:
    cell = arm["per_language"][language]
    fields = {k: v for k, v in cell.items() if k in ServingLanguageResult.model_fields}
    # Only Gemini records `latency_n` (one call excluded); elsewhere latency covers all n calls.
    fields.setdefault("latency_n", cell["n"])
    return ServingLanguageResult(**fields, failures=arm["failures_per_language"][language])


def serving_results(path: Path | None = None) -> ServingResults:
    """The comparison table, per arm and language, as `comparison.json` holds it."""
    path = path or SERVING_PATH
    data = _read(path)
    arms = [
        ServingArmResult(
            id=arm_id,
            label=arm["label"],
            hardware=arm["hardware"],
            models=arm["models"],
            deployment="cloud" if arm_id == CLOUD_ARM else "self_hosted",
            throughput=ServingThroughput(
                **{k: v for k, v in arm["throughput"].items() if k in ServingThroughput.model_fields}
            ),
            **{language.value: _serving_cell(arm, language.value) for language in Language},
        )
        for arm_id, arm in data["arms"].items()
    ]
    pricing = data["pricing"]
    return ServingResults(
        source=str(path),
        methods=SERVING_METHODS,
        pricing={k: v for k, v in pricing.items() if isinstance(v, str)},
        actual_spend_usd=pricing["actual_spend_usd"],
        arms=arms,
    )


def _verdict_code(verdict: str) -> str:
    """The report's verdict text as a stable code: confirms, contradicts, inconclusive, incomplete."""
    return verdict.split(" ", 1)[0]


def memory_results(path: Path | None = None) -> MemoryResults:
    """The memory comparison's per-language result and verdict, as the report holds it."""
    path = path or MEMORY_PATH
    data = _read(path)
    repeats = sorted(data["by_repeat"])
    languages = {}
    for language in Language:
        verdict = data["verdict"][language.value]
        per_arm = {
            arm: [data["by_repeat"][r][arm][language.value]["accuracy"] for r in repeats] for arm in _ARMS
        }
        languages[language.value] = MemoryLanguageResult(
            n_paired_per_repeat=verdict["n_paired_per_repeat"],
            accuracy_off_per_repeat=per_arm["memory_off"],
            accuracy_on_per_repeat=per_arm["memory_on_35_loo"],
            delta_per_repeat=verdict["delta_per_repeat"],
            delta=verdict["effect"],
            noise=verdict["noise"],
            phase4_delta=verdict["phase4_delta"],
            verdict=_verdict_code(verdict["verdict"]),
        )
    calls = sum(
        data["by_repeat"][r][arm][language.value]["n"]
        for r in repeats
        for arm in _ARMS
        for language in Language
    )
    return MemoryResults(
        source=str(path),
        method=MEMORY_METHOD,
        provider=MEMORY_PROVIDER,
        hardware=MEMORY_HARDWARE,
        calls=calls,
        **languages,
    )


def load_results(serving: Path | None = None, memory: Path | None = None) -> ResultsResponse:
    """Both sections. Raises `ResultsUnavailableError` if a file is missing."""
    return ResultsResponse(serving=serving_results(serving), memory=memory_results(memory))
