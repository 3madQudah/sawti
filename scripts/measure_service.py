"""Measure the containerized service: latency, throughput, retries, failures, cold start, image size.

Phase 6.2 done-when numbers. For each worker count, this script:

  1. recreates a scratch database (`sawti_service_measure`) so runs never mix
     with each other or with the dev data;
  2. (re)starts the stack against it — `docker compose --profile service up
     --scale worker=N --force-recreate --wait` — so every worker cold-starts;
  3. waits until N workers answer a Celery ping;
  4. submits a burst of synthetic calls over HTTP, equal counts per language,
     all at once, and polls `GET /calls/{id}` until each is terminal;
  5. reads each call's timestamps from the API.

Reported per language category (never blended): queue wait
(`started_at - queued_at`) and processing time (`finished_at - started_at`),
p50/p95 by linear interpolation (numpy's default) — with small n per cell,
p95 is close to the max and should be read as such. Throughput is the whole
burst's (calls / (last finished_at - first queued_at)), since workers serve
all languages from one queue. Retry rate = share of calls with attempts > 1;
failure rate = share ending `failed`.

A burst measures capacity under backlog: queue wait grows with a call's
position in the burst, and that is what it shows — not idle-system latency.

Transcripts: `data/synthetic/`, the first N per language whose lines all
carry a speaker prefix (the API rejects the 10 phase 1 calls with the known
prefix defect rather than guess).

Usage:
    uv run python scripts/measure_service.py --provider fake --fake-latency 5 \
        --workers 1 2 4 --per-language 20
    uv run python scripts/measure_service.py --provider gemini --workers 1 --per-language 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import psycopg

from sawti.api.views import split_turns
from sawti.config import get_settings
from sawti.data.transcript_parser import UnattributableLineError
from sawti.db.checkpointer import checkpointer_conninfo
from sawti.privacy.redaction import redact
from sawti.schemas import Language

MEASURE_DB = "sawti_service_measure"
SYNTHETIC_DIR = Path("data/synthetic")
RESULTS_DIR = Path("data/service_measurements")
SERVICES = ("migrate", "api", "worker", "beat", "langfuse")
TERMINAL = {"awaiting_review", "completed", "failed", "reviewed"}
_COLD_START = re.compile(r"embedding model loaded in ([0-9.]+)s")


def pick_transcripts(per_language: int) -> list[tuple[Language, str, str]]:
    """The first `per_language` usable synthetic transcripts per category, interleaved ar/en/mixed.

    Interleaved, not grouped: in a burst, queue wait grows with submission
    position, so submitting one language first would show up as a fake
    per-language difference in queue wait.
    """
    by_language: dict[Language, list[tuple[Language, str, str]]] = {}
    for language in Language:
        chosen = by_language.setdefault(language, [])
        for path in sorted(SYNTHETIC_DIR.glob(f"call_*_{language.value}.txt")):
            text = path.read_text(encoding="utf-8")
            try:
                # Not `path.stem`: that would apply the synthetic corpus's
                # hand-recorded SPEAKER_REPAIRS, which the API never sees.
                split_turns(redact(text).redacted_text, "measure")
            except UnattributableLineError:
                continue
            chosen.append((language, path.stem, text))
            if len(chosen) == per_language:
                break
    return [item for group in zip(*by_language.values(), strict=False) for item in group]


def recreate_database(name: str) -> None:
    admin = checkpointer_conninfo(get_settings().database_url.rsplit("/", 1)[0] + "/postgres")
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        conn.execute(f'CREATE DATABASE "{name}"')


def compose(env: dict[str, str], *args: str) -> str:
    result = subprocess.run(
        ["docker", "compose", "--profile", "service", *args],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout + result.stderr


def wait_for_workers(count: int, timeout: float = 180.0) -> None:
    from sawti.api.tasks import celery_app

    deadline = time.monotonic() + timeout
    replies: list[Any] = []
    while time.monotonic() < deadline:
        try:
            replies = celery_app.control.ping(timeout=2.0) or []
        except Exception:  # broker connection reset while the stack restarts
            replies = []
        if len(replies) >= count:
            return
        time.sleep(1.0)
    raise TimeoutError(f"only {len(replies)} of {count} workers answered")


def run_burst(
    base_url: str, headers: dict[str, str], items: list[tuple[Language, str, str]]
) -> list[dict[str, Any]]:
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        ids = []
        for language, stem, text in items:
            response = client.post(
                "/calls",
                json={"transcript": text, "language": language.value, "agent_external_id": f"measure-{stem}"},
                headers=headers,
            )
            response.raise_for_status()
            ids.append(response.json()["call_id"])
        pending, done = set(ids), {}
        while pending:
            for call_id in list(pending):
                detail = client.get(f"/calls/{call_id}", headers=headers).json()
                if detail["status"] in TERMINAL:
                    done[call_id] = detail
                    pending.discard(call_id)
            time.sleep(1.0)
    return [done[i] for i in ids]


def _seconds(later: str | None, earlier: str | None) -> float | None:
    if later is None or earlier is None:
        return None
    return (datetime.fromisoformat(later) - datetime.fromisoformat(earlier)).total_seconds()


def _pct(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    per_language: dict[str, Any] = {}
    for language in Language:
        rows = [d for d in details if d["language"] == language.value]
        if not rows:
            continue
        wait = [w for d in rows if (w := _seconds(d["started_at"], d["queued_at"])) is not None]
        proc = [p for d in rows if (p := _seconds(d["finished_at"], d["started_at"])) is not None]
        per_language[language.value] = {
            "n": len(rows),
            "queue_wait_p50": _pct(wait, 50),
            "queue_wait_p95": _pct(wait, 95),
            "processing_p50": _pct(proc, 50),
            "processing_p95": _pct(proc, 95),
            "retry_rate": sum(d["attempts"] > 1 for d in rows) / len(rows),
            "failure_rate": sum(d["status"] == "failed" for d in rows) / len(rows),
            "statuses": {s: sum(d["status"] == s for d in rows) for s in sorted({d["status"] for d in rows})},
        }
    finished = [d["finished_at"] for d in details if d["finished_at"]]
    span = _seconds(max(finished), min(d["queued_at"] for d in details)) if finished else None
    return {
        "per_language": per_language,
        "calls": len(details),
        "wall_seconds": span,
        "throughput_per_min": (len(finished) / span * 60.0) if span else None,
    }


def cold_starts(env: dict[str, str]) -> list[float]:
    return [float(m) for m in _COLD_START.findall(compose(env, "logs", "worker"))]


def image_sizes() -> dict[str, int]:
    sizes = {}
    for image in ("sawti-api", "sawti-worker"):
        out = subprocess.run(
            ["docker", "image", "inspect", image, "--format", "{{.Size}}"],
            capture_output=True,
            text=True,
            check=True,
        )
        sizes[image] = int(out.stdout.strip())
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--provider", choices=["fake", "gemini"], required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--per-language", type=int, default=20)
    parser.add_argument("--fake-latency", type=float, default=0.0)
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    key = get_settings().api_key
    if key is None or not key.get_secret_value():
        raise SystemExit("SAWTI_API_KEY must be set (in .env) to measure the service")
    headers = {"X-API-Key": key.get_secret_value(), "X-Reviewer-Id": "measure-script"}
    items = pick_transcripts(args.per_language)
    env = {
        "SAWTI_SERVICE_DB": MEASURE_DB,
        "SAWTI_LLM_PROVIDER": args.provider,
        "SAWTI_FAKE_LLM_LATENCY_SECONDS": str(args.fake_latency),
    }

    runs = []
    for workers in args.workers:
        recreate_database(MEASURE_DB)
        # Recreate only the service containers: recreating postgres/redis too
        # would drop every client connection mid-run.
        compose(
            env,
            "up",
            "-d",
            "--build",
            "--force-recreate",
            "--wait",
            "--scale",
            f"worker={workers}",
            *SERVICES,
        )
        wait_for_workers(workers)
        details = run_burst(args.base_url, headers, items)
        run = {"workers": workers, **summarize(details), "cold_start_seconds": cold_starts(env)}
        runs.append(run)
        print(json.dumps({k: v for k, v in run.items() if k != "per_language"}, indent=1))

    # Back to the default stack: dev database, one worker, provider from .env.
    compose({}, "up", "-d", "--force-recreate", "--wait", "--scale", "worker=1", *SERVICES)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = RESULTS_DIR / f"{stamp}_{args.provider}.json"
    out.write_text(
        json.dumps(
            {
                "provider": args.provider,
                "fake_latency_seconds": args.fake_latency if args.provider == "fake" else None,
                "per_language": args.per_language,
                "transcripts": [stem for _, stem, _ in items],
                "image_size_bytes": image_sizes(),
                "runs": runs,
            },
            indent=2,
        )
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
