"""Save the dashboard's API responses from a real run, for the demo video.

The video is recorded by replaying these responses in the browser
(demo_video/), so the numbers, transcript and claims on screen are the ones a
real analysis produced, while the recording itself is scripted and repeatable.

Two stages, around one real review done by hand in the dashboard:

    # 1. upload a call in the dashboard, wait until it is awaiting_review
    uv run python scripts/capture_demo_api.py before --call-id <id> --audio-name <file.wav>
    # 2. review it in the dashboard (confirm most claims, correct one), submit
    uv run python scripts/capture_demo_api.py after --call-id <id>

If only `after` was captured (the review was submitted first), the replay
rebuilds the pre-review call from it: the same analysis with the review removed
and the status set back to awaiting_review (frontend/demo_video/source.ts).

Writes data/demo_capture/{before,after}.json. Reads SAWTI_API_KEY from the
environment or the repo's .env. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "demo_capture"

Fetch = Callable[[str], dict[str, Any]]


def env_value(key: str, env_file: Path) -> str | None:
    """`key` from the process environment, else from a dotenv-style file."""
    if os.environ.get(key):
        return os.environ[key]
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.strip().partition("=")
        if sep and name.strip() == key:
            return value.strip().strip("'\"")
    return None


def endpoints(stage: str, call_id: str) -> dict[str, str]:
    """The GET paths the dashboard reads at this stage, keyed by path."""
    call = f"/calls/{call_id}"
    if stage == "before":
        paths = ["/results", "/stats", "/reviews?limit=500", "/memory/rules", "/memory/conflicts", call]
    elif stage == "after":
        # /results too: a capture taken only after the review must still carry it.
        paths = ["/results", "/stats", "/reviews?limit=500", "/memory/rules", "/memory/conflicts", call]
    else:
        raise ValueError(f"unknown stage {stage!r}")
    return {p.split("?")[0]: p for p in paths}


def check(stage: str, call: dict[str, Any]) -> None:
    """Refuse a capture taken at the wrong moment, so the video can't show a state that never happened."""
    status = call.get("status")
    if stage == "before" and status != "awaiting_review":
        raise SystemExit(f"call is {status!r}; capture 'before' once it is awaiting_review")
    if stage == "after" and (status == "awaiting_review" or not call.get("review")):
        raise SystemExit(f"call is {status!r} with no review yet; submit the review first")


def capture(stage: str, call_id: str, fetch: Fetch, audio_name: str | None = None) -> dict[str, Any]:
    responses = {key: fetch(path) for key, path in endpoints(stage, call_id).items()}
    check(stage, responses[f"/calls/{call_id}"])
    return {
        "stage": stage,
        "call_id": call_id,
        "audio_name": audio_name,
        "captured_at": datetime.now(UTC).isoformat(),
        "responses": responses,
    }


def http_fetch(base_url: str, api_key: str, reviewer: str) -> Fetch:
    def fetch(path: str) -> dict[str, Any]:
        request = urllib.request.Request(
            base_url.rstrip("/") + path, headers={"X-API-Key": api_key, "X-Reviewer-Id": reviewer}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body: dict[str, Any] = json.load(response)
            return body

    return fetch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stage", choices=["before", "after"])
    parser.add_argument("--call-id", required=True)
    parser.add_argument("--audio-name", help="file name shown in the upload step")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--reviewer", default="qa-demo")
    args = parser.parse_args(argv)

    api_key = env_value("SAWTI_API_KEY", REPO / ".env")
    if not api_key:
        print("SAWTI_API_KEY not set (environment or .env)", file=sys.stderr)
        return 1
    fetch = http_fetch(args.base_url, api_key, args.reviewer)
    data = capture(args.stage, args.call_id, fetch, args.audio_name)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.stage}.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    call = data["responses"][f"/calls/{args.call_id}"]
    print(f"wrote {out.relative_to(REPO)}: call {call.get('status')}, {len(call.get('claims', []))} claims")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
