"""Write the API's OpenAPI schema to `frontend/openapi.json`.

Phase 6.4 stage A. Stage B (the React dashboard) generates its TypeScript types
from this file. `tests/scripts/test_export_openapi.py` fails if the committed
file is stale, so regenerate it (`make openapi`) with every API change.

Usage:
    uv run python scripts/export_openapi.py [--out frontend/openapi.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sawti.api.main import create_app

DEFAULT_OUT = Path("frontend/openapi.json")


def openapi_document() -> dict[str, Any]:
    """The schema FastAPI generates for the app, exactly."""
    document: dict[str, Any] = create_app().openapi()
    return document


def render(document: dict[str, Any]) -> str:
    """Stable text: sorted keys, two-space indent, Arabic kept as is, trailing newline."""
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(openapi_document()), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
