"""Render the endpoint and schema lists in `docs/06-API-SPECIFICATION.md` from `frontend/openapi.json`.

The API doc is generated, not hand-copied: everything between the two marker
comments is replaced by what this script renders from the committed OpenAPI
schema (itself checked against the app by `tests/scripts/test_export_openapi.py`).
`tests/scripts/test_render_api_doc.py` fails if the doc is stale, so run
`make api-doc` after `make openapi`.

Usage:
    uv run python scripts/render_api_doc.py [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_SCHEMA = Path("frontend/openapi.json")
DEFAULT_DOC = Path("docs/06-API-SPECIFICATION.md")
BEGIN = "<!-- BEGIN GENERATED from frontend/openapi.json by scripts/render_api_doc.py; do not edit -->"
END = "<!-- END GENERATED -->"
# FastAPI's own validation-error envelope: the same on every route, not worth a row.
_SKIPPED_SCHEMAS = frozenset({"HTTPValidationError", "ValidationError"})


def type_name(schema: dict[str, Any]) -> str:
    """A short, Markdown-table-safe name for a JSON-schema fragment."""
    if "$ref" in schema:
        return str(schema["$ref"]).rsplit("/", 1)[-1]
    if "anyOf" in schema:
        return " \\| ".join(type_name(option) for option in schema["anyOf"])
    if "enum" in schema:
        return " \\| ".join(f"`{value}`" for value in schema["enum"])
    if "const" in schema:
        return f"`{schema['const']}`"
    kind = schema.get("type")
    if kind == "array":
        return f"{type_name(schema.get('items', {}))}[]"
    if kind == "object" and isinstance(schema.get("additionalProperties"), dict):
        return f"map<{type_name(schema['additionalProperties'])}>"
    if kind == "string" and "format" in schema:
        return f"string ({schema['format']})"
    return str(kind) if kind else "any"


def _endpoint_rows(document: dict[str, Any]) -> list[str]:
    header = ("Method", "Path", "Summary", "Headers", "Path / query parameters", "Request body", "Success")
    rows = ["| " + " | ".join((*header, "Codes in schema")) + " |", "|---" * (len(header) + 1) + "|"]
    for path, operations in document["paths"].items():
        for method, operation in operations.items():
            params = operation.get("parameters", [])
            headers = ", ".join(f"`{p['name']}`" for p in params if p["in"] == "header") or "—"
            other = (
                ", ".join(
                    f"`{p['name']}` ({p['in']}{'' if p.get('required') else ', optional'})"
                    for p in params
                    if p["in"] in ("path", "query")
                )
                or "—"
            )
            content = operation.get("requestBody", {}).get("content", {})
            body = ", ".join(f"{type_name(v['schema'])} (`{k}`)" for k, v in content.items()) or "—"
            codes = list(operation["responses"])
            success = next(code for code in codes if code.startswith("2"))
            returned = operation["responses"][success].get("content", {}).get("application/json", {})
            rows.append(
                f"| {method.upper()} | `{path}` | {operation.get('summary', '')} | {headers} | {other} | "
                f"{body} | {success} {type_name(returned.get('schema', {}))} | {', '.join(codes)} |"
            )
    return rows


def _schema_lines(document: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for name, schema in document["components"]["schemas"].items():
        if name in _SKIPPED_SCHEMAS:
            continue
        properties = schema.get("properties")
        if not properties:
            lines.append(f"- **{name}**: {type_name(schema)}")
            continue
        required = set(schema.get("required", []))
        fields = ", ".join(
            f"`{field}`{'' if field in required else '?'}: {type_name(value)}"
            for field, value in properties.items()
        )
        lines.append(f"- **{name}**: {fields}")
    return lines


def render(document: dict[str, Any]) -> str:
    """The generated block, markers included."""
    parts = [
        BEGIN,
        "",
        "### Endpoints",
        "",
        *_endpoint_rows(document),
        "",
        "### Schemas",
        "",
        "`?` marks an optional field.",
        "",
        *_schema_lines(document),
        "",
        END,
    ]
    return "\n".join(parts)


def splice(doc_text: str, block: str) -> str:
    """Replace the marked block in `doc_text` with `block`.

    Raises:
        ValueError: If either marker is missing or they are out of order.
    """
    start = doc_text.find(BEGIN)
    end = doc_text.find(END)
    if start == -1 or end == -1 or end < start:
        raise ValueError("the doc has no well-formed generated block (both markers, in order)")
    return doc_text[:start] + block + doc_text[end + len(END) :]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--check", action="store_true", help="exit 1 if the doc is stale; write nothing")
    args = parser.parse_args()

    document = json.loads(args.schema.read_text(encoding="utf-8"))
    current = args.doc.read_text(encoding="utf-8")
    updated = splice(current, render(document))
    if args.check:
        if updated != current:
            print(f"{args.doc} is stale: run `make api-doc`", file=sys.stderr)
            return 1
        return 0
    args.doc.write_text(updated, encoding="utf-8")
    print(f"wrote {args.doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
