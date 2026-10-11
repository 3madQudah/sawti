"""`docs/06-API-SPECIFICATION.md`'s endpoint and schema lists are generated from `frontend/openapi.json`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import render_api_doc

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "frontend" / "openapi.json"
DOC = ROOT / "docs" / "06-API-SPECIFICATION.md"

_REF = "#/components/schemas/"
TINY = {
    "paths": {
        "/things/{thing_id}": {
            "post": {
                "summary": "Make Thing",
                "parameters": [
                    {"name": "thing_id", "in": "path", "required": True},
                    {"name": "x-api-key", "in": "header", "required": False},
                    {"name": "limit", "in": "query", "required": False},
                ],
                "requestBody": {"content": {"application/json": {"schema": {"$ref": _REF + "In"}}}},
                "responses": {
                    "202": {"content": {"application/json": {"schema": {"$ref": _REF + "Out"}}}},
                    "422": {},
                },
            }
        }
    },
    "components": {
        "schemas": {
            "In": {
                "properties": {
                    "name": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "when": {"anyOf": [{"type": "string", "format": "date-time"}, {"type": "null"}]},
                },
                "required": ["name"],
            },
            "Out": {"properties": {"kind": {"enum": ["a", "b"]}}, "required": ["kind"]},
            "Colour": {"enum": ["red", "green"]},
            "HTTPValidationError": {"properties": {"detail": {"type": "array"}}},
        }
    },
}


def test_the_committed_doc_is_current() -> None:
    document = json.loads(SCHEMA.read_text(encoding="utf-8"))
    current = DOC.read_text(encoding="utf-8")
    assert (
        render_api_doc.splice(current, render_api_doc.render(document)) == current
    ), "docs/06-API-SPECIFICATION.md is stale: run `make api-doc`"


def test_every_endpoint_in_the_schema_has_a_row() -> None:
    document = json.loads(SCHEMA.read_text(encoding="utf-8"))
    text = DOC.read_text(encoding="utf-8")
    for path, operations in document["paths"].items():
        for method in operations:
            assert f"| {method.upper()} | `{path}` |" in text


def test_an_endpoint_row_carries_headers_parameters_body_and_success() -> None:
    block = render_api_doc.render(TINY)
    assert (
        "| POST | `/things/{thing_id}` | Make Thing | `x-api-key` "
        "| `thing_id` (path), `limit` (query, optional) | In (`application/json`) | 202 Out | 202, 422 |"
    ) in block


def test_schema_lines_mark_optional_fields_and_render_types() -> None:
    block = render_api_doc.render(TINY)
    assert "- **In**: `name`: string, `tags`?: string[], `when`?: string (date-time) \\| null" in block
    assert "- **Out**: `kind`: `a` \\| `b`" in block
    assert "- **Colour**: `red` \\| `green`" in block
    assert "HTTPValidationError" not in block


def test_splice_replaces_only_the_marked_block() -> None:
    doc = f"intro\n{render_api_doc.BEGIN}\nold\n{render_api_doc.END}\noutro\n"
    assert render_api_doc.splice(doc, "NEW") == "intro\nNEW\noutro\n"


@pytest.mark.parametrize(
    "doc",
    [
        "no markers at all",
        f"{render_api_doc.BEGIN}\nno end",
        f"{render_api_doc.END}\nthen {render_api_doc.BEGIN}",
    ],
)
def test_splice_refuses_a_doc_without_a_well_formed_block(doc: str) -> None:
    with pytest.raises(ValueError, match="well-formed"):
        render_api_doc.splice(doc, "NEW")
