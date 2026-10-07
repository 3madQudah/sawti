"""Tests for `scripts/build_kaggle_bundles.py` (phase 6.3)."""

from __future__ import annotations

import json
from pathlib import Path

import build_kaggle_bundles as bundles

from sawti.privacy.redaction import redact


def test_bundles_hold_only_redacted_transcripts_the_adapter_files_and_the_package(tmp_path: Path) -> None:
    sizes = bundles.build(tmp_path / "kaggle", "someone")
    calls = tmp_path / "kaggle/sawti-comparison-calls"
    transcripts = sorted((calls / "transcripts").glob("*.txt"))
    assert len(transcripts) == 45
    for path in transcripts:
        assert redact(path.read_text(encoding="utf-8")).redaction_count == 0
    assert not list(calls.rglob("*ground_truth*"))  # scoring is local

    adapter = tmp_path / "kaggle/sawti-qlora-adapter"
    assert sorted(p.name for p in adapter.iterdir()) == [
        "adapter_config.json",
        "adapter_model.safetensors",
        "dataset-metadata.json",
    ]
    assert sizes["adapter"]["mb"] < 120

    repo = tmp_path / "kaggle/sawti-repo"
    assert (repo / "src/sawti/eval/serving_benchmark.py").is_file()
    assert not list(repo.rglob("__pycache__"))
    client = (repo / "requirements-client.txt").read_text().split()
    assert "langgraph==0.2.76" in client and all("==" in pin for pin in client)
    manifest = json.loads((repo / "BUNDLE.json").read_text())
    assert manifest["git_commit"] and "src/sawti/eval/serving_benchmark.py" in manifest["sha256"]
    for directory in (adapter, calls, repo):
        meta = json.loads((directory / "dataset-metadata.json").read_text())
        assert meta["id"].startswith("someone/sawti-")


def test_locked_versions_come_from_uv_lock() -> None:
    assert bundles.locked_versions(("langgraph",)) == ["langgraph==0.2.76"]
