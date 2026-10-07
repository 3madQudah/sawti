"""Build the three Kaggle datasets the phase 6.3 benchmark notebook reads.

Writes `dist/kaggle/` (gitignored), one directory per dataset, each with the
`dataset-metadata.json` the Kaggle CLI wants:

  * `sawti-qlora-adapter/` — the Phase 5 LoRA adapter: `adapter_config.json`
    and `adapter_model.safetensors` only (~87 MB; vLLM's `--lora-modules`
    needs nothing else — the 895 MB on disk is mostly `checkpoint-*/`).
  * `sawti-comparison-calls/` — the 45 comparison calls
    (`data/serving_benchmark/call_set.json`), **redacted transcripts only**.
    The build fails if `redact()` finds anything left to redact in them. No
    ground truth: scoring happens locally, after download.
  * `sawti-repo/` — `src/sawti` as it is in this checkout, so the notebook runs
    the service's exact prompt builder, schema, grounding and graph; plus
    `requirements-client.txt` (the 9 pinned packages the client side needs,
    versions from `uv.lock`, verified in a clean venv) and
    `requirements-hf.txt` (the HF-baseline pins, = pyproject's `finetune`
    extra), and `BUNDLE.json` with the git commit and every file's sha256.

Upload: Kaggle → Datasets → New Dataset, once per directory (drag the folder;
keep the slugs), or with the CLI: `kaggle datasets create -p dist/kaggle/<dir>`.
All three stay **private**.

Usage:
    uv run python scripts/build_kaggle_bundles.py --kaggle-user <your-username>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sawti.privacy.redaction import redact

REPO = Path(__file__).resolve().parent.parent
ADAPTER_DIR = REPO / "data/finetune/qlora_adapter"
CALL_SET = REPO / "data/serving_benchmark/call_set.json"
SYNTHETIC_DIR = REPO / "data/synthetic"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
CLIENT_PACKAGES = (
    "pydantic",
    "pydantic-settings",
    "langgraph",
    "langgraph-checkpoint",
    "langchain-core",
    "openai",
    "httpx",
    "numpy",
    "python-dotenv",
)
HF_PACKAGES = ("transformers", "peft", "accelerate")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def locked_versions(names: tuple[str, ...], lock_path: Path = REPO / "uv.lock") -> list[str]:
    """`name==version` for each package, exactly as `uv.lock` resolved it."""
    lock = lock_path.read_text(encoding="utf-8")
    pins = []
    for name in names:
        match = re.search(r'\[\[package\]\]\nname = "' + re.escape(name) + r'"\nversion = "([^"]+)"', lock)
        if match is None:
            raise ValueError(f"{name} not in uv.lock")
        pins.append(f"{name}=={match.group(1)}")
    return pins


def _metadata(directory: Path, user: str, slug: str, title: str) -> None:
    (directory / "dataset-metadata.json").write_text(
        json.dumps({"title": title, "id": f"{user}/{slug}", "licenses": [{"name": "other"}]}, indent=2) + "\n"
    )


def build_adapter(out: Path, user: str) -> Path:
    target = out / "sawti-qlora-adapter"
    target.mkdir(parents=True)
    for name in ADAPTER_FILES:
        shutil.copy2(ADAPTER_DIR / name, target / name)
    config = json.loads((target / "adapter_config.json").read_text())
    if config.get("base_model_name_or_path") != "Qwen/Qwen3-8B":
        raise ValueError(f"unexpected base model {config.get('base_model_name_or_path')!r}")
    _metadata(target, user, "sawti-qlora-adapter", "Sawti Phase 5 QLoRA adapter (Qwen3-8B, r=16)")
    return target


def build_calls(out: Path, user: str) -> Path:
    target = out / "sawti-comparison-calls"
    (target / "transcripts").mkdir(parents=True)
    call_set = json.loads(CALL_SET.read_text(encoding="utf-8"))
    files: dict[str, str] = {}
    for language, call_ids in call_set.items():
        if language == "_meta":
            continue
        for call_id in call_ids:
            text = redact((SYNTHETIC_DIR / f"{call_id}.txt").read_text(encoding="utf-8")).redacted_text
            if redact(text).redaction_count:
                raise ValueError(f"{call_id}: PII-shaped text survived redaction")
            path = target / "transcripts" / f"{call_id}.txt"
            path.write_text(text, encoding="utf-8")
            files[f"transcripts/{call_id}.txt"] = _sha256(path)
    shutil.copy2(CALL_SET, target / "call_set.json")
    (target / "MANIFEST.json").write_text(
        json.dumps(
            {
                "redacted": True,
                "calls": sum(len(v) for k, v in call_set.items() if k != "_meta"),
                "sha256": files,
            },
            indent=2,
        )
        + "\n"
    )
    _metadata(target, user, "sawti-comparison-calls", "Sawti 6.3 comparison calls (redacted, synthetic)")
    return target


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.strip()


def build_repo(out: Path, user: str) -> Path:
    target = out / "sawti-repo"
    shutil.copytree(
        REPO / "src/sawti", target / "src/sawti", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copy2(REPO / "pyproject.toml", target / "pyproject.toml")
    shutil.copy2(REPO / "uv.lock", target / "uv.lock")
    (target / "requirements-client.txt").write_text("\n".join(locked_versions(CLIENT_PACKAGES)) + "\n")
    (target / "requirements-hf.txt").write_text("\n".join(locked_versions(HF_PACKAGES)) + "\n")
    notebook = REPO / "notebooks/kaggle_vllm_benchmark.ipynb"
    if notebook.exists():
        shutil.copy2(notebook, target / notebook.name)
    files = {
        str(path.relative_to(target)): _sha256(path)
        for path in sorted(target.rglob("*"))
        if path.is_file() and path.name != "BUNDLE.json"
    }
    (target / "BUNDLE.json").write_text(
        json.dumps(
            {
                "built_at": datetime.now(UTC).isoformat(),
                "git_commit": _git("rev-parse", "HEAD"),
                "git_dirty": bool(_git("status", "--porcelain", "--", "src", "pyproject.toml", "uv.lock")),
                "sha256": files,
            },
            indent=2,
        )
        + "\n"
    )
    _metadata(target, user, "sawti-repo", "Sawti repo snapshot (src/sawti) for the 6.3 benchmark")
    return target


def build(out: Path, user: str) -> dict[str, Any]:
    if out.exists():
        shutil.rmtree(out)
    built = {
        "adapter": build_adapter(out, user),
        "calls": build_calls(out, user),
        "repo": build_repo(out, user),
    }
    return {
        name: {
            "path": str(path),
            "mb": round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 1),
        }
        for name, path in built.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--kaggle-user", default="YOUR_KAGGLE_USERNAME")
    parser.add_argument("--out", type=Path, default=REPO / "dist/kaggle")
    args = parser.parse_args()
    print(json.dumps(build(args.out, args.kaggle_user), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
