"""Guards on `notebooks/kaggle_vllm_benchmark.ipynb` (phase 6.3): it cannot run in CI, so check its shape."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/kaggle_vllm_benchmark.ipynb"


def _cells() -> list[dict[str, object]]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _code() -> list[str]:
    return ["".join(c["source"]) for c in _cells() if c["cell_type"] == "code"]  # type: ignore[arg-type]


def test_every_code_cell_is_valid_python() -> None:
    for source in _code():
        ast.parse(source)


def test_the_first_code_cell_is_the_feasibility_check_and_everything_after_depends_on_it() -> None:
    first, second = _code()[0], _code()[1]
    for flag in (
        '"--tensor-parallel-size", "2"',
        '"--dtype", "float16"',
        '"--enable-lora"',
        "vllm=={VLLM_VERSION}",
    ):
        assert flag in first
    assert 'VLLM_VERSION = "0.18.1"' in first and "FEASIBLE = True" in first
    assert "assert FEASIBLE" in second


def test_the_hf_token_comes_only_from_kaggle_secrets_and_is_never_printed() -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert 'UserSecretsClient().get_secret("HF_TOKEN")' in "\n".join(_code())
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", text)
    assert not re.search(r"print\([^)]*HF_TOKEN", text)


def test_the_arms_match_the_plan() -> None:
    joined = "\n".join(_code())
    for arm in ('"vllm_adapter_c1"', '"vllm_adapter_c16"', '"vllm_base_c16"', '"hf_adapter_c1"'):
        assert arm in joined
    assert '"--concurrency", "16"' in joined and '"--limit", "15"' in joined and '"--limit", "3"' in joined
    assert "sawti.eval.kaggle_benchmark" in joined  # the tested driver, not inline reimplementation
    assert "sawti_kaggle_results.json" in joined


def test_the_gpu_time_estimate_is_stated_at_the_top() -> None:
    top = "".join(_cells()[0]["source"])  # type: ignore[arg-type]
    assert "Estimated GPU time" in top
    assert "T4×2" in top
