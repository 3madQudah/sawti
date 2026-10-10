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


def _backend_pattern() -> re.Pattern[str]:
    match = re.search(r're\.findall\(r"(.+?)", server_log\)', _code()[0])
    assert match, "the feasibility cell must parse the backend from vllm_server.log"
    return re.compile(match.group(1).replace("\\\\", "\\"))


def test_triton_attention_is_forced_with_the_0_18_1_flag_and_flashinfer_paths_are_closed() -> None:
    first = _code()[0]
    assert 'ATTENTION_BACKEND = "TRITON_ATTN"' in first
    assert '"--attention-backend", ATTENTION_BACKEND' in first
    assert 'environ["VLLM_ATTENTION_BACKEND"]' not in first  # gone in 0.18.1: the CLI flag is the control
    assert 'os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"' in first
    assert "libcuda.so" in first and "LIBRARY_PATH" in first  # the -lcuda safety net


def test_the_backend_assertion_reads_both_0_18_1_log_formats() -> None:
    pattern = _backend_pattern()
    explicit = (
        "(EngineCore_DP0 pid=81) INFO 10-10 06:12:01 [cuda.py:257] "
        "Using AttentionBackendEnum.TRITON_ATTN backend."
    )
    auto = (
        "(Worker_TP1 pid=90) INFO 10-10 06:12:01 [cuda.py:318] Using FLASHINFER attention backend out of "
        "potential backends: ['FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']."
    )
    assert pattern.findall(explicit) == ["TRITON_ATTN"]
    assert pattern.findall(auto) == ["FLASHINFER"]  # so a silent auto-pick fails the assertion
    assert pattern.findall("INFO Using xgrammar for structured outputs") == []


def test_the_backend_is_asserted_before_the_probe_call() -> None:
    first = _code()[0]
    assert first.index("assert set(chosen) == {ATTENTION_BACKEND}") < first.index(
        "urllib.request.urlopen(req"
    )
