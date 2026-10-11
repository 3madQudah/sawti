"""Upload → **real** ASR → redaction → analysis, on one Phase 3 recording (phase 6.4).

Marked `asr_local`, deselected by default: it needs Apple Silicon (mlx-whisper
on Metal), the gated pyannote weights (`HF_TOKEN`), and the Phase 3 audio,
which is not in git. Run it on the Mac with:

    uv run pytest -m asr_local

Everything else is the in-process service of the other API tests: eager
Celery, the fake LLM, the test database. The recording is copied into the
per-test upload directory, so the original is never deleted.
"""

from __future__ import annotations

import importlib.util
import platform
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sawti.config import get_settings

RECORDING = Path(__file__).resolve().parents[2] / "data" / "audio" / "call_0006_en.wav"

pytestmark = [
    pytest.mark.asr_local,
    pytest.mark.skipif(
        platform.system() != "Darwin" or platform.machine() != "arm64",
        reason="mlx-whisper needs Apple Silicon",
    ),
    pytest.mark.skipif(importlib.util.find_spec("mlx_whisper") is None, reason="needs the asr extra"),
    pytest.mark.skipif(not RECORDING.is_file(), reason=f"{RECORDING} is not present (make audio)"),
]


def test_a_real_recording_is_transcribed_diarized_redacted_and_analyzed(
    client: TestClient,
    headers: dict[str, str],
    upload: Callable[..., Any],
    llm: Any,
    audio_dir: Path,
) -> None:
    if not get_settings().hf_token:
        pytest.skip("HF_TOKEN is not set (gated pyannote weights)")

    response = upload(RECORDING.read_bytes(), filename=RECORDING.name, language_hint="en")
    assert response.status_code == 202, response.text

    detail = client.get(f"/calls/{response.json()['call_id']}", headers=headers).json()
    assert detail["failure_reason"] is None
    assert detail["status"] in {"awaiting_review", "completed"}
    assert {turn["speaker"] for turn in detail["turns"]} == {"Agent", "Customer"}
    assert len(detail["turns"]) >= 4
    assert detail["turns"][0]["speaker"] == "Agent", "first-speaker-is-agent convention"
    assert all(turn["language"] == "en" for turn in detail["turns"])
    assert list(audio_dir.iterdir()) == [], "the uploaded copy is deleted after transcription"
    assert RECORDING.is_file(), "the Phase 3 original is untouched"
