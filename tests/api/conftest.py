"""Fixtures for the service tests: a clean database, eager Celery, a fake LLM, an API client.

Phase 6.2. Everything runs in-process against the suite's real Postgres test
database (the checkpointer included); Celery runs tasks eagerly inside
`.delay()`, so a request to `POST /calls` has finished its analysis by the
time it returns. The LLM is `sawti.llm.fake_provider.FakeProvider`, wrapped to
count calls and to fail on demand.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from sawti.agent.nodes import extract as extract_module
from sawti.api import asr_tasks, tasks
from sawti.api.main import create_app
from sawti.asr import pipeline as asr_pipeline
from sawti.asr.pipeline import AsrResult, AsrTimings
from sawti.config import get_settings
from sawti.db.session import get_engine
from sawti.llm.fake_provider import FakeProvider
from sawti.memory import conflict, consolidate, induction

API_KEY = "test-api-key"
REVIEWER = "qa-reviewer-7"

TRANSCRIPT_EN = (
    "Agent: Good morning, this is Sara from Zain, how can I help?\n"
    "Customer: My internet has been down since yesterday.\n"
    "Agent: I will send a technician tomorrow before noon.\n"
    "Customer: Thank you, please call me on 0791234567 first.\n"
)
TRANSCRIPT_AR = (
    "Agent: مرحبا، معك سارة من زين، كيف بقدر أساعدك؟\n"
    "Customer: النت عندي مقطوع من مبارح.\n"
    "Agent: رح أبعتلك فني بكرا الصبح.\n"
    "Customer: تمام، شكراً كثير.\n"
)
TRANSCRIPT_MIXED = (
    "Agent: Hello, معك سارة من Zain، how can I help?\n"
    "Customer: الـ internet عندي slow كثير.\n"
    "Agent: I will reset the router من عندي now.\n"
    "Customer: Okay تمام.\n"
)


@dataclass
class CountingProvider:
    """`FakeProvider` that counts calls and can raise queued errors first."""

    ungrounded: int = 1
    errors: list[BaseException] = field(default_factory=list)
    calls: int = 0

    async def structured_complete(self, prompt: str, **kwargs: Any) -> Any:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return await FakeProvider(ungrounded_claims=self.ungrounded).structured_complete(prompt, **kwargs)


@pytest.fixture(autouse=True)
def _clean_tables() -> None:
    """Every service test starts from empty application tables (queue counts depend on it)."""
    with get_engine().begin() as conn:
        conn.execute(
            sa.text(
                "TRUNCATE memory_conflicts, reviewer_actions, review_submissions, call_analyses, calls, "
                "agents, memory_rules CASCADE"
            )
        )
    tasks._store, tasks._store_rule_ids = None, frozenset()


@pytest.fixture(autouse=True)
def _service_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """API key configured, no embedding preload, fast retries — and never a real LLM.

    `SAWTI_LLM_PROVIDER=fake` is the backstop: any code path that reaches
    `get_llm_provider()` without going through the `llm` fixture's patch gets
    the offline fake instead of the `.env` provider. (A first version patched
    only `extract`, and the memory loop's induction quietly called Gemini.)
    """
    monkeypatch.setenv("SAWTI_LLM_PROVIDER", "fake")
    monkeypatch.setenv("SAWTI_API_KEY", API_KEY)
    monkeypatch.setenv("SAWTI_WORKER_PRELOAD_EMBEDDINGS", "false")
    monkeypatch.setenv("SAWTI_TASK_MAX_RETRIES", "2")
    monkeypatch.setenv("SAWTI_TASK_RETRY_BACKOFF_SECONDS", "0")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def audio_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Uploads go to a per-test directory; audio is deleted after transcription by default."""
    directory = tmp_path / "uploads"
    monkeypatch.setenv("SAWTI_AUDIO_DIR", str(directory))
    monkeypatch.delenv("SAWTI_KEEP_AUDIO", raising=False)
    get_settings.cache_clear()
    return directory


#: Minimal file headers per accepted format: enough for the upload's content check.
AUDIO_HEADERS = {
    ".wav": b"RIFF\x24\x00\x00\x00WAVEfmt ",
    ".mp3": b"ID3\x04\x00\x00\x00\x00\x00\x00",
    ".m4a": b"\x00\x00\x00\x20ftypM4A \x00\x00",
}


def audio_bytes(suffix: str = ".wav", size: int = 4096) -> bytes:
    """A fake recording of `size` bytes that passes the content check for `suffix`."""
    header = AUDIO_HEADERS[suffix]
    return header + b"\x00" * (size - len(header))


@pytest.fixture
def make_audio() -> Callable[..., bytes]:
    """`audio_bytes`, for test modules (the tests directory is not a package)."""
    return audio_bytes


@pytest.fixture
def api_key() -> str:
    return API_KEY


@dataclass
class FakeAsr:
    """Stands in for `sawti.asr.pipeline.transcribe_call`: returns `transcript`, or raises `error`."""

    transcript: str = TRANSCRIPT_EN
    error: BaseException | None = None
    calls: list[tuple[Path, str]] = field(default_factory=list)

    def __call__(self, audio_path: Path, *, category: str, diarization_pipeline: Any = None) -> AsrResult:
        self.calls.append((audio_path, category))
        assert audio_path.is_file(), "the worker must read the stored upload"
        if self.error is not None:
            raise self.error
        timings = AsrTimings(decode_s=0.1, transcribe_s=1.0, diarize_s=0.5, total_s=1.6)
        return AsrResult(
            raw_transcript=self.transcript, audio_seconds=30.0, decoded_language=category, timings=timings
        )


@pytest.fixture
def fake_asr(monkeypatch: pytest.MonkeyPatch) -> FakeAsr:
    """No Whisper, no pyannote: the ASR task gets `FakeAsr`'s transcript."""
    fake = FakeAsr()
    monkeypatch.setattr(asr_pipeline, "transcribe_call", fake)
    monkeypatch.setattr(asr_tasks, "_pipeline", lambda: None)
    return fake


@pytest.fixture
def upload(client: TestClient) -> Callable[..., Any]:
    """POST /calls/audio; returns the response."""

    def _upload(
        content: bytes | None = None,
        *,
        filename: str = "call.wav",
        language_hint: str | None = "en",
        agent_external_id: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        data = {}
        if language_hint is not None:
            data["language_hint"] = language_hint
        if agent_external_id is not None:
            data["agent_external_id"] = agent_external_id
        files = {"file": (filename, content if content is not None else audio_bytes(Path(filename).suffix))}
        sent = {"X-API-Key": API_KEY} if headers is None else headers
        return client.post("/calls/audio", data=data, files=files, headers=sent)

    return _upload


@pytest.fixture(autouse=True)
def _eager_celery() -> Iterator[None]:
    conf = tasks.celery_app.conf
    previous = conf.task_always_eager, conf.task_eager_propagates
    conf.task_always_eager, conf.task_eager_propagates = True, True
    yield
    conf.task_always_eager, conf.task_eager_propagates = previous


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> CountingProvider:
    """The provider every LLM caller in the service uses — extraction and the memory loop.

    Extraction: two grounded claims and one ungrounded (escalates).
    """
    provider = CountingProvider()
    for module in (extract_module, induction, conflict, consolidate):
        monkeypatch.setattr(module, "get_llm_provider", lambda: provider)
    return provider


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
def headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY, "X-Reviewer-Id": REVIEWER}


@pytest.fixture
def transcripts() -> dict[str, str]:
    """One small call per language category, keyed by `Language` value."""
    return {"en": TRANSCRIPT_EN, "ar": TRANSCRIPT_AR, "mixed": TRANSCRIPT_MIXED}


@pytest.fixture
def submit(client: TestClient, headers: dict[str, str]) -> Callable[..., str]:
    """POST a call and return its id. Celery is eager, so analysis has already run."""

    def _submit(transcript: str = TRANSCRIPT_EN, language: str = "en") -> str:
        response = client.post(
            "/calls",
            json={"transcript": transcript, "language": language, "agent_external_id": "agent-42"},
            headers=headers,
        )
        assert response.status_code == 202, response.text
        return str(response.json()["call_id"])

    return _submit


# --- Memory conflicts (phase 6.4) ---------------------------------------------------

#: PII in the conflict judge's reasoning: it must be redacted before storage.
JUDGE_PHONE = "0795551234"


def _bag_of_words(text: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    vector = np.zeros(256)
    for word in re.findall(r"[a-z]+", text.lower()):
        vector[zlib.crc32(word.encode()) % 256] += 1.0
    return vector


@pytest.fixture
def contradicting_judge(monkeypatch: pytest.MonkeyPatch, llm: CountingProvider) -> None:
    """Every conflict check finds a contradiction, and a bag-of-words embedder makes similarity real."""
    monkeypatch.setattr(tasks, "_induction_embed", _bag_of_words)
    monkeypatch.setattr(
        conflict,
        "_judges_contradiction",
        lambda candidate, existing: conflict._ConflictJudgment(
            contradicts=True, reasoning=f"Opposite guidance for the same case; see {JUDGE_PHONE}."
        ),
    )


@pytest.fixture
def review_call(client: TestClient, headers: dict[str, str]) -> Callable[[str], None]:
    """Submit a review rejecting the call's first claim and confirming the rest (one correction)."""

    def _review(call_id: str) -> None:
        claims = client.get(f"/calls/{call_id}", headers=headers).json()["claims"]
        verdicts = [
            {"claim_id": claim["claim_id"], "verdict": "reject" if i == 0 else "confirm"}
            for i, claim in enumerate(claims)
        ]
        response = client.post(
            f"/reviews/{call_id}/corrections", json={"verdicts": verdicts}, headers=headers
        )
        assert response.status_code == 202, response.text

    return _review


@pytest.fixture
def make_conflicts(
    submit: Callable[..., str],
    review_call: Callable[[str], None],
    transcripts: dict[str, str],
    contradicting_judge: None,
) -> Callable[..., list[str]]:
    """Run the real loop: the first review stores a rule, each later one conflicts with it.

    Returns the conflicts' source call ids, in order; the first call (the
    stored rule's source) is not among them.
    """

    def _make(*languages: str, first: str = "en") -> list[str]:
        review_call(submit(transcripts[first], first))
        sources = []
        for language in languages:
            call_id = submit(transcripts[language], language)
            review_call(call_id)
            sources.append(call_id)
        return sources

    return _make
