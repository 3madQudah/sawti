"""Fixtures for the service tests: a clean database, eager Celery, a fake LLM, an API client.

Phase 6.2. Everything runs in-process against the suite's real Postgres test
database (the checkpointer included); Celery runs tasks eagerly inside
`.delay()`, so a request to `POST /calls` has finished its analysis by the
time it returns. The LLM is `sawti.llm.fake_provider.FakeProvider`, wrapped to
count calls and to fail on demand.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from sawti.agent.nodes import extract as extract_module
from sawti.api import tasks
from sawti.api.main import create_app
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
                "TRUNCATE reviewer_actions, review_submissions, call_analyses, calls, agents, memory_rules "
                "CASCADE"
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
