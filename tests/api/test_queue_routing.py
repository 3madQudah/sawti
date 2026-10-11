"""Queue routing: the native ASR worker and the Docker workers never take each other's tasks (phase 6.4).

Consumption is decided by two things, and both are checked: where each task is
routed (`celery_app.conf.task_routes`), and which queues each worker command
subscribes to (`-Q`): the Docker worker's `CMD` and `make asr-worker`.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from sawti.api.tasks import ASR_QUEUE, DEFAULT_QUEUE, TRANSCRIBE_TASK, celery_app

ROOT = Path(__file__).resolve().parents[2]

DOCKER_TASKS = [
    "sawti.analyze_call",
    "sawti.resume_call",
    "sawti.induce_rules",
    "sawti.consolidate_rules",
    "sawti.reap_stuck_calls",
]


def _queue(task_name: str) -> str:
    route = celery_app.amqp.router.route({}, task_name)
    return str(route["queue"].name)


def test_the_asr_task_goes_to_the_asr_queue() -> None:
    import sawti.api.asr_tasks  # noqa: F401  (registers the task)

    assert _queue(TRANSCRIBE_TASK) == ASR_QUEUE == "asr"


@pytest.mark.parametrize("task_name", DOCKER_TASKS)
def test_every_other_task_goes_to_the_default_queue(task_name: str) -> None:
    assert task_name in celery_app.tasks
    assert _queue(task_name) == DEFAULT_QUEUE != ASR_QUEUE


def _queues_of(command: list[str]) -> set[str]:
    assert "-Q" in command, f"{command} must name its queues explicitly"
    return set(command[command.index("-Q") + 1].split(","))


def test_the_docker_worker_consumes_only_the_default_queue() -> None:
    cmd_line = next(
        line for line in (ROOT / "Dockerfile").read_text().splitlines() if line.startswith('CMD ["celery"')
    )
    command = yaml.safe_load(cmd_line.removeprefix("CMD "))
    assert command[:3] == ["celery", "-A", "sawti.api.tasks"]
    assert _queues_of(command) == {DEFAULT_QUEUE}

    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    assert "command" not in compose["services"]["worker"], "the worker must use the image's CMD"


def test_the_asr_worker_consumes_only_the_asr_queue_in_one_process() -> None:
    lines = (ROOT / "Makefile").read_text().splitlines()
    recipe = lines[lines.index("asr-worker:") + 1].strip()
    command = shlex.split(recipe)
    command = command[command.index("celery") :]
    assert command[:3] == ["celery", "-A", "sawti.api.asr_tasks"]
    assert _queues_of(command) == {ASR_QUEUE}
    assert "--pool=solo" in command


def test_the_docker_worker_app_does_not_even_register_the_asr_task() -> None:
    """`-A sawti.api.tasks` never imports `sawti.api.asr_tasks` (nor Whisper or pyannote)."""
    probe = (
        "import sys, sawti.api.tasks as t; t.celery_app.loader.import_default_modules(); "
        f"print({TRANSCRIBE_TASK!r} in t.celery_app.tasks, 'sawti.api.asr_tasks' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
    assert out.split() == ["False", "False"]
