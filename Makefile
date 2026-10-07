.PHONY: install test lint typecheck up down fmt stack e2e measure audio transcribe asr-eval asr-forcing

install:
	uv sync --all-extras --dev

test:
	uv run pytest

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

typecheck:
	uv run mypy src

up:
	docker compose up -d

down:
	docker compose --profile service down

# --- Phase 6.2: the service in containers ---------------------------------
# Whole stack: migrate (one-shot), api, worker. Provider from .env.
stack:
	docker compose --profile service up -d --build --wait

# End-to-end over HTTP, with the deterministic fake LLM so the call escalates.
# Runs against its own fresh database (E2E_DB), never the dev data, and puts
# the stack back on the dev database afterwards whether or not it passed.
E2E_DB ?= sawti_e2e
SERVICE_CONTAINERS = migrate api worker beat langfuse
e2e:
	docker compose up -d --wait postgres redis
	docker compose exec -T postgres psql -q -U sawti -d postgres \
		-c 'DROP DATABASE IF EXISTS $(E2E_DB) WITH (FORCE)' -c 'CREATE DATABASE $(E2E_DB)'
	SAWTI_SERVICE_DB=$(E2E_DB) SAWTI_LLM_PROVIDER=fake \
		docker compose --profile service up -d --build --force-recreate --wait $(SERVICE_CONTAINERS)
	uv run pytest -m integration tests/e2e -v; status=$$?; \
		docker compose --profile service up -d --force-recreate --wait $(SERVICE_CONTAINERS); \
		exit $$status

# Latency / throughput numbers into eval_results.md (see the script's docstring).
measure:
	uv run python scripts/measure_service.py

# --- Phase 3: audio pipeline ---------------------------------------------
# Order matters: audio -> transcribe -> asr-eval. Each writes what the next
# one reads. `audio` and `transcribe` are long (~45 min and ~2 h on an M1);
# both resume, so a re-run after a failure picks up where it stopped.

# Synthesize call audio from the Phase 1 transcripts, with the speaker-timeline
# manifest the WER and diarization checks are scored against. Needs ffmpeg.
audio:
	uv run python -m sawti.data.generate_audio

# Transcribe the corpus with the language forced, and write per-language WER.
# Produces data/asr_transcripts/ for the propagation eval.
transcribe:
	uv run python -m sawti.eval.experiments.asr_propagation --transcribe-only

# Rerun the Phase 2 agent over the ASR transcripts and append the scores to
# eval_results.md. This one costs LLM calls.
asr-eval:
	uv run python -m sawti.eval.experiments.asr_propagation

# Forced-Arabic vs Whisper auto-detect, on the code-switched calls.
asr-forcing:
	uv run python -m sawti.eval.experiments.asr_language_forcing --categories mixed --limit 15
