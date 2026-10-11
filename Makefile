.PHONY: install test lint typecheck up down fmt stack e2e measure audio transcribe asr-eval asr-forcing asr-worker asr-latency openapi api-doc demo

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

# --- Phase 6.4: audio ingestion and the dashboard contract --------------------
# The ASR worker runs natively, not in Docker: mlx-whisper needs Metal. It
# consumes only the `asr` queue (the Docker workers only the default one),
# one file at a time in-process (`--pool=solo`: no fork after Metal init).
# Needs `make up` (or the stack) for Postgres + Redis, and HF_TOKEN in .env.
asr-worker:
	SAWTI_WORKER_PRELOAD_EMBEDDINGS=false uv run celery -A sawti.api.asr_tasks worker -Q asr --pool=solo --loglevel=INFO

# The demo: the stack (Docker), the native ASR worker and the dashboard dev
# server, together. Ctrl-C stops the worker and the dashboard; the stack keeps
# running (`make down` stops it). The dashboard gets the API key from .env, so
# there is nothing to copy into frontend/.env.local. See docs/DEMO.md.
DEMO_REVIEWER ?= qa-demo
demo: stack
	@test -d frontend/node_modules || (cd frontend && npm ci)
	@key=$$(sed -n 's/^SAWTI_API_KEY=//p' .env); \
	test -n "$$key" || { echo "SAWTI_API_KEY is not set in .env"; exit 1; }; \
	echo "dashboard: http://localhost:5173   API docs: http://localhost:8000/docs   Langfuse: http://localhost:3000"; \
	trap 'kill 0' INT TERM; \
	$(MAKE) --no-print-directory asr-worker & \
	(cd frontend && VITE_SAWTI_API_KEY="$$key" VITE_SAWTI_REVIEWER_ID=$(DEMO_REVIEWER) npm run dev) & \
	wait

# ASR latency and real-time factor per language, on Phase 3 audio (eval_results.md).
asr-latency:
	uv run python scripts/measure_asr_latency.py

# The OpenAPI schema stage B generates its types from.
openapi:
	uv run python scripts/export_openapi.py

# The generated endpoint/schema lists in docs/06-API-SPECIFICATION.md, from openapi.json.
api-doc:
	uv run python scripts/render_api_doc.py

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
