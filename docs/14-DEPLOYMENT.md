# 14 — Deployment (local only)

Sawti is **not deployed**, and nothing in the repo deploys it. This file
explains how to run it on one machine and why it stops there. For the
component layout see [04-ARCHITECTURE.md](04-ARCHITECTURE.md). For a demo
walkthrough see [DEMO.md](DEMO.md).

## Prerequisites

- An Apple Silicon Mac. The ASR worker uses mlx-whisper on Metal. Every
  measurement in `eval_results.md` ran on an M1 with 8 GB.
- Docker Desktop, [`uv`](https://docs.astral.sh/uv/), Python 3.11, Node 20+,
  `ffmpeg`.
- A Hugging Face token with the terms of `pyannote/segmentation-3.0` and
  `pyannote/speaker-diarization-3.1` accepted. Without it, audio uploads fail
  at diarization. Transcript submission still works.
- A Gemini API key, unless you use `SAWTI_LLM_PROVIDER=fake`.

## Commands

| Command | What it does |
|---|---|
| `make install` | `uv sync --all-extras --dev` |
| `make up` | Postgres and Redis only, for tests and scripts on the host. |
| `make stack` | Builds and starts the `service` profile: `migrate` (one-shot), `api`, `worker`, `beat`, `langfuse`, and waits until healthy. |
| `make asr-worker` | The native ASR worker (`celery -A sawti.api.asr_tasks worker -Q asr --pool=solo`). Needs Postgres and Redis up. |
| `make demo` | `make stack`, then runs the ASR worker and the dashboard dev server together. Ctrl-C stops those two and leaves the stack running. |
| `make e2e` | Runs the stack on a scratch database (`sawti_e2e`) with the fake provider, runs `tests/e2e`, then puts the stack back on the dev database. |
| `make down` | Stops everything. |
| `make openapi`, `make api-doc` | Regenerate `frontend/openapi.json` and the generated part of [06-API-SPECIFICATION.md](06-API-SPECIFICATION.md). |

URLs while it runs: dashboard <http://localhost:5173>, API docs
<http://localhost:8000/docs>, Langfuse <http://localhost:3000>.

The first audio upload downloads the Whisper and pyannote weights (a few
GB). The Results page needs `data/serving_benchmark/comparison.json` and
`data/memory_comparison/report.json`, which are both tracked in git.

## Environment variables

Copy `.env.example` to `.env` and fill it in. The names are listed below,
never their values. The defaults are in `src/sawti/config.py`.

| Group | Variables |
|---|---|
| App | `SAWTI_ENV`, `SAWTI_LOG_LEVEL` |
| LLM provider | `SAWTI_LLM_PROVIDER` (`anthropic` \| `vllm` \| `gemini` \| `fake`), `SAWTI_LLM_MODEL`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `SAWTI_GEMINI_MODEL`, `VLLM_BASE_URL`, `VLLM_MODEL` |
| Routing | `SAWTI_CONFIDENCE_THRESHOLD` (default 0.7) |
| Postgres | `DATABASE_URL`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `SAWTI_SERVICE_DB` |
| Redis / Celery | `REDIS_URL`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND` |
| API | `SAWTI_API_KEY` (required, or every protected route returns 503), `SAWTI_API_HOST`, `SAWTI_API_PORT`, `SAWTI_CORS_ORIGINS` |
| Tasks | `SAWTI_TASK_MAX_RETRIES`, `SAWTI_TASK_RETRY_BACKOFF_SECONDS`, `SAWTI_TASK_RETRY_BACKOFF_MAX_SECONDS`, `SAWTI_REAPER_INTERVAL_SECONDS`, `SAWTI_REAPER_QUEUED_TIMEOUT_SECONDS`, `SAWTI_REAPER_PROCESSING_TIMEOUT_SECONDS` |
| Memory | `SAWTI_MEMORY_TOP_K`, `SAWTI_WORKER_PRELOAD_EMBEDDINGS` |
| Audio | `SAWTI_AUDIO_DIR`, `SAWTI_MAX_AUDIO_BYTES`, `SAWTI_KEEP_AUDIO`, `WHISPER_MODEL`, `WHISPER_LANGUAGE`, `WHISPER_LANGUAGE_OVERRIDES`, `HF_TOKEN` |
| Fake provider | `SAWTI_FAKE_LLM_LATENCY_SECONDS`, `SAWTI_FAKE_LLM_UNGROUNDED_CLAIMS` |
| Langfuse | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (tracing is off unless both are set), `LANGFUSE_HOST`, `LANGFUSE_NEXTAUTH_SECRET`, `LANGFUSE_SALT`, `LANGFUSE_ENCRYPTION_KEY`, `LANGFUSE_INIT_USER_EMAIL`, `LANGFUSE_INIT_USER_PASSWORD` |
| Fine-tuning | `SAWTI_FINETUNE_BASE_MODEL` (read only by `scripts/train_qlora.py`) |

`make demo` reads `SAWTI_API_KEY` from `.env` and passes it to the dashboard
as `VITE_SAWTI_API_KEY`, which puts it in the browser bundle (see
[13-SECURITY.md](13-SECURITY.md)).

## Self-hosted vLLM (Kaggle, benchmark only)

The 6.3 self-hosted numbers ran Qwen3-8B on vLLM 0.18.1 in a free Kaggle
notebook (2× T4) with tensor parallelism 2. This was a benchmark, not
serving for the stack.

1. `uv run python scripts/build_kaggle_bundles.py --kaggle-user <username>`
   writes three private datasets to `dist/kaggle/`: the adapter, the 45
   redacted comparison calls, and a snapshot of `src/sawti`.
2. Upload them to Kaggle and run `notebooks/kaggle_vllm_benchmark.ipynb`.
   It starts `vllm serve` with `--enable-lora` and the T4 settings recorded in
   [eval_results.md](../eval_results.md) (2026-10-10).
3. Download `sawti_kaggle_results.json` and score it locally:
   `uv run python scripts/assemble_serving_comparison.py --kaggle sawti_kaggle_results.json`.

To point the service at any OpenAI-compatible vLLM server, set
`SAWTI_LLM_PROVIDER=vllm`, `VLLM_BASE_URL` and `VLLM_MODEL`. The stack doesn't
start one, and this configuration has not been measured inside the stack.

## Why it isn't deployed

- **It was decided.** The 6.4 deliverable is a local demo video
  ([09-DECISIONS.md](09-DECISIONS.md), 2026-10-10).
- **No real auth.** A shared key that ships in the browser bundle is
  acceptable only on localhost.
- **ASR needs Metal.** mlx-whisper can't run in a Linux container, and CPU
  Whisper was the ~0.1× real-time path that was rejected. A server deployment
  would need a GPU ASR path, which doesn't exist.
- **Redaction isn't good enough for real calls.** It is regex-only, with no
  names or addresses ([13-SECURITY.md](13-SECURITY.md)).
- **Provider quotas.** The free Gemini tier is 15 requests/min and 500/day,
  which stopped several runs ([09-DECISIONS.md](09-DECISIONS.md),
  2026-09-25 incidents).
- **Ports and credentials.** Compose binds on all interfaces with default
  database credentials. That is fine for a laptop and not for a server.
