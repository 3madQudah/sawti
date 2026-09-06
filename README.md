# Sawti

Sawti is a self-improving agent for bilingual (Arabic / English / code-switched)
contact center call analysis. It transcribes and diarizes call recordings,
extracts quote-grounded claims (commitments, sentiment, compliance flags, QA
rubric scores), routes low-confidence results to human review, and learns
from supervisor corrections over time via an induced-rule memory system.

Every agent output is a validated Pydantic model. Every extracted claim
carries a verbatim quote from the transcript — a claim without a quote is
invalid, not "low confidence." Evaluation is always reported per language
category (`ar`, `en`, `mixed`), never as a single blended number.

## Setup

Requires Python 3.11 and [`uv`](https://docs.astral.sh/uv/).

```bash
cp .env.example .env        # fill in API keys / DB config
make install                # uv sync --all-extras --dev
make up                     # start postgres + redis (docker compose)
make test                   # run the test suite
make lint                   # ruff check
make typecheck              # mypy
```

Run the API locally:

```bash
uv run uvicorn sawti.api.main:create_app --factory --reload
```

## Repo map

```
src/sawti/
├── config.py         # pydantic-settings, loads .env
├── schemas.py         # source of truth for all output shapes
├── llm/                # provider abstraction (Anthropic / vLLM, swappable by config)
├── privacy/            # PII redaction, runs before any model call
├── data/                # synthetic call generation, ground truth loading
├── asr/                 # Whisper transcription + speaker diarization
├── agent/               # LangGraph state, graph assembly, and nodes
│   └── nodes/            # extract -> ground -> score -> compliance -> confidence -> escalate
├── memory/              # capture, induce, store, conflict-check, consolidate, retire rules
├── eval/                 # per-language-category metrics, runner, experiments
├── db/                    # SQLAlchemy models (incl. agents — records, not logins) + sessions
└── api/                    # FastAPI app, routes, Celery tasks

docs/                # numbered project docs (product, technical, architecture, ...)
tests/                # mirrors src/sawti/ exactly
data/                  # synthetic/, ground_truth/, audio/ (gitignored contents)
scripts/              # one-off operational scripts
notebooks/           # exploratory notebooks
```

See `docs/` for the full product and technical specification, roadmap, and
architectural decisions. `eval_results.md` at the repo root is written by
the eval runner (`sawti.eval.runner`) — not by hand.

## Results

Populated by `sawti.eval.runner`. See `eval_results.md` for the full history.

| Metric | ar | en | mixed |
| --- | --- | --- | --- |
| Accuracy | | | |
| Rubric agreement | | | |
| Grounding precision | | | |
| Compliance recall | | | |
