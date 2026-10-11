# 15 — Contributing

How to set up, test and commit. Code and docs conventions are in
[16-STYLEGUIDE.md](16-STYLEGUIDE.md). What the tests do and don't prove is in
[12-TESTING.md](12-TESTING.md).

## Setup

```bash
cp .env.example .env     # fill in values; see 14-DEPLOYMENT.md for the list
make install             # uv sync --all-extras --dev
make up                  # Postgres + Redis for the DB-backed tests
cd frontend && npm ci    # dashboard dependencies
```

Python 3.11 and [`uv`](https://docs.astral.sh/uv/). `uv.lock` pins every
version, and both Docker images install exactly the locked set.

## The rule: no code without a test

Every module ships with a test file that mirrors its path
(`src/sawti/agent/nodes/ground.py` → `tests/agent/nodes/test_ground.py`).
At minimum it covers the happy path, at least one failure path, and the
invariant the module exists to protect, asserted directly. Scripts in
`scripts/` are tested under `tests/scripts/`. Frontend code ships with a
Vitest test next to it, and each page has a Playwright spec in English and
Arabic.

Tests never call a live model. Every LLM caller is patched, and the service
tests set `SAWTI_LLM_PROVIDER=fake`, because a test once leaked about 4 real
Gemini requests ([09-DECISIONS.md](09-DECISIONS.md), 2026-10-07). Model
quality is measured by eval runs, which write to `eval_results.md`. Tests
don't measure it.

## Checks before a commit

| Check | Command | State at `1cf741f` (2026-10-11) |
|---|---|---|
| Python tests | `make test` | 934 passed, 10 skipped (9 placeholder skips tagged "phase 1", plus the NER test), 2 deselected |
| Lint | `make lint` (`ruff check .`) | clean |
| Types | `make typecheck` (`mypy src`, strict) | clean, 86 files |
| Frontend unit | `cd frontend && npm test` | 40 passed |
| Frontend types | `cd frontend && npm run typecheck` | also fails if `src/api/schema.d.ts` is stale |
| Frontend e2e | `cd frontend && npm run e2e` | Playwright against a mocked API. Not re-run for this table. |
| Container e2e | `make e2e` | Needs Docker. Not re-run for this table. |
| Real ASR | `uv run pytest -m asr_local` | Mac with `HF_TOKEN` only |

After changing an API route, run `make openapi api-doc` (and `npm run gen:api`
in `frontend/`). Tests fail if `frontend/openapi.json` or
[06-API-SPECIFICATION.md](06-API-SPECIFICATION.md) is stale.

**Formatting.** The repo is not `ruff format`-clean: 28 files would be
reformatted. Don't run `ruff format .`, because it rewrites unrelated files.
Format only the files you create, or files that are already clean.

## Results and decisions

- **Numbers go in `eval_results.md`**, as a new dated round per run, per
  language (`ar` / `en` / `mixed`) and never blended. Each round states
  n, hardware and method. Don't edit old rounds. Mark them superseded or
  "pre-fix" in place instead.
- **Decisions go in [09-DECISIONS.md](09-DECISIONS.md)**: what was decided,
  why, what it costs, and the alternatives rejected. Record a known bug as an
  open TODO with its planned fix, rather than leaving it out.
- Every number cited in a doc links to the round or data file it comes from.

## Commits

The convention, from `git log`:

- The subject names the phase and what changed:
  `Phase 6.4 stage B: dashboard Queue + Review (React/Vite, EN/AR, mocked-API tests)`,
  or `Fix …` for a fix.
- The body lists what changed, and states what was **not** verified when a
  full check wasn't run (see `78f0367`).
- Commits made with Claude Code end with a `Co-Authored-By:` trailer.
- Work goes on a phase branch (for example `phase-6.4`), not on `main`.
- Never commit `.env`, `data/uploads/`, `data/backups/`, audio, or anything
  holding raw transcripts. `.gitignore` covers these.
