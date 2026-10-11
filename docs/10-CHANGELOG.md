# 10 — Changelog

Built from `git log`, grouped by phase and newest first. Each phase's numbers
are in [`eval_results.md`](../eval_results.md) and its reasoning in
[09-DECISIONS.md](09-DECISIONS.md). Some commit subjects undersell what they
contain. Those commits have a note.

## Phase 6.4 — QA dashboard, local only (2026-10-10 → 10-11)

- `e0c9c5f` 2026-10-11. Review actions are locked when the call isn't
  `awaiting_review`. Adds `make demo` and [DEMO.md](DEMO.md). Records the
  finding that `call_0003_mixed` came back garbled but grounded at 1.00, and a
  TODO to route on ASR quality.
- `f164b95` 2026-10-11. Dashboard pages New call, Memory and Results (EN/AR,
  mocked-API tests, Playwright).
- `d9df6de` 2026-10-11. Dashboard Queue and Review (React 18, Vite, typed
  client generated from `openapi.json`, RTL), plus structured `EscalationInfo`.
- `78f0367` 2026-10-11. Backend: `POST /calls/audio` with a native ASR worker
  (mlx-whisper + pyannote). Migration 0005 (`transcribing`, `failure_reason`,
  `memory_conflicts`). Conflict resolution, `/stats`, `/results`, and
  `torchaudio` pinned to 2.4. ASR RTF is 1.22–1.27. The 11 legacy conflicts were
  re-checked: 3 are still conflicts and 8 were inserted, and the run's
  consolidation took the active rules from 35 to 23. The commit message notes
  that the full test, lint and typecheck run wasn't repeated on its final state.

## Phase 6.3 — Observability and serving comparison (2026-10-07 → 10-10)

- `5e8c966` 2026-10-10. Final cloud vs self-hosted table (Gemini vs vLLM on
  Kaggle T4×2), and the final memory comparison (600/600).
- `46938c8` 2026-10-10. Records `finish_reason`, adds a per-call timeout,
  makes the throughput calculation account for host sleep, adds a failure
  report, and pre-registers the memory verdict rule.
- `55ae70f` 2026-10-10. Kaggle cell forces `TRITON_ATTN` on the T4.
- `cab12a5` 2026-10-10. Gemini cloud arm 45/45. Memory comparison partial.
- `93af91f` 2026-10-07. Langfuse v2 tracing with PII masking, the vLLM
  provider, the Kaggle benchmark harness, and the memory comparison harness.

## Phase 6.1–6.2 — Persistence and service (2026-10-06 → 10-07)

- `08581d2` 2026-10-07. Consolidation fixed to use complete linkage. Single
  linkage had chained 81 rules into one, and the store went from 9 to 35 active
  rules.
- `ecad741` 2026-10-07. Postgres checkpointer, Alembic 0001–0004, FastAPI,
  Celery with retries and an idempotency guard, the reaper, containers, the
  durable memory loop, sentiment grounding, grounding metrics re-scored against
  redacted text, and the dev database converted to 0001.

## Phase 5 — Fine-tuning (2026-09-25 → 10-06)

- `8c9daca` 2026-10-06. Corrected held-out eval, base vs tuned (`ar`,
  n=10).
- `ce611be` 2026-10-06. Relaxed the `numpy` pin to `<3`, because `<2.0` broke
  Colab on Python 3.13.
- `b7cad1b`, `b31d5ba`, `837d03b` 2026-10-02. Notebook fixes: an import smoke
  test, `--no-deps` on reinstall, and per-session fixes made permanent.
- `ee12ea6`, `92404ab` 2026-09-28. Pre-fix held-out numbers recorded as
  preliminary. Three Colab issues documented.
- `a0ed242`, `874ae60`, `f5c0169`, `33b0eb1`, `26b45a5` 2026-09-27. Fixed the
  held-out eval's denominators, and the eval scoring zero calls. Roadmap
  numbering reconciled. Real Colab training and forgetting results. Finetune
  stack re-pinned.
- `115b285` 2026-09-25. `requires-python` widened to `<3.14` for Colab.
- `cc35314` 2026-09-25. Fine-tuning dataset, QLoRA script, forgetting check.
  *This commit also contains the Phase 4 memory pipeline
  (`sawti.memory.*`), the five-batch experiment, and the Phase 3 eval
  rounds.*

## Phase 4 — Memory (2026-09-22 → 09-25)

- `8ead491` 2026-09-22. Synthetic corrections generator.
- The memory modules and the five-batch result landed in `cc35314` (above).

## Phases 2 and 3 — Grounded agent and audio (2026-09-22)

- `0a7b2e5` 2026-09-22. *The subject says "tests for transcript parsing and
  ASR experiments", but the commit contains the LangGraph pipeline (all six
  nodes), redaction, the ASR modules (`transcribe`, `diarization`), audio
  generation, and the Phase 2 eval round.*

## Phase 1 — Data and baseline (2026-09-06 → 09-21)

- `3bfa0c4` 2026-09-21. LLM-generated reference labels and the plain-prompt
  baseline.
- `8cb51e2` 2026-09-08. Ground-truth authoring tooling: the rubric, 150 blank
  templates, and the quote-offset finder. It was never used for human labels.
- `38ce9f8` 2026-09-06. Fixed dialect and script violations (28 files
  regenerated).
- `61f19ca` 2026-09-06. Gemini provider and 150 synthetic calls (40/20/40).

## Phase 0 — Foundations (2026-09-06)

- `cb119fd` 2026-09-06. `.gitignore`.
- `dd9c1a4` 2026-09-06. Scaffold, config and Pydantic schemas.

## Docs

- `09a529f` 2026-10-11. Filled docs 01–04.
