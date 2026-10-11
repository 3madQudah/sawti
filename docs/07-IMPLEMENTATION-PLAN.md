# 07 — Implementation plan (as built)

This is what was built, in order, with dates from `git log`. The plan and its
done-when criteria are in [08-ROADMAP.md](08-ROADMAP.md). The roadmap's
status markers lag behind the code: Phase 4 still shows 🔮 and 6.4 shows 🔨.
This file reflects the repo as of commit `e0c9c5f` (2026-10-11).

Commits are coarse. Phase 2 and the Phase 3 ASR code landed in one commit
(`0a7b2e5`), and Phase 4 landed with the start of Phase 5 (`cc35314`). Where a
phase has no commit of its own, the dates of its rounds in
[`eval_results.md`](../eval_results.md) are given as well.

| Phase | Built | Key commits | Done-when (from the roadmap) | Outcome |
|---|---|---|---|---|
| 0 Foundations | 2026-09-06 | `dd9c1a4` | Every output shape is a validated Pydantic model, and no module reads `os.environ` directly. | Met. |
| 1 Data and baseline | 2026-09-06 → 09-21 | `61f19ca`, `38ce9f8`, `8cb51e2`, `3bfa0c4` | A baseline number for every metric, per language. | Met (2026-09-21/22 rounds). The reference labels are LLM-generated, not human-reviewed. |
| 2 Grounded agent | 2026-09-22 | `0a7b2e5` | Unsupported claims drop versus Phase 1, per language. | Met. 0.032/0.045/0.036 → 0.000/0.000/0.004 (ar/en/mixed). The drop is structural: `ground` deletes unsupported claims. |
| 3 Audio | 2026-09-22 → 09-25 | `0a7b2e5` (code), `cc35314` (rounds) | Audio transcribed with the language forced per category, and scored for WER and the extraction delta. | Met for ASR. WER 0.153/0.064/0.433. Diarization was not measured: it was gated in Phase 3, and in 6.4 it runs live but still isn't scored. |
| 4 Memory | 2026-09-22 → 09-25 | `8ead491`, `cc35314` | Memory on beats memory off over five batches, per language. | Met for `ar` only in Phase 4. A later 600-call re-run with corrected consolidation: +0.021/+0.022/+0.042, with `en` within noise (2026-10-10). |
| 5 Fine-tuning (QLoRA) | 2026-09-25 → 10-06 | `cc35314` … `8c9daca` | The adapter trains, the loss is recorded, the forgetting check is clean, and the held-out delta is measured. | Met. `ar` n=10: no measurable degradation, and no real gain. 18 synthetic training examples. |
| 6.1 Persistence | 2026-10-06 → 10-07 | `ecad741` | A paused review survives a process restart. | Met (`tests/db/test_checkpointer.py`). |
| 6.2 Service | 2026-10-07 | `ecad741`, `08581d2` | e2e over HTTP in containers, plus latency, throughput and retry numbers per language. | Met (`make e2e`, 2026-10-07 round). |
| 6.3 Observability + serving comparison | 2026-10-07 → 10-10 | `93af91f`, `cab12a5`, `55ae70f`, `46938c8`, `5e8c966` | Langfuse tracing, and a cloud vs self-hosted table (accuracy, cost, latency, egress). | Met (2026-10-10 rounds). |
| 6.4 QA dashboard (local only) | 2026-10-10 → 10-11 | `78f0367`, `d9df6de`, `f164b95`, `e0c9c5f` | Phase 6: a human reviews through the dashboard and the run resumes. | Built and tested against a mocked API, with `make demo` for a live run. No real-reviewer session is recorded in `eval_results.md`. |

## What each phase produced

- **0.** `sawti.schemas` (claims must carry evidence, low confidence forces
  human review) and `sawti.config`.
- **1.** 150 synthetic Jordanian calls (60 `ar`, 30 `en`, 60 `mixed`) from
  Gemini, a dialect and script check with 28 files regenerated, Gemini-generated
  reference labels with quote offsets computed rather than trusted, and a
  plain-prompt baseline plus a per-language metrics harness.
- **2.** A deterministic LangGraph pipeline (`extract → ground → score →
  compliance → assess_confidence → escalate`), regex PII redaction, and an
  in-process checkpointer.
- **3.** A TTS audio corpus (edge-tts, 352.6 min), forced-language Whisper,
  WER per category, and the propagation experiment with oracle speakers.
  The diarization code was written but couldn't run.
- **4.** Correction capture, rule induction, conflict check, embedding store,
  consolidation and lifecycle, plus the five-batch experiment.
- **5.** A fine-tuning dataset (22 examples, 15 calls held out), a QLoRA
  script, the Colab notebook, a forgetting check and the held-out eval.
- **6.1–6.2.** Alembic schema, Postgres checkpointer, FastAPI, Celery with
  retries and an idempotency guard, the reaper, containers, the durable memory
  loop, sentiment grounding and the metric fix.
- **6.3.** Self-hosted Langfuse, the vLLM provider, the Kaggle benchmark, the
  cloud vs self-hosted table, and the corrected memory comparison.
- **6.4.** Audio upload with a native ASR worker (pyannote runs for the first
  time), the conflicts table and resolution, `/stats`, `/results`, and the React
  dashboard (EN/AR) with `make demo`.

## Not done (open items from the decision log)

| Item | Status | Source |
|---|---|---|
| Route on ASR quality, not only grounding coverage | Planned, not built | [09-DECISIONS.md](09-DECISIONS.md), 2026-10-11 TODO |
| Keep consolidation from retiring rules that open conflicts point at | Known bug, not fixed | [09-DECISIONS.md](09-DECISIONS.md), 2026-10-11 TODO |
| `disable_any_whitespace` Kaggle re-run | Proposed, not run | [eval_results.md](../eval_results.md), 2026-10-10 |
| Rebuild fine-tuning targets as full analyses and retrain | Not done | [eval_results.md](../eval_results.md), 2026-10-06 |
| NER redaction (names, addresses) | Not built | [09-DECISIONS.md](09-DECISIONS.md), 2026-09-22 |
| Diarization accuracy against the manifest timeline | Not measured | [09-DECISIONS.md](09-DECISIONS.md), 2026-09-22 |
| Real authentication (OIDC) | Not built | [09-DECISIONS.md](09-DECISIONS.md), 2026-10-06 |
