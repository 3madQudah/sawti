# 08-ROADMAP

Sawti is a self-improving agent for bilingual (Arabic / English / code-switched)
contact center call analysis. It ingests a call, extracts what was promised and
what went wrong, grounds every claim in a verbatim quote, scores the call against
a fixed QA rubric, and routes anything it is not confident about to a human.

Every phase reports its results **per language category** — `ar`, `en`, `mixed` —
never as one blended number.

---

## Status note — numbering reconciled 2026-09-22

This file used to number Phase 3 *Service* (FastAPI/Postgres/Celery, not
started) and Phase 5 *Deployment* (containerization, Postgres checkpointing,
Langfuse, and the ASR front end). The ASR front end was actually built and
labelled `phase-3` throughout the code and decision records
(`sawti.asr`, `docs/09-DECISIONS.md`, `eval_results.md`), which disagreed
with this file. Resolved in favor of what the code already says: **audio is
Phase 3**. Service folds into Phase 5, next to the other not-yet-started
infrastructure work it belongs with. See `docs/09-DECISIONS.md` for the
decision record.

---

## Phase 0 — Foundations ✅

Schemas, configuration, and the LLM provider abstraction.

`sawti.schemas` is the source of truth for every value crossing a node boundary,
and it encodes the project's hard rules structurally: every `Claim` carries a
mandatory non-empty `evidence` quote, and `CallAnalysis` refuses to exist with a
confidence below threshold unless `requires_human_review` is set.

**Done when:** every output shape is a validated Pydantic model, and no module
reads `os.environ` directly.

---

## Phase 1 — Data and baseline ✅

150 synthetic Jordanian contact-center transcripts (60 `ar`, 30 `en`, 60
`mixed`), reference labels, a plain-prompt baseline extractor, and the
per-language evaluation harness.

**Known limitation, disclosed:** the reference labels are LLM-generated, not
human-reviewed. Phase 1's accuracy and rubric-agreement figures measure agreement
between two LLM-driven processes, not accuracy against human judgment. Grounding
metrics are the exception — they check predictions against the transcripts
themselves. See `docs/09-DECISIONS.md`.

**Done when:** a baseline number exists for every metric, per language.

---

## Phase 2 — The grounded agent ✅

A deterministic LangGraph pipeline that makes "an unsupported claim is rejected"
a property of the system rather than a hope about the model.

```
START → extract → ground → score → compliance → assess_confidence
                                                   │
                                   confidence ≥ threshold ──→ END
                                   confidence < threshold ──→ escalate → (interrupt)
```

- **extract** — the only node that talks to a model. Redacts PII first, then
  asks for claims with verbatim evidence.
- **ground** — verifies every claim's quote against the transcript and computes
  its real offsets. Unsupported claims are **dropped**, never down-weighted.
- **score** — reconciles rubric scores against the seven canonical criteria.
- **compliance** — a documented passthrough; see `docs/09-DECISIONS.md`.
- **assess_confidence** — `confidence = grounding_coverage`, compared against a
  named, configurable threshold.
- **escalate** — suspends the graph with `interrupt()`, resumable via the
  checkpointer. An escalated run cannot reach the automated exit.

**Done when:** unsupported claims measurably drop versus the phase 1 baseline,
per language. Results in `eval_results.md`.

---

## Phase 3 — Audio 🟡

The ASR front end (`sawti.asr`) so the system takes audio rather than
transcripts. Built 2026-09-22.

- [x] Synthesized audio corpus for all 150 calls, two distinct voices per call,
      with an exact speaker-per-segment manifest (`data/audio/manifest.json`).
      No recorded audio existed, so it had to be generated first.
- [x] Whisper transcription with `language` **forced**, not auto-detected
      (`sawti.asr.transcribe`), plus a measured forced-vs-auto-detect
      comparison on the `mixed` category.
- [x] WER measured per language category, normalized and raw
      (`sawti.eval.metrics.wer_by_category`).
- [x] Extraction accuracy delta, clean text vs ASR text, per language
      (`sawti.eval.experiments.asr_propagation`) — the phase's headline number.
- [~] Speaker diarization (`sawti.asr.diarization`): **implemented and unit
      tested, but never run.** pyannote's pretrained pipelines are gated on
      HuggingFace and no `HF_TOKEN` exists. Needs licence acceptance on
      `pyannote/segmentation-3.0` *and* `pyannote/speaker-diarization-3.1`.
      See `docs/09-DECISIONS.md`.

**Done when:** a call's audio can be transcribed, forced per language
category, and scored (WER, extraction-accuracy delta) against the same
per-language harness as every other phase. Diarization is implemented but
its "done when" (verified against ground truth) is blocked on `HF_TOKEN`.

---

## Phase 4 — Memory and self-improvement 🔮

The part that makes the system *self-improving*: reviewer corrections are
captured, induced into general rules, checked for conflicts, stored in a vector
index, and retrieved to inform later extractions. Rules earn or lose standing
based on whether they help.

The `extract` node has a marked, tested absence where retrieved rules will be
injected.

**Carried debt this phase must clear first:** `sawti.db.session` is still an
unimplemented stub, and correction capture persists via it. Tracked in
`docs/09-DECISIONS.md`.

**Done when:** five successive batches with memory on beat the same five with
memory off, per language.

---

## Phase 5 — Deployment and service 🔮

Everything needed to run this as a real deployed system rather than a
collection of eval scripts: FastAPI service, Postgres persistence, Celery
workers, the review API that lets a human resume an interrupted run,
containerization, Postgres-backed LangGraph checkpointing, and observability
via Langfuse.

**Carried debt this phase must clear:** the graph's checkpointer is
process-local `MemorySaver` — a review queue that evaporates on restart.
Tracked in `docs/09-DECISIONS.md`.

**Done when:** a call can be submitted over HTTP, analyzed asynchronously,
escalated, reviewed by a human, and resumed — with the paused state surviving
a process restart — all running in containers with Langfuse observability.

---

## Standing rules

These do not change between phases:

1. Every agent output is a validated Pydantic schema.
2. Every claim is grounded in a verbatim quote. An unsupported claim is
   rejected outright — never softened, never down-weighted.
3. Low-confidence cases route to a human. Never an automated verdict.
4. No code without a corresponding pytest test.
5. PII redaction runs before any text reaches a model.
6. Arabic, English, and code-switched results are reported separately,
   everywhere.
