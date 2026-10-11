# 03 — Technical specification

How a call becomes a routed analysis. For component layout see
[04-ARCHITECTURE.md](04-ARCHITECTURE.md). For why each choice was made, see
[09-DECISIONS.md](09-DECISIONS.md).

## Pipeline

```
audio ─→ ASR (Whisper, forced language) ─→ diarization (pyannote) ─→ role labelling ─┐
                                                                                      ├→ redaction ─→ LangGraph agent
transcript (POST /calls) ─────────────────────────────────────────────────────────────┘

LangGraph agent:  extract → ground → score → compliance → assess_confidence ─┬→ END (completed)
                                                                             └→ escalate → interrupt() (awaiting_review)
```

| Stage | Module | What it does |
|---|---|---|
| Decode | `sawti.asr.pipeline.load_audio` | Uses ffmpeg to convert the file to 16 kHz mono float32. |
| ASR | `sawti.asr.transcribe` | Runs mlx-whisper `large-v3-turbo` with the language **forced**: `en` → `en`, `ar` and `mixed` → `ar` (`Settings.whisper_language_for`). It never auto-detects. |
| Diarization | `sawti.asr.diarization` | pyannote `speaker-diarization-3.1` with 2 speakers. Each Whisper segment takes the speaker it overlaps most, and a segment that overlaps nothing takes the nearest turn. |
| Role labelling | `assign_roles_by_first_speaker` | Labels the first speaker `Agent` and everyone else `Customer`. This is a convention, not detection, and it is wrong for any call the customer opens. |
| Redaction | `sawti.privacy.redaction.redact`, called from `sawti.api.ingest` | Regex replacement with `[EMAIL]`, `[IBAN]`, `[CARD]`, `[NATIONAL_ID]`, `[PHONE]`. Only the redacted text is stored or put in graph state. |
| `extract` | `sawti.agent.nodes.extract` | The only node that calls an LLM. It folds the retrieved memory rules into the system prompt and asks for an `ExtractionProposal` with verbatim evidence. A transient provider error is re-raised so Celery retries it. Any other error is recorded in state. |
| `ground` | `sawti.agent.nodes.ground` | Finds each claim's quote verbatim in the redacted transcript and computes the offsets itself. Claims that don't match are **dropped** into `rejected_claims`, and non-verbatim sentiment points are dropped as well. `grounding_coverage = survived / total` (1.0 when there are no claims). |
| `score` | `sawti.agent.nodes.score` | Reconciles rubric scores against the 7 canonical criteria (`RUBRIC_CRITERIA`). It never invents a score for a missing criterion. |
| `compliance` | `sawti.agent.nodes.compliance` | Passes state through unchanged (documented in [09-DECISIONS.md](09-DECISIONS.md), 2026-09-22). |
| `assess_confidence` | `sawti.agent.nodes.confidence` | Sets `confidence = grounding_coverage`. A run with an `error` scores 0.0. A call auto-passes only if `confidence ≥ threshold` (default 0.7). |
| `escalate` | `sawti.agent.nodes.escalate` | Calls `interrupt()`. The Postgres checkpointer persists the paused run, and `POST /reviews/{id}/corrections` resumes it. |

**What grounding does not check.** Grounding verifies that the quote exists.
It does not check that the quote supports the claim, or that the transcript is
usable. `call_0003_mixed` came back from ASR garbled and still grounded at 1.00
([eval_results.md](../eval_results.md), 2026-10-11). An ASR-quality routing
signal is planned, not built
([09-DECISIONS.md](09-DECISIONS.md), 2026-10-11 TODO).

**Diarization limits.** Speakers are attributed per Whisper segment, so a
segment that spans a speaker change goes entirely to one speaker. Diarization
quality has never been measured against the manifest's speaker timeline. The
Phase 3 propagation numbers used oracle speakers and are a lower bound on the
cost of ASR ([eval_results.md](../eval_results.md), 2026-09-25).

## Models

| Role | Model | Where it runs | Used in |
|---|---|---|---|
| ASR | mlx-whisper `large-v3-turbo` | Native on the Mac (Metal) | Phase 3 eval, ASR worker |
| Diarization | pyannote `speaker-diarization-3.1` (gated, needs `HF_TOKEN`) | Native on the Mac (CPU) | ASR worker (6.4) only |
| Extraction (default) | Gemini `gemini-flash-lite-latest`, which resolved to `gemini-3.5-flash-lite` in the 6.3 runs | Google API | Every phase, service |
| Extraction (self-hosted) | Qwen3-8B fp16, with and without the Phase 5 QLoRA adapter, on vLLM 0.18.1 | Kaggle T4×2 | 6.3 benchmark only |
| Fine-tuning | Qwen3-8B + QLoRA (4-bit NF4, r=16, α=32) | Colab T4 | Phase 5 |
| Rule embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers) | Docker worker (baked into the image) | Retrieval, conflict narrowing, consolidation |
| Offline stand-in | `sawti.llm.fake_provider` | Anywhere | Tests, `make e2e`, load measurement |

The provider is selected with `SAWTI_LLM_PROVIDER` (`anthropic` | `vllm` | `gemini` | `fake`).
No result in `eval_results.md` used the Anthropic provider.

## Schemas (`sawti.schemas`)

| Model | Key fields | Invariant enforced by the schema |
|---|---|---|
| `Language` | `ar`, `en`, `mixed` | The axis every metric is split on. |
| `CallStatus` | `transcribing`, `queued`, `processing`, `awaiting_review`, `reviewed`, `completed`, `failed` | Matches the `calls.status` CHECK constraint. |
| `Quote` | `text`, `speaker`, `start_char`, `end_char` | `end_char − start_char == len(text)`. |
| `Claim` | `evidence: Quote` | Evidence is mandatory and non-empty. |
| `Commitment(Claim)` | `promised_by`, `description`, `deadline?` | — |
| `ComplianceFlag(Claim)` | `rule_id`, `severity` (low…critical), `description` | — |
| `RubricScore(Claim)` | `criterion`, `score ∈ [0,1]`, `justification` | — |
| `SentimentPoint` | `quote`, `speaker`, `score ∈ [−1,1]`, `timestamp_sec` | Not a `Claim`, but grounded since 2026-10-07. |
| `SentimentTrajectory` | `points` | Points must be chronological. This is the most common terminal failure in Phase 5. |
| `ExtractionProposal` | summary and claim lists | What `extract` returns, before grounding. |
| `CallAnalysis` | claim lists, `confidence`, `requires_human_review` | `requires_human_review` must be true when `confidence < threshold`. |
| `Correction` | `original`, `corrected`, `error_location`, `note?`, `reviewer_id`, `timestamp` | Reviewer id and timestamp are mandatory. |

The API's own request and response models (`sawti.api.schemas`) are listed in
[06-API-SPECIFICATION.md](06-API-SPECIFICATION.md).

## Memory

- **Store.** Rules live in the `memory_rules` table. The worker rebuilds an
  in-process `MemoryStore` whenever the set of active rules changes, and
  retrieves the top 5 (`SAWTI_MEMORY_TOP_K`) against the redacted transcript.
- **Loop.** `sawti.memory.persistence.induce_and_store`: induce → conflict
  check → store or record a conflict → consolidate. A Postgres advisory lock
  serializes runs.
- **Consolidation.** Complete linkage at cosine ≥ 0.7. Single linkage, used
  until 2026-10-07, chained 81 rules into one
  ([eval_results.md](../eval_results.md), 2026-10-07 pre-fix and post-fix
  rounds).

## Retries and failure handling

- Transient provider errors (429, 408, 5xx, timeouts) put the call back in
  `queued` and retry it with exponential backoff (`SAWTI_TASK_MAX_RETRIES`,
  default 5). Once retries run out the call is `failed`.
- A non-transient extraction error escalates to a human (`analysis_error`)
  and produces an empty analysis record.
- A `beat` reaper re-enqueues calls stuck in `queued` or `processing` for 15
  minutes (`SAWTI_REAPER_*`).
- Every provider call has a 120 s deadline (`sawti.llm.provider.run_with_timeout`).
