# 08-ROADMAP

Sawti is a self-improving agent for bilingual (Arabic / English / code-switched)
contact center call analysis. It ingests a call, extracts what was promised and
what went wrong, grounds every claim in a verbatim quote, scores the call against
a fixed QA rubric, and routes anything it is not confident about to a human.

Every phase reports its results **per language category** — `ar`, `en`, `mixed` —
never as one blended number.

See also: [07-IMPLEMENTATION-PLAN.md](07-IMPLEMENTATION-PLAN.md) (phases as built, with dates),
[11-CHECKPOINT.md](11-CHECKPOINT.md) (current state and open issues).

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

## Status note — numbering reconciled again, 2026-09-25

QLoRA fine-tuning work started under an ad hoc "Phase 5" label in code
comments and `docs/09-DECISIONS.md` before this file had a matching phase —
this roadmap's own Phase 5 was still *Deployment*. Resolved by giving
fine-tuning an actual phase rather than leaving two different things
sharing one number: **fine-tuning is now Phase 5**, and deployment/service
(the note above) **renumbers to Phase 6**. Every existing "Phase 5" mention
of deployment across the codebase (comments, `docs/09-DECISIONS.md`'s prior
entries) was written when deployment held that number — see
`docs/09-DECISIONS.md` for the decision record and which references were
updated vs. left as historical.

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
      (`sawti.eval.metrics.wer_by_category`). Results: `ar` 0.153, `en` 0.064,
      `mixed` 0.433 normalized.
- [x] Forced language is per language category, not one global value
      (`Settings.whisper_language_overrides`). Forcing `ar` on English audio
      cost one call a total loss; `en` WER halved once corrected.
- [x] Extraction accuracy delta, clean text vs ASR text, per language
      (`sawti.eval.experiments.asr_propagation`) — the phase's headline number.
      Rubric agreement falls `mixed` −0.149, `ar` −0.037, `en` −0.012, tracking
      WER. **A lower bound**: speaker labels are oracle, so diarization error is
      not included.
- [x] The escalation path fired on real data for the first time (4 calls),
      having never triggered on clean text in phase 2.
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

## Phase 5 — Fine-tuning (QLoRA) ✅

Distills the Phase 4 finding — reviewer corrections clearly and
consistently help `ar` extraction, per the five-batch done-when result —
into the model itself via QLoRA fine-tuning, rather than relying solely on
retrieval at inference time. Started 2026-09-25.

- [x] Base model checkpoint confirmed explicitly, not guessed:
      `Qwen/Qwen3-8B` (instruct), wired as `Settings.finetune_base_model`
      (`SAWTI_FINETUNE_BASE_MODEL`). See `docs/09-DECISIONS.md`.
- [x] Fine-tuning dataset built (`scripts/build_finetune_dataset.py`): 22
      examples (18 train / 4 val) from 45 of the 60 `ar` calls, 15 held out
      before any training pair was built and reserved for the step-4
      comparison below. Every example is synthetic — manufactured from an
      agent-output-vs-ground-truth diff, never a real QA reviewer's
      judgment. See `docs/09-DECISIONS.md` and `eval_results.md`'s
      2026-09-25 entry.
- [x] QLoRA training script written, unit-tested, and **run for real on
      Colab (T4), 2026-09-27**: 3 epochs, eval loss 1.9185 → 1.7573 →
      1.7259, eval mean token accuracy 0.6018 → 0.6314 → 0.6356,
      `train_runtime` 1588.27s. Hyperparameters (4-bit NF4, LoRA r=16/α=32
      over every linear layer) are the QLoRA paper's documented defaults,
      not tuned against this dataset. See `docs/09-DECISIONS.md` and
      `eval_results.md`.
- [x] Catastrophic-forgetting check written, unit-tested, and **run for
      real on Colab, 2026-09-27**: 7/7 generic prompts, 0 flagged, base and
      tuned responses near-identical. Not a full eval suite — a handful of
      prompts checked for a large regression only.
- [x] **Held-out-calls accuracy delta — measured 2026-10-06** (corrected
      re-run, both models on the same 10-call intersection; supersedes the
      2026-09-27 preliminary numbers). `ar`, n=10: accuracy 0.640 → 0.652,
      rubric agreement 0.873 → 0.899, grounding precision 0.899 → 0.907,
      unsupported-claim rate 0.132 → 0.086. Extraction success 11/15 →
      13/15. Read as **no measurable degradation**, not an improvement —
      n=10, one run, synthetic training data. Main finding is qualitative:
      the tuned model drifts toward the training targets' single-field
      output shape (train/eval task mismatch), recovering via sampled
      retries. See `eval_results.md` (2026-10-06 round) and
      `docs/09-DECISIONS.md` (2026-10-06 entries).

**Done when:** the QLoRA adapter trains end to end on Colab, its train/val
loss is recorded, the forgetting check shows no large regression on
generic prompts, **and** the held-out 15 calls' accuracy delta (base vs.
tuned) is measured and recorded — all three done (2026-10-06). The
result validates the fine-tuning *mechanism*; its effect on `ar` accuracy
is within noise at this sample size — 18 training examples, all
synthetic, so even a positive delta is a small-sample signal, not a scaled
result. See `docs/09-DECISIONS.md`.

---

## Phase 6 — Deployment and service 🔮

Everything needed to run this as a real deployed system rather than a
collection of eval scripts: FastAPI service, Postgres persistence, Celery
workers, the review API that lets a human resume an interrupted run,
containerization, Postgres-backed LangGraph checkpointing, and observability
via Langfuse — **plus how it's actually served and reviewed by a person**:
`sawti.llm.vllm_provider` self-hosted serving as an alternative to the
cloud provider (the `LLMProvider` abstraction already exists for exactly
this swap — see `README.md`), a QA review dashboard (React, RTL/LTR — the
human-facing surface the review API above has no UI for yet), and a
measured comparison of the two serving modes: accuracy, cost per 1000
calls, latency, and data egress, cloud vs. self-hosted. Folded into this
phase rather than split into a separate one — vLLM serving and the review
dashboard are both "how this runs as a deployed system," the same question
the rest of the phase already answers, and the comparison table is this
phase's own done-when measurement once both serving modes exist. See
`docs/09-DECISIONS.md`, 2026-09-27.

**Carried debt this phase must clear:** ~~the graph's checkpointer is
process-local `MemorySaver` — a review queue that evaporates on restart.~~
✅ **Cleared in 6.1 (2026-10-06):** Postgres checkpointer
(`sawti.db.checkpointer`, `langgraph-checkpoint-postgres==2.0.21`); a run
interrupted for review in one process is resumed and finished by another.
Tracked in `docs/09-DECISIONS.md`.

**Progress:**
- ✅ 6.1 — persistence and durable human-in-the-loop: Alembic schema,
  session factory, Postgres checkpointer, restart test.
- ✅ 6.2 — async service in containers (2026-10-07): FastAPI review API,
  Celery worker with retries and an idempotency guard, `api` / `worker` /
  `migrate` containers; e2e passes over HTTP in containers; latency,
  throughput, retry/failure rates, image sizes and cold start recorded per
  language in `eval_results.md`.
- ✅ 6.3 — observability and cloud vs self-hosted comparison:
  ✅ self-hosted Langfuse v2 tracing every node and LLM call, no raw PII;
  ✅ cloud vs self-hosted table per language (free Kaggle T4×2, vLLM 0.18.1,
  Gemini free tier; every number's method, n and hardware stated —
  `eval_results.md`, 2026-10-10); ✅ corrected memory comparison (600/600): memory helps in every
  language; vs Phase 4 the hypothesis is confirmed for mixed, inconclusive
  for en, contradicted for ar (`eval_results.md`, 2026-10-10) Proposed follow-up:
  a `disable_any_whitespace` Kaggle run.
- 🔨 6.4 — QA dashboard, local only (demo video; `docs/09-DECISIONS.md`,
  2026-10-10). Stage A, backend (built, awaiting review): audio upload with a
  native ASR worker on the Mac (`make asr-worker`, `asr` queue), recorded and
  resolvable memory conflicts, `/stats`, `/results`, `frontend/openapi.json`.
  Stage B, the React dashboard (English + Arabic RTL), starts after the
  mockups are approved.

**Done when:** a call can be submitted over HTTP, analyzed asynchronously,
escalated, reviewed by a human through the dashboard, and resumed — with
the paused state surviving a process restart — all running in containers
with Langfuse observability, **and** a comparison table exists showing
cloud vs. self-hosted (vLLM) across accuracy, cost per 1000 calls,
latency, and data egress.

---

## Phase 7 — Improvements 🔨

Four fixes for weaknesses recorded in Phases 3–6, done one at a time on the
`improvements-7` branch. **Pre-registered on 2026-10-11, before any
implementation or measurement.** Each item's metric, its per-language
breakdown, and the bars for success, inconclusive and failure are fixed here.
A later change to any of them must be logged as a dated amendment below the
item, with the reason, *before* the measurement it affects.

Common rules:

- Results go per language (`ar`, `en`, `mixed`) with n, in
  `eval_results.md`.
- A language whose n is below the stated minimum is **inconclusive** for that
  language. It never counts as success or failure.
- No Gemini calls unless an item says so.

### 7.1 — Consolidation must not retire a rule an open conflict points at ✅

- **Changes.** Consolidation (`induce_and_store`, `consolidate_store`) leaves
  out every active rule named by an unresolved `memory_conflicts` row. Such a
  rule becomes eligible again once its conflict is resolved. This fixes the
  2026-10-11 TODO in `docs/09-DECISIONS.md`.
- **Metric.**
  1. A regression test where a conflicted rule would otherwise cluster. It
     must fail on the pre-fix code and pass after.
  2. A data check on the dev database: open conflicts that name a retired
     rule, out of n open conflicts.
- **Per language.** Open conflicts are counted by the language of their
  source correction.
- **Success.** The regression test fails before and passes after, the full
  suite, ruff and mypy are green, and the data check finds 0 open conflicts
  naming a retired rule.
- **Inconclusive.** The data check has n = 0 open conflicts, so it says
  nothing. The tests alone carry the result. (Known at registration time: the
  3 conflicts recorded so far are all resolved, so n = 0 today.)
- **Failure.** The regression test can't be made to pass, or any open
  conflict names a retired rule after the fix.
- **Result (2026-10-11): inconclusive on data, supported by tests.**
  - Both regression tests fail on the pre-fix code and pass after.
  - The data check found 0 of n = 0 open conflicts naming a retired rule, so
    it is vacuous.
  - See `eval_results.md` and `docs/09-DECISIONS.md`, 2026-10-11.

### 7.2 — Transcript-quality signal in routing

- **Changes.** A quality score computed from the stored redacted transcript
  only, with no ASR re-run. A call below the threshold escalates to human
  review with a new escalation kind (`low_transcript_quality`), even when
  grounding coverage passes. Signals:
  - *Foreign-script ratio.* Letters outside the scripts the call's language
    allows (Arabic + Latin for `ar` and `mixed`, Latin for `en`) ÷ all
    letters.
  - *Repetition.* The zlib compression ratio of the transcript text, which is
    the repetition-loop signal.
  - *Language mismatch* vs `language_hint`. For an `en` call, the share of
    Arabic letters. For an `ar` call, the share of Latin letters. Not defined
    for `mixed`.
- **Data.** The 150 Phase 3 ASR transcripts with known WER
  (`data/asr_transcripts/wer_report.json`). WER bands at registration
  (high = WER > 0.4, low = WER < 0.2):

  | | high | mid | low |
  |---|---|---|---|
  | ar | 4 | 4 | 52 |
  | en | 1 | 4 | 25 |
  | mixed | 27 | 30 | 3 |

- **Split.** The calls are split in half, stratified by language × WER band,
  with seed `20261011`. The split file is committed **before any signal is
  computed**. Thresholds are chosen on the selection half only:
  - a call is flagged if any signal crosses its threshold;
  - maximize `mixed` high-WER recall, subject to a false-escalation rate
    ≤ 0.10 on low-WER `ar` calls and on low-WER `en` calls;
  - break ties by the fewest flagged calls.

  Every result is reported on the held-out half.
- **Metric**, per language on the held-out half:
  - recall of high-WER calls;
  - false-escalation rate on low-WER calls;
  - the escalation rate on mid-band calls, reported descriptively.
- **Minimum n.** Recall needs ≥ 5 held-out high-WER calls, so only `mixed`
  qualifies; `ar` (about 2) and `en` (0–1) are reported descriptively. The
  false-escalation rate needs ≥ 10 held-out low-WER calls, so `ar` and `en`
  qualify and `mixed` (1–2) is descriptive.
- **Success.** Held-out `mixed` recall ≥ 0.60, **and** false escalation
  ≤ 0.10 in both `ar` and `en`.
- **Failure.** `mixed` recall < 0.30, **or** false escalation > 0.20 in `ar`
  or `en`.
- **Inconclusive.** Anything in between.
- **Also reported, not part of the verdict.** Whether the live garbled
  `call_0003_mixed` transcript from the dashboard (2026-10-11) is flagged.

### 7.3 — Name redaction with a local NER model

- **Changes.** A local, free, CPU-only multilingual NER model (ar + en) adds
  a `[NAME]` placeholder for person names, after the regex patterns. The first
  candidate is `Davlan/xlm-roberta-base-ner-hrl`. Its licence and CPU latency
  are checked before adoption, and a model change is logged as an amendment
  before measurement.
- **Gold.** A list of every personal-name occurrence in the 150 clean
  transcripts (`data/synthetic`). It is built from candidate lines (pattern
  search, then a read of each candidate line) and committed, with its hash,
  **before the model is first run on the corpus**. No structured name data
  exists in the generator, so this list *is* the ground truth. It can be
  checked by a reviewer.
- **Metric**, per language:
  - *recall* = gold name occurrences replaced by `[NAME]` ÷ gold occurrences;
  - *false redactions* = `[NAME]` spans that are not gold names, as a count
    and as a share of all `[NAME]` spans;
  - stored claims whose evidence quote the new redactor would change, out of
    all stored claims;
  - CPU time per call.

  ASR transcripts are reported descriptively (`[NAME]` spans per call). They
  have no gold, because ASR spells names inconsistently.
- **Minimum n.** ≥ 20 gold name occurrences per language.
- **Success.** Recall ≥ 0.90 **and** false-redaction share ≤ 0.20, in every
  language that meets the minimum n.
- **Failure.** Recall < 0.70, **or** false-redaction share > 0.40, in any
  language.
- **Inconclusive.** Anything in between.

### 7.4 — Claim-support check with a local NLI model

- **Changes.** A local, free multilingual NLI model scores whether each
  grounded claim is supported by its evidence. The first candidate is
  `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`. The premise is
  the full transcript turn(s) holding the evidence quote. The hypothesis is the
  claim's text: the commitment description, the flag description, or the
  rubric justification. **No Gemini.** This item only measures. Whether a low
  score should route to a human is decided after the result.
- **Labels.** 60 claims, 20 per language, drawn at random with seed
  `20261011` from all stored analyses. There are no hand-picked additions,
  and `call_0003_mixed`'s claims enter only if drawn. The labeling file has
  claim, quote, context turns and an empty label column (`supported` /
  `not_supported` / `unclear`). It is prepared by Claude and **labeled by the
  user**. The model's scores aren't shown to the labeler, and they are
  computed only after the labels are frozen.
- **Metric**, per language: precision and recall for detecting
  `not_supported`, at a decision rule fixed now. A claim is predicted
  `not_supported` when the model's entailment probability is < 0.5. A
  precision-recall curve is reported alongside it. `unclear` labels are left
  out and counted.
- **Minimum n.** ≥ 5 `not_supported` labels in a language. Below that, the
  language is inconclusive, and its prevalence is reported.
- **Success.** Recall ≥ 0.70 **and** precision ≥ 0.50 in every language that
  meets the minimum n.
- **Failure.** Recall < 0.40, **or** precision < 0.25, in any language that
  meets it.
- **Inconclusive.** Anything in between, or no language meeting the minimum n.

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
