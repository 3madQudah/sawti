# 11 — Checkpoint

**As of 2026-10-11, branch `improvements-7`.** Phases 0–6.4 are done (6.4 is
the local QA dashboard, merged to `main`). Phase 7 is pre-registered in
[08-ROADMAP.md](08-ROADMAP.md), and 7.1 is done. The phase history is in
[07-IMPLEMENTATION-PLAN.md](07-IMPLEMENTATION-PLAN.md).

## Checks (run 2026-10-11 on this branch)

| Check | Result |
|---|---|
| `make test` | 1,020 passed, 10 skipped, 2 deselected (`integration`, `asr_local`), after 7.1 and the UUID/timestamp redaction fix. Two solo runs; running two suites at once on the shared test database gives false failures. |
| `make lint` | clean |
| `make typecheck` | clean (86 files, strict) |
| Frontend `vitest run` | 40 passed |
| `make e2e`, Playwright, `-m asr_local` | not re-run for this checkpoint |

## What works

| Capability | Evidence |
|---|---|
| Transcript in → grounded, scored, routed analysis | Phase 2 round; `tests/agent/` |
| Audio in (wav/mp3/m4a) → native ASR + diarization → redacted transcript → analysis | `call_0003_mixed` uploaded through New call ([eval_results.md](../eval_results.md), 2026-10-11); `tests/api/`, `tests/asr/` |
| A low-confidence call pauses, survives a restart, and resumes after review | `tests/db/test_checkpointer.py`, `make e2e` |
| Corrections → induced rules → retrieved into later calls | `tests/api/test_memory_loop.py`; the 600-call memory comparison |
| Memory conflicts recorded and resolvable (4 actions); their rules held out of consolidation while open (7.1) | Migration 0005; legacy re-check: 3 conflicts, 8 inserted; regression tests in `tests/api/routes/test_memory.py` |
| Retries on transient provider errors, reaper for stuck calls | `tests/api/test_tasks.py` |
| Self-hosted Langfuse tracing with PII masking | `tests/observability/test_tracing.py` |
| Redaction of phone formats (3-3-4, 3-4, Jordanian, Arabic-Indic) and any 7+ digit run | 69 cases in `tests/privacy/test_redaction.py`. Verified live: `call_0006_en` re-uploaded with `[PHONE]` in place of the number, and the number is in neither the transcript nor its Langfuse trace ([eval_results.md](../eval_results.md), 2026-10-11) |
| Dashboard: Queue, Review, New call, Memory, Results, in EN and AR (RTL) | Vitest and Playwright against a mocked API; `make demo` |

## Headline numbers (per language)

| | ar | en | mixed | Source |
|---|---|---|---|---|
| ASR WER, normalized | 0.153 | 0.064 | **0.433** | [eval_results.md](../eval_results.md), 2026-09-25 |
| Rubric agreement lost to ASR (clean → ASR) | −0.037 | −0.012 | −0.149 | same, Phase 2 → 3 |
| Memory on − off, accuracy | +0.021 | +0.022 (within noise) | +0.042 | 2026-10-10, 600/600 |
| ASR real-time factor (M1, pooled) | 1.22 | 1.25 | 1.27 | 2026-10-11 A1 (n = 3 each) |
| Reviewer agreement | n = 0 | n = 0 | n = 0 | 2026-10-07 |
| PII spans caught on the Phase 3 ASR transcripts, before → after today's fix | 2 → 19 | 1 → 26 | 10 → 16 | 2026-10-11 (two rounds) |
| 7+ digit runs left unredacted, before → after | 17 → 0 | 25 → 0 | 6 → 0 | same |

## Known issues

| Issue | Impact | Status | Source |
|---|---|---|---|
| **Routing ignores ASR quality.** `call_0003_mixed` came back with Spanish, Japanese and Cyrillic fragments, grounded at 1.00, and auto-completed. | The worst transcripts are the least likely to reach a human. | Planned fix (Whisper scores + foreign-script ratio, thresholds measured per language). Not built. | [eval_results.md](../eval_results.md), [09-DECISIONS.md](09-DECISIONS.md), 2026-10-11 |
| **Grounding proves a quote exists, not that it supports the claim.** | "Skipped identity verification" was backed by `Give me a second, Bess.` | No fix designed. | same |
| ~~Consolidation can retire a rule an open conflict points at.~~ | It made `keep_new` and `discard` no-ops for 2 of the 3 conflicts recorded. (The UI never disabled those actions; an earlier version of this file said it did.) | **Fixed in 7.1.** Held rules are excluded from consolidation, and regression tests fail on the old code. The data check is vacuous: 0 open conflicts today. | [09-DECISIONS.md](09-DECISIONS.md), 2026-10-11 |
| **Redaction mangled UUIDs and ISO timestamps** (regression from the phone fix and catch-all). | Call ids in Langfuse metadata were corrupted (1 trace and its 6 observations), and the tracing test was flaky. | **Fixed** (UUIDs and timestamps are protected). Corpus redaction counts are unchanged. | [09-DECISIONS.md](09-DECISIONS.md), 2026-10-11 |
| **Diarization is unmeasured and misaligned.** Each Whisper segment goes wholly to one speaker, roles are first-speaker-is-agent, and accuracy has never been scored against the manifest timeline. On the uploaded `call_0006_en`, turns #3–#4 (and #7–#8) mix agent and customer speech. | A misattributed turn can move a commitment between agent and customer. The Phase 3 deltas used oracle speakers, so they are a lower bound. | Known issue, not fixed. Not measured. | [09-DECISIONS.md](09-DECISIONS.md), 2026-10-11; [eval_results.md](../eval_results.md), 2026-09-25 caveat 1 |
| **`mixed` WER 0.433**, and 4 of 150 calls hit Whisper repetition loops. | Code-switched calls are the weakest link at every stage. | No mitigation tried. | [eval_results.md](../eval_results.md), 2026-09-25 |
| **The live ASR path differs from Phase 3.** `call_0003_mixed` was clean in Phase 3 (WER 0.185) and garbled through New call. | Phase 3 WER doesn't guarantee live quality. | Cause not investigated. | [eval_results.md](../eval_results.md), 2026-10-11 |
| **Redaction is regex-only.** It doesn't catch personal names or addresses (no NER), or numbers shorter than 7 digits. | Not fit for real calls. | Known gap. | [13-SECURITY.md](13-SECURITY.md) |
| **PII leak found in production, fixed.** An uploaded `call_0006_en` kept a 3-3-4 phone number in its "redacted" transcript. It reached Gemini, the stored analysis, the checkpoint and one Langfuse trace. | The redact-before-model rule was broken. 11 stored `en` calls hold 14 such spans. | Phone patterns widened and a `[NUMBER]` catch-all added (`bfadca6`, `7f8cce9`), then verified live. Stored data deliberately not rewritten (synthetic). | [eval_results.md](../eval_results.md), [09-DECISIONS.md](09-DECISIONS.md), [13-SECURITY.md](13-SECURITY.md), 2026-10-11 |
| **Earlier PII bug, fixed.** Raw phone numbers survived inside `reviewer_actions` JSON. | They would have outlived transcript redaction. | Fixed in the 0001 conversion. Its "0 PII quotes left" check used the pre-fix patterns, so it undercounted. The raw backup dump remains on disk, gitignored. | [09-DECISIONS.md](09-DECISIONS.md), 2026-10-07 |
| **Not real auth.** A shared key ships in the browser bundle, and the reviewer id is self-asserted. | Local only. | By decision. | [09-DECISIONS.md](09-DECISIONS.md), 2026-10-06 |
| **Labels and audio are synthetic.** The reference labels are Gemini-generated and the audio is edge-tts. | Accuracy means agreement with Gemini, and WER is optimistic. | Disclosed in every round. | [09-DECISIONS.md](09-DECISIONS.md), 2026-09-21, 2026-09-22 |
| **The active rule set changed after the measurement.** The memory comparison used 35 frozen rules. The live store has 23 after the re-check's consolidation. | Nobody has measured whether 23 merged rules help as much. | Open. | [eval_results.md](../eval_results.md), 2026-10-11 |
| **Stale docs.** The `README.md` results table is empty and its provider list is outdated. [12-TESTING.md](12-TESTING.md) predates the service. The roadmap's markers lag. | Misleading for a new reader. | Not fixed in this pass. Only cross-links were changed in those files. | — |

## Next steps (from the decision log, in order)

1. Measure ASR-quality signals on the 150 Phase 3 calls, per language
   (high-WER recall and false-escalation rate), then add a
   `low_asr_quality` escalation ([09-DECISIONS.md](09-DECISIONS.md),
   2026-10-11).
2. ~~Exclude conflicted rules from consolidation.~~ Done (7.1). The 2
   affected conflicts had already been resolved. Next are Phase 7.2–7.4, as
   pre-registered in [08-ROADMAP.md](08-ROADMAP.md): transcript-quality routing,
   NER name redaction, and the NLI claim-support check.
3. Run the first real human reviews through the dashboard, so reviewer
   agreement has an n.
4. Score diarization against `data/audio/manifest.json`.
5. Proposed and not run: the `disable_any_whitespace` Kaggle re-run, and
   rebuilding the fine-tuning targets as full analyses
   ([eval_results.md](../eval_results.md), 2026-10-10 and 2026-10-06).
