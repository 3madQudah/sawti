# Project audit

This is a self-review of Sawti as of 2026-10-11 (`phase-6.4`), written for
someone deciding how much to trust it. Every number links to its source. For
the full list of open issues, see [11-CHECKPOINT.md](11-CHECKPOINT.md).

## What this is, in one paragraph

Sawti is a bilingual (Arabic / English / code-switched) call-analysis
pipeline. It runs ASR, PII redaction and a fixed LangGraph agent that keeps
only claims it can quote verbatim. It routes low-confidence calls to a human
and learns rules from reviewer corrections. It also includes a FastAPI and
Celery service, a durable review queue, self-hosted tracing, a cloud vs
self-hosted serving comparison, and a React review dashboard in EN and AR.
**It runs locally on synthetic data, and no real reviewer has used it.**

## What's strong

- **The core rules are built into the structure.** A `Claim` can't exist
  without evidence. `CallAnalysis` refuses to exist below the threshold unless
  it is flagged for review. `ground` deletes claims whose quote isn't in the
  transcript, and runs before anything reads them. The router defaults to
  "escalate" when state is missing. The pipeline is a fixed graph rather
  than a ReAct agent, so the model can't skip verification
  ([09-DECISIONS.md](09-DECISIONS.md), 2026-09-22).
- **Measurement discipline.** Every result is per language, with n,
  hardware and method. Superseded numbers stay in place, marked "pre-fix" or
  "preliminary". The memory verdict rule was written down before the
  memory-on data existed ([eval_results.md](../eval_results.md), 2026-10-10).
- **Measuring found real bugs, and they were fixed:**
  - Forcing Arabic decoding on English audio. Fixing it halved `en` WER, from
    0.130 to 0.064, and recovered one call from 1.000 to 0.024 (2026-09-25).
  - Single-linkage consolidation chaining 81 rules into one. Fixing it took
    the store from 9 to 35 active rules (2026-10-07).
  - A held-out eval comparing mismatched denominators (2026-09-27).
  - Grounding metrics checking raw text while `ground` checked redacted text
    (2026-10-07).
  - A Phase 4 verdict that compared memory's own trend instead of the
    memory − control delta (2026-09-25).
- **Privacy by design, within its limits.** Redaction happens at the
  boundary. The raw transcript is never stored or put in graph state (which
  the checkpointer persists). Raw audio is deleted after ASR. Langfuse
  payloads are re-redacted, and a test pushes known PII through the pipeline.
- **A production leak was handled the way the rest of the project works:**
  found, measured, fixed, verified and written down the same day
  ([eval_results.md](../eval_results.md), 2026-10-11).
  - Root cause: both phone patterns needed a `+`/`00` prefix or a leading `0`.
  - Measured per language on redaction alone, with no ASR or LLM, against the
    pre-fix code loaded from git.
  - Fixed with 61 new test cases.
  - Re-measured. Leftover 7+ digit runs on the ASR transcripts went from
    17 / 25 / 6 to 0 / 0 / 0 (ar / en / mixed).
  - Verified live: the re-uploaded call shows `[PHONE]`, and its trace doesn't
    contain the number.
- **Service engineering.**
  - An idempotent `queued → processing` claim.
  - Transient provider errors are retried. Other failures go to a human.
  - A reaper for lost messages.
  - A restart test proving that a paused review survives the process.
  - The container e2e runs on a scratch database.
  - The API image carries no torch.
  - The API types are generated, with staleness tests.
  - 1,009 Python tests, strict mypy, 40 frontend unit tests
    ([15-CONTRIBUTING.md](15-CONTRIBUTING.md)).
- **The bilingual UI is done properly.** Full RTL, per-turn text direction
  independent of the UI language, logical CSS only, and an EN/AR string parity
  test.

## What's weak

- **The labels aren't ground truth.** The reference labels are
  Gemini-generated and not human-reviewed
  ([09-DECISIONS.md](09-DECISIONS.md), 2026-09-21). "Accuracy" and "rubric
  agreement" mean agreement with an earlier Gemini run. That makes the
  cross-provider accuracy columns unfair by construction (the cloud arm gets
  0.86–0.93 and Qwen3 gets 0.52–0.63; [eval_results.md](../eval_results.md),
  2026-10-10).
- **Everything is synthetic.** That covers the transcripts, the edge-tts
  audio (clean, two voices, no channel noise), and every correction (diffs
  against those labels). Reviewer agreement is **n = 0**.
- **Grounding is weaker than it sounds.** It proves that a quote exists. It
  doesn't prove the quote supports the claim, or that the transcript is
  usable. `call_0003_mixed` came back from ASR with Spanish, Japanese and
  Cyrillic fragments, grounded at 1.00, and auto-completed with a claim backed
  by `Give me a second, Bess.` ([eval_results.md](../eval_results.md),
  2026-10-11). Routing uses grounding coverage alone, so the worst ASR failures
  are the least likely to reach a human.
- **The threshold is untested.** 0.7 escalated **0 of 150** calls on clean
  text and **4 of 30 `en`** calls (0 `ar`, 0 `mixed`) on ASR text
  ([eval_results.md](../eval_results.md), 2026-09-25). It has never been tuned
  against outcomes.
- **Code-switched audio is the weak point:**
  - `mixed` WER is 0.433, against 0.153 for `ar` and 0.064 for `en`.
  - Rubric agreement drops by 0.149 on `mixed` ASR text.
  - Repetition loops hit 4 of 150 calls, 3 of them `mixed`.
  - No mitigation has been tried.
- **Diarization is unmeasured.** Each Whisper segment is assigned wholly to
  one speaker. Roles come from "first speaker is the agent". Accuracy has never
  been scored against the manifest. The Phase 3 deltas used oracle speakers,
  so they are lower bounds.
- **Memory works only modestly:**
  - Accuracy +0.021 / +0.022 / +0.042 (ar / en / mixed), with `en` within noise.
  - The `ar` result contradicts Phase 4's larger effect.
  - Consolidation could retire a rule that an open conflict pointed at. This
    was fixed in Phase 7.1 by holding such rules out, with a regression test.
    It was found and fixed only after it had affected 2 of 3 conflicts.
  - The live store went from 35 to 23 rules after a re-check, and that set
    hasn't been measured
    ([eval_results.md](../eval_results.md), 2026-10-10, 2026-10-11).
- **Fine-tuning didn't pay off.**
  - On `ar`, n=10: no measurable change.
  - In the 6.3 benchmark, the adapter was within ±0.05 of the base model in
    every language, with 4 schema failures against 0 and 39% lower
    throughput.
  - The cause is known: the training targets were single-field fragments
    taken from 18 synthetic examples, and the training text was raw rather
    than redacted ([eval_results.md](../eval_results.md), 2026-10-06,
    2026-10-10).
- **Security is demo-grade.**
  - The shared API key ships in the browser bundle, and the reviewer id is
    self-asserted.
  - Redaction is regex-only, with no personal names or addresses (no NER).
  - **The redactor leaked in production.** An uploaded call's phone number
    (`ddd-ddd-dddd`) passed redaction and reached Gemini, the stored analysis,
    the checkpoint and one Langfuse trace. The 0001 conversion's earlier "0 PII
    left" check used the same blind patterns, so it was wrong too. The data is
    synthetic, so the stored copies were deliberately left in place
    ([13-SECURITY.md](13-SECURITY.md)). The fix was a regex patch tuned to one
    synthetic corpus, and real callers will use formats that aren't in it.
  - Compose binds on all interfaces with default database credentials
    ([13-SECURITY.md](13-SECURITY.md)).
- **Hygiene.**
  - Commits are coarse, and one has a misleading subject: `0a7b2e5`, "tests
    for transcript parsing", contains the whole Phase 2 agent and the ASR
    code.
  - The README results table is empty, 12-TESTING predates the service, and
    the roadmap's status markers lag behind the code.
  - 28 files aren't `ruff format`-clean.
  - Nine placeholder skips tagged "phase 1" remain.
  - `78f0367` was committed without a full check run. The suite has since
    been run and is green.

## What the numbers do and don't show

| Claim | Supported? | Why |
|---|---|---|
| "Unsupported claims don't reach the output." | **Yes, by construction.** | Phase 2 output: 0.000 / 0.000 / 0.004 (pre-fix metric). The model still *proposes* unsupported claims at 1.2% (16 of 1,296), and the gate removes them ([eval_results.md](../eval_results.md), 2026-09-22). |
| "Extracted claims are correct." | **No.** | Grounding checks quote existence, and accuracy is measured against LLM labels. Nothing measures correctness against a human. |
| "ASR mostly hurts code-switched calls." | **Yes, as a lower bound.** | Rubric agreement −0.149 `mixed`, −0.037 `ar`, −0.012 `en`. That uses oracle speakers and clean synthetic audio, so real-world loss would be larger. |
| "Memory improves extraction." | **Partly.** | It is positive in every language and both repeats. `ar` and `mixed` exceed noise, `en` doesn't. The baseline is Gemini labels, and the rules come from synthetic corrections. |
| "Self-hosting is cost-competitive." | **On cost, at concurrency 16.** | Base Qwen3-8B on vLLM costs $3.76–4.34 per 1,000 calls, against $4.39–4.62 for Gemini, priced as if paid. Actual spend was $0. The rental price comes from an aggregator. T4 p50 latency is 200–350 s, so it is not interactive. Accuracy isn't comparable across providers. |
| "Fine-tuning helped." | **No.** | It was within noise on `ar`, n=10, and in 6.3 it was slower and less schema-valid than the base model. |
| "Human-in-the-loop works." | **The mechanism, yes. The practice, no.** | Interrupt, persist, resume and correct are tested end to end. Zero real reviews have happened, and the threshold is untuned. |
| "PII never reaches the model." | **Not until 2026-10-11, and only for structured identifiers.** | The pre-fix redactor let 3-3-4 / 3-4 phones and every non-phone-shaped 7+ digit run through: 17 / 25 / 6 such runs on the ASR transcripts. After the fix there are 0 / 0 / 0, verified on one live re-upload. Names are still sent: there is no NER. |
| "It handles real calls." | **Not shown.** | No recorded telephony audio was ever used. |

## What I would do next

These are recommendations from this review. Only items 1 and 2 are in the
decision log as planned work.

1. **Route on ASR quality.** Keep Whisper's `avg_logprob`,
   `compression_ratio` and `no_speech_prob`, and add a foreign-script ratio.
   Measure high-WER recall and false escalations per language on the 150
   known-WER calls before choosing thresholds
   ([09-DECISIONS.md](09-DECISIONS.md), 2026-10-11).
2. ~~**Fix the conflict/consolidation bug.**~~ Done in Phase 7.1: conflicted
   rules are excluded from consolidation, with regression tests.
3. **Get human labels for a per-language subset**, and use them to re-anchor
   accuracy, rubric agreement and the escalation threshold. Until then, every
   accuracy number is agreement with Gemini.
4. **Check that each quote supports its claim**, as a separate signal from
   quote existence, and feed it into routing.
5. **Score diarization** against `data/audio/manifest.json`, then rerun the
   propagation experiment with diarized speakers instead of oracle ones.
6. **Before anything leaves localhost:** real auth (OIDC, as decision 3
   anticipated), NER redaction, ports bound to `127.0.0.1`, and non-default
   credentials.
7. **Retrain the adapter or drop it.** If retrained, use full-analysis
   targets on redacted text. Otherwise serve the base model, which was faster
   and as accurate in 6.3.
8. **Test on real recorded calls**, even a few per language, because every
   current WER is optimistic.
