# Eval Results

Written by `sawti.eval.runner`. Do not hand-edit — append a new dated round
per run. Every metric is reported per language category; never a single
blended number across `ar` / `en` / `mixed`.

<!-- TEMPLATE — copy the block below for each new round, filled in by the runner -->

## Round: YYYY-MM-DD — <experiment name>

- Ground truth: `<path>`
- Agent config / model: `<provider>` / `<model>`
- Memory: `<on|off>`
- N calls: ar=`<n>`, en=`<n>`, mixed=`<n>`

| Metric | ar | en | mixed |
| --- | --- | --- | --- |
| Accuracy | | | |
| Rubric agreement | | | |
| Grounding precision | | | |
| Unsupported claim rate | | | |

## Round: 2026-09-21 — phase 1 baseline (plain-prompt extraction, no memory)

> **Reference labels for this run are LLM-generated, not human-reviewed — see docs/09-DECISIONS.md.** These figures measure agreement between two LLM-driven processes, not accuracy against human judgment. Grounding precision is the exception: it is checked against the transcripts themselves and needs no reference labels.

- Ground truth: `data/ground_truth` (LLM-generated reference labels, `human_reviewed: false`)
- Agent config / model: `gemini` / `gemini-flash-lite-latest`
- Extraction: plain-prompt baseline (`sawti.eval.plain_extraction`), no memory
- Memory: `off`
- N calls scored: ar=`60`, en=`30`, mixed=`60` (total `150`)

| Metric | ar | en | mixed |
| --- | --- | --- | --- |
| Accuracy | 0.893 | 0.858 | 0.874 |
| Rubric agreement | 0.954 | 0.954 | 0.910 |
| Grounding precision | 0.965 | 0.961 | 0.971 |

## Round: 2026-09-22 — phase 1 baseline (plain-prompt extraction, no memory)

> **Reference labels for this run are LLM-generated, not human-reviewed — see docs/09-DECISIONS.md.** These figures measure agreement between two LLM-driven processes, not accuracy against human judgment. Grounding precision is the exception: it is checked against the transcripts themselves and needs no reference labels.

- Ground truth: `data/ground_truth` (LLM-generated reference labels, `human_reviewed: false`)
- Agent config / model: `gemini` / `gemini-flash-lite-latest`
- Extraction: `phase 1 baseline (plain-prompt extraction, no memory)`
- Memory: `off`
- N calls scored: ar=`59`, en=`29`, mixed=`60` (total `148`)
- Extraction failures: `2` call(s) produced no valid `CallAnalysis` and are excluded from every metric above (call_0060_ar, call_0101_en)

| Metric | ar | en | mixed |
| --- | --- | --- | --- |
| Accuracy | 0.886 | 0.859 | 0.868 |
| Rubric agreement | 0.952 | 0.946 | 0.910 |
| Grounding precision | 0.974 | 0.957 | 0.973 |
| Unsupported claim rate | 0.032 | 0.045 | 0.036 |

## Round: 2026-09-22 — phase 2 grounded agent (LangGraph, grounding gate, no memory)

> **Reference labels for this run are LLM-generated, not human-reviewed — see docs/09-DECISIONS.md.** These figures measure agreement between two LLM-driven processes, not accuracy against human judgment. Grounding precision is the exception: it is checked against the transcripts themselves and needs no reference labels.

- Ground truth: `data/ground_truth` (LLM-generated reference labels, `human_reviewed: false`)
- Agent config / model: `gemini` / `gemini-flash-lite-latest`
- Extraction: `phase 2 grounded agent (LangGraph, grounding gate, no memory)`
- Memory: `off`
- N calls scored: ar=`60`, en=`30`, mixed=`60` (total `150`)

| Metric | ar | en | mixed |
| --- | --- | --- | --- |
| Accuracy | 0.866 | 0.922 | 0.858 |
| Rubric agreement | 0.962 | 0.960 | 0.933 |
| Grounding precision | 0.990 | 1.000 | 0.986 |
| Unsupported claim rate | 0.000 | 0.000 | 0.004 |

### Phase 1 → Phase 2 comparison

Both rounds above ran on 2026-09-22, over the same 150-call reference set, with
the same model and the same metric definitions. The phase 1 baseline was
**re-run rather than quoted** from 2026-09-21, so the unsupported-claim rate is
computed by the identical formula on both sides. (The re-run reproduced the
original baseline closely — accuracy `ar` 0.886 vs 0.893 — which is the
consistency check that makes the comparison trustworthy.)

**Unsupported claim rate — the phase 2 criterion. Lower is better.**

| | ar | en | mixed |
| --- | --- | --- | --- |
| Phase 1 baseline | 0.032 | 0.045 | 0.036 |
| Phase 2 grounded agent | **0.000** | **0.000** | **0.004** |
| Change | −100% | −100% | −89% |

It drops in all three language categories. Grounding precision rose alongside it
(`ar` 0.974→0.990, `en` 0.957→1.000, `mixed` 0.973→0.986) and rubric agreement
improved slightly in all three (`ar` 0.952→0.962, `en` 0.946→0.960,
`mixed` 0.910→0.933).

#### How to read this, honestly

**The near-zero is structural, not a better model.** `ground` drops claims whose
evidence does not verify, so unsupported claims *cannot* reach the output. A
pipeline that rejects bad claims and one that never produces them both score
0.000 here. What actually improved is what downstream consumers receive, which
is the thing phase 2 set out to fix — not the model's honesty.

The grounding gate's own counts separate the two:

| | ar | en | mixed | total |
| --- | --- | --- | --- | --- |
| Claims proposed | 525 | 248 | 523 | 1296 |
| Claims rejected | 5 | 1 | 10 | 16 |
| Rejection rate | 0.010 | 0.004 | 0.019 | 0.012 |

So the model proposed 1296 claims and 16 of them (1.2%) were unsupported. That
1.2% is the model's real error rate on this data, and it is unchanged by phase 2
— the gate removes those claims rather than preventing them. `mixed` is the
worst category at 1.9%, roughly double `ar` and five times `en`, which is the
per-language signal worth carrying into phase 4.

#### Three caveats

1. **`mixed` does not reach exactly 0.000, and the residual 0.004 is most likely
   a redaction artifact rather than a leaked bad claim.** `ground` verifies
   quotes against the *redacted* transcript the model saw, while this metric
   checks the *raw* transcript on disk. A quote spanning a redacted phone number
   is verbatim in the former and not in the latter. Supporting evidence: 20% of
   `mixed` transcripts contain redactable PII versus 3% of `en` — and `en`
   scored exactly 0.000. This was not traced to the individual claim, so treat
   it as the likely explanation, not a confirmed one.

2. **Zero calls escalated.** At `confidence_threshold = 0.7` and a ~1% rejection
   rate, no call came close to the threshold, so the human-in-the-loop path
   never fired on real data. It is tested end to end (interrupt, checkpoint,
   resume) but this run is not evidence that it behaves well under load, and it
   is not evidence that 0.7 is the right threshold — only that it is not being
   hit. Tuning it needs the escalation outcomes phase 4 produces.

3. **Accuracy moved unevenly**: `en` 0.859→0.922, `ar` 0.886→0.866,
   `mixed` 0.868→0.858. The `ar` and `mixed` declines are small and expected in
   direction — dropping unsupported claims can remove a commitment or compliance
   flag that happened to agree with the reference label, and `accuracy_by_category`
   rewards that agreement regardless of whether the claim was supported. A
   correct rejection can therefore cost accuracy. Worth watching, not worth
   reversing.

Phase 1 also had 2 extraction failures (`call_0060_ar`, `call_0101_en`, excluded
from its metrics); phase 2 had none, so it scored 150 calls against phase 1's 148.

## Round: 2026-09-23 — phase 3 grounded agent on ASR transcripts (oracle speakers)

> **Reference labels for this run are LLM-generated, not human-reviewed — see docs/09-DECISIONS.md.** These figures measure agreement between two LLM-driven processes, not accuracy against human judgment. Grounding precision is the exception: it is checked against the transcripts themselves and needs no reference labels.

- Ground truth: `data/ground_truth` (LLM-generated reference labels, `human_reviewed: false`)
- Agent config / model: `gemini` / `gemini-flash-lite-latest`
- Extraction: `phase 3 grounded agent on ASR transcripts (oracle speakers)`
- Memory: `off`
- N calls scored: ar=`58`, en=`28`, mixed=`57` (total `143`)
- Extraction failures: `7` call(s) produced no valid `CallAnalysis` and are excluded from every metric above (call_0060_ar, call_0061_mixed, call_0064_en, call_0065_en, call_0066_ar, call_0067_mixed, call_0068_mixed)

| Metric | ar | en | mixed |
| --- | --- | --- | --- |
| Accuracy | 0.882 | 0.862 | 0.781 |
| Rubric agreement | 0.925 | 0.890 | 0.784 |
| Grounding precision | 0.997 | 0.992 | 0.983 |
| Unsupported claim rate | 0.000 | 0.000 | 0.002 |

## Round: 2026-09-25 — phase 3 ASR quality (WER, forced language, per category)

Word error rate of `sawti.asr.transcribe` against the reference transcripts the
audio was synthesized from. Not an extraction metric — this is the input
quality that the propagation round below is explained by.

- Audio: `data/audio` (150 synthesized calls, 352.6 min), manifest-linked to
  `data/synthetic` references. Synthetic TTS speech, not recorded telephony —
  see `docs/09-DECISIONS.md` for what that does and does not represent.
- ASR: `mlx-whisper` / `large-v3-turbo` on an 8 GB M1, language **forced**
  per category (`ar`→`ar`, `en`→`en`, `mixed`→`ar`).
- Normalization: headline is orthographically normalized (diacritics, alef and
  ta-marbuta folds, digits, punctuation, word-internal apostrophes); raw is the
  same pairs unnormalized. Rule in `docs/09-DECISIONS.md`.
- N: ar=`60`, en=`30`, mixed=`60` (total `150`). Transcription failures: `0`.
- Throughput: 352.6 min of audio in 118.3 min wall clock — **2.98x realtime**.
- Cost: zero. TTS (`edge-tts`) and ASR (local MLX) are both free; no metered
  API was used for audio.

| Metric | ar | en | mixed |
| --- | --- | --- | --- |
| **WER, normalized** | **0.153** | **0.064** | **0.433** |
| WER, raw | 0.328 | 0.094 | 0.572 |
| WER median | 0.145 | 0.021 | 0.371 |
| WER max | 0.512 | 0.651 | 1.000 |

**`mixed` is 2.8x worse than `ar` and 6.8x worse than `en`.** That gap is the
headline ASR finding and it is not an artifact of aggregation — the medians
(0.371 / 0.145 / 0.021) show the same ordering as the means.

**Normalization matters most where the writing system has the most freedom.**
Raw-to-normalized: `ar` 0.328→0.153 (−53%), `mixed` 0.572→0.433 (−24%), `en`
0.094→0.064 (−32%). Over half the apparent Arabic error was orthography —
hamza seating, ta-marbuta, absent diacritics — not misrecognition. An
unnormalized Arabic WER would have overstated the error by a factor of two.

**Four of 150 calls (2.7%) hit Whisper repetition loops** — degenerate output
where one token is 25-64% of the transcript. Three are `mixed`, one was `en`
(fixed, see below). `mixed`'s WER max of 1.000 is one such call
(`call_0095_mixed`). No mitigation has been tried; carried debt.

### Forced `ar` vs forced `en` on the English calls

The first corpus run forced `ar` globally, sending English audio through the
Arabic decoder. Re-transcribing the 30 `en` calls with `en` forced:

| | en WER (normalized) | en WER (raw) | median | max |
| --- | --- | --- | --- | --- |
| Forced `ar` (global) | 0.130 | 0.166 | 0.031 | **1.000** |
| Forced `en` (per category) | **0.064** | **0.094** | 0.021 | 0.651 |

**Halved.** The single biggest contributor was `call_0070_en` —
**1.000 → 0.024**, a 615-word call that had come back as Arabic
repetition-loop gibberish. Four others improved materially:
`call_0142_en` 0.662→0.043, `call_0098_en` 0.398→0.009,
`call_0128_en` 0.252→0.049, `call_0046_en` 0.172→0.011.

**It is not uniformly better: 5 of 30 calls got slightly worse**, the worst
being `call_0063_en` (+0.124) and `call_0072_en` (+0.102). So forcing `en` is
a large net win driven by removing catastrophic failures, not a uniform
improvement on every call. `ar` and `mixed` were not re-transcribed and their
numbers above are unchanged.

## Round: 2026-09-25 — phase 3 grounded agent on ASR transcripts (corrected `en`)

> **Reference labels for this run are LLM-generated, not human-reviewed — see docs/09-DECISIONS.md.** These figures measure agreement between two LLM-driven processes, not accuracy against human judgment. Grounding precision is the exception: it is checked against the transcripts themselves and needs no reference labels.

Supersedes the `en` column of the 2026-09-23 round above, which was scored
against transcripts produced with `ar` forced globally. The `ar` and `mixed`
columns are **carried forward unchanged** from that round: their transcripts
are byte-identical before and after the forcing fix, and the metrics group by
language independently, so re-extracting them could not have changed anything.

- Ground truth: `data/ground_truth` (LLM-generated reference labels, `human_reviewed: false`)
- Agent config / model: `gemini` / `gemini-flash-lite-latest`
- Extraction: phase 2 grounded agent, unchanged, over ASR transcripts
- Speaker labels: **oracle** (manifest ground truth), not diarization — see below
- Memory: `off`
- N calls scored: ar=`58`, en=`30`, mixed=`57` (total `145`)

| Metric | ar | en | mixed |
| --- | --- | --- | --- |
| Accuracy | 0.882 | 0.911 | 0.781 |
| Rubric agreement | 0.925 | 0.948 | 0.784 |
| Grounding precision | 0.997 | 0.963 | 0.983 |
| Unsupported claim rate | 0.000 | 0.000 | 0.002 |

### Phase 2 → Phase 3: what the audio front end costs

**This is the phase's headline number.** Same agent, same model, same metrics,
same reference labels — the only thing that changed is that the transcript is
ASR output instead of the clean text.

| Metric | | ar | en | mixed |
| --- | --- | --- | --- | --- |
| **Accuracy** | clean text | 0.866 | 0.922 | 0.858 |
| | ASR text | 0.882 | 0.911 | **0.781** |
| | change | **+0.016** | −0.011 | **−0.077** |
| **Rubric agreement** | clean text | 0.962 | 0.960 | 0.933 |
| | ASR text | 0.925 | 0.948 | **0.784** |
| | change | −0.037 | −0.012 | **−0.149** |
| **Grounding precision** | clean text | 0.990 | 1.000 | 0.986 |
| | ASR text | 0.997 | 0.963 | 0.983 |
| | change | +0.007 | −0.037 | −0.003 |
| **Unsupported claim rate** | clean text | 0.000 | 0.000 | 0.004 |
| | ASR text | 0.000 | 0.000 | 0.002 |

**The degradation tracks WER, and it is concentrated in `mixed`.** Rubric
agreement falls 0.149 for `mixed` against 0.037 for `ar` and 0.012 for `en` —
the same ordering as WER (0.433 / 0.153 / 0.064). Code-switched audio is the
weakest link at every stage of this pipeline, and the audio front end widens
the gap rather than narrowing it.

**`en` barely degrades at all once the language is forced correctly**:
accuracy 0.922→0.911, rubric 0.960→0.948. At 0.064 WER the extractor hardly
notices. Note what this means for the earlier round: the apparent `en`
degradation there (accuracy 0.862) was mostly an artifact of the wrong forced
language, not a cost of ASR. Measuring the pipeline exposed a bug in the
pipeline's configuration, which is the main argument for having measured it.

**`ar` accuracy went up (+0.016), which is noise, not improvement.** Three
reasons to read it that way: it is smaller than the phase 1 baseline's own
run-to-run variation (`ar` 0.893 vs 0.886 on identical inputs), rubric
agreement for the same category moved the other way (−0.037), and this column
is scored on 58 calls against phase 2's 60. Nothing here supports a claim that
ASR text helps Arabic extraction.

### The grounding gate caught more, and the human-review path fired for the first time

| | ar | en | mixed |
| --- | --- | --- | --- |
| Claims proposed, clean | 525 | 248 | 523 |
| Claims proposed, ASR | 496 | 253 | 499 |
| Claims rejected, clean | 5 | 1 | 10 |
| Claims rejected, ASR | 8 | **22** | 14 |
| Rejection rate, clean | 0.010 | 0.004 | 0.019 |
| Rejection rate, ASR | 0.016 | **0.087** | 0.028 |

The gate rejected more in every category, which is the architecture behaving
correctly: ASR text breaks exact quote matching, unsupported claims are dropped
rather than softened, and the unsupported-claim rate stays at 0.000/0.002 even
though the model proposed more bad claims.

**Escalations: 0 of 150 on clean text, 4 of 30 `en` calls on ASR text.** Phase
2's writeup noted that the human-in-the-loop path had never fired on real data,
so the 0.7 threshold was untested in practice. It fires now
(e.g. `call_0034_en` at confidence 0.500). ASR noise breaks grounding, coverage
falls below the threshold, and the call routes to a human instead of receiving
an automated verdict. That is the first real-data evidence that the escalation
path and the threshold do something.

**One result here is genuinely unexplained.** `en` has the *lowest* WER (0.064)
but by far the *highest* rejection rate (0.087, versus 0.028 for `mixed` at
0.433 WER). Recognition quality and grounding failure are not tracking each
other, and that inversion is not accounted for. A plausible mechanism — the
extractor selecting longer English quote spans, where any single ASR slip
breaks a verbatim match, while Arabic quotes are shorter — was **not tested**.
Treat it as an open question, not a finding.

### Caveats, in order of how much they should change your reading

1. **Speaker labels are oracle, so this delta is a lower bound.** Labels come
   from the audio manifest's true speaker timeline, not from diarization,
   deliberately: the clean transcripts are speaker-labelled, so unlabelled ASR
   text would have confounded losing the labels with mis-recognizing the words.
   Real deployment stacks diarization error on top, and a mis-attributed turn
   can move a commitment from the customer to the agent — exactly what the
   rubric is sensitive to. **Diarization was never run** (gated pyannote
   weights, no `HF_TOKEN`), so that second arm is unmeasured.

2. **The denominators do not match: ar=58 and mixed=57 against phase 2's 60/60.**
   Seven calls were lost to a Gemini free-tier `429 RESOURCE_EXHAUSTED` window
   between 09:29 and 09:58 on 2026-09-23 (`call_0060_ar`, `call_0061_mixed`,
   `call_0064_en`, `call_0065_en`, `call_0066_ar`, `call_0067_mixed`,
   `call_0068_mixed`). The two `en` calls were recovered by the `en` re-run; the
   five `ar`/`mixed` ones were not, because `run_eval` holds predictions in
   memory and writes only the metrics table, so recovering them would have
   meant re-extracting all 120. The exclusions are quota-driven and occurred in
   arbitrary call-id order, so they are effectively random with respect to
   content rather than a biased subset — but they are a real mismatch. Carried
   debt: persist predictions in `run_eval`.

3. **The audio is synthetic.** No channel noise, no codec artifacts, no
   overlapping speech, one voice per role across the whole corpus. Every WER and
   every delta above is optimistic relative to real call-centre audio.

4. **Reference labels remain LLM-generated.** Unchanged from phase 1, and it
   applies to accuracy and rubric agreement here exactly as it did there.

## Round: 2026-09-25 — phase 4 five-batch memory experiment (memory-on vs. no-memory control)

> **Reference labels for this run are LLM-generated, not human-reviewed — see docs/09-DECISIONS.md.**

- Ground truth: `data/ground_truth` (LLM-generated reference labels, `human_reviewed: false`)
- Agent config / model: `gemini` / `gemini-flash-lite-latest`
- Extraction: phase 2 grounded agent (LangGraph, grounding gate); memory-on arm retrieves top-5 rules per call and captures corrections between batches, control arm runs identically with memory retrieval and capture both off
- Batches: 5, sequential, each language spread evenly across all of them
- Batch composition: batch 1 (ar=`12`, en=`6`, mixed=`12`); batch 2 (ar=`12`, en=`6`, mixed=`12`); batch 3 (ar=`12`, en=`6`, mixed=`12`); batch 4 (ar=`12`, en=`6`, mixed=`12`); batch 5 (ar=`12`, en=`6`, mixed=`12`)

### ar — Accuracy per batch

| Batch | Memory | Control | Δ |
| --- | --- | --- | --- |
| 1 | 0.922 | 0.907 | +0.015 |
| 2 | 0.960 | 0.890 | +0.070 |
| 3 | 0.925 | 0.917 | +0.007 |
| 4 | 0.943 | 0.883 | +0.060 |
| 5 | 0.909 | 0.780 | +0.129 |

Memory trend: **falling** (0.922 → 0.909). Control trend: **falling** (0.907 → 0.780). **Δ (memory − control) trend: rising** (+0.015 → +0.129), beats control every batch.

- Rubric agreement (first batch → last batch): memory 0.987 → 0.927, control 0.983 → 0.905
- Grounding precision (first batch → last batch): memory 0.975 → 0.988, control 0.987 → 0.989
- Unsupported claim rate (first batch → last batch): memory 0.000 → 0.000, control 0.000 → 0.000

### en — Accuracy per batch

| Batch | Memory | Control | Δ |
| --- | --- | --- | --- |
| 1 | 0.857 | 0.858 | -0.001 |
| 2 | 0.892 | 0.967 | -0.075 |
| 3 | 0.920 | 0.921 | -0.001 |
| 4 | 0.942 | 0.913 | +0.029 |
| 5 | 0.945 | 0.873 | +0.072 |

Memory trend: **rising** (0.857 → 0.945). Control trend: **rising** (0.858 → 0.873). **Δ (memory − control) trend: rising** (-0.001 → +0.072), does not beat control every batch.

- Rubric agreement (first batch → last batch): memory 0.952 → 0.962, control 0.939 → 0.967
- Grounding precision (first batch → last batch): memory 1.000 → 1.000, control 1.000 → 1.000
- Unsupported claim rate (first batch → last batch): memory 0.000 → 0.000, control 0.000 → 0.000

### mixed — Accuracy per batch

| Batch | Memory | Control | Δ |
| --- | --- | --- | --- |
| 1 | 0.853 | 0.839 | +0.013 |
| 2 | 0.894 | 0.871 | +0.022 |
| 3 | 0.858 | 0.899 | -0.040 |
| 4 | 0.866 | 0.855 | +0.010 |
| 5 | 0.874 | 0.879 | -0.005 |

Memory trend: **rising** (0.853 → 0.874). Control trend: **rising** (0.839 → 0.879). **Δ (memory − control) trend: falling** (+0.013 → -0.005), does not beat control every batch.

- Rubric agreement (first batch → last batch): memory 0.938 → 0.946, control 0.910 → 0.939
- Grounding precision (first batch → last batch): memory 0.994 → 0.988, control 0.994 → 0.976
- Unsupported claim rate (first batch → last batch): memory 0.000 → 0.009, control 0.000 → 0.010

### Memory pipeline activity

| Batch | Corrections captured | Active rules after batch |
| --- | --- | --- |
| 1 | 10 | 7 |
| 2 | 8 | 6 |
| 3 | 15 | 10 |
| 4 | 10 | 10 |
| 5 | 15 | 14 |

### Finding

Judged on the memory-vs-control **advantage** (Δ), not on memory's own raw trajectory — see `docs/09-DECISIONS.md` for why that is the correct comparison when the control itself is not flat batch to batch, which it was not here.

**Memory's advantage over control does not grow consistently in every language category.** Category/categories where it does not: en, mixed.

- **ar: the clean result.** Memory beats control in **every single batch**, by a margin that grows monotonically from +0.015 to +0.129. This is the strongest, least ambiguous signal in the run.
- **en: a plausible but partial win.** Memory trails or ties control in batches 1-3 (Δ -0.001, -0.075, -0.001), then pulls ahead in batches 4-5 (Δ +0.029, +0.072) — consistent with the memory store needing a few batches of accumulated corrections (7→6→10 active rules by batch 3) before its retrieved guidance starts helping more than it costs. But three of five batches show no advantage, so this does not meet the "beats control every batch" bar `ar` does.
- **mixed: no consistent effect.** Δ oscillates with no trend (+0.013, +0.022, -0.040, +0.010, -0.005) and ends slightly negative. Memory neither reliably helps nor hurts here.

**The done-when criterion (`docs/08-ROADMAP.md`) is met for one of three language categories, not all three.** Stated plainly rather than reframed: this is the finding, and Phase 5's LoRA dataset plan depends on whether it is true, not on it being written down as true.

#### A bug this result caught, worth stating

The first version of this report's automated verdict called `ar` a **failure** — it compared memory's own raw accuracy trajectory (which fell slightly, 0.922→0.909) against a "rising" bar, without accounting for the fact that the no-memory control fell *further* (0.907→0.780) over the same five batches. The control was assumed to be roughly flat when the experiment was designed; it was not, in this run — batch-to-batch variance in call difficulty moved both arms together. Measuring the delta (memory − control) instead of memory's absolute score isolates the effect actually being tested and reverses the verdict for `ar` from "failed" to "the clearest win in the dataset." The code and this table were both corrected before this round was reported; see `docs/09-DECISIONS.md` for the fix and the regression test that pins it down.

#### Caveats

1. **Only `ar` meets the per-batch bar; `en`'s win is real but partial, and `mixed` shows nothing.** Do not round this up to "memory works" — it worked clearly in one category, plausibly in a second after a delay, and not at all in the third, on this run.
2. **One run, not a replicated result.** Gemini's `gemini-flash-lite-latest` extraction is not deterministic (no fixed seed, temperature > 0 in the extraction prompt), and this experiment was run once. The `ar` margin is large enough that noise is an unlikely full explanation, but `en`'s later-batch turnaround and `mixed`'s null result have not been checked for replication.
3. **Corrections and induced rules are synthetic-review artifacts, not real QA judgments** (`scripts/generate_synthetic_corrections.py`'s stand-in role applies identically here — every correction this run captured was auto-generated from an agent/ground-truth diff, reviewer_id `five-batch-experiment`). If the LLM-generated reference labels encode a systematic quirk, the induced rules learn to match that quirk, not necessarily a real QA improvement.
4. **Rubric agreement and grounding precision moved in the same direction as accuracy for `ar`** (rubric 0.987→0.927 memory vs. 0.983→0.905 control — memory still ahead at the end) **but the unsupported-claim rate stayed at 0.000 for both arms in `ar`/`en`**, so the grounding gate's own guarantee is unaffected by memory either way, as designed.
5. **Reference labels remain LLM-generated**, not human-reviewed — unchanged from every prior round; see `docs/09-DECISIONS.md`.

## Round: 2026-09-25 — phase 5 step 2, fine-tuning dataset build (`scripts/build_finetune_dataset.py`)

Not an accuracy round — no model was trained or evaluated yet. This records
the dataset-composition numbers the QLoRA training run (step 3) and the
catastrophic-forgetting eval (step 4) will cite, per the ground-in-real-
numbers rule for this phase. See `docs/09-DECISIONS.md`'s 2026-09-25 "Phase
5 fine-tuning dataset" entry for the full reasoning.

- Ground truth corpus: `data/ground_truth`, `ar` category only (default —
  `en`/`mixed` are opt-in, unused this round).
- Sources: existing Postgres `ReviewerAction` rows (`five-batch-experiment`,
  `synthetic-day1`) + a fresh live diff pass this script ran itself
  (`phase1-2-diff-augmentation`, 45 real `gemini-flash-lite-latest` calls).
- Every example, from both sources, is synthetic — see the caveat below.

| | ar |
| --- | --- |
| Corpus size | 60 |
| Held out (never trained/validated on; reserved for step 4) | 15 |
| Eligible for training pairs | 45 |
| Examples from Postgres `ReviewerAction` rows | 16 |
| Examples from the fresh diff-augmentation pass | 6 |
| Dropped for failing the grounding re-check | 0 |
| **Total examples** | **22** |
| Train / val split (stratified by error-location topic) | 18 / 4 |

Topic distribution (both splits combined): `commitments` (incl.
`commitments[i].deadline`) 19, `compliance_flags` 2, `rubric_scores` 1.

Held-out call_ids: `data/finetune/held_out_call_ids.json` (seed 20260925,
25% of the corpus, fixed before any training pair was built). Zero overlap
with `train.jsonl`/`val.jsonl` — enforced in code (`build_dataset()` raises
on any leak), not just checked after the fact.

#### Caveat

**Every example in this dataset is synthetic.** Both sources — the existing
Postgres corrections and this run's own diff-augmentation pass — are
produced by diffing an agent output against an LLM-generated reference
label (`data/ground_truth`, itself not human-reviewed; see caveat 5 above
and `docs/09-DECISIONS.md`). None of the 22 examples reflects a real QA
reviewer's judgment. Read the eventual QLoRA result (steps 3-4) as
validating the fine-tuning *mechanism* end-to-end, not as evidence of
real-world learning capacity — 18 training examples is a small-sample proof
of mechanism, not a scaled result.

#### Steps 3-4: pending

`scripts/train_qlora.py` (step 3) and `scripts/check_forgetting.py` (step
4) are written and unit-tested (`tests/scripts/test_train_qlora.py`,
`tests/scripts/test_check_forgetting.py`) but **have never run** — both
need CUDA, which this dev machine does not have. `notebooks/
qlora_train.ipynb` is the execution path (Colab, T4 minimum). Once run,
this section should be replaced with: the per-step train/eval loss from
`data/finetune/train_log.csv`, and the forgetting-check flags from
`data/finetune/forgetting_eval.json` — real numbers, not the runtime
estimate currently in `docs/09-DECISIONS.md`.
