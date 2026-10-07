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

> **Pre-fix metric (added 2026-10-07; numbers above unchanged).** Grounding
> precision and unsupported claim rate in this round checked quotes against
> the *raw* transcript, while `ground` verifies against the *redacted* one the
> model saw, so a legitimately grounded quote containing a placeholder
> (`[PHONE]` …) scored as unsupported. The bias only makes these two metrics
> look worse. This round's predictions were not saved, so it cannot be
> re-scored; the exact bound instead:
>
> - **Unsupported claim rate:** `ar` 0.000 and `en` 0.000 are exact (nothing
>   to over-count). `mixed` 0.004 is exactly **2 of 513** emitted claims
>   (523 proposed − 10 rejected; 2/513 = 0.0039, and no other count rounds to
>   0.004). The fixed value is in **[0.000, 0.004]**: at most 2 claims, all
>   `mixed`. Re-scoring the phase 4 analyses that were saved (round below,
>   2026-10-07) moved `mixed` from 0.0039 to exactly 0.000.
> - **Grounding precision:** `en` 1.000 is exact. `ar` 0.990 and `mixed`
>   0.986 are lower bounds: true values in [0.990, 1.000] and [0.986, 1.000].
>   This metric also counts sentiment-point quotes, whose model paraphrases
>   are a second cause of shortfall, and per-quote counts were not recorded,
>   so the placeholder share cannot be separated out. Only calls whose
>   transcript contains PII can be affected: ar 9/60, en 1/30, mixed 12/60.
>   In the re-scored phase 4 sample the whole shift was `mixed`; `ar` did not
>   move.
>
> 6.3 produces fresh evals with the fixed metric.

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
- *(Pre-fix metric — grounding precision and unsupported claim rate here checked the raw transcript. The stored subset of this run's analyses is re-scored with the fixed metric in the 2026-10-07 round below: `mixed` unsupported claim rate 0.0039 → 0.000.)*

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

#### Steps 3-4: run on Colab 2026-09-27 (T4)

Real numbers, from `data/finetune/train_log.csv` and
`data/finetune/forgetting_eval.json`, both now checked into this repo. The
trained LoRA adapter itself is not (it's on Google Drive at
`sawti-phase5/` — see `notebooks/qlora_train.ipynb` — and is a build
artifact, not a result record, the same distinction `data/finetune/
_diff_cache/` draws).

**Training** — 3 epochs over the 18-example train set (batch size 1,
gradient accumulation 4, so 5 optimizer steps/epoch, 15 total, matching
`_warmup_steps()`'s own step-count math):

| Step | Epoch | Train loss |
| ---: | ---: | ---: |
| 1 | 0.222 | 2.1449 |
| 2 | 0.444 | 2.3660 |
| 3 | 0.667 | 2.4276 |
| 4 | 0.889 | 2.1048 |
| 5 | 1.000 | 2.0978 |
| 6 | 1.222 | 2.0233 |
| 7 | 1.444 | 1.7504 |
| 8 | 1.667 | 1.9545 |
| 9 | 1.889 | 1.8562 |
| 10 | 2.000 | 1.3269 |
| 11 | 2.222 | 1.6273 |
| 12 | 2.444 | 1.5460 |
| 13 | 2.667 | 1.7049 |
| 14 | 2.889 | 1.9141 |
| 15 | 3.000 | 1.4161 |

Noisy step to step (an 18-example set gives each step's loss high
variance), but the per-epoch eval numbers show a clear, monotonic trend:

| Epoch | Eval loss | Eval mean token accuracy |
| ---: | ---: | ---: |
| 1 | 1.9185 | 0.6018 |
| 2 | 1.7573 | 0.6314 |
| 3 | 1.7259 | 0.6356 |

Overall `train_loss` (trainer's own weighted average) **1.8840**,
`train_runtime` **1588.27s (~26m28s)** on a T4. This is the actual
`trainer.train()` wall-clock alone — separate from, and much larger than,
`scripts/train_qlora.py`'s own pre-run *estimate* of "well under a minute
of GPU compute" for training, which assumed a T4 would move through an 8B
model's forward/backward pass much faster than it did in practice
(~106s/step). See `docs/09-DECISIONS.md`'s 2026-09-27 entry for the
correction.

**Catastrophic-forgetting check** — 7/7 generic prompts checked, **0
flagged** (verified directly against `data/finetune/forgetting_eval.json`,
now checked into the repo alongside `train_log.csv`, not just a console
summary). Base and tuned response lengths are comparable on every prompt
(e.g. 778 vs. 881 chars on the capital-of-France prompt, 948 vs. 938 on
the Romeo-and-Juliet summary) — no length-collapse or empty-response
regression on any of the 7, and reading the full text confirms the same:
near-identical reasoning traces (both models work through the same
`<think>` steps: breaking `17 × 23` into `17×20 + 17×3`, second-guessing
haiku line count against the "two-line" instruction, converging on Paris
via the same landmarks) and the same recalled facts throughout.

#### Caveat

Every training/val example remains synthetic (see the dataset-build round
above and `docs/09-DECISIONS.md`, 2026-09-25) — this result validates that
QLoRA training runs correctly end to end and does not visibly damage
general instruction-following on 7 generic prompts. It is not evidence
that the fine-tuned model is better at call analysis than the base model;
no held-out-call comparison (the 15 calls reserved in
`held_out_call_ids.json`) has been scored yet.

#### Held-out-calls accuracy comparison: run on Colab 2026-09-27 — **PRELIMINARY, UNCORRECTED**

> **Superseded** by the corrected re-run on 2026-10-06 (next round below).
> This section is kept as the historical record; its raw JSON is in git at
> `git show ee12ea6:data/finetune/held_out_eval.json` (the working-tree file
> now holds the 2026-10-06 result).

> **These numbers were measured before a denominator-mismatch bug in
> `scripts/eval_held_out_calls.py` was found and fixed (see
> `docs/09-DECISIONS.md`, 2026-09-27, "`eval_held_out_calls.py` compared
> mismatched denominators"). `base_metrics` and `tuned_metrics` below were
> each computed from that model's own full extraction-success list, not
> necessarily the same 10 calls — `N calls scored: ar=10` is the
> *intersection* size, correctly reported, but the metric values above it
> may each reflect a different, larger set. Read the deltas below as
> indicative, not validated. A corrected re-run (using the already-fixed
> `_compare()`, which restricts both sides to the shared intersection) is
> still needed before this checklist item can be marked done — see
> `docs/08-ROADMAP.md`.**

N calls scored: `ar=10` (5 of the 15 held-out calls excluded — extraction
failed on the base model, the tuned model, or both, varying by call).

| Metric | Base | Tuned | Δ (tuned − base) |
| --- | --- | --- | --- |
| Accuracy | 0.693 | 0.671 | -0.022 |
| Rubric agreement | 0.893 | 0.878 | -0.014 |
| Grounding precision | 0.808 | 0.885 | +0.076 |
| Unsupported claim rate | 0.231 | 0.122 | -0.110 |

**Qualitative finding, independent of the denominator bug**: the tuned
model failed structured-output validation (`LocalHFProvider.
structured_complete`'s JSON-schema parse/validation, missing required
fields such as `rubric_scores`, `sentiment_points`, `confidence`) more
often than the base model across multiple retries on several calls — part
of why only 10/15 calls were scored on both sides. This observation
doesn't depend on which denominator the accuracy numbers used, so it
survives the bug; whether it replicates on a clean re-run is still open.

#### Caveat

**Preliminary and uncorrected, stated as plainly as possible**: the four
deltas above are not a validated same-10-calls comparison — see the
warning block above this table. Do not cite the specific delta values
(-0.022, -0.014, +0.076, -0.110) as a measured effect of fine-tuning until
a corrected re-run replaces this section. The higher tuned-model
structured-output failure rate is the one finding from this run that can
be read with more confidence, independent of the bug.

## Round: 2026-10-06 — phase 5 step 4, held-out-calls accuracy (base vs. QLoRA-tuned), corrected re-run

- Script: `scripts/eval_held_out_calls.py` with the fixed `_compare()` — both
  models scored on the **same** intersection of calls (supersedes the
  2026-09-27 preliminary round above)
- Base model: Qwen3-8B, 4-bit (bitsandbytes), greedy first attempt, up to 3
  sampled retries on JSON/schema failure (`LocalHFProvider`)
- Tuned model: same base + `data/finetune/qlora_adapter` (restored from Drive)
- Held-out set: 15 `ar` calls (`data/finetune/held_out_call_ids.json`), never
  trained or validated on
- Ground truth: `data/ground_truth` (LLM-generated reference labels,
  `human_reviewed: false`) — unchanged from every prior round
- Hardware: Colab free tier, T4 16 GB; both models resident on GPU
  (13.0–13.7 GB GPU RAM, no CPU offload)
- Raw result: `data/finetune/held_out_eval.json`

N calls scored: `ar=10` (shared successes; 4 base-side failures, 2
tuned-side failures — per-call outcomes in the JSON's `outcomes`).

| Metric (ar, n=10) | Base | Tuned | Δ (tuned − base) |
| --- | --- | --- | --- |
| Accuracy | 0.640 | 0.652 | +0.012 |
| Rubric agreement | 0.873 | 0.899 | +0.026 |
| Grounding precision | 0.899 | 0.907 | +0.008 |
| Unsupported claim rate (lower is better) | 0.132 | 0.086 | −0.046 |

`en` and `mixed`: not measured — the held-out set is `ar` only by design.

### Extraction outcomes (all 15 calls)

| | Base | Tuned |
| --- | --- | --- |
| Extracted | 11/15 | 13/15 |
| Terminal failures | 4 — all `SentimentTrajectory` not chronological (`call_0088`, `0118`, `0122`, `0129`) | 2 — both `SentimentTrajectory` not chronological (`call_0088`, `0100`) |
| Calls needing ≥1 retry that then succeeded | 2 (`0100`, `0112`) | 5 (`0031`, `0048`, `0053`, `0094`, `0118`) |
| Retryable failed attempts (JSON/schema) | 2 | 13 |
| …of which: training-format fragment (`{'call_id': <invented>, …'procedure_adherence': x}`, `rubric_scores`/`sentiment_points`/`confidence` missing) | 0 | 11 |
| …of which: echoed the JSON schema itself (`$defs`) instead of data | 0 | 1 |
| …of which: malformed JSON syntax | 1 | 1 |
| …of which: missing `sentiment_points` only | 1 | 0 |

Per-call outcome disagreements: tuned succeeded where base failed on
`call_0118`, `call_0122`, `call_0129`; base succeeded where tuned failed on
`call_0100`; both failed on `call_0088`.

**Source of the attempt counts**: transcribed from the Colab cell log, not
machine-recorded — `held_out_eval.json` stores only each call's final
outcome. Treat the retry rows as careful hand counts, not instrumented data.

### Latency (T4, from the cell's elapsed timer — approximate)

| Stage | Time |
| --- | --- |
| Download Qwen3-8B (5 shards, ~16 GB) | ~3 min |
| Load + 4-bit quantize, two copies | ~2.3 min |
| Base, 15 calls incl. retries | ~70 min (~4.6 min/call) |
| Tuned, 15 calls incl. retries | ~104 min (~6.9 min/call) |
| Whole cell | ~3 h 0 min |
| Compute cost | $0 (Colab free tier) |

The tuned model's higher per-call time is retries, not slower generation.

### Finding

1. **No regression on the shared calls; all four deltas point the right
   way, but none is shown to be a real effect.** n=10, a single run,
   18 synthetic training examples, LLM-generated references, no
   confidence intervals. A one-call swing moves these metrics by more than
   the observed deltas. Report as "no measurable degradation", not
   "fine-tuning improved accuracy".
2. **Fine-tuning shifted the model's default output format toward its
   training target.** Every training example's output is a *single
   corrected field* (16/18 `commitments`, 1 `compliance_flags`, 1
   `rubric_scores[procedure_adherence]`); this eval asks for a *full*
   `PlainAnalysis`. The tuned model's greedy first attempt repeatedly
   emitted a fragment shaped like the training targets (invented
   `call_id`, a lone `procedure_adherence` score) — 11 of its 13 retryable
   failures. Sampled retries usually recovered the full schema. This
   replicates the 2026-09-27 qualitative finding (higher tuned-side
   structured-output failure rate) and supplies a mechanism for it: a
   train/eval task mismatch, not general damage to the model.
3. **Self-reported confidence is not a usable signal here.** Fragment
   outputs missing required fields came with `confidence` 0.95–0.98.
   Routing must keep depending on schema validation and grounding
   verification, not on the model's own confidence.
4. **The one terminal failure mode for both models is chronological
   ordering of sentiment points**, which `LocalHFProvider`'s retry loop
   never sees (it is raised later, in `CallAnalysis` assembly). Once the
   tuned model produced a full schema it hit this less often (2 vs. 4).

### Caveats

1. **Sample size.** 10 scored calls, one language, one run. Not
   significance-tested.
2. **The 2026-09-27 preliminary numbers are not a valid comparison point**
   for direction of effect — they used mismatched denominators, so the
   difference in sign between that run and this one says nothing either
   way.
3. **Reference labels remain LLM-generated**, not human-reviewed.
4. **Attempt counts are hand-transcribed** from the log (see above).

### How the next step would be measured (not done — deadline)

Rebuild the fine-tuning targets as full, corrected `PlainAnalysis`
objects (so train and eval tasks match), retrain, and re-run this exact
script on the same 15 calls. Success criterion: tuned retryable-failure
count ≤ base's (2), with no metric regressing on the shared calls; record
per-call attempt counts in the JSON rather than by hand.

---

## Round: 2026-10-07 — phase 6.2, the service in containers: latency, throughput, retries, cold start

Measured with `scripts/measure_service.py` against the full compose stack
(`migrate` → `api` + N × `worker`, Postgres, Redis) on Docker Desktop,
Apple M1 8 GB host, Docker VM 8 vCPU / 3.8 GiB, `linux/arm64` images.
Raw per-call results: `data/service_measurements/20261007T100929_gemini.json`
and `data/service_measurements/20261007T102541_fake.json`.

**How to read these.** Each run is a **burst**: every call is submitted at
once, ar/en/mixed interleaved, and the run ends when all are terminal. Queue
wait therefore measures a call's position in a backlog, not idle latency.
Processing time is one task's own duration (claim → record written). p50/p95
use linear interpolation; with n = 3 or 20 per cell, p95 sits close to the
max. Each worker runs one task at a time (`--concurrency=1`).

Two providers, for two different questions:

- **`gemini`** (the real provider, `gemini-flash-lite-latest`, free tier) —
  1 worker, 3 calls per language. What a real call costs. Not scaled up:
  the free tier's 15 requests/min would cap throughput at ~15 calls/min no
  matter how many workers, which measures Google's quota, not this system.
- **`fake`** (`sawti.llm.fake_provider`, offline, deterministic) with a
  simulated **5.0 s** model latency — matched to Gemini's measured ~5 s
  processing p50 — at 1, 2 and 4 workers, 20 calls per language. What the
  system itself adds and how it scales. Every fake call escalates by
  construction (2 of 3 claims grounded), so these runs also exercise the
  checkpoint write of an interrupted run on every call.

### Real provider (gemini), 1 worker, n = 3 per language

| Language | Queue wait p50 / p95 (s) | Processing p50 / p95 (s) | Retry rate | Failure rate | Outcomes |
|---|---|---|---|---|---|
| ar    | 16.6 / 31.1 | 5.76 / 5.94 | 0 / 3 | 0 / 3 | 3 completed |
| en    | 22.6 / 35.9 | 5.12 / 5.85 | 0 / 3 | 0 / 3 | 3 completed |
| mixed | 27.6 / 41.0 | 5.19 / 5.26 | 0 / 3 | 0 / 3 | 3 completed |

Throughput: 11.25 calls/min (9 calls in 48.0 s). The queue-wait gradient
ar < en < mixed is submission order within the interleaved burst (each
language's k-th call is submitted one and two slots after ar's), not a
language effect. All 9 auto-passed (no escalation) — a small sample, not a
grounding result.

### System capacity (fake provider, 5.0 s simulated latency), n = 20 per language per run

| Workers | Language | Queue wait p50 / p95 (s) | Processing p50 / p95 (s) | Retry rate | Failure rate |
|---|---|---|---|---|---|
| 1 | ar    | 145.5 / 276.7 | 5.08 / 5.19 | 0 | 0 |
| 1 | en    | 150.7 / 281.8 | 5.08 / 5.19 | 0 | 0 |
| 1 | mixed | 155.9 / 286.8 | 5.08 / 5.13 | 0 | 0 |
| 2 | ar    |  71.7 / 137.9 | 5.09 / 5.17 | 0 | 0 |
| 2 | en    |  74.2 / 138.2 | 5.08 / 5.18 | 0 | 0 |
| 2 | mixed |  76.7 / 143.1 | 5.08 / 5.12 | 0 | 0 |
| 4 | ar    |  39.2 /  72.3 | 5.12 / 5.88 | 0 | 0 |
| 4 | en    |  41.8 /  72.3 | 5.12 / 5.23 | 0 | 0 |
| 4 | mixed |  44.3 /  77.2 | 5.12 / 5.24 | 0 | 0 |

| Workers | Throughput (calls/min) | Ideal at 5.08 s/call | Efficiency | Wall time (60 calls) |
|---|---|---|---|---|
| 1 | 11.72 | 11.8 | 99% | 307.1 s |
| 2 | 23.42 | 23.6 | 99% | 153.7 s |
| 4 | 42.90 | 47.2 | 91% | 83.9 s |

**Overhead per task: ~0.08 s** above the simulated model latency (processing
p50 5.08 s vs 5.00 s): DB claim, memory-rule lookup, graph run on the
Postgres checkpointer (including the interrupt write), record insert. At 4
workers p50 rises to 5.12 s and one ar p95 to 5.88 s — contention on the
shared 3.8 GiB / 8 vCPU VM, which is also where the 9% efficiency loss goes.

### Retries and failures

0 retries and 0 failures across all 189 calls. That is an honest result for
these runs and **not** evidence the retry path works under load: the fake
provider never errors, and the 9 real calls hit no 429. The retry path is
covered by tests (`tests/api/test_tasks.py`: a 429 re-queues and the retry
finishes; retries exhausted ends `failed`; a non-transient error escalates
instead of retrying), not by this measurement.

### Images and cold start

| Image | Uncompressed (`docker images`) | Compressed content (`docker image inspect .Size`) |
|---|---|---|
| `sawti-api`    | 932 MB | 197 MB |
| `sawti-worker` | 3.9 GB | 1.03 GB |

The api image carries no torch / transformers / sentence-transformers
(`uv export --prune sentence-transformers`). The worker image bakes in the
`paraphrase-multilingual-MiniLM-L12-v2` weights, so cold start is a load
from local disk.

**Worker cold start** (embedding model load in `worker_process_init`, from
the worker's own log line): **3.9 – 5.4 s** for one or two workers starting
together (3.88, 4.22, 5.27, 5.37 in the recorded runs); **~12.0 s** each
(12.03 – 12.09) when four start at once on the same VM. One worker measured
~845 MiB resident with the model loaded (`docker stats`); four would be
~3.4 GiB of the VM's 3.8 GiB by that estimate (not measured during the run)
— no worker was OOM-killed, but four is about the ceiling on this VM.

### Caveats

1. **Synthetic transcripts, synthetic load.** 60 calls per language
   category at most; the real-provider sample is 3 per language.
2. **Fake latency is a constant.** Real model latency varies (Gemini
   p95/p50 here ≈ 1.03 – 1.14); a constant 5.0 s hides tail effects.
3. **`memory_rules` was empty** in every run (no rule store existed before
   6.2), so retrieval returned immediately and the loaded embedding model
   was never queried. Retrieval cost per call is not in these numbers.
4. **One machine.** Workers, API, Postgres and Redis share one Docker VM.


---

## Round: 2026-10-07 — grounding metrics re-scored against the redacted text (pre-fix vs fixed)

**What changed.** `grounding_precision_by_category` and
`unsupported_claim_rate_by_category` now check quotes against the redacted
transcript — the string the model saw and the one `ground` verifies — instead
of the raw file. See `docs/09-DECISIONS.md`, 2026-10-07. The old numbers in
the rounds above are **not** overwritten; they are marked "pre-fix metric".

**What was re-scored.** Phase 2's and phase 4's full prediction sets were never
saved. What *was* saved is the `original` analysis of each of the 103 stored
corrections (`call_analyses` in the dev database): 85 from the phase 4
five-batch run (memory arm, calls where the agent disagreed with the
reference) and 18 from phase 4's synthetic-correction step (`synthetic-day1`).
Scored here twice — raw vs redacted — so the two columns differ by the metric
alone. Zero LLM calls. `scripts/rescore_stored_analyses.py`; raw output
`data/rescore/2026-10-07_stored_phase4_analyses.json`.

This is a **subset selected for disagreement**, not phase 4's population:
read the change between the columns, not the absolute values, as the finding.

### All 103 stored analyses

| Language | n | Grounding precision, pre-fix → fixed | Unsupported claim rate, pre-fix → fixed |
|---|---|---|---|
| ar    | 27 | 0.9973 → 0.9973 | 0.0000 → 0.0000 |
| en    | 19 | 1.0000 → 1.0000 | 0.0000 → 0.0000 |
| mixed | 57 | 0.9861 → **0.9987** | 0.0039 → **0.0000** |

### By source

| Source | Language | n | Grounding precision, pre-fix → fixed | Unsupported claim rate, pre-fix → fixed |
|---|---|---|---|---|
| five-batch (phase 4 run) | ar    | 23 | 0.9969 → 0.9969 | 0.0000 → 0.0000 |
| five-batch (phase 4 run) | en    | 14 | 1.0000 → 1.0000 | 0.0000 → 0.0000 |
| five-batch (phase 4 run) | mixed | 48 | 0.9880 → 0.9985 | 0.0023 → 0.0000 |
| synthetic-day1           | ar    |  4 | 1.0000 → 1.0000 | 0.0000 → 0.0000 |
| synthetic-day1           | en    |  5 | 1.0000 → 1.0000 | 0.0000 → 0.0000 |
| synthetic-day1           | mixed |  9 | 0.9760 → 1.0000 | 0.0123 → 0.0000 |

**Finding.** Every unsupported claim the old metric reported in this sample
was a redaction artifact: with the fixed metric the rate is 0.000 in all three
categories, which is what the grounding gate guarantees by construction. The
whole shift is `mixed` (20% of `mixed` transcripts contain PII). The residual
precision shortfall (`ar` 0.9973, `mixed` 0.9987) is the sentiment-point
paraphrases below, which grounding precision counts and `ground` did not
check until today.

### Sentiment points the new `ground` rule drops (same 103 analyses)

As of 2026-10-07 `ground` drops sentiment points whose quote is not verbatim
in the redacted transcript (`docs/09-DECISIONS.md`). Applied to the stored
analyses, which predate the rule:

| Language | Sentiment points | Would be dropped | Rate |
|---|---|---|---|
| ar    | 142 | 1 | 0.7% |
| en    |  96 | 0 | 0.0% |
| mixed | 281 | 1 | 0.4% |

The two are the quotes recorded in the 2026-10-07 dev-database conversion
entry (`call_0028_ar`: dropped quotation marks; `call_0073_mixed`: حاليأ for
حالياً). Routing is unaffected: sentiment drops do not count toward grounding
coverage.

---

## Round: 2026-10-07 — reviewer agreement with the agent (service reviews)

`scripts/reviewer_agreement.py` over `review_submissions` in the dev database;
definitions in `sawti.eval.metrics.reviewer_agreement_by_category`:
**claim agreement** = confirmed claims / judged claims, **call agreement** =
reviews confirming every claim / reviews. Escalated calls only — reviewers see
`awaiting_review` calls, selected for low grounding coverage — so this will
never describe the auto-passed population.

| Language | Reviews | Claims judged | Claim agreement | Call agreement |
|---|---|---|---|---|
| ar    | 0 | 0 | — | — |
| en    | 0 | 0 | — | — |
| mixed | 0 | 0 | — | — |

**No number yet, honestly.** No human has reviewed a call through the service:
the only submissions ever made were the e2e's synthetic ones, which were
deleted from the dev database (`docs/09-DECISIONS.md`, 2026-10-07), and the
phase 4 corrections are ground-truth diffs, not per-claim verdicts, so they
cannot be turned into an agreement rate. The metric and script are in place
and tested; the first real rate needs the 6.4 dashboard and real reviewers.

---

## Round: 2026-10-07 — memory-rule backfill over the 103 stored corrections (PARTIAL: 81 / 103)

`scripts/induce_rules_from_corrections.py`: the service's memory loop
(`sawti.memory.persistence.induce_and_store` — induce → conflict check →
store → consolidate) over every correction in the dev database, oldest
first, Gemini `gemini-flash-lite-latest`, throttled to one request per 4.5 s.
**Stopped by the free tier's 500-requests/day cap after 81 corrections**;
re-running the same command after the reset resumes at correction 82.
Counts read from the database, per language category of the corrected call.

| Language | Corrections | Induced | → rule stored | → conflict (not stored) | Pending | Active rules from this language |
|---|---|---|---|---|---|---|
| ar    | 27 | 21 | 20 | 1 |  6 | 20 |
| en    | 19 | 17 | 15 | 2 |  2 | 15 |
| mixed | 57 | 43 | 36 | 7 | 14 | 36 |
| **total** | 103 | 81 | 71 | 10 | 22 | 71 |

**Not final, and not yet consolidated.** `consolidate()` runs once at the end
of a run that finishes, so these 71 rules include near-duplicates the merge
step will collapse (phase 4's five-batch run ended with 14 active rules from
58 corrections after per-batch consolidation). The final count, after the
remaining 22 corrections and consolidation, will be recorded here.

**Conflicts (10)** are candidates the LLM contradiction check judged
incompatible with an existing rule; by phase 4's `insert_rule` policy neither
side wins automatically, so they are logged and left for a human
(`reviewer_actions.induction_outcome = 'conflict'`). 7 of the 10 are `mixed`.

One `TimeoutError` occurred and was retried once after 65 s; no other
transient errors.

---

## Round: 2026-10-07 — memory-rule backfill, FINAL (103 / 103) — **PRE-FIX** (single-linkage consolidation, superseded)

> **Pre-fix.** The 9-rule result below came from single-linkage consolidation,
> since found to be a correctness bug and fixed the same day. Kept as recorded;
> the post-fix result is the "complete-linkage re-consolidation" round below.

Completes the partial round above. The last 22 corrections were run by the
user on a **different Gemini API key / Google Cloud project** (the first
key's 500/day cap was spent); same model alias, same script, same code.
Report: `data/memory_backfill/2026-10-07_final_report.json`
(`scripts/induce_rules_from_corrections.py --report-only`).

### Corrections

| Language | Corrections | → rule stored | → conflict (not stored) |
|---|---|---|---|
| ar    | 27 | 26 | 1 |
| en    | 19 | 17 | 2 |
| mixed | 57 | 49 | 8 |
| **total** | **103** | **92** | **11** |

### Rule rows — how 92 inserted becomes 9 active

| | Rows |
|---|---|
| inserted by induction (one per `inserted` correction) | 92 |
| created by consolidation (merges, ≥ 2 sources each) | 3 |
| retired by consolidation (merged away) | 86 |
| **active** = 92 + 3 − 86 | **9** |

`92 − 86 = 6` misses the 3 rows consolidation itself creates: each merged
cluster becomes a *new* rule and every member is retired (all 86 retired rows
are members of the 3 merged rules; none is orphaned).

### Per language — why 3 + 3 + 6 = 12, not 9

A rule counts once for each language among its source corrections. Exact
split of the 9 active rules by language set: `ar` 1, `en` 2, `mixed` 4,
`ar+mixed` 1, `ar+en+mixed` 1 — the two multi-language rules contribute
3 extra memberships (12 − 9).

| Language | Active rules drawing on it |
|---|---|
| ar    | 3 |
| en    | 3 |
| mixed | 6 |

### The 9 active rules

| # | Rule text | Source corrections (ar / en / mixed) |
|---|---|---|
| 1 | Carefully audit the entire dialogue to comprehensively capture and count every distinct, explicit commitment and promise made by the agent or customer without omission or duplication, while strictly applying compliance flags only when procedural violations are explicitly verified. | **81** (24 / 13 / 44) |
| 2 | Utilize the specific compliance flag 'refusing_manager_escalation' instead of any generalized refusal flag whenever a request for a manager is denied. | 3 (0 / 3 / 0) |
| 3 | Verify that agent procedure adherence and name disclosure rules are strictly evaluated against established guidelines and actual audio transcripts before assigning non-compliance flags. | 2 (1 / 0 / 1) |
| 4 | Assign the 'none' compliance flag when no resolution confirmation issues are present in the interaction. | 1 (mixed) |
| 5 | Do not trigger politeness failure compliance flags when agents maintain appropriate professional etiquette throughout the interaction. | 1 (en) |
| 6 | Do not identify general statements of policy or conversational closing remarks as actionable customer commitments. | 1 (ar) |
| 7 | Flag interactions as containing third-party blame whenever representatives deflect responsibility or attribute faults to external entities, systems, or other departments. | 1 (mixed) |
| 8 | Accurately identify and flag any instances of inappropriate language used during the interaction. | 1 (mixed) |
| 9 | Exclude automatically generated system messages or placeholders from being classified as valid commitments. | 1 (mixed) |

### Over-merging: rule 1 is a chaining artifact, and it lost specific detail

Consolidation is **single-linkage** clustering at cosine **≥ 0.7**
(`sawti.memory.consolidate.DEFAULT_SIMILARITY_THRESHOLD`): connected
components of the "similar" graph, so A ~ B and B ~ C merges A with C however
far apart they are. Measured with the real embedding model on each merged
rule's members:

| Rule | Members | Member–member cosine: min / median / share ≥ 0.7 | Member → merged text: min / median |
|---|---|---|---|
| 1 | 81 | 0.07 / 0.50 / **13%** | 0.34 / 0.60 |
| 2 |  3 | 0.78 / 0.79 / 100% | 0.87 / 0.90 |
| 3 |  2 | 0.70 / 0.70 / 100% | 0.83 / 0.89 |

- **Rule 2 is a clean merge**: three phrasings of one principle, and the
  merged text keeps the specific flag name.
- **Rule 3 is borderline**: exactly at the threshold, and it joins two
  different checks (procedure-adherence scoring, a rubric item, and
  name-disclosure flagging). Both survive in the text, but as one vague
  instruction.
- **Rule 1 is over-merged.** 87% of its member pairs are *below* the
  threshold; the cluster spans two topics (53 commitment corrections, 27
  compliance-flag, 1 rubric) and all three languages. The merged text keeps
  two generic themes (count explicit commitments; flag only verified
  violations) and drops the specifics 33 of its 81 members carried. Examples
  of retired members whose content is not in it:
  - "Do not flag identity verification as late if it was completed within the acceptable timeframe."
  - "Do not trigger compliance flags for identity verification failure unless the customer genuinely fails to pass the mandatory authentication steps."
  - "Ensure agent identification flags are only raised when the agent genuinely fails to provide their name or identifier according to standard greeting protocols." (cosine 0.34 to the merged text)
  - "Ensure that casual mentions of future actions or conditional statements without explicit intent to follow through are not classified as formal commitments."

  Retrieval returns top-5, so in practice every call now gets rule 1 plus
  four near-singletons, and identity-verification or agent-identification
  guidance is gone from the store.

**Why phase 4 did not show this:** it consolidated *per batch*, over at most
~15 active rules at a time; the backfill consolidated 92 rules in one pass,
giving single-linkage enough points to chain through. Nothing is lost
irreversibly — the 86 retired rows are intact, so the set can be
re-consolidated (see `docs/09-DECISIONS.md`, 2026-10-07, for the proposed fix,
not yet applied).

### Not the rule set phase 4 measured

Phase 4's memory-on results (2026-09-25 round) were measured with rules
induced **inside that run** — an in-process store built from that run's own
58 corrections, consolidated per batch, ending at 14 active rules, never
persisted. These 9 rules come from a different process and different input:
all 103 stored corrections — 85 `five-batch-experiment` rows (27 more than
the 58 the recorded run reports capturing; most likely persisted by the
interrupted attempts earlier on 2026-09-25, recorded as incidents 1–3 in
`docs/09-DECISIONS.md`, but not verified row by row) plus phase 4's 18
`synthetic-day1` corrections — induced again with the service pipeline,
consolidated once, with the last 22 on a different API project. Phase 4's numbers say nothing
about how *these* rules perform; that is a 6.3 measurement.


---

## Round: 2026-10-07 — memory rules re-consolidated with complete linkage (POST-FIX)

**The bug.** `consolidate()` clustered with single linkage (connected components
of the cosine ≥ 0.7 graph). Chains merge rules that are not similar: the
backfill's one pass turned 92 rules into 9, one of which absorbed 81 —
memory was effectively one generic rule. **The fix:** complete linkage — every
pair inside a cluster ≥ 0.7, same threshold, same embedder
(`sawti.memory.consolidate._cluster_indices`; `docs/09-DECISIONS.md`,
2026-10-07). **Re-run:** the 86 retired originals un-retired, the 3
single-linkage merges retired, the 92 single-correction rules re-clustered and
merged (25 Gemini calls for merge text, zero for clustering)
(`scripts/reconsolidate_memory_rules.py`). Results:
`data/memory_backfill/2026-10-07_reconsolidation_result.json`; plan with
every cluster's members: `..._reconsolidation_plan.json`.

### Before / after

| | Pre-fix (single linkage) | Post-fix (complete linkage) |
|---|---|---|
| Active rules | 9 | **35** |
| Cluster sizes (size × count) | 81×1, 3×1, 2×1, 1×6 | 8×2, 5×2, 4×4, 3×6, 2×11, 1×10 |
| Largest cluster | 81 | 8 |
| Lowest min-pairwise cosine in any cluster | **0.068** | **0.700** |
| Multi-member clusters with min pairwise < 0.7 | 1 (the 81) | 0 |

Min pairwise cosine per multi-member cluster, post-fix (sizes 8, 8, 5, 5, 4, 4,
4, 4, 3 ×6, 2 ×11): 0.759, 0.779, 0.731, 0.757, 0.720, 0.738, 0.788, 0.759,
0.745, 0.779, 0.793, 0.807, 0.724, 0.778, 0.806, 0.751, 0.805, 0.921, 0.700,
0.714, 0.925, 0.825, 0.868, 0.771, 0.895 — all ≥ 0.70 by construction.

### Active rules per language (a rule counts once per language among its source corrections)

| Language | Pre-fix | Post-fix |
|---|---|---|
| ar    | 3 | 16 |
| en    | 3 | 11 |
| mixed | 6 | 25 |

Exact post-fix split by language set: ar 3, en 4, mixed 14, ar+en 3,
ar+mixed 7, en+mixed 1, ar+en+mixed 3 (35). Row accounting: 120 rows =
92 induced + 28 created by consolidation (3 old merges + 25 new), 85 retired
(82 merged originals + the 3 old merges), 35 active.

### Is specific detail kept now?

Spot-checked the two largest clusters (8 each): both are eight phrasings of
"capture every commitment, including secondary ones", and the merged text
says exactly that. Guidance the 81-rule merge had erased is back as its own
rules, e.g. "Apply the missing identity verification compliance flag only
when the agent genuinely fails to perform the mandatory security validation
process…". Remaining imperfection, stated: those two 8-clusters are close to
each other but not merged (some cross pairs < 0.7) — redundancy, not loss.

### Phase 4 — could its results have been affected?

Phase 4 consolidated per batch and persisted neither rules nor clusters, so its
max cluster size is unknown. Two pieces of evidence that it chained too:
its own counts (batch 2: up to 15 candidate rules → 6 active), and a replay of
its batch sizes with the backfill's re-induced rules and real embeddings
(`scripts/simulate_phase4_consolidation.py`, zero LLM calls;
`data/memory_backfill/2026-10-07_phase4_consolidation_replay.json`):

| Batch | Single linkage: max cluster / active after | Complete linkage: max cluster / active after | Phase 4 recorded active after |
|---|---|---|---|
| 1 | 3 / 5   | 2 / 6  | 7 |
| 2 | 6 / 8   | 3 / 11 | 6 |
| 3 | **13** / 6 | 4 / 15 | 10 |
| 4 | 5 / 9   | 4 / 16 | 10 |
| 5 | **10** / 11 | 4 / 18 | 14 |

The single-linkage replay tracks phase 4's recorded active counts roughly
(same scale, somewhat lower), so phase 4's memory-on arm very likely ran with
chained merges of 10+ rules from batch 3 on. **Yes, its results could have
been affected.** Likely direction: over-merged rules carry vaguer guidance,
which would *weaken* memory, so phase 4's memory − control deltas are more
likely understated than inflated — but that is an inference, not a
measurement, and the replay uses re-induced rule text, not phase 4's. A
corrected measurement of memory's effect belongs in 6.3.
