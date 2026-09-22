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
