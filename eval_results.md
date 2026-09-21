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
| Compliance recall | | | |
| % requiring human review | | | |

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
