# 01 — Product requirements

Sawti analyzes contact-center calls in Arabic, English and code-switched
Arabic/English. For each call it pulls out commitments, compliance flags,
rubric scores and sentiment, backs every claim with a verbatim quote, and
sends any call it is not confident about to a human reviewer. Reviewer
corrections become rules that later analyses retrieve.

Sources for everything below: the code, [`eval_results.md`](../eval_results.md),
[`09-DECISIONS.md`](09-DECISIONS.md), [`08-ROADMAP.md`](08-ROADMAP.md).

## Problem

- QA reviewers score calls against a fixed rubric by listening to them. That
  doesn't scale, and an automated score is useless if a reviewer can't check
  what it is based on.
- LLM extraction makes claims the transcript doesn't support. On clean text,
  the Phase 2 model proposed 1,296 claims, and 16 of them (1.2%) had no
  verbatim support: `ar` 0.010, `en` 0.004, `mixed` 0.019
  ([eval_results.md](../eval_results.md), 2026-09-22 phase 2 round).
- Code-switched calls are the hardest case at every stage. ASR word error rate
  (normalized) is `ar` 0.153, `en` 0.064, `mixed` 0.433
  ([eval_results.md](../eval_results.md), 2026-09-25 ASR quality round).

## Users

| User | What they do in Sawti |
|---|---|
| QA reviewer (primary) | Works the review queue, confirms, corrects or rejects each grounded claim, resolves memory-rule conflicts. |
| Operator (the person running the stack) | Uploads recordings or submits transcripts, reads the per-language results. In this project that is the developer. |

Nothing has been tested with real QA reviewers. The dev database has
**0** review submissions, so there is no reviewer-agreement number yet
([eval_results.md](../eval_results.md), 2026-10-07 reviewer agreement round).

## Goals

These come from the roadmap's standing rules:

1. Every agent output is a validated Pydantic model.
2. Every claim has a verbatim quote. A claim whose quote isn't in the
   transcript is dropped, not down-weighted.
3. Low-confidence calls go to a human. They never get an automated verdict.
4. PII is redacted before any text reaches a model, and only redacted text
   is stored.
5. Every result is reported per language category (`ar`, `en`, `mixed`).
6. Reviewer corrections improve later analyses through induced rules.

## Non-goals (what is not built)

- **Deployment.** The stack is local only, and the deliverable is a demo
  video ([09-DECISIONS.md](09-DECISIONS.md), 2026-10-10).
- **Authentication.** There is one shared API key plus a self-asserted
  reviewer id. It is not authentication
  ([09-DECISIONS.md](09-DECISIONS.md), 2026-10-06, decision 3).
- **Name and address redaction.** Redaction is regex-only. It does no NER.
- **Language auto-detection.** The caller must choose the call's language,
  because auto-detection garbles code-switched Arabic.
- **Real telephony audio.** All audio is synthetic TTS.
- **A compliance rule engine.** The `compliance` node is a documented
  passthrough.
- **Checking that a quote supports a claim.** Grounding checks only that the
  quote exists in the transcript.

## Success metrics

Every number is per language and comes from the source named in its row.
Accuracy and rubric agreement are measured against **LLM-generated** reference
labels, not human labels. Read them as agreement between two LLM processes.

| Metric | Criterion | ar | en | mixed | Source |
|---|---|---|---|---|---|
| Unsupported claim rate, Phase 1 → Phase 2 | Must drop (Phase 2 done-when) | 0.032 → 0.000 | 0.045 → 0.000 | 0.036 → 0.004 | [eval_results.md](../eval_results.md), 2026-09-22 comparison (pre-fix metric; `mixed` true value in [0.000, 0.004]) |
| ASR WER, normalized | No target set | 0.153 | 0.064 | 0.433 | [eval_results.md](../eval_results.md), 2026-09-25 ASR quality |
| Rubric agreement, clean → ASR text | No target set | −0.037 | −0.012 | −0.149 | [eval_results.md](../eval_results.md), 2026-09-25 Phase 2 → 3 |
| Memory on − off, accuracy | Positive in each language | +0.021 | +0.022 (within noise) | +0.042 | [eval_results.md](../eval_results.md), 2026-10-10 memory comparison, 600/600 |
| Reviewer agreement with the agent | Not set | n = 0 | n = 0 | n = 0 | [eval_results.md](../eval_results.md), 2026-10-07 |
| Escalations on ASR text | Low confidence routes to a human | 0 | 4 of 30 | 0 | [eval_results.md](../eval_results.md), 2026-09-25 Phase 2 → 3 |

**Known gap in the routing metric.** Routing depends only on grounding
coverage. A garbled ASR transcript (`call_0003_mixed` uploaded through the
dashboard) still grounded at 1.00 and auto-completed
([eval_results.md](../eval_results.md), 2026-10-11). The worst ASR failures are
the least likely to reach a human. A fix is planned but not built
([09-DECISIONS.md](09-DECISIONS.md), 2026-10-11 TODO).
