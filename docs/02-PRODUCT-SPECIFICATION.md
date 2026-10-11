# 02 — Product specification

The QA dashboard (`frontend/`, React + Vite) is local only, in English by
default with an Arabic toggle that switches the whole UI to RTL. It talks to
the API in [06-API-SPECIFICATION.md](06-API-SPECIFICATION.md). See
[DEMO.md](DEMO.md) to run it. The page designs it follows are in
[`design/`](design/).

## Pages

| Route | Page | What it shows | API |
|---|---|---|---|
| `/` | **Queue** | Calls in `awaiting_review`, with a pending count per language, a language filter, grounding coverage against the threshold, why each call escalated, and how long it has waited. Refreshes every 15 s. | `GET /reviews` |
| `/calls/:callId` | **Review** | The redacted transcript, each turn in its own text direction. The grounded claims, each linked to its evidence span. A "Rejected by grounding" panel. The memory rules retrieved for this call. Call summary, coverage, and a PII-redacted count. | `GET /calls/{id}`, `POST /reviews/{id}/corrections` |
| `/new` | **New call** | Audio upload (`.wav`/`.mp3`/`.m4a`, up to 50 MB). The language is required and the agent id is optional. Shows the pipeline's progress, plus recent uploads from this browser only. | `POST /calls/audio`, `GET /calls/{id}` |
| `/memory` | **Memory** | Per-language counts of active rules and pending conflicts, and the conflict detector's false-positive rate. Pending conflicts sit beside the rules they contradict, with four resolutions. Also the table of active rules. | `GET /stats`, `GET /memory/rules`, `GET /memory/conflicts`, `POST /memory/conflicts/{id}/resolve` |
| `/results` | **Results** | The Phase 6.3 numbers as recorded, per language: memory effect against Phase 4 ± noise with the pre-registered verdict, and the cloud vs self-hosted serving table with n, hardware, cost and egress. It recomputes nothing. | `GET /results` |

The backend sends codes and numbers, never UI text (for example
`EscalationInfo.kind` = `low_grounding` | `analysis_error`). The frontend
translates them ([09-DECISIONS.md](09-DECISIONS.md), 2026-10-10).

## Call lifecycle

```
POST /calls/audio → transcribing ─┐
POST /calls ──────────────────────┴→ queued → processing ─┬→ completed        (coverage ≥ threshold)
                                                          ├→ awaiting_review → reviewed
                                                          └→ failed           (retries exhausted, ASR error)
```

Statuses come from `sawti.schemas.CallStatus`. The default threshold is 0.7
(`SAWTI_CONFIDENCE_THRESHOLD`).

## Review flow

1. The reviewer opens a call from the queue.
2. Each grounded claim needs exactly one verdict:
   - **Confirm.** Nothing is stored beyond the verdict.
   - **Correct.** The reviewer edits the claim's fields. The evidence quote can't be
     edited. A correction that changes nothing is refused (422).
   - **Reject.** The claim is wrong.
3. Submit. The submission must cover exactly the call's grounded claims (422
   otherwise). A second submission gets 409. The paused graph run resumes and the
   call moves to `reviewed`.
4. The verdict buttons and submit are disabled whenever the call is not
   `awaiting_review`, or has just been submitted, and the page shows the reason
   next to them.

## Correction → memory flow

Each **reject** and **correct** verdict becomes a `Correction`. Confirm does
not. The memory loop runs asynchronously on the worker:

```
Correction → induce a candidate rule (LLM) → conflict check against similar active rules (LLM)
   ├─ no conflict → store as an active rule → consolidate the store (complete linkage, cosine ≥ 0.7)
   └─ conflict    → hold in memory_conflicts for a human (not stored)
```

Later analyses retrieve the top 5 active rules (`SAWTI_MEMORY_TOP_K`) by
similarity to the redacted transcript and add them to the extraction prompt.

## Conflict resolution

| Action | Candidate rule | Conflicting existing rules |
|---|---|---|
| Keep existing | dropped | unchanged |
| Keep new | activated | retired |
| Discard both | dropped | retired |
| Not a real conflict | activated | unchanged (counts as a detector false positive) |

`keep_both` is refused (422), because two contradicting active rules could be
retrieved into the same prompt.

**Rules named by an open conflict are held out of consolidation** (fixed in
Phase 7.1). Before the fix, consolidation could merge such a rule away, and
then *Keep new* and *Discard both* retired nothing
([09-DECISIONS.md](09-DECISIONS.md), 2026-10-11).

An existing rule can still show an **"already retired"** badge. That happens
when resolving another conflict retired it. Every action stays selectable;
the dashboard never disabled any of them.

## What the UI does not claim

- A grounded claim only means its quote is in the transcript. It doesn't mean
  the quote supports the claim
  ([eval_results.md](../eval_results.md), 2026-10-11, `call_0003_mixed`).
- A `completed` call has not been checked for ASR quality. A garbled
  transcript can auto-complete (same source).
- "Recent uploads" lives in this browser tab's `sessionStorage` (last 5). The API has no endpoint for listing calls.
