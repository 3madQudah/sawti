# 13 — Security

**Scope.** Sawti runs on one developer machine, on synthetic data. It has no
real authentication and has never been deployed. Nothing here makes it safe
for real customer calls. This file lists what is protected, what isn't, and
the bugs found along the way.

## PII redaction

`sawti.privacy.redaction.redact` is regex-only
([09-DECISIONS.md](09-DECISIONS.md), 2026-09-22).

| Covered (placeholder) | Pattern |
|---|---|
| `[EMAIL]` | standard address shape |
| `[IBAN]` | `JO` + 2 check digits + 4 bank characters + up to 18 alphanumerics |
| `[CARD]` | 13–19 digits, optionally grouped by spaces or dashes |
| `[NATIONAL_ID]` | exactly 10 consecutive digits (wins over phone) |
| `[PHONE]` | `+`/`00` international forms, Jordanian `0…`/`٠…` + 8–9 digits (any single separators), 3-3-4 and 3-4 with separators (widened 2026-10-11 after a production miss) |
| `[NUMBER]` | catch-all, applied last: any other run of 7+ digits, with at most one space or dash between digits. Dots don't join a run. (Added 2026-10-11.) |
| *(never redacted)* | UUIDs and ISO-8601 timestamps with a time part pass through unchanged. They aren't PII, and the digit patterns used to break them inside Langfuse traces (fixed 2026-10-11). |

Python's `\d` also matches Arabic-Indic digits, so both digit sets are covered.

**What it misses:**

- **Personal names are not redacted.** The redactor is regex-based and has no
  NER, so names (for example a caller giving their own name in Arabic or
  English), addresses, employers, and anything that is only identifiable from
  context reach the model and the database. Catching these needs NER, and
  `tests/privacy/test_redaction.py::test_redact_removes_names_in_arabic_and_english` stays
  skipped so the gap shows in every test run.
- Identifiers shorter than 7 digits (PINs, short reference numbers), and
  long numbers joined by dots. After the catch-all, 0 / 0 / 0 runs of 7+
  digits are left in the ar / en / mixed ASR transcripts. The clean `mixed`
  text has 1 left, a dotted private router address that isn't PII
  ([eval_results.md](../eval_results.md), 2026-10-11 catch-all round).
- Numbers written out as words. Every pattern is digit-based.
- Anything ASR mangles into a shape the regex doesn't see.

**Where it runs.** At the `POST /calls` boundary, in the ASR worker before
anything is stored, in `persist_analysis_record` (the eval scripts' path),
defensively in `extract`, on rule text before storage, and on every Langfuse
payload string as a backstop (`tests/observability/test_tracing.py` sends
known PII and asserts it reaches no trace).

**The trade-off.** The raw transcript is never stored. That also means a
missed redaction is permanent, and a false positive can't be undone
([09-DECISIONS.md](09-DECISIONS.md), 2026-10-06, decision 1).

**Raw audio** is deleted once transcription succeeds, unless
`SAWTI_KEEP_AUDIO=true`. `data/uploads/` is gitignored. The ASR result's
`repr` hides its text, so logging it by accident can't leak the transcript,
and failure reasons never carry transcript text.

## PII bugs found, and what was done

| Bug | Effect | Fix | Source |
|---|---|---|---|
| 7 quotes inside `reviewer_actions` payloads held **raw phone numbers**. They came from ground truth written against raw transcripts. | Raw PII would have survived in the JSON after the transcripts were redacted. | Redacted and re-anchored during the 0001 conversion, which then found 0 quotes with PII left in 4,286. | [09-DECISIONS.md](09-DECISIONS.md), 2026-10-07 dev-DB conversion |
| **Found in production, 2026-10-11.** Phone numbers in 3-3-4 / 3-4 format (and several Jordanian and Arabic-Indic groupings) passed redaction. An uploaded call sent one to Gemini and Langfuse. | The redact-before-model rule was broken. 11 stored `en` calls hold 14 such spans. | Phone patterns widened (42 tests), then a 7+ digit catch-all added (19 tests). On the ASR transcripts, en spans went from 1 to 26 and leftover runs to 0 in every language. **Stored data was deliberately not rewritten** (see below). | [eval_results.md](../eval_results.md), [09-DECISIONS.md](09-DECISIONS.md), 2026-10-11 |
| Redaction mangled UUIDs and ISO timestamps (regression from the 2026-10-11 phone fix and catch-all). The trace mask runs `redact()` on every string. | Not a leak: an integrity bug. One trace and its 6 observations carry a corrupted call id in their metadata (its `session_id` is intact), and the tracing test was flaky. | UUIDs and timestamps are protected, with 8 tests. Corpus redaction counts are unchanged. | [09-DECISIONS.md](09-DECISIONS.md), 2026-10-11 |
| Grounding metrics checked the raw text while `ground` checked the redacted text. | A legitimate quote containing `[PHONE]` scored as unsupported. This was a measurement bias, not a leak. | The metrics now redact before checking. `mixed` unsupported rate went from 0.0039 to 0.000 on the 103 stored analyses. | [eval_results.md](../eval_results.md), 2026-10-07 re-score |
| A test patched the provider only in `extract`, so induction called the **real** Gemini API, about 4 requests carrying synthetic test text. | Test text left the machine. | Every LLM caller is now patched, and service tests set `SAWTI_LLM_PROVIDER=fake`. | [09-DECISIONS.md](09-DECISIONS.md), 2026-10-07 closing 6.2 |
| The Phase 5 adapter was trained on raw transcripts. 2 of 18 training records contain a PII-shaped span (synthetic). | The training data held raw PII. The pattern would matter with real data. | **Not fixed.** The adapter was not retrained. | [09-DECISIONS.md](09-DECISIONS.md), 2026-10-07 |
| `data/backups/sawti-pre-0001-*.dump` holds the raw, unredacted transcripts. | The only raw copy left. | Gitignored. It is still on disk. | [09-DECISIONS.md](09-DECISIONS.md), 2026-10-07 |

### The leaked number: where it went, and why it was left there

The phone number missed on 2026-10-11 (an uploaded `call_0006_en`, call
`89c66124…`) **reached Gemini**, in the analysis request. It is also stored in
three places in the local stack:

- the call's stored analysis and redacted transcript in Postgres;
- the LangGraph **checkpoint** for that run;
- **one Langfuse trace**. One observation in it, the LLM input, contains the
  number, because the trace mask used the same pre-fix `redact()`.

The same pre-fix patterns also left 13 more spans in 10 other stored `en`
calls, which came from the eval scripts. That text went to Gemini when those
calls were analyzed.

**None of it was rewritten, by decision (2026-10-11).** Every one of these
numbers is synthetic: the calls are TTS renderings of generated transcripts,
and no real person's data is involved. Rewriting would mean the following:

- re-redacting stored text and re-anchoring every quote offset, as the 0001
  conversion did;
- editing checkpoint state and a trace;
- changing the audit trail behind numbers already recorded in
  `eval_results.md`.

That is a lot of risk for no privacy gain on synthetic data. **With real data,
the answer would be the opposite.** Rewrite or delete every stored copy, ask
the provider to delete the request where its terms allow, and purge the
trace. The Gemini request itself can't be recalled from this side.

**Verified fixed, live (2026-10-11).** After the fix, the stack was rebuilt
and the native ASR worker restarted. Then `call_0006_en` was uploaded again
(call `8b768603…`):

- Its stored transcript shows `[PHONE]` where the number was, and
  `pii_redacted_count` is 1.
- No 3-3-4 number is left in it.
- Its one Langfuse trace has 6 observations, and 0 of them contain the number.
  The LLM input shows `[PHONE]`.

`make stack` alone was not enough. The native ASR worker, which redacts
uploaded audio, kept the old patterns until it was restarted.

## Known limits (not fixed)

| Limit | Why it matters here | Source |
|---|---|---|
| **Personal names are not redacted.** Regex only, no NER. Addresses and employers are missed too. | Names in a transcript go to Gemini, into the database, and into traces. Not fit for real calls. | [09-DECISIONS.md](09-DECISIONS.md), 2026-09-22; the skipped `test_redact_removes_names_in_arabic_and_english` |
| Numbers under 7 digits that don't match a specific pattern (PINs, short references), dotted runs, numbers spoken as words. | They pass redaction. | [eval_results.md](../eval_results.md), 2026-10-11 catch-all round |
| **Diarization misalignment.** On the uploaded `call_0006_en`, turns #3–#4 (and #7–#8) mix agent and customer speech, because each Whisper segment goes wholly to one speaker. | Redaction still runs on every turn, so nothing is missed because of it. But the transcript attributes speech, including PII the customer spoke, to the wrong party, and an evidence quote can carry the wrong speaker. Accuracy has never been measured. | [09-DECISIONS.md](09-DECISIONS.md), 2026-10-11 known issue |
| Data already leaked (above) stays in the database, checkpoint and one trace. | It's synthetic. With real data, this would have to be purged. | this file |

## Identity and access

- `X-API-Key` is one shared key (`SAWTI_API_KEY`), compared in constant time.
  Unset or empty means 503 everywhere (fail closed).
- `X-Reviewer-Id` is **self-asserted**. Anyone with the key can act as anyone,
  including resolving conflicts that retire memory rules.
- `make demo` puts the key into the dashboard's JavaScript bundle, so anyone
  who can load the page can read it.
- This is not authentication, and the decision log says so
  ([09-DECISIONS.md](09-DECISIONS.md), 2026-10-06, decision 3; 2026-10-10).
  There are no roles and no OIDC.

## Secrets handling

- Secrets live in `.env` (gitignored). `.env.example` lists the names with
  generation hints and no values. See [14-DEPLOYMENT.md](14-DEPLOYMENT.md).
- `SAWTI_API_KEY` is a pydantic `SecretStr`. The other keys (`GEMINI_API_KEY`,
  `HF_TOKEN`, `LANGFUSE_*`) are plain settings read from the environment.
- `eval_results.md` identifies Gemini keys by a 12-character fingerprint,
  never by the key itself.
- Langfuse must stay self-hosted. Traces carry redacted transcript text, and
  `.env.example` warns never to point it at Langfuse Cloud.

## Network exposure (local stack)

- Compose publishes Postgres, Redis (6379), the API (8000) and Langfuse
  (3000) on **all interfaces**, not only `127.0.0.1`. On a shared network,
  other machines can reach them.
- Postgres falls back to `sawti`/`sawti` credentials when `.env` doesn't set
  them. Redis has no password.
- CORS defaults to no allowed origins. The dashboard goes through the Vite
  proxy instead.

## Data leaving the machine

| Destination | What | When |
|---|---|---|
| Gemini API | Redacted transcripts, memory-rule text, induction and conflict prompts | Every analysis and induction |
| Kaggle (Google) | Redacted synthetic transcripts of the 45 benchmark calls, uploaded as a private dataset, plus the adapter | 6.3 benchmark only ([eval_results.md](../eval_results.md), 2026-10-10) |
| Hugging Face | Model downloads only (Whisper, pyannote, embeddings) | First run |

## Threat model (brief)

| Threat | Mitigated? | How / gap |
|---|---|---|
| PII sent to the LLM provider | Partly | Structured identifiers are redacted. Names and addresses are not. |
| PII at rest in the database | Partly | Only redacted text is stored, with the same regex gap. The backup dump is raw. |
| PII in traces or logs | Yes, for structured identifiers | Langfuse payloads are re-redacted. ASR text is never logged. |
| Unauthorized API use | Weakly | A shared key, visible in the browser bundle. Acceptable only on localhost. |
| Reviewer impersonation | No | The reviewer id is self-asserted. |
| Model hallucination presented as fact | Partly | Unquoted claims are dropped. A quote that exists but doesn't support the claim passes ([eval_results.md](../eval_results.md), 2026-10-11). |
| Bad ASR silently auto-completing | No | Routing ignores ASR quality. A fix is planned ([09-DECISIONS.md](09-DECISIONS.md), 2026-10-11 TODO). |
| Prompt injection via transcript content | Not assessed | Output is schema-validated and quotes are grounded. No injection test exists. |
| Exposed infrastructure ports | No | Ports bind on all interfaces, with default credentials. |
