# 06 — API specification

FastAPI app `sawti.api.main:create_app`, served on `http://localhost:8000`.
Interactive docs are at `/docs` while the stack runs. The dashboard reaches
the API through the Vite proxy at `/api`.

The endpoint and schema lists below are **generated** from
[`frontend/openapi.json`](../frontend/openapi.json) by
`scripts/render_api_doc.py`. Don't edit them by hand. After an API change, run:

```bash
make openapi api-doc
```

`tests/scripts/test_export_openapi.py` fails if `openapi.json` is stale, and
`tests/scripts/test_render_api_doc.py` fails if this file is.

## Identity (not authentication)

| Header | Checked by | Routes |
|---|---|---|
| `X-API-Key` | `require_api_key`. Must equal `SAWTI_API_KEY` (constant-time compare). | Everything except `/health` |
| `X-Reviewer-Id` | `get_current_reviewer`. Must be non-blank and at most 255 characters. | Every route except `/health`, `POST /calls`, `POST /calls/audio` |

The reviewer id is self-asserted, and the one shared key ends up in the
dashboard's browser bundle. This keeps casual callers out of a local stack,
nothing more ([09-DECISIONS.md](09-DECISIONS.md), 2026-10-06, decision 3).
If the key is unset or empty, every protected route answers **503** (fail
closed).

## Status codes raised in code

FastAPI's generated schema lists only 2xx and 422. The routes and
dependencies also raise the codes below (from `src/sawti/api/routes/` and
`src/sawti/api/deps.py`):

| Code | When |
|---|---|
| 401 | Missing or wrong `X-API-Key`, or a missing or invalid `X-Reviewer-Id`. |
| 404 | Unknown call (`GET /calls/{id}`, `POST /reviews/{id}/corrections`) or conflict (`…/resolve`). |
| 409 | Review submitted for a call that isn't `awaiting_review`, or that already has a submission. Also a conflict that is already resolved. |
| 413 | Audio larger than `SAWTI_MAX_AUDIO_BYTES` (default 50 MB). |
| 415 | Audio that isn't `.wav`/`.mp3`/`.m4a`, or whose content doesn't match its extension. |
| 422 | Request validation. Also: a transcript line with no `Agent:`/`Customer:` prefix, an empty audio file, verdicts that don't match the call's grounded claims exactly, a `correct` verdict that changes nothing or a field the claim kind doesn't have, `keep_both`. |
| 503 | `SAWTI_API_KEY` not configured, the task could not be enqueued, the memory write lock is busy (`…/resolve`), or the result files for `/results` are missing. `/health` also returns 503 when Postgres or Redis is down. |

## Behaviour notes

- `POST /calls` and `POST /calls/audio` return **202** with a `call_id`.
  Analysis is asynchronous, so poll `GET /calls/{id}`.
- `POST /calls/audio` requires `language_hint` (`ar` | `en` | `mixed`), which
  picks the forced Whisper language. It doesn't auto-detect.
- Claim ids are `"<list>[<index>]"` (for example `commitments[0]`).
- `GET /results` serves recorded numbers from
  `data/serving_benchmark/comparison.json` and
  `data/memory_comparison/report.json`. It recomputes nothing, and a test ties
  every served number to `eval_results.md`.
- `GET /memory/rules`: `analyses_used` counts only service analyses. In the
  dev database it is 0 for every rule, because the eval runs never recorded
  retrieval.

## Generated reference

<!-- BEGIN GENERATED from frontend/openapi.json by scripts/render_api_doc.py; do not edit -->

### Endpoints

| Method | Path | Summary | Headers | Path / query parameters | Request body | Success | Codes in schema |
|---|---|---|---|---|---|---|---|
| POST | `/calls` | Submit Call | `x-api-key` | — | CallSubmitRequest (`application/json`) | 202 CallSubmitResponse | 202, 422 |
| POST | `/calls/audio` | Submit Audio | `x-api-key` | — | Body_submit_audio_calls_audio_post (`multipart/form-data`) | 202 CallSubmitResponse | 202, 422 |
| GET | `/calls/{call_id}` | Get Call | `x-reviewer-id`, `x-api-key` | `call_id` (path) | — | 200 CallDetail | 200, 422 |
| GET | `/health` | Health | — | — | — | 200 HealthResponse | 200 |
| GET | `/memory/conflicts` | List Conflicts | `x-reviewer-id`, `x-api-key` | — | — | 200 MemoryConflictListResponse | 200, 422 |
| POST | `/memory/conflicts/{conflict_id}/resolve` | Resolve | `x-reviewer-id`, `x-api-key` | `conflict_id` (path) | ConflictResolveRequest (`application/json`) | 200 ConflictResolveResponse | 200, 422 |
| GET | `/memory/rules` | List Rules | `x-reviewer-id`, `x-api-key` | — | — | 200 MemoryRuleListResponse | 200, 422 |
| GET | `/results` | Results | `x-reviewer-id`, `x-api-key` | — | — | 200 ResultsResponse | 200, 422 |
| GET | `/reviews` | Review Queue | `x-reviewer-id`, `x-api-key` | `status` (query, optional), `language` (query, optional), `limit` (query, optional) | — | 200 ReviewQueueResponse | 200, 422 |
| POST | `/reviews/{call_id}/corrections` | Submit Review | `x-reviewer-id`, `x-api-key` | `call_id` (path) | ReviewSubmission (`application/json`) | 202 ReviewAccepted | 202, 422 |
| GET | `/stats` | Stats | `x-reviewer-id`, `x-api-key` | — | — | 200 StatsResponse | 200, 422 |

### Schemas

`?` marks an optional field.

- **Body_submit_audio_calls_audio_post**: `agent_external_id`?: string \| null, `file`: string (binary), `language_hint`: Language
- **CallDetail**: `agent_name`: string, `attempts`: integer, `call_id`: string (uuid), `claims`?: GroundedClaim[], `confidence`?: number \| null, `confidence_threshold`?: number \| null, `escalation`?: EscalationInfo \| null, `escalation_reason`?: string \| null, `failure_reason`?: string \| null, `finished_at`: string (date-time) \| null, `grounding_coverage`?: number \| null, `language`: Language, `memory_rules`?: MemoryRuleUsed[], `pii_redacted_count`: integer, `queued_at`: string (date-time) \| null, `redacted_transcript`: string, `rejected_claims`?: RejectedClaim[], `review`?: ReviewSubmission \| null, `started_at`: string (date-time) \| null, `status`: CallStatus, `summary`?: string \| null, `turns`: TranscriptTurn[]
- **CallStatus**: `transcribing` \| `queued` \| `processing` \| `awaiting_review` \| `reviewed` \| `completed` \| `failed`
- **CallSubmitRequest**: `agent_external_id`: string, `language`: Language, `transcript`: string
- **CallSubmitResponse**: `call_id`: string (uuid)
- **ClaimChanges**: `deadline`?: string (date-time) \| null, `description`?: string \| null, `justification`?: string \| null, `promised_by`?: string \| null, `rule_id`?: string \| null, `score`?: number \| null, `severity`?: Severity \| null
- **ClaimVerdict**: `changes`?: ClaimChanges \| null, `claim_id`: string, `note`?: string \| null, `verdict`: `confirm` \| `correct` \| `reject`
- **ConflictResolveRequest**: `action`: `keep_existing` \| `keep_new` \| `discard` \| `not_a_conflict`
- **ConflictResolveResponse**: `action`: `keep_existing` \| `keep_new` \| `discard` \| `not_a_conflict`, `activated_rule_id`: string (uuid) \| null, `conflict_id`: string (uuid), `consolidation_queued`: boolean, `resolved_at`: string (date-time), `resolved_by`: string, `retired_rule_ids`: string (uuid)[]
- **ConflictSourceView**: `call_id`: string (uuid), `corrected_at`: string (date-time), `correction_id`: string (uuid), `language`: Language, `reviewer_id`: string
- **ConflictingRuleView**: `active`: boolean, `reasoning`: string, `rule_id`: string (uuid), `similarity`: number, `text`: string \| null
- **EscalationInfo**: `grounding_coverage`: number \| null, `kind`: `low_grounding` \| `analysis_error`, `rejected_claims`: integer, `threshold`: number \| null, `total_claims`: integer
- **GroundedClaim**: `claim_id`: string, `criterion`?: string \| null, `deadline`?: string (date-time) \| null, `description`: string, `end_char`: integer, `kind`: `commitment` \| `compliance_flag` \| `rubric_score`, `promised_by`?: string \| null, `quote`: string, `rule_id`?: string \| null, `score`?: number \| null, `severity`?: Severity \| null, `speaker`: string, `start_char`: integer, `turn_index`: integer \| null
- **HealthResponse**: `database`: `ok` \| `error`, `redis`: `ok` \| `error`, `status`: `ok` \| `degraded`
- **Language**: `ar` \| `en` \| `mixed`
- **LanguageStats**: `active_rules`: integer, `conflict_false_positive_rate`: Rate, `pending_conflicts`: integer, `pending_reviews`: integer, `reviewer_agreement`: Rate
- **MemoryConflictListResponse**: `conflicts`: MemoryConflictView[], `pending`: PendingCounts
- **MemoryConflictView**: `candidate_rule_id`: string (uuid), `candidate_text`: string, `conflicting`: ConflictingRuleView[], `detected_at`: string (date-time), `id`: string (uuid), `source`: ConflictSourceView
- **MemoryLanguageResult**: `accuracy_off_per_repeat`: number[], `accuracy_on_per_repeat`: number[], `delta`: number, `delta_per_repeat`: number[], `n_paired_per_repeat`: integer[], `noise`: number, `phase4_delta`: number, `verdict`: `confirms` \| `contradicts` \| `inconclusive` \| `incomplete`
- **MemoryResults**: `ar`: MemoryLanguageResult, `calls`: integer, `en`: MemoryLanguageResult, `hardware`: string, `method`: string, `mixed`: MemoryLanguageResult, `provider`: string, `source`: string
- **MemoryRuleListResponse**: `rules`: MemoryRuleView[]
- **MemoryRuleUsed**: `id`: string (uuid), `text`: string
- **MemoryRuleView**: `analyses_used`: integer, `created_at`: string (date-time), `id`: string (uuid), `languages`: Language[], `source_corrections`: integer, `text`: string
- **PendingCounts**: `ar`: integer, `en`: integer, `mixed`: integer
- **Rate**: `n`: integer, `numerator`: integer, `value`: number \| null
- **RejectedClaim**: `attempted_quote`: string, `description`: string, `kind`: `commitment` \| `compliance_flag` \| `rubric_score`, `speaker`: string
- **ResultsResponse**: `memory`: MemoryResults, `serving`: ServingResults
- **ReviewAccepted**: `call_id`: string (uuid), `corrections_recorded`: integer, `submission_id`: string (uuid)
- **ReviewQueueItem**: `agent_name`: string, `call_id`: string (uuid), `escalation`?: EscalationInfo \| null, `escalation_reason`: string \| null, `grounding_coverage`: number \| null, `language`: Language, `waiting_since`: string (date-time) \| null
- **ReviewQueueResponse**: `items`: ReviewQueueItem[], `pending`: PendingCounts
- **ReviewSubmission**: `note`?: string \| null, `verdicts`: ClaimVerdict[]
- **ServingArmResult**: `ar`: ServingLanguageResult, `deployment`: `cloud` \| `self_hosted`, `en`: ServingLanguageResult, `hardware`: string, `id`: string, `label`: string, `mixed`: ServingLanguageResult, `models`: string[], `throughput`: ServingThroughput
- **ServingLanguageResult**: `accuracy`: number \| null, `cost_per_1000_usd`: number \| null, `cost_per_1000_usd_sensitivity`?: number \| null, `failures`: integer, `grounding_precision`: number \| null, `latency_n`: integer, `latency_p50_s`: number \| null, `latency_p95_s`: number \| null, `n`: integer, `rubric_agreement`: number \| null, `schema_valid`: integer, `schema_valid_rate`: number, `third_party_transcript_bytes_per_call`: number \| null, `unsupported_claim_rate`: number \| null
- **ServingResults**: `accuracy_comparable_across_providers`?: `False`, `actual_spend_usd`: number, `arms`: ServingArmResult[], `methods`: map<string>, `pricing`: map<string>, `source`: string
- **ServingThroughput**: `calls`: integer, `calls_per_min`: number, `concurrency`: integer, `excluded_as_host_sleep`: string[], `method`: string, `output_tokens_per_s`: number \| null
- **Severity**: `low` \| `medium` \| `high` \| `critical`
- **StatsResponse**: `active_rules_distinct`: integer, `ar`: LanguageStats, `en`: LanguageStats, `mixed`: LanguageStats
- **TranscriptTurn**: `end_char`: integer, `index`: integer, `language`: Language, `speaker`: `Agent` \| `Customer`, `start_char`: integer, `text`: string

<!-- END GENERATED -->
