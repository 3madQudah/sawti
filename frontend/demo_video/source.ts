/**
 * Where the demo's data comes from.
 *
 * With `data/demo_capture/after.json` (and `before.json` if taken; scripts/capture_demo_api.py)
 * the browser is answered with the responses of a real run: the transcript,
 * claims, coverage, PII count, memory and numbers are what Sawti produced.
 * The only synthesized parts are the pipeline's in-between statuses, rebuilt
 * from the real call with its results hidden until it reaches them.
 *
 * Without a capture it falls back to the e2e mock, clearly labelled.
 */
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import type { CallDetail, ClaimVerdict } from "../src/api/client";
import { MockApi, type MockResponse } from "../src/test/mockApi";

export interface CorrectionStep {
  claimId: string;
  field: string;
  value: string;
  evidence: string;
}

/** Numbers the captions quote, computed from the data on screen rather than typed in. */
export interface DemoFacts {
  coverage: number | null;
  threshold: number | null;
  pii: number;
  rejected: number;
  /** Rejected quotes that are in the transcript verbatim once speaker labels are removed: split by a turn boundary. */
  boundarySplit: number;
  rulesActive: number;
  rulesUsed: number;
  /** The 1-based turn where the first split quote starts (`turn-N` on screen), if any. */
  splitTurn: number | null;
}

export interface DemoPlan {
  kind: "replay" | "mock";
  facts: DemoFacts;
  audioName: string;
  language: CallDetail["language"];
  /** The verdicts to click, in order: the real review's when replaying. */
  verdicts: { claimId: string; verdict: ClaimVerdict["verdict"] }[];
  correction: CorrectionStep | null;
}

export interface DemoSource {
  plan: DemoPlan;
  handle(method: string, url: string, headers: Record<string, string>, body: unknown): MockResponse;
}

const CAPTURE_DIR = resolve(process.env.DEMO_CAPTURE ?? "../data/demo_capture");

export interface Capture {
  call_id: string;
  audio_name: string | null;
  responses: Record<string, unknown>;
}

/** The verified evidence travels in the verdict note: `... (verbatim, turn n, chars a-b): «text»`. */
export function evidenceFromNote(note: string | null | undefined): string {
  const m = note?.match(/«([^»]*)»/);
  return m ? m[1]! : "";
}

export function correctionFrom(verdicts: readonly ClaimVerdict[]): CorrectionStep | null {
  const v = verdicts.find((x) => x.verdict === "correct" && x.changes);
  if (!v || !v.changes) return null;
  const [field, value] = Object.entries(v.changes).find(([, val]) => val !== null && val !== undefined) ?? [];
  if (!field) return null;
  return { claimId: v.claim_id, field, value: String(value), evidence: evidenceFromNote(v.note) };
}

/** The real call as it looked at an earlier status: results that did not exist yet are hidden. */
export function callAt(call: CallDetail, status: CallDetail["status"]): CallDetail {
  if (status === call.status) return call;
  const beforeAnalysis = status === "transcribing" || status === "queued";
  return {
    ...call,
    status,
    queued_at: status === "transcribing" ? null : call.queued_at,
    started_at: beforeAnalysis ? null : call.started_at,
    finished_at: null,
    pii_redacted_count: status === "transcribing" ? 0 : call.pii_redacted_count,
    redacted_transcript: status === "transcribing" ? "" : call.redacted_transcript,
    turns: status === "transcribing" ? [] : call.turns,
    claims: [],
    rejected_claims: [],
    memory_rules: [],
    grounding_coverage: null,
    escalation: null,
    review: null,
  };
}

/** The transcript as one text without the `Speaker: ` labels a diarization boundary inserts. */
export function withoutSpeakerLabels(transcript: string): string {
  return transcript.replace(/\n(?:Agent|Customer): /g, " ");
}

export function factsFrom(call: CallDetail, rules: { rules: unknown[] } | undefined): DemoFacts {
  const flat = withoutSpeakerLabels(call.redacted_transcript);
  const rejected = call.rejected_claims ?? [];
  const split = rejected
    .map((r) => (r as { attempted_quote?: string }).attempted_quote ?? "")
    .filter((q) => q !== "" && !call.redacted_transcript.includes(q) && flat.includes(q));
  const head = split[0]?.slice(0, 25);
  const turn = head ? call.turns.find((t) => t.text.includes(head)) : undefined;
  return {
    splitTurn: turn ? turn.index + 1 : null,
    coverage: call.grounding_coverage ?? null,
    threshold: call.confidence_threshold ?? null,
    pii: call.pii_redacted_count,
    rejected: rejected.length,
    boundarySplit: split.length,
    rulesActive: rules?.rules.length ?? 0,
    rulesUsed: call.memory_rules?.length ?? 0,
  };
}

const PROGRESS: CallDetail["status"][] = ["transcribing", "transcribing", "queued", "processing", "processing"];

function replay(before: Capture, after: Capture): DemoSource {
  const callId = before.call_id;
  const realBefore = before.responses[`/calls/${callId}`] as CallDetail;
  const realAfter = after.responses[`/calls/${callId}`] as CallDetail;
  const verdicts = realAfter.review?.verdicts ?? [];
  let uploaded = false;
  let polls = 0;
  let submitted = false;
  let pollsAfterSubmit = 0;
  const snap = () => (submitted ? after : before).responses;

  return {
    plan: {
      kind: "replay",
      facts: factsFrom(realAfter, after.responses["/memory/rules"] as { rules: unknown[] }),
      audioName: before.audio_name ?? after.audio_name ?? `${callId.slice(0, 8)}.wav`,
      language: realBefore.language,
      verdicts: verdicts.map((v) => ({ claimId: v.claim_id, verdict: v.verdict })),
      correction: correctionFrom(verdicts),
    },
    handle(method, url) {
      const path = new URL(url, "http://localhost").pathname.replace(/^\/api/, "");
      if (method === "GET" && path === "/results") return { status: 200, body: before.responses["/results"] };
      if (method === "GET" && ["/stats", "/reviews", "/memory/rules", "/memory/conflicts"].includes(path)) {
        return { status: 200, body: snap()[path] };
      }
      if (method === "POST" && path === "/calls/audio") {
        uploaded = true;
        return { status: 202, body: { call_id: callId } };
      }
      if (method === "GET" && path === `/calls/${callId}`) {
        if (submitted) {
          pollsAfterSubmit += 1;
          return { status: 200, body: pollsAfterSubmit > 1 ? realAfter : { ...realBefore, review: realAfter.review } };
        }
        if (!uploaded) return { status: 200, body: realBefore };
        const status = PROGRESS[polls] ?? realBefore.status;
        polls += 1;
        return { status: 200, body: callAt(realBefore, status) };
      }
      if (method === "POST" && path === `/reviews/${callId}/corrections`) {
        submitted = true;
        const corrections = verdicts.filter((v) => v.verdict !== "confirm").length;
        return { status: 202, body: { call_id: callId, submission_id: "00000000-0000-4000-8000-000000000000", corrections_recorded: corrections } };
      }
      return { status: 404, body: { detail: `demo replay has no response for ${method} ${path}` } };
    },
  };
}

function mock(): DemoSource {
  const api = new MockApi();
  api.audioProgress = ["transcribing", "transcribing", "queued", "processing", "processing", "awaiting_review"];
  const others = ["commitments[1]", "compliance_flags[0]", "rubric_scores[0]", "rubric_scores[1]", "rubric_scores[2]"];
  let uploadedId = "";
  return {
    plan: {
      kind: "mock",
      facts: factsFrom([...api.calls.values()][0]!, { rules: api.rules }),
      audioName: "call_0067_mixed.wav",
      language: "mixed",
      verdicts: [{ claimId: "commitments[0]", verdict: "correct" }, ...others.map((claimId) => ({ claimId, verdict: "confirm" as const }))],
      correction: { claimId: "commitments[0]", field: "description", value: "Refund credited within 3 business days", evidence: "خلال ٣ أيام عمل" },
    },
    handle(method, url, headers, body) {
      // The mock's uploaded call is empty; show the fixture call's results once it is routed.
      const response = api.handle(method, url, headers, body);
      if (method === "POST" && url.includes("/calls/audio") && response.status === 202) {
        uploadedId = (response.body as { call_id: string }).call_id;
      }
      if (uploadedId && method === "GET" && url.includes(`/calls/${uploadedId}`)) {
        const call = response.body as CallDetail;
        if (call.status === "awaiting_review" || call.status === "reviewed") {
          const fixture = [...api.calls.values()][0]!;
          const { status, review, queued_at, started_at, finished_at } = call;
          return { status: 200, body: { ...fixture, call_id: uploadedId, status, review, queued_at, started_at, finished_at } };
        }
      }
      return response;
    },
  };
}

/**
 * The pre-review state, rebuilt from a capture taken after the review: the same
 * analysis with the review removed and the status set back to awaiting_review.
 * Only the call is rebuilt; memory, stats and the queue are not shown before
 * the submission in the video, so their post-review values are never misplaced.
 */
export function beforeFromAfter(after: Capture): Capture {
  const call = after.responses[`/calls/${after.call_id}`] as CallDetail;
  return {
    ...after,
    responses: { ...after.responses, [`/calls/${after.call_id}`]: { ...call, status: "awaiting_review", review: null } },
  };
}

export function demoSource(): DemoSource {
  const before = resolve(CAPTURE_DIR, "before.json");
  const after = resolve(CAPTURE_DIR, "after.json");
  const read = (p: string) => JSON.parse(readFileSync(p, "utf-8")) as Capture;
  if (existsSync(after)) {
    const a = read(after);
    const b = existsSync(before) ? read(before) : beforeFromAfter(a);
    if (a.call_id !== b.call_id) throw new Error(`before.json is call ${b.call_id}, after.json is ${a.call_id}`);
    return replay(b, a);
  }
  if (process.env.DEMO_REQUIRE_CAPTURE) throw new Error(`no capture in ${CAPTURE_DIR}`);
  return mock();
}
