/**
 * An in-memory stand-in for the Sawti API, shared by Vitest (as a `fetch`
 * stub) and Playwright (as a `page.route` handler). It answers only what the
 * dashboard calls and records every submission, so tests can assert the exact
 * request bodies.
 */
import type { CallDetail, ConflictAction, Schemas } from "../api/client";
import { memoryConflicts, memoryRules, mixedCall, queue, stats } from "./fixtures";
import resultsJson from "./results.json" with { type: "json" };

/** `GET /results` exactly as the real API serves it (pinned by tests/api/routes/test_results.py). */
export const results = resultsJson as Schemas["ResultsResponse"];

/** What the dashboard's upload form sends to `POST /calls/audio`. */
export interface AudioUpload {
  fileName: string;
  size: number;
  language_hint: string | null;
  agent_external_id: string | null;
}

const AUDIO_TYPES = [".wav", ".mp3", ".m4a"];
let uploadCounter = 0;

export interface MockResponse {
  status: number;
  body: unknown;
}

export class MockApi {
  readonly calls = new Map<string, CallDetail>();
  readonly submissions: { callId: string; body: Schemas["ReviewSubmission"] }[] = [];
  readonly requests: { method: string; path: string; headers: Record<string, string> }[] = [];
  /** GETs of a reviewed call before it reports `reviewed`: the worker's resume is asynchronous. */
  resumeAfterPolls = 1;
  private pendingResume = new Map<string, number>();
  readonly uploads: AudioUpload[] = [];
  /** The statuses an uploaded call moves through, one per GET after the upload; the last one sticks. */
  audioProgress: CallDetail["status"][] = ["transcribing", "queued", "processing", "awaiting_review"];
  private audioSteps = new Map<string, number>();
  rules = memoryRules();
  conflicts = memoryConflicts();
  readonly resolutions: { conflictId: string; action: ConflictAction; reviewer: string }[] = [];
  /** Set to answer `GET /results` with 503, as the API does when the 6.3 files are missing. */
  resultsMissing = false;

  constructor(private readonly now: Date = new Date()) {
    const call = mixedCall();
    this.calls.set(call.call_id, call);
  }

  handle(method: string, url: string, headers: Record<string, string>, body: unknown): MockResponse {
    const path = new URL(url, "http://localhost").pathname.replace(/^\/api/, "");
    this.requests.push({ method, path, headers });
    if (headers["x-api-key"] === undefined || headers["x-api-key"] === "") {
      return { status: 401, body: { detail: "missing or invalid API key" } };
    }
    if (method === "GET" && path === "/reviews") return { status: 200, body: queue(this.now) };
    if (method === "GET" && path === "/stats") return { status: 200, body: stats() };
    if (method === "GET" && path === "/results") {
      return this.resultsMissing ? { status: 503, body: { detail: "results not recorded yet" } } : { status: 200, body: results };
    }
    if (method === "GET" && path === "/memory/rules") return { status: 200, body: { rules: this.rules } };
    if (method === "GET" && path === "/memory/conflicts") {
      const pending = { ar: 0, en: 0, mixed: 0 };
      for (const c of this.conflicts) pending[c.source.language] += 1;
      return { status: 200, body: { conflicts: this.conflicts, pending } };
    }
    const resolveMatch = path.match(/^\/memory\/conflicts\/([^/]+)\/resolve$/);
    if (method === "POST" && resolveMatch) return this.resolve(resolveMatch[1]!, body as Schemas["ConflictResolveRequest"], headers);
    if (method === "POST" && path === "/calls/audio") return this.upload(body);

    const callMatch = path.match(/^\/calls\/([^/]+)$/);
    if (method === "GET" && callMatch) {
      const call = this.calls.get(callMatch[1]!);
      if (!call) return { status: 404, body: { detail: "no such call" } };
      const step = this.audioSteps.get(call.call_id);
      if (step !== undefined) {
        this.advanceUpload(call, step);
        this.audioSteps.set(call.call_id, step + 1);
      }
      const left = this.pendingResume.get(call.call_id);
      if (left !== undefined) {
        if (left <= 0) {
          call.status = "reviewed";
          this.pendingResume.delete(call.call_id);
        } else this.pendingResume.set(call.call_id, left - 1);
      }
      return { status: 200, body: call };
    }

    const reviewMatch = path.match(/^\/reviews\/([^/]+)\/corrections$/);
    if (method === "POST" && reviewMatch) {
      const call = this.calls.get(reviewMatch[1]!);
      if (!call) return { status: 404, body: { detail: "no such call" } };
      if (call.status !== "awaiting_review") return { status: 409, body: { detail: "already reviewed" } };
      const submission = body as Schemas["ReviewSubmission"];
      this.submissions.push({ callId: call.call_id, body: submission });
      call.review = submission;
      this.pendingResume.set(call.call_id, this.resumeAfterPolls);
      const corrections = submission.verdicts.filter((v) => v.verdict !== "confirm").length;
      return {
        status: 202,
        body: { call_id: call.call_id, submission_id: "5ab0c1d2-0000-4000-8000-0000000000ff", corrections_recorded: corrections },
      };
    }
    return { status: 404, body: { detail: `mock API has no route for ${method} ${path}` } };
  }

  private resolve(conflictId: string, request: Schemas["ConflictResolveRequest"], headers: Record<string, string>): MockResponse {
    const conflict = this.conflicts.find((c) => c.id === conflictId);
    if (!conflict) return { status: 404, body: { detail: "no such conflict" } };
    const action = request.action;
    this.conflicts = this.conflicts.filter((c) => c.id !== conflictId);
    this.resolutions.push({ conflictId, action, reviewer: headers["x-reviewer-id"] ?? "" });
    const activates = action === "keep_new" || action === "not_a_conflict";
    const retires = action === "keep_new" || action === "discard";
    const retired = retires ? conflict.conflicting.map((r) => r.rule_id) : [];
    this.rules = this.rules.filter((r) => !retired.includes(r.id));
    if (activates) {
      this.rules = [
        { id: conflict.candidate_rule_id, text: conflict.candidate_text, source_corrections: 1, languages: [conflict.source.language], analyses_used: 0, created_at: this.now.toISOString() },
        ...this.rules,
      ];
    }
    const response: Schemas["ConflictResolveResponse"] = {
      conflict_id: conflictId,
      action,
      resolved_by: headers["x-reviewer-id"] ?? "",
      resolved_at: this.now.toISOString(),
      activated_rule_id: activates ? conflict.candidate_rule_id : null,
      retired_rule_ids: retired,
      consolidation_queued: activates,
    };
    return { status: 200, body: response };
  }

  /** `POST /calls/audio`, with the API's own checks on type and form fields. */
  private upload(body: unknown): MockResponse {
    // Duck-typed: under Vitest the parsed form and its files are undici's classes, not jsdom's globals.
    const form = body as FormData | undefined;
    if (typeof form?.get !== "function") return { status: 422, body: { detail: "expected multipart/form-data" } };
    const file = form.get("file") as File | string | null;
    const language = form.get("language_hint");
    if (file === null || typeof file === "string" || typeof language !== "string") {
      return { status: 422, body: { detail: "file and language_hint are required" } };
    }
    const fileName = file.name ?? "";
    const suffix = fileName.slice(fileName.lastIndexOf(".")).toLowerCase();
    if (!AUDIO_TYPES.includes(suffix)) return { status: 415, body: { detail: `unsupported audio type ${suffix}` } };
    const agent = form.get("agent_external_id");
    this.uploads.push({ fileName, size: file.size, language_hint: language, agent_external_id: typeof agent === "string" ? agent : null });
    uploadCounter += 1;
    const callId = `5e1f0a00-0000-4000-8000-${String(uploadCounter).padStart(12, "0")}`;
    this.calls.set(callId, {
      call_id: callId,
      status: "transcribing",
      language: language as CallDetail["language"],
      agent_name: typeof agent === "string" ? agent : "unattributed",
      queued_at: null,
      started_at: null,
      finished_at: null,
      pii_redacted_count: 0,
      attempts: 0,
      failure_reason: null,
      redacted_transcript: "",
      turns: [],
      claims: [],
      rejected_claims: [],
      memory_rules: [],
      review: null,
    });
    this.audioSteps.set(callId, 0);
    return { status: 202, body: { call_id: callId } };
  }

  private advanceUpload(call: CallDetail, step: number): void {
    const status = this.audioProgress[Math.min(step, this.audioProgress.length - 1)]!;
    call.status = status;
    // "failed" here is an ASR failure: no transcript, never queued for analysis.
    if (status !== "transcribing" && status !== "failed") {
      call.pii_redacted_count = 3;
      call.queued_at = "2026-10-11T09:00:41Z";
    }
    if (status === "processing" || status === "awaiting_review" || status === "completed") call.started_at = "2026-10-11T09:00:42Z";
    if (status === "awaiting_review" || status === "completed" || status === "failed") call.finished_at = "2026-10-11T09:00:50.100Z";
    if (status === "awaiting_review") {
      call.grounding_coverage = 0.67;
      call.confidence_threshold = 0.7;
      call.escalation = { kind: "low_grounding", rejected_claims: 3, total_claims: 9, grounding_coverage: 0.67, threshold: 0.7 };
    }
    if (status === "failed") call.failure_reason = "asr: no speech detected";
  }

  /** A `fetch` replacement for Vitest. */
  fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const request = input instanceof Request ? input : new Request(new URL(String(input), "http://localhost"), init);
    const headers = Object.fromEntries(request.headers.entries());
    const { status, body } = this.handle(request.method, request.url, headers, await readBody(request));
    return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  };
}

/** A request body as the mock wants it: parsed JSON, `FormData` for multipart, or undefined. */
export async function readBody(request: Request): Promise<unknown> {
  if (request.method === "GET") return undefined;
  if ((request.headers.get("content-type") ?? "").startsWith("multipart/form-data")) return request.formData();
  const text = await request.text();
  return text ? JSON.parse(text) : undefined;
}
