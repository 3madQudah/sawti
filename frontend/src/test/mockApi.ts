/**
 * An in-memory stand-in for the Sawti API, shared by Vitest (as a `fetch`
 * stub) and Playwright (as a `page.route` handler). It answers only what the
 * dashboard calls and records every submission, so tests can assert the exact
 * request bodies.
 */
import type { CallDetail, Schemas } from "../api/client";
import { mixedCall, queue } from "./fixtures";

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

    const callMatch = path.match(/^\/calls\/([^/]+)$/);
    if (method === "GET" && callMatch) {
      const call = this.calls.get(callMatch[1]!);
      if (!call) return { status: 404, body: { detail: "no such call" } };
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

  /** A `fetch` replacement for Vitest. */
  fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const request = input instanceof Request ? input : new Request(new URL(String(input), "http://localhost"), init);
    const text = request.method === "GET" ? "" : await request.text();
    const headers = Object.fromEntries(request.headers.entries());
    const { status, body } = this.handle(request.method, request.url, headers, text ? JSON.parse(text) : undefined);
    return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  };
}
