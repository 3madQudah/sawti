/**
 * The typed API client. Every request and response type comes from
 * `schema.d.ts`, generated from `frontend/openapi.json` (`npm run gen:api`):
 * there are no hand-written copies of the API's shapes.
 *
 * Auth is the local shared key plus a self-asserted reviewer id, from env vars.
 * It is not authentication, and the key is visible in the bundle: acceptable
 * only because this dashboard never leaves localhost (docs/09-DECISIONS.md, 2026-10-10).
 */
import createClient from "openapi-fetch";

import type { components, paths } from "./schema";

export type Schemas = components["schemas"];
export type Language = Schemas["Language"];
export type CallDetail = Schemas["CallDetail"];
export type GroundedClaim = Schemas["GroundedClaim"];
export type TranscriptTurn = Schemas["TranscriptTurn"];
export type ClaimChanges = Schemas["ClaimChanges"];
export type ClaimVerdict = Schemas["ClaimVerdict"];
export type ClaimKind = GroundedClaim["kind"];
export type Verdict = ClaimVerdict["verdict"];

export const LANGUAGES: readonly Language[] = ["ar", "en", "mixed"];

export const reviewerId: string = import.meta.env.VITE_SAWTI_REVIEWER_ID ?? "";

export const api = createClient<paths>({
  // Same origin; the dev server proxies `/api` to the local API. Absolute,
  // because `Request` outside a browser (Vitest) rejects relative URLs.
  baseUrl: new URL("/api", window.location.origin).toString(),
  // Resolved per request, not captured at import, so tests can stub `fetch`.
  fetch: (request) => globalThis.fetch(request),
  headers: {
    "X-API-Key": import.meta.env.VITE_SAWTI_API_KEY ?? "",
    "X-Reviewer-Id": reviewerId,
  },
});

/** An API error with the server's `detail`, so the UI can show it. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    detail: unknown,
  ) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
}

/** Unwrap an openapi-fetch result: the data, or an `ApiError`. */
export function unwrap<T>(result: { data?: T; error?: unknown; response: Response }): T {
  if (result.data === undefined) {
    const detail = (result.error as { detail?: unknown } | undefined)?.detail ?? result.response.statusText;
    throw new ApiError(result.response.status, detail);
  }
  return result.data;
}
