import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, unwrap, type CallDetail, type ClaimVerdict, type ConflictAction, type Language } from "./client";

/** The whole `awaiting_review` queue; the page filters by language client-side, so counts stay stable. */
export function useQueue() {
  return useQuery({
    queryKey: ["queue"],
    queryFn: async () => unwrap(await api.GET("/reviews", { params: { query: { limit: 500 } } })),
    refetchInterval: 15_000,
  });
}

/** Statuses a call can still leave without a person: an upload's pipeline is running. */
export const RUNNING: readonly CallDetail["status"][] = ["transcribing", "queued", "processing"];

/**
 * One call. `pollUntilReviewed`: after a submission, poll until the resumed
 * run has finished (`reviewed`), which happens asynchronously in the worker.
 * `pollWhileRunning`: after an upload, poll until ASR and analysis are done.
 */
export function useCall(callId: string, { pollUntilReviewed = false, pollWhileRunning = false } = {}) {
  return useQuery({
    queryKey: ["call", callId],
    queryFn: async () =>
      unwrap(await api.GET("/calls/{call_id}", { params: { path: { call_id: callId } } })),
    enabled: callId !== "",
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (pollUntilReviewed && status === "awaiting_review") return 1500;
      if (pollWhileRunning && (status === undefined || RUNNING.includes(status))) return 1500;
      return false;
    },
  });
}

export function useSubmitReview(callId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (verdicts: ClaimVerdict[]) =>
      unwrap(
        await api.POST("/reviews/{call_id}/corrections", {
          params: { path: { call_id: callId } },
          body: { verdicts },
        }),
      ),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["call", callId] });
      void client.invalidateQueries({ queryKey: ["queue"] });
    },
  });
}

export interface AudioUploadInput {
  file: File;
  language: Language;
  agentExternalId: string;
}

/** `POST /calls/audio`: multipart, so the browser sets the boundary. */
export function useUploadAudio() {
  return useMutation({
    mutationFn: async ({ file, language, agentExternalId }: AudioUploadInput) => {
      const form = new FormData();
      form.append("file", file);
      form.append("language_hint", language);
      if (agentExternalId.trim()) form.append("agent_external_id", agentExternalId.trim());
      return unwrap(
        await api.POST("/calls/audio", {
          // The generated type describes the form fields; the body actually sent is `form`.
          body: { file: "", language_hint: language },
          bodySerializer: () => form,
        }),
      );
    },
  });
}

export function useStats() {
  return useQuery({ queryKey: ["stats"], queryFn: async () => unwrap(await api.GET("/stats")) });
}

export function useRules() {
  return useQuery({ queryKey: ["rules"], queryFn: async () => unwrap(await api.GET("/memory/rules")) });
}

export function useConflicts() {
  return useQuery({ queryKey: ["conflicts"], queryFn: async () => unwrap(await api.GET("/memory/conflicts")) });
}

export function useResolveConflict() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async ({ conflictId, action }: { conflictId: string; action: ConflictAction }) =>
      unwrap(
        await api.POST("/memory/conflicts/{conflict_id}/resolve", {
          params: { path: { conflict_id: conflictId } },
          body: { action },
        }),
      ),
    onSuccess: () => {
      for (const key of ["conflicts", "rules", "stats"]) void client.invalidateQueries({ queryKey: [key] });
    },
  });
}

/** The recorded 6.3 numbers; they never change while the page is open. */
export function useResults() {
  return useQuery({
    queryKey: ["results"],
    queryFn: async () => unwrap(await api.GET("/results")),
    staleTime: Infinity,
  });
}
