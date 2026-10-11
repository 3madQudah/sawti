import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, unwrap, type ClaimVerdict } from "./client";

/** The whole `awaiting_review` queue; the page filters by language client-side, so counts stay stable. */
export function useQueue() {
  return useQuery({
    queryKey: ["queue"],
    queryFn: async () => unwrap(await api.GET("/reviews", { params: { query: { limit: 500 } } })),
    refetchInterval: 15_000,
  });
}

/**
 * One call. `pollUntilReviewed`: after a submission, poll until the resumed
 * run has finished (`reviewed`), which happens asynchronously in the worker.
 */
export function useCall(callId: string, { pollUntilReviewed = false } = {}) {
  return useQuery({
    queryKey: ["call", callId],
    queryFn: async () =>
      unwrap(await api.GET("/calls/{call_id}", { params: { path: { call_id: callId } } })),
    refetchInterval: (query) =>
      pollUntilReviewed && query.state.data?.status === "awaiting_review" ? 1500 : false,
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
