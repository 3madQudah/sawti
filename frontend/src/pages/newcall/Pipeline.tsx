import { Link } from "react-router-dom";

import type { CallDetail } from "../../api/client";
import { useCall } from "../../api/hooks";
import type { StringKey } from "../../i18n/strings";
import { useI18n } from "../../i18n/I18nProvider";
import { fixed, shortId } from "../../lib/format";

type StepState = "done" | "active" | "pending" | "failed";
type Step = { title: StringKey; detail: string; state: StepState; time?: string };

/** Index of the step a status is in: 1 transcribing, 2 queued, 3 analyzing, 4 routed. */
const POSITION: Record<CallDetail["status"], number> = {
  transcribing: 1,
  queued: 2,
  processing: 3,
  awaiting_review: 4,
  reviewed: 4,
  completed: 4,
  failed: -1,
};

const seconds = (from: string | null | undefined, to: string | null | undefined) =>
  from && to ? (new Date(to).getTime() - new Date(from).getTime()) / 1000 : null;

/** Where a failed call stopped: no queue time means ASR never finished, no start means analysis never ran. */
function failedAt(call: CallDetail): number {
  if (!call.queued_at) return 1;
  if (!call.started_at) return 2;
  return 3;
}

function steps(call: CallDetail | undefined, t: (key: StringKey, vars?: Record<string, string | number>) => string): Step[] {
  const failed = call?.status === "failed";
  const at = call === undefined ? 1 : failed ? failedAt(call) : POSITION[call.status];
  const state = (i: number): StepState => {
    if (failed && i === at) return "failed";
    if (i < at || (i === 4 && at === 4)) return "done";
    return i === at ? "active" : "pending";
  };
  const analysis = seconds(call?.started_at, call?.finished_at);
  const routed: Record<string, StringKey> = {
    awaiting_review: "newCall.step.routedReview",
    completed: "newCall.step.routedCompleted",
    reviewed: "newCall.step.routedReviewed",
  };
  return [
    { title: "newCall.step.uploaded", detail: t("newCall.step.uploadedDetail"), state: "done" },
    {
      title: "newCall.step.transcribing",
      state: state(1),
      detail: state(1) === "done" && call ? t("newCall.step.transcribingDone", { count: call.pii_redacted_count }) : t("newCall.step.transcribingDetail"),
    },
    { title: "newCall.step.queued", detail: t("newCall.step.queuedDetail"), state: state(2) },
    {
      title: "newCall.step.analyzing",
      detail: t("newCall.step.analyzingDetail"),
      state: state(3),
      time: state(3) === "done" && analysis !== null ? t("newCall.seconds", { seconds: fixed(analysis, 1) }) : undefined,
    },
    {
      title: "newCall.step.routed",
      detail: call && routed[call.status] ? t(routed[call.status]!) : t("newCall.step.routedDetail"),
      state: failed ? "pending" : state(4),
    },
  ];
}

const DOT: Record<StepState, string> = {
  done: "bg-teal text-white",
  active: "bg-amber text-white",
  pending: "bg-line text-muted",
  failed: "bg-danger text-white",
};

export function Pipeline({ callId }: { callId: string }) {
  const { t } = useI18n();
  const call = useCall(callId, { pollWhileRunning: true });
  const detail = call.data;

  return (
    <div className="flex flex-col gap-4">
      <div dir="ltr" className="self-start font-mono text-xs text-muted" title={callId}>
        {t("newCall.call", { id: shortId(callId) })}
      </div>
      <ol className="m-0 flex list-none flex-col gap-0 p-0">
        {steps(detail, t).map((step, i) => (
          <li key={step.title} data-testid={`step-${i}`} data-state={step.state} className="flex gap-3.5 border-b border-line-soft py-3 last:border-b-0">
            <span className={`inline-flex h-7 w-7 flex-none items-center justify-center rounded-full text-[13px] font-semibold ${DOT[step.state]}`}>
              {step.state === "done" ? "✓" : step.state === "failed" ? "!" : i + 1}
            </span>
            <span className="flex flex-1 flex-col gap-0.5">
              <span className={`font-semibold ${step.state === "pending" ? "text-muted" : "text-ink"}`}>
                {step.state === "failed" ? `${t(step.title)} · ${t("newCall.step.failed")}` : t(step.title)}
              </span>
              <span className="text-[13px] text-muted">{step.detail}</span>
            </span>
            <span dir="ltr" className="font-mono text-[13px] text-muted">{step.state === "active" ? "…" : step.time}</span>
          </li>
        ))}
      </ol>

      {detail?.status === "awaiting_review" && (
        <div role="status" className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-amber-edge bg-amber-tint px-4 py-3">
          <span className="text-sm text-amber-text">
            {detail.escalation?.kind === "analysis_error"
              ? t("newCall.escalatedError")
              : t("newCall.escalated", {
                  coverage: fixed(detail.escalation?.grounding_coverage ?? detail.grounding_coverage),
                  threshold: fixed(detail.escalation?.threshold ?? detail.confidence_threshold),
                })}
          </span>
          <Link
            to={`/calls/${callId}`}
            className="inline-flex min-h-10 items-center rounded-lg bg-teal px-4 font-medium text-white no-underline hover:bg-teal-dark hover:text-white"
          >
            {t("newCall.openReview")}
          </Link>
        </div>
      )}
      {detail?.status === "failed" && (
        <p role="alert" className="m-0 rounded-lg border border-danger/30 px-4 py-3 text-sm text-danger">
          {t("newCall.failed", { reason: detail.failure_reason ?? "—" })}
        </p>
      )}
    </div>
  );
}
