import { useState } from "react";
import { Link, useParams } from "react-router-dom";

import type { CallDetail, ClaimVerdict, GroundedClaim, TranscriptTurn, Verdict } from "../../api/client";
import { useCall, useSubmitReview } from "../../api/hooks";
import { CoverageBar } from "../../components/CoverageBar";
import { LangChip } from "../../components/LangChip";
import { LoadError, Loading } from "../../components/Status";
import { useI18n } from "../../i18n/I18nProvider";
import { toChange } from "../../lib/claims";
import { evidenceNote, findVerbatim } from "../../lib/evidence";
import { fixed, shortId } from "../../lib/format";
import { ClaimCard } from "./ClaimCard";
import { correctionReady, emptyCorrection, type Correction } from "./CorrectionForm";
import { RejectedClaims, RulesUsed } from "./SidePanels";
import { Transcript } from "./Transcript";

interface Draft {
  verdict?: Verdict;
  correction: Correction;
}

/**
 * The submission for `drafts`, or null while any claim lacks a verdict or a
 * correction is incomplete. A `correct` verdict carries its one structured
 * change; verified corrected evidence goes in the note.
 */
export function buildVerdicts(
  claims: readonly GroundedClaim[],
  drafts: Record<string, Draft>,
  transcript: string,
  turns: readonly TranscriptTurn[],
): ClaimVerdict[] | null {
  const verdicts: ClaimVerdict[] = [];
  for (const claim of claims) {
    const draft = drafts[claim.claim_id];
    if (draft?.verdict === undefined) return null;
    if (draft.verdict !== "correct") {
      verdicts.push({ claim_id: claim.claim_id, verdict: draft.verdict });
      continue;
    }
    const c = draft.correction;
    if (!correctionReady(claim, c, transcript, turns)) return null;
    const { changes } = toChange(claim, c.field, c.value);
    const match = c.evidence === "" ? null : findVerbatim(transcript, turns, c.evidence);
    verdicts.push({
      claim_id: claim.claim_id,
      verdict: "correct",
      changes: changes!,
      ...(match ? { note: evidenceNote(c.evidence, match) } : {}),
    });
  }
  return verdicts;
}

export function ReviewPage() {
  const { callId = "" } = useParams();
  const [submitted, setSubmitted] = useState(false);
  const call = useCall(callId, { pollUntilReviewed: submitted });
  const { t } = useI18n();

  return (
    <main className="mx-auto flex w-full max-w-[1320px] flex-col gap-5 px-8 pb-10 pt-6">
      <div className="flex flex-wrap items-center gap-3 text-muted">
        <Link to="/" className="font-medium no-underline">
          <span aria-hidden className="inline-block rtl:rotate-180">←</span> {t("review.back")}
        </Link>
        <span>/</span>
        <span dir="ltr" title={callId} className="font-mono text-ink">
          {shortId(callId)}
        </span>
      </div>
      {call.isPending && <Loading />}
      {call.isError && <LoadError error={call.error} retry={() => void call.refetch()} />}
      {call.isSuccess && <Review call={call.data} submitted={submitted} onSubmitted={() => setSubmitted(true)} />}
    </main>
  );
}

function Review({ call: detail, submitted, onSubmitted }: { call: CallDetail; submitted: boolean; onSubmitted: () => void }) {
  const { t } = useI18n();
  // List fields have server-side defaults, so the generated type marks them optional.
  const call = {
    ...detail,
    claims: detail.claims ?? [],
    rejected_claims: detail.rejected_claims ?? [],
    memory_rules: detail.memory_rules ?? [],
  };
  const submit = useSubmitReview(call.call_id);
  const [selected, setSelected] = useState<string | null>(call.claims[0]?.claim_id ?? null);
  const [drafts, setDrafts] = useState<Record<string, Draft>>(() =>
    Object.fromEntries(call.claims.map((c) => [c.claim_id, { correction: emptyCorrection(c) }])),
  );

  const awaiting = call.status === "awaiting_review";
  const readOnly = !awaiting || submitted;
  const lockedReason = !awaiting
    ? t("review.locked", { status: t(`review.status.${call.status}`) })
    : submitted
      ? t("review.lockedSubmitted")
      : undefined;
  const transcript = call.redacted_transcript;
  const verdicts = buildVerdicts(call.claims, drafts, transcript, call.turns);
  const decided = call.claims.filter((c) => drafts[c.claim_id]?.verdict !== undefined).length;
  const total = call.claims.length + call.rejected_claims.length;

  // Every correction's verified evidence stays outlined in the transcript, not only the selected one's.
  const corrected = call.claims.flatMap((claim) => {
    const draft = drafts[claim.claim_id];
    if (draft?.verdict !== "correct" || draft.correction.evidence === "") return [];
    const match = findVerbatim(transcript, call.turns, draft.correction.evidence);
    return match ? [match] : [];
  });

  const update = (claimId: string, next: Partial<Draft>) =>
    setDrafts((current) => ({ ...current, [claimId]: { ...current[claimId]!, ...next } }));

  // A review submitted earlier (or in this session) shows its stored verdicts.
  const stored = Object.fromEntries((call.review?.verdicts ?? []).map((v) => [v.claim_id, v.verdict]));

  return (
    <>
      <section
        aria-label={t("review.summary")}
        className="flex flex-wrap items-center gap-7 rounded-[10px] border border-line bg-white px-6 py-5"
      >
        <SummaryItem label={t("review.agent")}>
          <span className="font-medium">{call.agent_name}</span>
        </SummaryItem>
        <SummaryItem label={t("review.language")}>
          <LangChip language={call.language} />
        </SummaryItem>
        <div className="flex min-w-[240px] flex-[1_1_260px] flex-col gap-1.5">
          <div className="flex justify-between">
            <SummaryLabel>{t("review.coverage")}</SummaryLabel>
            <span className="whitespace-nowrap font-mono">
              <strong>
                <bdi>{fixed(call.grounding_coverage)}</bdi>
              </strong>{" "}
              {call.confidence_threshold != null && (
                <span className="text-muted">{t("review.thresholdOf", { threshold: `\u2066${fixed(call.confidence_threshold)}\u2069` })}</span>
              )}
            </span>
          </div>
          <CoverageBar coverage={call.grounding_coverage} threshold={call.confidence_threshold} className="h-2 w-full" showTick />
          <span className="text-[13px] text-muted">
            {t("review.survived", { kept: call.claims.length, total })}
          </span>
        </div>
        <SummaryItem label={t("review.privacy")}>
          <span className="rounded-md bg-teal-tint px-2.5 py-1 text-[13px] font-medium text-teal-dark">
            {t("review.redacted", { count: call.pii_redacted_count })}
          </span>
        </SummaryItem>
      </section>

      {submitted && (
        <div role="status" className="rounded-[10px] border border-teal-mint bg-teal-tint px-5 py-3.5 font-medium text-teal-dark">
          {call.status === "reviewed" ? t("review.submittedDone") : t("review.submittedPending")}
        </div>
      )}
      {!awaiting && !submitted && (
        <div role="status" className="rounded-[10px] border border-line bg-white px-5 py-3.5 text-muted">
          {t("review.notAwaiting", { status: t(`review.status.${call.status}`) })}
        </div>
      )}
      {submit.isError && (
        <div role="alert" className="rounded-[10px] border border-danger/40 bg-white px-5 py-3.5 text-danger">
          {t("review.submitFailed", { message: submit.error.message })}
        </div>
      )}

      <div className="flex flex-wrap items-start gap-5">
        <Transcript
          turns={call.turns}
          claims={call.claims}
          selectedId={selected}
          corrected={corrected}
          onSelect={setSelected}
        />
        <aside aria-label={t("review.claims")} className="flex min-w-0 flex-[1_1_420px] flex-col gap-3.5">
          <div className="flex items-baseline justify-between">
            <h2 className="m-0 text-base font-semibold">{t("review.claims")}</h2>
            <span className="text-[13px] text-muted">{t("review.decided", { decided, total: call.claims.length })}</span>
          </div>
          {call.claims.length === 0 && <p className="m-0 text-muted">{t("review.noClaims")}</p>}
          {call.claims.map((claim) => (
            <ClaimCard
              key={claim.claim_id}
              claim={claim}
              verdict={drafts[claim.claim_id]?.verdict ?? stored[claim.claim_id]}
              correction={drafts[claim.claim_id]!.correction}
              selected={claim.claim_id === selected}
              readOnly={readOnly}
              lockedReason={lockedReason}
              onSelect={() => setSelected(claim.claim_id)}
              onVerdict={(verdict) => {
                update(claim.claim_id, { verdict });
                setSelected(claim.claim_id);
              }}
              onCorrection={(correction) => update(claim.claim_id, { correction })}
              transcript={transcript}
              turns={call.turns}
            />
          ))}
          <RejectedClaims claims={call.rejected_claims} />
          <RulesUsed rules={call.memory_rules} />
          {!submitted && (
            <div className="sticky bottom-0 flex flex-col gap-2 bg-paper pt-1.5">
              <button
                type="button"
                aria-describedby="submit-help"
                disabled={!awaiting || verdicts === null || submit.isPending}
                onClick={() => {
                  if (verdicts === null) return;
                  submit.mutate(verdicts, { onSuccess: onSubmitted });
                }}
                className="min-h-12 rounded-[10px] border-none bg-ink text-[15px] font-semibold text-white disabled:cursor-not-allowed disabled:bg-field disabled:text-subtle"
              >
                {submit.isPending ? t("review.submitting") : t("review.submit")}
              </button>
              <span id="submit-help" data-testid="submit-help" className="text-center text-xs text-muted">
                {lockedReason ?? t("review.submitHelp")}
              </span>
            </div>
          )}
        </aside>
      </div>
    </>
  );
}

function SummaryLabel({ children }: { children: React.ReactNode }) {
  return <span className="text-xs uppercase tracking-[0.06em] text-muted">{children}</span>;
}

function SummaryItem({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex min-w-[140px] flex-col gap-1.5">
      <SummaryLabel>{label}</SummaryLabel>
      <div>{children}</div>
    </div>
  );
}
