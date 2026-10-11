import type { GroundedClaim, TranscriptTurn, Verdict } from "../../api/client";
import { useI18n } from "../../i18n/I18nProvider";
import { fixed } from "../../lib/format";
import { CorrectionForm, type Correction } from "./CorrectionForm";

const KIND_TAG: Record<GroundedClaim["kind"], string> = {
  commitment: "bg-teal-soft text-teal-dark",
  compliance_flag: "bg-amber-chip text-amber-text",
  rubric_score: "bg-rubric text-subtle",
};

const VERDICT_ON: Record<Verdict, string> = {
  confirm: "border-teal bg-teal text-white",
  correct: "border-amber bg-amber text-white",
  reject: "border-danger bg-danger text-white",
};

export function ClaimCard({
  claim,
  verdict,
  correction,
  selected,
  readOnly,
  onSelect,
  onVerdict,
  onCorrection,
  transcript,
  turns,
}: {
  claim: GroundedClaim;
  verdict: Verdict | undefined;
  correction: Correction;
  selected: boolean;
  readOnly: boolean;
  onSelect: () => void;
  onVerdict: (verdict: Verdict) => void;
  onCorrection: (next: Correction) => void;
  transcript: string;
  turns: readonly TranscriptTurn[];
}) {
  const { t } = useI18n();
  const kind =
    claim.kind === "rubric_score"
      ? t("review.kind.rubric_score", { criterion: claim.criterion ?? "", score: fixed(claim.score) })
      : t(`review.kind.${claim.kind}`);
  const turn = claim.turn_index == null ? "—" : claim.turn_index + 1;

  return (
    <article
      data-testid={`claim-${claim.claim_id}`}
      className={`flex flex-col gap-2.5 rounded-[10px] bg-white px-[18px] py-4 ${selected ? "border-2 border-amber" : "border border-line"}`}
    >
      <div className="flex items-center justify-between gap-2.5">
        <span className={`rounded-full px-[9px] py-[3px] text-xs font-semibold ${KIND_TAG[claim.kind]}`}>{kind}</span>
        <span className="font-mono text-[11px] text-muted">
          {/* Numbers isolated (LRI…PDI), so a range keeps its order inside Arabic text. */}
          {t("review.claimMeta", { turn, start: `\u2066${claim.start_char}`, end: `${claim.end_char}\u2069` })}
        </span>
      </div>
      <div dir="auto" className="text-start text-[15px] font-semibold">
        {claim.description}
      </div>
      <button
        type="button"
        onClick={onSelect}
        aria-label={t("review.showEvidence")}
        dir="auto"
        className="min-h-11 cursor-pointer rounded-md border-none bg-paper px-3 py-2.5 text-start font-arabic text-[15px] leading-relaxed text-ink"
      >
        “{claim.quote}”
      </button>
      <div role="group" aria-label={t("review.verdict")} className="flex flex-wrap gap-1.5">
        {(["confirm", "correct", "reject"] as const).map((v) => (
          <button
            key={v}
            type="button"
            aria-pressed={verdict === v}
            disabled={readOnly}
            onClick={() => onVerdict(v)}
            className={`min-h-10 rounded-lg border px-3.5 font-medium disabled:cursor-not-allowed ${verdict === v ? VERDICT_ON[v] : "border-field bg-white text-ink"}`}
          >
            {t(`review.${v}`)}
          </button>
        ))}
      </div>
      {verdict === "correct" && !readOnly && (
        <CorrectionForm claim={claim} correction={correction} onChange={onCorrection} transcript={transcript} turns={turns} />
      )}
    </article>
  );
}
