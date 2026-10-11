import { useEffect } from "react";

import type { GroundedClaim, TranscriptTurn } from "../../api/client";
import { useI18n } from "../../i18n/I18nProvider";
import { segmentTurn } from "../../lib/claims";
import { turnDirection } from "../../lib/direction";

/**
 * The redacted transcript, one row per turn. Each turn sets its own `dir` from
 * its language; the UI language does not affect it. Evidence spans come from
 * the API's offsets into `redacted_transcript`.
 */
export function Transcript({
  turns,
  claims,
  selectedId,
  corrected,
  onSelect,
}: {
  turns: readonly TranscriptTurn[];
  claims: readonly GroundedClaim[];
  selectedId: string | null;
  corrected: readonly { start: number; end: number }[];
  onSelect: (claimId: string) => void;
}) {
  const { t } = useI18n();
  const selected = claims.find((c) => c.claim_id === selectedId) ?? null;

  useEffect(() => {
    if (selected?.turn_index == null) return;
    document.getElementById(`turn-${selected.turn_index}`)?.scrollIntoView?.({ block: "nearest" });
  }, [selected]);

  return (
    <section aria-label={t("review.transcript")} className="min-w-0 flex-[999_1_560px] overflow-hidden rounded-[10px] border border-line bg-white">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-5 py-4">
        <h2 className="m-0 text-base font-semibold">
          {t("review.transcript")} <span className="font-normal text-muted">{t("review.transcriptRedacted")}</span>
        </h2>
        <div className="flex flex-wrap gap-3.5 text-xs text-muted">
          <Legend swatch="bg-mark-selected" label={t("review.legendSelected")} />
          <Legend swatch="bg-mark-other" label={t("review.legendOther")} />
          {corrected.length > 0 && <Legend swatch="h-0 rounded-none border-b-2 border-dashed border-amber" label={t("review.legendCorrected")} />}
        </div>
      </div>
      <ol className="m-0 list-none py-2 ps-0">
        {turns.map((turn) => {
          const segments = segmentTurn(turn.text, turn.start_char, claims, corrected);
          const holdsSelected = selected !== null && selected.turn_index === turn.index;
          return (
            <li
              key={turn.index}
              id={`turn-${turn.index}`}
              data-testid={`turn-${turn.index + 1}`}
              className={`flex items-start gap-4 border-s-[3px] px-5 py-3.5 ${holdsSelected ? "border-amber" : "border-transparent"} ${turn.speaker === "Customer" ? "bg-customer" : "bg-white"}`}
            >
              <div className="flex w-[92px] flex-none flex-col gap-1">
                <span className="text-[13px] font-semibold">{t(`speaker.${turn.speaker}`)}</span>
                <span dir="ltr" className="text-start font-mono text-[11px] text-muted">
                  {t("review.turnMeta", { n: turn.index + 1, lang: turn.language })}
                </span>
              </div>
              <p
                dir={turnDirection(turn.language)}
                className={`m-0 min-w-0 flex-auto text-start text-base leading-[1.75] ${turn.language === "en" ? "font-sans" : "font-arabic"}`}
              >
                {segments.map((segment, i) => {
                  const on = selectedId !== null && segment.claims.includes(selectedId);
                  const mark = on ? "rounded-[3px] bg-mark-selected shadow-[0_0_0_2px_#FFD98A]" : segment.claims.length > 0 ? "rounded-[3px] bg-mark-other" : "";
                  const dashed = segment.corrected ? "underline decoration-amber decoration-dashed decoration-2 underline-offset-[6px]" : "";
                  if (segment.claims.length === 0)
                    return <span key={i} data-corrected={segment.corrected || undefined} className={dashed}>{segment.text}</span>;
                  return (
                    <mark
                      key={i}
                      data-selected={on}
                      data-corrected={segment.corrected || undefined}
                      onClick={() => onSelect(segment.claims[0]!)}
                      className={`cursor-pointer text-inherit ${mark} ${dashed}`}
                    >
                      {segment.text}
                    </mark>
                  );
                })}
              </p>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function Legend({ swatch, label }: { swatch: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={`h-3 w-3 rounded-[3px] ${swatch}`} />
      {label}
    </span>
  );
}
