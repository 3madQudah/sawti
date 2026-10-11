import type { GroundedClaim, TranscriptTurn } from "../../api/client";
import { useI18n } from "../../i18n/I18nProvider";
import { EDITABLE_FIELDS, SEVERITIES, originalValue, toChange, type EditableField } from "../../lib/claims";
import { findVerbatim } from "../../lib/evidence";

export interface Correction {
  field: EditableField;
  value: string;
  evidence: string;
}

export function emptyCorrection(claim: GroundedClaim): Correction {
  const field = EDITABLE_FIELDS[claim.kind][0]!;
  return { field, value: "", evidence: "" };
}

const input = "min-h-10 rounded-md border border-field bg-white px-2.5 text-ink";

/**
 * A `correct` verdict's structured change: one field, its original, the new
 * value, and optional corrected evidence that must be in the transcript
 * verbatim. The API cannot change a claim's quote, so verified evidence travels
 * in the verdict's note (docs/09-DECISIONS.md, 2026-10-11).
 */
export function CorrectionForm({
  claim,
  correction,
  onChange,
  transcript,
  turns,
}: {
  claim: GroundedClaim;
  correction: Correction;
  onChange: (next: Correction) => void;
  transcript: string;
  turns: readonly TranscriptTurn[];
}) {
  const { t } = useI18n();
  const id = (part: string) => `${claim.claim_id}-${part}`;
  const original = originalValue(claim, correction.field);
  const result = correction.value === "" ? null : toChange(claim, correction.field, correction.value);
  const match = correction.evidence === "" ? null : findVerbatim(transcript, turns, correction.evidence);

  return (
    <div className="flex flex-col gap-2.5 rounded-lg border border-amber-edge bg-amber-tint p-3">
      <span className="text-xs font-semibold tracking-[0.04em] text-amber-text">{t("correction.title")}</span>
      <div className="grid grid-cols-2 gap-2.5">
        <label htmlFor={id("field")} className="flex flex-col gap-1 text-[13px] text-subtle">
          {t("correction.field")}
          <select
            id={id("field")}
            value={correction.field}
            onChange={(e) => onChange({ ...correction, field: e.target.value as EditableField, value: "" })}
            className={input}
          >
            {EDITABLE_FIELDS[claim.kind].map((field) => (
              <option key={field} value={field}>
                {t(`field.${field}`)}
              </option>
            ))}
          </select>
        </label>
        <label htmlFor={id("original")} className="flex flex-col gap-1 text-[13px] text-subtle">
          {t("correction.original")}
          <input
            id={id("original")}
            readOnly
            dir="auto"
            value={original === "" ? t("correction.originalEmpty") : correction.field === "severity" ? t(`severity.${original as "low"}`) : original}
            className="min-h-10 rounded-md border border-line bg-paper px-2.5 text-muted"
          />
        </label>
      </div>
      <div className="flex flex-col gap-1 text-[13px] text-subtle">
        <label htmlFor={id("value")}>{t("correction.value")}</label>
        <ValueInput
          id={id("value")}
          describedBy={result?.problem ? id("value-problem") : undefined}
          field={correction.field}
          value={correction.value}
          onChange={(value) => onChange({ ...correction, value })}
        />
        {result?.problem && (
          <span id={id("value-problem")} className="text-xs font-medium text-danger">
            {t(result.problem === "unchanged" ? "correction.valueUnchanged" : "correction.valueMissing")}
          </span>
        )}
      </div>
      <div className="flex flex-col gap-1 text-[13px] text-subtle">
        <label htmlFor={id("evidence")}>{t("correction.evidence")}</label>
        <input
          id={id("evidence")}
          dir="auto"
          value={correction.evidence}
          onChange={(e) => onChange({ ...correction, evidence: e.target.value })}
          aria-invalid={correction.evidence !== "" && match === null}
          aria-describedby={correction.evidence === "" ? undefined : id("evidence-status")}
          className={`${input} font-arabic ${correction.evidence === "" ? "" : match ? "border-teal" : "border-danger"}`}
        />
        {correction.evidence !== "" && (
          <span id={id("evidence-status")} role="status" className={`text-xs font-medium ${match ? "text-teal-dark" : "text-danger"}`}>
            {match === null
              ? t("correction.evidenceMissing")
              : match.occurrences > 1
                ? t("correction.evidenceFoundMore", { turn: match.turn, more: match.occurrences - 1 })
                : t("correction.evidenceFound", { turn: match.turn })}
          </span>
        )}
      </div>
    </div>
  );
}

function ValueInput({
  id,
  describedBy,
  field,
  value,
  onChange,
}: {
  id: string;
  describedBy: string | undefined;
  field: EditableField;
  value: string;
  onChange: (value: string) => void;
}) {
  const { t } = useI18n();
  if (field === "severity") {
    return (
      <select id={id} aria-describedby={describedBy} value={value} onChange={(e) => onChange(e.target.value)} className={input}>
        <option value="" disabled>
          —
        </option>
        {SEVERITIES.map((s) => (
          <option key={s} value={s}>
            {t(`severity.${s}`)}
          </option>
        ))}
      </select>
    );
  }
  if (field === "deadline") {
    return <input id={id} aria-describedby={describedBy} type="datetime-local" value={value} onChange={(e) => onChange(e.target.value)} className={input} />;
  }
  if (field === "score") {
    return <input id={id} aria-describedby={describedBy} type="number" min={0} max={1} step={0.05} dir="ltr" value={value} onChange={(e) => onChange(e.target.value)} className={input} />;
  }
  return <input id={id} aria-describedby={describedBy} dir="auto" value={value} onChange={(e) => onChange(e.target.value)} className={input} />;
}

/** Whether a correction can be submitted: a valid changed value, and evidence (if any) found verbatim. */
export function correctionReady(claim: GroundedClaim, c: Correction, transcript: string, turns: readonly TranscriptTurn[]): boolean {
  const result = toChange(claim, c.field, c.value);
  if (result.problem !== null) return false;
  return c.evidence === "" || findVerbatim(transcript, turns, c.evidence) !== null;
}
