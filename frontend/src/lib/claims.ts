import type { ClaimChanges, ClaimKind, GroundedClaim, Schemas } from "../api/client";

export type EditableField = keyof ClaimChanges;

/**
 * Which `ClaimChanges` fields a `correct` verdict may set, per claim kind. Mirrors
 * `EDITABLE_FIELDS` in `sawti.api.schemas` (not exported in the OpenAPI schema);
 * the field names are type-checked against the generated `ClaimChanges`. The
 * evidence quote is never editable.
 */
export const EDITABLE_FIELDS: Record<ClaimKind, readonly EditableField[]> = {
  commitment: ["description", "promised_by", "deadline"],
  compliance_flag: ["rule_id", "severity", "description"],
  rubric_score: ["score", "justification"],
};

export const SEVERITIES: readonly Schemas["Severity"][] = ["low", "medium", "high", "critical"];

/** The claim's current value for `field`, as the form shows it ("" when unset). */
export function originalValue(claim: GroundedClaim, field: EditableField): string {
  switch (field) {
    case "justification":
    case "description":
      return claim.description;
    case "deadline":
      return claim.deadline ? claim.deadline.slice(0, 16) : "";
    case "score":
      return claim.score === null || claim.score === undefined ? "" : String(claim.score);
    default:
      return claim[field] ?? "";
  }
}

export type ChangeProblem = "missing" | "unchanged" | null;

/** Turn the form's text into the `ClaimChanges` value the API expects, or say what is wrong. */
export function toChange(
  claim: GroundedClaim,
  field: EditableField,
  value: string,
): { changes: ClaimChanges; problem: null } | { changes: null; problem: Exclude<ChangeProblem, null> } {
  const trimmed = value.trim();
  if (trimmed === "") return { changes: null, problem: "missing" };
  if (trimmed === originalValue(claim, field).trim()) return { changes: null, problem: "unchanged" };
  if (field === "score") {
    const score = Number(trimmed);
    if (!Number.isFinite(score) || score < 0 || score > 1) return { changes: null, problem: "missing" };
    return { changes: { score }, problem: null };
  }
  if (field === "deadline") return { changes: { deadline: new Date(trimmed).toISOString() }, problem: null };
  if (field === "severity") return { changes: { severity: trimmed as Schemas["Severity"] }, problem: null };
  return { changes: { [field]: trimmed }, problem: null };
}

export interface Segment {
  text: string;
  /** Claim ids whose evidence covers this segment. */
  claims: string[];
  /** Inside the reviewer's corrected-evidence span. */
  corrected: boolean;
}

/**
 * Split one turn's text into segments by the evidence spans that overlap it.
 * Offsets are into the whole redacted transcript, as the API gives them.
 */
export function segmentTurn(
  turnText: string,
  turnStart: number,
  claims: readonly Pick<GroundedClaim, "claim_id" | "start_char" | "end_char">[],
  corrected: readonly { start: number; end: number }[],
): Segment[] {
  const turnEnd = turnStart + turnText.length;
  const spans = [
    ...claims.map((c) => ({ start: c.start_char, end: c.end_char, id: c.claim_id as string | null })),
    ...corrected.map((c) => ({ start: c.start, end: c.end, id: null })),
  ].filter((s) => s.start < turnEnd && s.end > turnStart);
  const cuts = new Set([turnStart, turnEnd]);
  for (const s of spans) {
    cuts.add(Math.max(s.start, turnStart));
    cuts.add(Math.min(s.end, turnEnd));
  }
  const points = [...cuts].sort((a, b) => a - b);
  const segments: Segment[] = [];
  for (let i = 0; i + 1 < points.length; i += 1) {
    const from = points[i]!;
    const to = points[i + 1]!;
    const covering = spans.filter((s) => s.start <= from && s.end >= to);
    segments.push({
      text: turnText.slice(from - turnStart, to - turnStart),
      claims: covering.flatMap((s) => (s.id === null ? [] : [s.id])),
      corrected: covering.some((s) => s.id === null),
    });
  }
  return segments;
}
