/**
 * Mock API data for Vitest and Playwright. Typed against the generated schema,
 * so a backend contract change breaks these at compile time instead of letting
 * them drift. Synthetic, like everything in this project: no real PII.
 *
 * Offsets are computed the way the API computes them: `redacted_transcript` is
 * `Speaker: text` lines, and every offset indexes that one string.
 */
import type { CallDetail, GroundedClaim, Language, Schemas, TranscriptTurn } from "../api/client";

type TurnSpec = { speaker: "Agent" | "Customer"; language: Language; text: string };
type ClaimSpec = Omit<GroundedClaim, "claim_id" | "start_char" | "end_char" | "turn_index" | "speaker"> & {
  list: "commitments" | "compliance_flags" | "rubric_scores";
};

/** A call detail whose list fields are present (the API always sends them; the schema marks them optional). */
export type FullCallDetail = CallDetail & Required<Pick<CallDetail, "claims" | "rejected_claims" | "memory_rules">>;

export function buildCall(
  base: Omit<FullCallDetail, "redacted_transcript" | "turns" | "claims">,
  turnSpecs: TurnSpec[],
  claimSpecs: ClaimSpec[],
): FullCallDetail {
  let transcript = "";
  const turns: TranscriptTurn[] = turnSpecs.map((spec, index) => {
    const prefix = `${spec.speaker}: `;
    const start = transcript.length + prefix.length;
    transcript += `${prefix}${spec.text}\n`;
    return { index, speaker: spec.speaker, language: spec.language, text: spec.text, start_char: start, end_char: start + spec.text.length };
  });
  const counters: Record<string, number> = {};
  const claims = claimSpecs.map(({ list, ...spec }) => {
    const start = transcript.indexOf(spec.quote);
    if (start === -1) throw new Error(`fixture quote not in transcript: ${spec.quote}`);
    const turn = turns.find((t) => t.start_char <= start && start < t.end_char)!;
    const n = counters[list] ?? 0;
    counters[list] = n + 1;
    return {
      ...spec,
      claim_id: `${list}[${n}]`,
      start_char: start,
      end_char: start + spec.quote.length,
      turn_index: turn.index,
      speaker: turn.speaker,
    };
  });
  return { ...base, redacted_transcript: transcript, turns, claims };
}

export const CALL_ID = "7f3a91c2-5d4e-4b8a-9c1f-2e6d8a0b3c47";
export const RULE_IDS = ["0b14e2a1-6c3d-4f5e-8a9b-1c2d3e4f5a6b", "2c22f3b4-7d8e-4a1b-9c0d-2e3f4a5b6c7d"];

const escalation: Schemas["EscalationInfo"] = {
  kind: "low_grounding",
  rejected_claims: 3,
  total_claims: 9,
  grounding_coverage: 0.67,
  threshold: 0.7,
};

export function mixedCall(): FullCallDetail {
  return buildCall(
    {
      call_id: CALL_ID,
      status: "awaiting_review",
      language: "mixed",
      agent_name: "agent-lina",
      queued_at: "2026-10-11T08:00:00Z",
      started_at: "2026-10-11T08:00:05Z",
      finished_at: "2026-10-11T08:00:40Z",
      pii_redacted_count: 2,
      attempts: 1,
      failure_reason: null,
      summary: "Customer disputes roaming charges; agent promises a refund and an SMS.",
      confidence: 0.55,
      grounding_coverage: 0.67,
      confidence_threshold: 0.7,
      escalation_reason: "3 of 9 claims had no verbatim evidence (grounding coverage 67%, threshold 70%)",
      escalation,
      rejected_claims: [
        { kind: "commitment", attempted_quote: "رح نعطيك خصم عالفاتورة الجاية", speaker: "Agent", description: "Agent offered a discount on next bill" },
        { kind: "commitment", attempted_quote: "I will cancel my line", speaker: "Customer", description: "Customer threatened to cancel" },
        { kind: "compliance_flag", attempted_quote: "سياسة التجوال بتنص على", speaker: "Agent", description: "Agent explained the roaming policy" },
      ],
      memory_rules: [
        { id: RULE_IDS[0]!, text: 'In Jordanian dialect, "بيوصلك … خلال" plus a time span is a commitment with a deadline.' },
        { id: RULE_IDS[1]!, text: "Asking only for the line number does not count as identity verification." },
      ],
      review: null,
    },
    [
      { speaker: "Agent", language: "ar", text: "مرحبا، معك [NAME] من خدمة العملاء، كيف بقدر أساعدك؟" },
      { speaker: "Customer", language: "mixed", text: "أهلين، الفاتورة هالشهر طالعة 45 دينار، وأنا my plan is only 20 JD." },
      { speaker: "Agent", language: "ar", text: "بعتذر منك. ممكن رقم الخط لو سمحت؟" },
      { speaker: "Customer", language: "ar", text: "[PHONE]" },
      { speaker: "Agent", language: "mixed", text: "تمام، شايف إنه في roaming charges من الأسبوع الماضي." },
      { speaker: "Customer", language: "ar", text: "بس أنا ما سافرت أبداً! هاد غلط منكم." },
      { speaker: "Agent", language: "ar", text: "معك حق، رح أرفع طلب استرجاع وبيوصلك الرصيد خلال ٣ أيام عمل." },
      { speaker: "Customer", language: "en", text: "Okay, but I want a confirmation SMS." },
      { speaker: "Agent", language: "ar", text: "أكيد، رح توصلك رسالة اليوم." },
    ],
    [
      { list: "commitments", kind: "commitment", quote: "رح أرفع طلب استرجاع وبيوصلك الرصيد خلال ٣ أيام", description: "Refund credited within 3 days", promised_by: "agent", deadline: null },
      { list: "commitments", kind: "commitment", quote: "رح توصلك رسالة اليوم", description: "Confirmation SMS sent today", promised_by: "agent", deadline: "2026-10-11T23:59:00Z" },
      { list: "compliance_flags", kind: "compliance_flag", quote: "ممكن رقم الخط لو سمحت؟", description: "Identity checked by line number only", rule_id: "identity_verification", severity: "medium" },
      { list: "rubric_scores", kind: "rubric_score", quote: "بعتذر منك.", description: "Apologised before troubleshooting", criterion: "empathy", score: 0.8 },
      { list: "rubric_scores", kind: "rubric_score", quote: "شايف إنه في roaming charges من الأسبوع الماضي", description: "Found the disputed charge", criterion: "diagnosis", score: 1 },
      { list: "rubric_scores", kind: "rubric_score", quote: "Okay, but I want a confirmation SMS.", description: "Customer asked for written confirmation", criterion: "resolution", score: 0.6 },
    ],
  );
}

type QueueRow = Omit<Schemas["ReviewQueueItem"], "waiting_since"> & { waitingMinutes: number };

const QUEUE_ROWS: QueueRow[] = [
  { call_id: CALL_ID, agent_name: "agent-lina", language: "mixed", grounding_coverage: 0.67, escalation_reason: null, escalation, waitingMinutes: 102 },
  { call_id: "2b18e4a0-0000-4000-8000-000000000002", agent_name: "agent-omar", language: "ar", grounding_coverage: 0.6, escalation_reason: null, escalation: { ...escalation, rejected_claims: 2, total_claims: 5, grounding_coverage: 0.6 }, waitingMinutes: 65 },
  { call_id: "91cc02b1-0000-4000-8000-000000000003", agent_name: "agent-sara", language: "en", grounding_coverage: 0.65, escalation_reason: null, escalation: { ...escalation, rejected_claims: 4, total_claims: 11, grounding_coverage: 0.65 }, waitingMinutes: 48 },
  { call_id: "4de7a0c3-0000-4000-8000-000000000004", agent_name: "agent-lina", language: "ar", grounding_coverage: 0, escalation_reason: null, escalation: { ...escalation, kind: "analysis_error", rejected_claims: 0, total_claims: 0, grounding_coverage: 0 }, waitingMinutes: 31 },
  { call_id: "0a6f5bd4-0000-4000-8000-000000000005", agent_name: "agent-yazan", language: "mixed", grounding_coverage: 0.69, escalation_reason: null, escalation: { ...escalation, rejected_claims: 3, total_claims: 10, grounding_coverage: 0.69 }, waitingMinutes: 22 },
  { call_id: "c3e811e5-0000-4000-8000-000000000006", agent_name: "agent-omar", language: "mixed", grounding_coverage: 0.64, escalation_reason: null, escalation: { ...escalation, rejected_claims: 3, total_claims: 8, grounding_coverage: 0.64 }, waitingMinutes: 9 },
];

export function queue(now: Date): Schemas["ReviewQueueResponse"] {
  const items = QUEUE_ROWS.map(({ waitingMinutes, ...row }) => ({
    ...row,
    waiting_since: new Date(now.getTime() - waitingMinutes * 60_000).toISOString(),
  }));
  return { items, pending: { ar: 2, en: 1, mixed: 3 } };
}
