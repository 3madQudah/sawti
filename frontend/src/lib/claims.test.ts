import { describe, expect, it } from "vitest";

import { mixedCall } from "../test/fixtures";
import { EDITABLE_FIELDS, originalValue, segmentTurn, toChange } from "./claims";

const call = mixedCall();
const [refund, sms, flag, empathy] = call.claims;

describe("structured corrections", () => {
  it("offers only the kind's editable fields, never the quote", () => {
    expect(EDITABLE_FIELDS.commitment).toEqual(["description", "promised_by", "deadline"]);
    expect(EDITABLE_FIELDS.compliance_flag).toEqual(["rule_id", "severity", "description"]);
    expect(EDITABLE_FIELDS.rubric_score).toEqual(["score", "justification"]);
    expect(Object.values(EDITABLE_FIELDS).flat()).not.toContain("quote");
  });

  it("shows the original value per field", () => {
    expect(originalValue(refund!, "description")).toBe("Refund credited within 3 days");
    expect(originalValue(refund!, "deadline")).toBe("");
    expect(originalValue(sms!, "deadline")).toBe("2026-10-11T23:59");
    expect(originalValue(flag!, "severity")).toBe("medium");
    expect(originalValue(empathy!, "score")).toBe("0.8");
    expect(originalValue(empathy!, "justification")).toBe("Apologised before troubleshooting");
  });

  it("builds the API change, or says why it cannot", () => {
    expect(toChange(refund!, "description", "Refund within 3 business days")).toEqual({
      changes: { description: "Refund within 3 business days" },
      problem: null,
    });
    expect(toChange(refund!, "description", "  ")).toEqual({ changes: null, problem: "missing" });
    expect(toChange(refund!, "description", "Refund credited within 3 days")).toEqual({ changes: null, problem: "unchanged" });
    expect(toChange(empathy!, "score", "0.4")).toEqual({ changes: { score: 0.4 }, problem: null });
    expect(toChange(empathy!, "score", "1.5").problem).toBe("missing");
    expect(toChange(flag!, "severity", "high")).toEqual({ changes: { severity: "high" }, problem: null });
    const deadline = toChange(refund!, "deadline", "2026-10-14T17:00");
    expect(deadline.changes?.deadline).toBe(new Date("2026-10-14T17:00").toISOString());
  });
});

describe("segmentTurn", () => {
  it("splits a turn at evidence spans, which are transcript offsets", () => {
    const turn = call.turns[6]!;
    const segments = segmentTurn(turn.text, turn.start_char, call.claims, []);
    expect(segments.map((s) => s.text).join("")).toBe(turn.text);
    const marked = segments.filter((s) => s.claims.length > 0);
    expect(marked.map((s) => s.text)).toEqual([refund!.quote]);
    expect(marked[0]!.claims).toEqual([refund!.claim_id]);
  });

  it("marks a corrected-evidence span separately", () => {
    const turn = call.turns[6]!;
    const start = call.redacted_transcript.indexOf("أيام عمل");
    const segments = segmentTurn(turn.text, turn.start_char, [], [{ start, end: start + "أيام عمل".length }]);
    expect(segments.filter((s) => s.corrected).map((s) => s.text)).toEqual(["أيام عمل"]);
  });
});
