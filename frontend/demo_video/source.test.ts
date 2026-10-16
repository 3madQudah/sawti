import type { CallDetail, ClaimVerdict } from "../src/api/client";
import { mixedCall } from "../src/test/fixtures";
import { beforeFromAfter, callAt, correctionFrom, evidenceFromNote, factsFrom } from "./source";

describe("demo replay", () => {
  it("reads the verified evidence out of a verdict note", () => {
    expect(evidenceFromNote("Corrected evidence (verbatim, turn 7, chars 3-17): «خلال ٣ أيام عمل»")).toBe("خلال ٣ أيام عمل");
    expect(evidenceFromNote(null)).toBe("");
    expect(evidenceFromNote("no evidence")).toBe("");
  });

  it("replays the first correction with its field, value and evidence", () => {
    const verdicts = [
      { claim_id: "a", verdict: "confirm" },
      { claim_id: "b", verdict: "correct", changes: { description: "New text" }, note: "… «quote»" },
    ] as ClaimVerdict[];
    expect(correctionFrom(verdicts)).toEqual({ claimId: "b", field: "description", value: "New text", evidence: "quote" });
  });

  it("has no correction to replay when every claim was confirmed", () => {
    expect(correctionFrom([{ claim_id: "a", verdict: "confirm" }] as ClaimVerdict[])).toBeNull();
  });

  it("hides results a call did not have yet at an earlier status", () => {
    const call = { ...mixedCall(), status: "awaiting_review" } as CallDetail;
    const transcribing = callAt(call, "transcribing");
    expect(transcribing.turns).toEqual([]);
    expect(transcribing.claims).toEqual([]);
    expect(transcribing.pii_redacted_count).toBe(0);
    const processing = callAt(call, "processing");
    expect(processing.turns).toEqual(call.turns);
    expect(processing.claims).toEqual([]);
    expect(processing.escalation).toBeNull();
    expect(callAt(call, "awaiting_review")).toBe(call);
  });

  it("rebuilds the pre-review call from a reviewed capture, analysis unchanged", () => {
    const reviewed = { ...mixedCall(), status: "reviewed", review: { verdicts: [] } } as unknown as CallDetail;
    const after = { call_id: reviewed.call_id, audio_name: "x.wav", responses: { [`/calls/${reviewed.call_id}`]: reviewed, "/results": {} } };
    const call = beforeFromAfter(after).responses[`/calls/${reviewed.call_id}`] as CallDetail;
    expect(call.status).toBe("awaiting_review");
    expect(call.review).toBeNull();
    expect(call.claims).toEqual(reviewed.claims);
    expect(call.grounding_coverage).toBe(reviewed.grounding_coverage);
  });

  it("counts rejected quotes that a speaker boundary split, and only those", () => {
    const call = {
      ...mixedCall(),
      redacted_transcript: "Customer: I need help with\nAgent: my bill please.\nAgent: Sure.",
      rejected_claims: [
        { kind: "rubric_score", attempted_quote: "I need help with my bill please.", speaker: "Customer", description: "split" },
        { kind: "rubric_score", attempted_quote: "I will cancel my line", speaker: "Customer", description: "invented" },
      ],
      memory_rules: [{}, {}],
      pii_redacted_count: 1,
    } as unknown as CallDetail;
    const facts = factsFrom(call, { rules: [1, 2, 3] });
    expect(facts.rejected).toBe(2);
    expect(facts.boundarySplit).toBe(1);
    expect(facts.rulesActive).toBe(3);
    expect(facts.rulesUsed).toBe(2);
    expect(facts.pii).toBe(1);
  });
});
