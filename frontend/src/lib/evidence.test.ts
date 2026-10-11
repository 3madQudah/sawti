import { describe, expect, it } from "vitest";

import { mixedCall } from "../test/fixtures";
import { evidenceNote, findVerbatim } from "./evidence";

const call = mixedCall();
const text = call.redacted_transcript;

describe("findVerbatim", () => {
  it("finds an exact substring and the turn it starts in (1-based)", () => {
    const match = findVerbatim(text, call.turns, "خلال ٣ أيام عمل");
    expect(match).not.toBeNull();
    expect(match!.turn).toBe(7);
    expect(text.slice(match!.start, match!.end)).toBe("خلال ٣ أيام عمل");
    expect(match!.occurrences).toBe(1);
  });

  it("is exact: no trimming, no case folding, no digit normalisation", () => {
    expect(findVerbatim(text, call.turns, " خلال ٣ أيام عمل  ")).toBeNull();
    expect(findVerbatim(text, call.turns, "okay, but")).toBeNull();
    expect(findVerbatim(text, call.turns, "خلال 3 أيام عمل")).toBeNull();
  });

  it("rejects empty evidence and a match inside a speaker prefix", () => {
    expect(findVerbatim(text, call.turns, "")).toBeNull();
    expect(findVerbatim(text, call.turns, "Customer: ")).toBeNull();
  });

  it("counts every occurrence and reports the first", () => {
    const match = findVerbatim(text, call.turns, "رح ");
    expect(match!.turn).toBe(7);
    expect(match!.occurrences).toBe(2); // turns 7 and 9
  });

  it("puts the verified span in the verdict note", () => {
    const match = findVerbatim(text, call.turns, "رسالة اليوم")!;
    expect(evidenceNote("رسالة اليوم", match)).toBe(
      `Corrected evidence (verbatim, turn 9, chars ${match.start}-${match.end}): «رسالة اليوم»`,
    );
  });
});
