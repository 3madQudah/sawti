import { describe, expect, it } from "vitest";

import { textDirection, turnDirection } from "./direction";

it("sets each turn's direction from its own language", () => {
  expect(turnDirection("ar")).toBe("rtl");
  expect(turnDirection("en")).toBe("ltr");
  expect(turnDirection("mixed")).toBe("auto");
});

describe("textDirection", () => {
  it("follows the majority script, not the first character", () => {
    expect(textDirection('"بكرا" sets the deadline to the next calendar day.')).toBe("ltr");
    expect(textDirection("رح أرفع طلب refund اليوم")).toBe("rtl");
    expect(textDirection("12:30")).toBe("ltr");
  });
});
