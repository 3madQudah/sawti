import { expect, it } from "vitest";

import { turnDirection } from "./direction";

it("sets each turn's direction from its own language", () => {
  expect(turnDirection("ar")).toBe("rtl");
  expect(turnDirection("en")).toBe("ltr");
  expect(turnDirection("mixed")).toBe("auto");
});
