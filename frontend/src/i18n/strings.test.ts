import { describe, expect, it } from "vitest";

import { ar, en, format } from "./strings";

const placeholders = (s: string) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();

describe("strings", () => {
  it("has every English key in Arabic, none empty, with the same placeholders", () => {
    expect(Object.keys(ar).sort()).toEqual(Object.keys(en).sort());
    for (const key of Object.keys(en) as (keyof typeof en)[]) {
      expect(ar[key].trim(), key).not.toBe("");
      expect(placeholders(ar[key]), key).toEqual(placeholders(en[key]));
    }
  });

  it("fills placeholders and leaves unknown ones visible", () => {
    expect(format("{a} of {b}", { a: 1, b: 2 })).toBe("1 of 2");
    expect(format("{a} of {b}", { a: 1 })).toBe("1 of {b}");
  });
});
