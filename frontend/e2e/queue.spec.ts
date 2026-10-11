import { expect, test } from "@playwright/test";

import { mockApi, open, SHOTS, type UiLang } from "./mockApi";

const COPY = {
  en: { title: "Awaiting human review", mixed: "Code-switched", reason: "3 of 9 claims had no verbatim evidence", review: "Review" },
  ar: { title: "بانتظار المراجعة البشرية", mixed: "مختلط", reason: "3 من 9 ادعاءات بلا دليل حرفي", review: "مراجعة" },
} as const;

for (const lang of ["en", "ar"] as UiLang[]) {
  test.describe(`queue (${lang})`, () => {
    test("lists the escalated calls per language, filters, and opens a review", async ({ page }) => {
      const api = await mockApi(page);
      await open(page, "/", lang);
      const copy = COPY[lang];

      await expect(page.locator("html")).toHaveAttribute("dir", lang === "ar" ? "rtl" : "ltr");
      await expect(page.getByRole("heading", { level: 1 })).toHaveText(copy.title);
      await expect(page.locator("tbody tr")).toHaveCount(6);
      await expect(page.getByText(copy.reason)).toBeVisible();
      await expect(page.getByTestId("pending-ar")).toHaveText("2");
      await expect(page.getByTestId("pending-en")).toHaveText("1");
      await expect(page.getByTestId("pending-mixed")).toHaveText("3");
      await page.screenshot({ path: `${SHOTS}/queue-${lang}.png`, fullPage: true });

      await page.getByRole("button", { name: new RegExp(copy.mixed) }).click();
      await expect(page.locator("tbody tr")).toHaveCount(3);
      await expect(page.getByTestId("pending-ar")).toHaveText("2");

      await page.getByRole("link", { name: copy.review, exact: true }).first().click();
      await expect(page).toHaveURL(/\/calls\/7f3a91c2/);
      expect(api.requests.every((r) => r.headers["x-api-key"] === "e2e-key")).toBe(true);
    });
  });
}

test("usable at 1024 px: no horizontal page scroll", async ({ page }) => {
  await page.setViewportSize({ width: 1024, height: 768 });
  await mockApi(page);
  for (const lang of ["en", "ar"] as UiLang[]) {
    await open(page, "/", lang);
    await expect(page.locator("tbody tr")).toHaveCount(6);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(1024);
    await page.goto("/calls/7f3a91c2-5d4e-4b8a-9c1f-2e6d8a0b3c47");
    await expect(page.getByTestId("turn-1")).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(1024);
    await page.screenshot({ path: `${SHOTS}/review-1024-${lang}.png` });
  }
});
