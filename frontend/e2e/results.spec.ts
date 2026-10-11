import { expect, test } from "@playwright/test";

import { mockApi, open, SHOTS, type UiLang } from "./mockApi";

const VERDICT = {
  en: { ar: "Contradicts Phase 4", en: "Inconclusive (within noise)", mixed: "Confirms the hypothesis", title: "What the measurements say" },
  ar: { ar: "يناقض المرحلة 4", en: "غير حاسم (ضمن التذبذب)", mixed: "يؤكد الفرضية", title: "ماذا تقول القياسات" },
} as const;

for (const lang of ["en", "ar"] as UiLang[]) {
  test(`results (${lang}): every number from GET /results, per language`, async ({ page }) => {
    const api = await mockApi(page);
    await open(page, "/results", lang);
    const copy = VERDICT[lang];
    await expect(page.locator("html")).toHaveAttribute("dir", lang === "ar" ? "rtl" : "ltr");
    await expect(page.getByRole("heading", { level: 1 })).toHaveText(copy.title);
    for (const [language, delta] of [["ar", "+0.021"], ["en", "+0.022"], ["mixed", "+0.042"]] as const) {
      await expect(page.getByTestId(`delta-${language}`)).toHaveText(delta);
      await expect(page.getByTestId(`verdict-${language}`)).toHaveText(copy[language]);
    }
    await expect(page.getByTestId("arm-gemini").getByTestId("accuracy")).toHaveText("0.857");
    await expect(page.getByTestId("tile-throughput")).toContainText("12.5×");
    await page.screenshot({ path: `${SHOTS}/results-${lang}.png`, fullPage: true });

    await page.getByRole("button", { name: "ar", exact: true }).click();
    await expect(page.getByTestId("arm-vllm_adapter_c16").getByTestId("schema-valid")).toHaveText("14 / 15");
    expect(api.requests.filter((r) => r.path === "/results")).toHaveLength(1);
  });
}

test("new pages are usable at 1024 px: no horizontal page scroll", async ({ page }) => {
  await page.setViewportSize({ width: 1024, height: 768 });
  await mockApi(page);
  for (const lang of ["en", "ar"] as UiLang[]) {
    for (const [path, ready] of [["/new", "step-0"], ["/memory", "existing-rule"], ["/results", "arm-gemini"]] as const) {
      await open(page, path, lang);
      if (path === "/new") await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
      else await expect(page.getByTestId(ready)).toBeVisible();
      expect(await page.evaluate(() => document.documentElement.scrollWidth), `${path} ${lang}`).toBeLessThanOrEqual(1024);
    }
    await page.screenshot({ path: `${SHOTS}/results-1024-${lang}.png` });
  }
});
