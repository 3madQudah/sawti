import { expect, test } from "@playwright/test";

import { mockApi, open, SHOTS, type UiLang } from "./mockApi";

const COPY = {
  en: { title: "Rule memory", keepNew: /Keep new/, resolve: "Resolve conflict", resolved: "Resolved: Keep new.", rules: /Active rules/ },
  ar: { title: "ذاكرة القواعد", keepNew: /اعتماد الجديدة/, resolve: "حسم التعارض", resolved: "حُسم: اعتماد الجديدة.", rules: /القواعد الفعّالة/ },
} as const;

for (const lang of ["en", "ar"] as UiLang[]) {
  test(`memory (${lang}): resolve a conflict, then see the active rules`, async ({ page }) => {
    const api = await mockApi(page);
    await open(page, "/memory", lang);
    const copy = COPY[lang];
    await expect(page.locator("html")).toHaveAttribute("dir", lang === "ar" ? "rtl" : "ltr");
    await expect(page.getByRole("heading", { level: 1 })).toHaveText(copy.title);
    await expect(page.getByTestId("existing-rule")).toContainText("0.84");
    await expect(page.getByTestId("rules-mixed")).toHaveText("2");
    await page.screenshot({ path: `${SHOTS}/memory-${lang}.png`, fullPage: true });

    await page.getByRole("radio", { name: copy.keepNew }).check();
    await page.getByRole("button", { name: copy.resolve }).click();
    await expect(page.getByRole("status")).toContainText(copy.resolved);
    expect(api.resolutions.map((r) => [r.action, r.reviewer])).toEqual([["keep_new", "qa-emad"]]);

    await page.getByRole("tab", { name: copy.rules }).click();
    await expect(page.locator("tbody tr")).toHaveCount(4);
    await page.screenshot({ path: `${SHOTS}/memory-rules-${lang}.png`, fullPage: true });
  });
}
