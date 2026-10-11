import { expect, test } from "@playwright/test";

import { mockApi, open, SHOTS, type UiLang } from "./mockApi";

const COPY = {
  en: { title: "Analyze a new call", mixed: /Code-switched/, start: "Upload and analyze", review: "Open review", input: "Call recording" },
  ar: { title: "تحليل مكالمة جديدة", mixed: /مختلط/, start: "رفع وتحليل", review: "فتح المراجعة", input: "تسجيل المكالمة" },
} as const;

// A RIFF/WAVE header: the mock, like the API, only looks at the name; the bytes are what a real upload would carry.
const WAV = Buffer.from("524946460000000057415645", "hex");

for (const lang of ["en", "ar"] as UiLang[]) {
  test(`new call (${lang}): upload, follow the pipeline, open the review`, async ({ page }) => {
    const api = await mockApi(page);
    await open(page, "/new", lang);
    const copy = COPY[lang];
    await expect(page.locator("html")).toHaveAttribute("dir", lang === "ar" ? "rtl" : "ltr");
    await expect(page.getByRole("heading", { level: 1 })).toHaveText(copy.title);
    await page.getByLabel(copy.input).setInputFiles({ name: "call_0067_mixed.wav", mimeType: "audio/wav", buffer: WAV });
    await page.getByRole("radio", { name: copy.mixed }).check();
    await page.screenshot({ path: `${SHOTS}/newcall-${lang}.png`, fullPage: true });

    await page.getByRole("button", { name: copy.start }).click();
    await expect(page.getByRole("link", { name: copy.review })).toBeVisible({ timeout: 10_000 });
    await expect(page.getByTestId("step-4")).toHaveAttribute("data-state", "done");
    expect(api.uploads).toEqual([{ fileName: "call_0067_mixed.wav", size: WAV.length, language_hint: "mixed", agent_external_id: null }]);
    await page.screenshot({ path: `${SHOTS}/newcall-routed-${lang}.png`, fullPage: true });
  });
}
