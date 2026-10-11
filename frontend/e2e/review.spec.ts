import { expect, test } from "@playwright/test";

import { CALL_ID } from "../src/test/fixtures";
import { mockApi, open, SHOTS, type UiLang } from "./mockApi";

const COPY = {
  en: {
    confirm: "Confirm",
    correct: "Correct",
    value: "Corrected value",
    evidence: "Corrected evidence (optional, must be verbatim)",
    found: "✓ Found verbatim in turn 7",
    submit: "Submit review and resume run",
    done: "finished as human-reviewed",
  },
  ar: {
    confirm: "تأكيد",
    correct: "تصحيح",
    value: "القيمة المصحَّحة",
    evidence: "الدليل المصحَّح (اختياري، يجب أن يكون حرفياً)",
    found: "✓ وُجد حرفياً في الدور 7",
    submit: "إرسال المراجعة واستئناف التحليل",
    done: "انتهى كمراجَع بشرياً",
  },
} as const;

const OTHERS = ["commitments[1]", "compliance_flags[0]", "rubric_scores[0]", "rubric_scores[1]", "rubric_scores[2]"];

for (const lang of ["en", "ar"] as UiLang[]) {
  test(`review (${lang}): structured correction with verbatim evidence, submit, resume`, async ({ page }) => {
    const api = await mockApi(page);
    await open(page, `/calls/${CALL_ID}`, lang);
    const copy = COPY[lang];

    // Each turn's direction comes from its own language, not the UI's.
    const turnText = (n: number) => page.getByTestId(`turn-${n}`).locator("p");
    await expect(turnText(1)).toHaveAttribute("dir", "rtl");
    await expect(turnText(2)).toHaveAttribute("dir", "auto");
    await expect(turnText(8)).toHaveAttribute("dir", "ltr");
    await expect(page.locator("html")).toHaveAttribute("dir", lang === "ar" ? "rtl" : "ltr");

    const refund = page.getByTestId("claim-commitments[0]");
    await refund.getByRole("button", { name: copy.correct }).click();
    await refund.getByLabel(copy.value).fill("Refund credited within 3 business days");
    await refund.getByLabel(copy.evidence).fill("خلال ٣ أيام عمل");
    await expect(refund.getByRole("status")).toHaveText(copy.found);

    const submit = page.getByRole("button", { name: copy.submit });
    await expect(submit).toBeDisabled();
    for (const id of OTHERS) await page.getByTestId(`claim-${id}`).getByRole("button", { name: copy.confirm }).click();
    await expect(submit).toBeEnabled();
    await page.screenshot({ path: `${SHOTS}/review-${lang}.png`, fullPage: true });
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: `${SHOTS}/review-viewport-${lang}.png` });
    // The corrected evidence stays outlined after other claims were judged.
    // It may be split where it crosses a claim's highlight; together the parts are the evidence.
    expect((await page.getByTestId("turn-7").locator("[data-corrected]").allTextContents()).join("")).toBe("خلال ٣ أيام عمل");

    await submit.click();
    await expect(page.getByRole("status").filter({ hasText: copy.done })).toBeVisible({ timeout: 10_000 });
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: `${SHOTS}/review-submitted-${lang}.png` });

    expect(api.submissions).toHaveLength(1);
    const [first, ...rest] = api.submissions[0]!.body.verdicts;
    expect(first).toEqual({
      claim_id: "commitments[0]",
      verdict: "correct",
      changes: { description: "Refund credited within 3 business days" },
      note: expect.stringMatching(/^Corrected evidence \(verbatim, turn 7, chars \d+-\d+\): «خلال ٣ أيام عمل»$/),
    });
    expect(rest.map((v) => v.verdict)).toEqual(Array(5).fill("confirm"));
  });
}
