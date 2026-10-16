/**
 * The ~90 s demo video, as a scripted browser session.
 *
 *   npm run demo:video        (records frames, then demo_video/render.sh encodes the MP4)
 *
 * Frames come from Chrome's screencast at full 1920×1080, so zoomed text is
 * re-rendered sharp rather than upscaled.
 */
import { mkdirSync, rmSync, writeFileSync } from "node:fs";

import { expect, type Locator, type Page, test } from "@playwright/test";

import { readBody } from "../src/test/mockApi";
import { overlayScript } from "./overlay";
import { demoSource } from "./source";

const FRAMES = "demo_video/out/frames";
const REPO_URL = process.env.DEMO_REPO_URL ?? "github.com/3madQudah/sawti";
const WAV = Buffer.from("524946460000000057415645", "hex");

test.setTimeout(240_000);

// ---------------------------------------------------------------- recording
let flushIndex: (() => void) | null = null;

// Keep the frames of a failed run too, so it can be rendered and inspected.
test.afterEach(() => flushIndex?.());

async function startScreencast(page: Page) {
  rmSync(FRAMES, { recursive: true, force: true });
  mkdirSync(FRAMES, { recursive: true });
  const cdp = await page.context().newCDPSession(page);
  const index: { file: string; t: number }[] = [];
  cdp.on("Page.screencastFrame", async ({ data, metadata, sessionId }) => {
    const file = `f${String(index.length).padStart(6, "0")}.jpg`;
    writeFileSync(`${FRAMES}/${file}`, Buffer.from(data, "base64"));
    index.push({ file, t: metadata.timestamp ?? Date.now() / 1000 });
    await cdp.send("Page.screencastFrameAck", { sessionId }).catch(() => {});
  });
  await cdp.send("Page.startScreencast", { format: "jpeg", quality: 92, maxWidth: 1920, maxHeight: 1080, everyNthFrame: 1 });
  flushIndex = () => writeFileSync(`${FRAMES}/index.json`, JSON.stringify({ frames: index }));
  return async () => {
    await cdp.send("Page.stopScreencast");
    flushIndex?.();
  };
}

// ------------------------------------------------------------------ director
const sleep = (page: Page, ms: number) => page.waitForTimeout(ms);
let cursor = { x: 960, y: 620 };

/** Ease the cursor to a point (real mouse events, so hover states show). */
async function glide(page: Page, x: number, y: number, ms = 650) {
  const steps = Math.max(12, Math.round(ms / 16));
  const from = { ...cursor };
  for (let i = 1; i <= steps; i++) {
    const k = i / steps;
    const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
    await page.mouse.move(from.x + (x - from.x) * e, from.y + (y - from.y) * e);
    await page.waitForTimeout(ms / steps);
  }
  cursor = { x, y };
}

async function centre(target: Locator) {
  await expect(target).toBeVisible();
  const box = (await target.boundingBox())!;
  return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
}

async function point(page: Page, target: Locator, ms?: number) {
  const c = await centre(target);
  await glide(page, c.x, c.y, ms);
}

async function click(page: Page, target: Locator, ms?: number) {
  await point(page, target, ms);
  await sleep(page, 120);
  await page.mouse.down();
  await sleep(page, 70);
  await page.mouse.up();
  await sleep(page, 180);
}

async function type(page: Page, field: Locator, text: string) {
  await click(page, field);
  await field.pressSequentially(text, { delay: 26 });
}

async function zoomTo(page: Page, target: Locator, opts: { maxScale?: number; fill?: number; focusY?: number } = {}) {
  await expect(target).toBeVisible();
  const box = (await target.boundingBox())!;
  await page.evaluate(([b, o]) => (window as any).__demo.zoomTo(b, o), [box, opts] as const);
  await sleep(page, 1200);
}

async function zoomOut(page: Page) {
  await page.evaluate(() => (window as any).__demo.zoomOut());
  await sleep(page, 1200);
}

async function caption(page: Page, html: string | null, kicker?: string) {
  await page.evaluate(([h, k]) => (window as any).__demo.caption(h, k), [html, kicker ?? null] as const);
}

async function card(page: Page, html: string | null) {
  await page.evaluate((h) => (window as any).__demo.card(h), html);
}

/** Smooth-scroll so the target sits near the top third; only at zoom 1. */
async function scrollTo(page: Page, target: Locator, offset = 140) {
  const y = await target.evaluate((el, off) => el.getBoundingClientRect().top + window.scrollY - off, offset);
  await page.evaluate((top) => window.scrollTo({ top, behavior: "smooth" }), y);
  await sleep(page, 1100);
}

async function scrollTop(page: Page) {
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "smooth" }));
  await sleep(page, 900);
}

/** Navigate by clicking the header link, like a person would. */
async function nav(page: Page, name: string) {
  await click(page, page.getByRole("navigation").getByRole("link", { name }).or(page.getByRole("banner").getByRole("link", { name })).first());
  await sleep(page, 700);
}

// --------------------------------------------------------------------- story
test("Sawti demo video", async ({ page }) => {
  const source = demoSource();
  const { plan } = source;
  const f = plan.facts;
  const num = (x: number | null) => (x === null ? "—" : x.toFixed(x * 100 % 1 === 0 ? 2 : 3));
  const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;
  console.log(`demo data: ${plan.kind}`);

  await page.addInitScript(overlayScript);
  await page.route((url) => url.pathname.startsWith("/api/"), async (route) => {
    const r = route.request();
    const headers = await r.allHeaders();
    const raw = r.postDataBuffer();
    const body = await readBody(new Request(r.url(), { method: r.method(), headers, body: raw ?? undefined }));
    const response = source.handle(r.method(), r.url(), headers, body);
    await route.fulfill({ status: response.status, contentType: "application/json", body: JSON.stringify(response.body) });
  });

  await page.goto("/new");
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await page.mouse.move(cursor.x, cursor.y);
  await card(
    page,
    `<div class="title">Sawti<span class="ar">صوتي</span></div>
     <p>An agent that reviews contact-center calls in Arabic, English, and the code-switched speech between them.</p>
     <div class="chips"><span>Whisper + diarization</span><span>LangGraph</span><span>Grounded claims</span><span>Human review</span><span>Rule memory</span></div>
     <div class="small">${plan.kind === "mock" ? "illustrative data" : "real run · synthetic call audio"}</div>`,
  );
  const stop = await startScreencast(page);
  const t0 = Date.now();
  const mark = (name: string) => console.log(`[t=${((Date.now() - t0) / 1000).toFixed(1)}s] ${name}`);
  await sleep(page, 3000);
  await card(page, null);
  await sleep(page, 800);

  mark("1 — New call");
  // 1 — New call ------------------------------------------------------------
  await caption(page, "Upload a call recording and pick its language.", "1 · New call");
  await page.getByLabel("Call recording").setInputFiles({ name: plan.audioName, mimeType: "audio/wav", buffer: WAV });
  await sleep(page, 600);
  const langRadio = page.getByRole("radio", { name: new RegExp({ ar: "Arabic", en: "English", mixed: "Code-switched" }[plan.language]) });
  await click(page, langRadio);
  await zoomTo(page, page.getByText("Auto-detect garbles", { exact: false }).locator(".."), { maxScale: 1.7, fill: 0.7 });
  await caption(page, "No auto-detect on purpose: it garbles code-switched Arabic.", "1 · New call");
  await sleep(page, 2200);
  await zoomOut(page);
  await click(page, page.getByRole("button", { name: "Upload and analyze" }));

  const pipeline = page.locator("section", { has: page.locator("#pipeline-title") });
  await zoomTo(page, pipeline, { maxScale: 1.6, fill: 0.85 });
  await caption(page, "Transcription and diarization run on this machine. PII is redacted before any model sees the text.", "Pipeline");
  const openReview = page.getByRole("link", { name: "Open review" });
  await expect(openReview).toBeVisible({ timeout: 20_000 });
  await sleep(page, 900);
  await caption(page, `Grounding coverage ${num(f.coverage)} is below the ${num(f.threshold)} threshold, so the call goes to a person, not an automated verdict.`, "Low confidence → human");
  await sleep(page, 3600);
  await zoomOut(page);
  await click(page, openReview);
  await caption(page, null);

  mark("2 — Review");
  // 2 — Review --------------------------------------------------------------
  const summary = page.locator('[aria-label="Call summary"]');
  await expect(page.getByRole("heading", { name: /^Transcript/ })).toBeVisible();
  await sleep(page, 700);
  await zoomTo(page, summary, { maxScale: 1.7, fill: 0.92 });
  await caption(page, `Coverage ${num(f.coverage)} against a ${num(f.threshold)} threshold. ${plural(f.pii, "PII span", "PII spans")} redacted before the model saw the text.`, "2 · Human review");
  await sleep(page, 2400);
  await zoomOut(page);

  const firstClaim = page.locator("article[data-testid^='claim-']").first();
  await click(page, firstClaim.getByRole("button", { name: "Show this evidence in the transcript" }));
  const selected = page.locator("[data-testid^='turn-']").filter({ has: page.locator("mark[data-selected='true']") }).first();
  await zoomTo(page, selected, { maxScale: 1.9, fill: 0.75 });
  await caption(page, "Every claim points at a verbatim quote in the transcript. Click a claim, its evidence lights up.", "Grounding");
  await sleep(page, 3200);
  await zoomOut(page);

  const rejected = page.getByText("Rejected by grounding").locator("xpath=ancestor::*[self::section or self::div][contains(@class,'rounded')][1]");
  await caption(page, null);
  await scrollTo(page, rejected, 260);
  await zoomTo(page, rejected, { maxScale: 1.7, fill: 0.8 });
  if (f.boundarySplit > 0) {
    // Say what actually happened on this call: the quotes are real, the turn boundaries are not.
    const which = f.boundarySplit === f.rejected ? `All ${f.rejected}` : `${f.boundarySplit} of ${f.rejected}`;
    await caption(page, `${which} rejected quotes are real. A diarization speaker boundary split each one mid-sentence, so the verbatim match failed.`, "Honest limitation");
    await sleep(page, 4200);
    await zoomOut(page);
    if (f.splitTurn !== null) {
      await caption(page, null);
      const a = page.getByTestId(`turn-${f.splitTurn}`), b = page.getByTestId(`turn-${f.splitTurn + 1}`);
      await scrollTo(page, a, 300);
      const box = await a.evaluate((el, next) => {
        const r1 = el.getBoundingClientRect(), r2 = document.querySelector(`[data-testid="${next}"]`)!.getBoundingClientRect();
        return { x: r1.x, y: r1.y, width: r1.width, height: r2.bottom - r1.top };
      }, `turn-${f.splitTurn + 1}`);
      await page.evaluate((bx) => (window as any).__demo.zoomTo(bx, { maxScale: 1.9, fill: 0.8 }), box);
      await sleep(page, 1200);
      void b;
    }
    await caption(page, "Dropping them was safe: a person reviews the call instead of trusting a wrong verdict. The boundary bug is logged as a known, open issue.", "Honest limitation");
    await sleep(page, 4400);
    await zoomOut(page);
  } else {
    await caption(page, "Claims whose quote isn't in the transcript are rejected and never shown as findings.", "Grounding");
    await sleep(page, 3200);
    await zoomOut(page);
  }
  await caption(page, null);
  await scrollTop(page);

  mark("The reviewer's verdicts");
  // The reviewer's verdicts, in order; the correction is typed out in full.
  for (const v of plan.verdicts) {
    const claimCard = page.getByTestId(`claim-${v.claimId}`);
    const label = { confirm: "Confirm", correct: "Correct", reject: "Reject" }[v.verdict];
    if (v.verdict === "correct" && plan.correction?.claimId === v.claimId) {
      await scrollTo(page, claimCard, 120);
      await click(page, claimCard.getByRole("button", { name: label, exact: true }));
      const form = claimCard.locator("div", { hasText: "STRUCTURED CORRECTION" }).last();
      await zoomTo(page, claimCard, { maxScale: 1.55, fill: 0.85, focusY: 0.45 });
      await caption(page, "A structured correction: the field, the new value, and evidence that must be verbatim.", "Correct a claim");
      const c = plan.correction;
      if (c.field !== "description") await claimCard.getByLabel("Field").selectOption(c.field);
      await type(page, claimCard.getByLabel("Corrected value"), c.value);
      if (c.evidence) {
        await type(page, claimCard.getByLabel(/Corrected evidence/), c.evidence);
        await expect(claimCard.getByRole("status")).toContainText("Found verbatim");
      }
      await sleep(page, 1600);
      await zoomOut(page);
      void form;
    } else {
      const button = claimCard.getByRole("button", { name: label, exact: true });
      const box = await button.boundingBox();
      if (!box || box.y < 70 || box.y + box.height > 900) await scrollTo(page, claimCard, 220);
      await click(page, button, 260);
    }
  }
  await caption(page, null);
  mark("submit");
  const submit = page.getByRole("button", { name: "Submit review and resume run" });
  await expect(submit).toBeEnabled();
  await click(page, submit);
  await caption(page, plan.correction
    ? "The paused run resumes with the reviewer's verdicts. The correction becomes a candidate memory rule."
    : "Every claim judged. The paused run resumes and finishes as human-reviewed.", "Resume");
  await expect(page.getByRole("status").filter({ hasText: /human-reviewed|Resuming/ }).first()).toBeVisible({ timeout: 15_000 });
  await scrollTop(page);
  await sleep(page, 2400);

  mark("3 — Memory");
  // 3 — Memory --------------------------------------------------------------
  await caption(page, null);
  await nav(page, "Memory");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Rule memory");
  await sleep(page, 600);
  const proposed = page.getByTestId("proposed-rule");
  if (await proposed.count()) {
    const conflict = proposed.locator("xpath=ancestor::section[1]");
    await zoomTo(page, conflict, { maxScale: 1.5, fill: 0.85 });
    await caption(page, "A new rule that contradicts an existing one is held for a person to resolve, never merged silently.", "3 · Memory");
    await sleep(page, 3600);
    await zoomOut(page);
  } else {
    await caption(page, `${plural(f.rulesActive, "active rule", "active rules")}, induced from reviewer corrections. This call's analysis retrieved ${f.rulesUsed} of them.`, "3 · Memory");
    await click(page, page.getByRole("tab", { name: /Active rules/ }));
    await sleep(page, 3400);
  }

  mark("4 — Results");
  // 4 — Results -------------------------------------------------------------
  await caption(page, null);
  await nav(page, "Results");
  await expect(page.getByText("Does rule memory improve accuracy?")).toBeVisible();
  await sleep(page, 500);
  const memoryPanel = page.locator("section, div").filter({ has: page.getByText("Does rule memory improve accuracy?") }).filter({ hasText: "Verdict rule" }).last();
  await zoomTo(page, memoryPanel, { maxScale: 1.35, fill: 0.95, focusY: 0.5 });
  await caption(page, "Every number is per language with its sample size. Arabic, English and code-switched are never blended.", "4 · Measured");
  await sleep(page, 3400);
  await zoomOut(page);
  await caption(page, null);
  const serving = page.getByText("Cloud vs self-hosted serving");
  await scrollTo(page, serving, 120);
  const servingPanel = serving.locator("xpath=ancestor::*[contains(@class,'rounded')][1]");
  await zoomTo(page, servingPanel, { maxScale: 1.3, fill: 0.95 });
  await caption(page, "Cloud API vs self-hosted Qwen3-8B + QLoRA on vLLM: throughput, cost, latency, schema failures.", "4 · Measured");
  await sleep(page, 1400);
  for (const l of ["ar"]) {
    await click(page, servingPanel.getByRole("button", { name: l, exact: true }), 330);
    await sleep(page, 900);
  }
  await zoomOut(page);
  await scrollTop(page);

  mark("5 — Arabic UI");
  // 5 — Arabic UI -----------------------------------------------------------
  await caption(page, "The whole interface is bilingual: one toggle flips it to Arabic, right to left.", "5 · عربي");
  await click(page, page.getByRole("button", { name: "عربي" }));
  await expect(page.locator("html")).toHaveAttribute("dir", "rtl");
  await sleep(page, 2800);
  await caption(page, null);

  await card(
    page,
    `<div class="title">Sawti<span class="ar">صوتي</span></div>
     <p>Grounded claims · human review when unsure · corrections that become rules</p>
     <div class="small">${REPO_URL}</div>`,
  );
  await sleep(page, 3600);
  await stop();
});
