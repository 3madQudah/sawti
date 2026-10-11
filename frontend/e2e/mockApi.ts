import type { Page } from "@playwright/test";

import { MockApi, readBody } from "../src/test/mockApi";

/** Answer every `/api/**` request in the browser from the shared in-memory mock. No backend runs. */
export async function mockApi(page: Page): Promise<MockApi> {
  const api = new MockApi();
  // The path must start with /api/: a glob like **/api/** also matches Vite's /src/api/*.ts modules.
  await page.route((url) => url.pathname.startsWith("/api/"), async (route) => {
    const request = route.request();
    const headers = await request.allHeaders();
    const raw = request.postDataBuffer();
    // Rebuild a fetch Request so JSON and multipart bodies are parsed the same way as in Vitest.
    const body = await readBody(new Request(request.url(), { method: request.method(), headers, body: raw ?? undefined }));
    const response = api.handle(request.method(), request.url(), headers, body);
    await route.fulfill({ status: response.status, contentType: "application/json", body: JSON.stringify(response.body) });
  });
  return api;
}

export type UiLang = "en" | "ar";

/** Open `path` in English, then switch with the header toggle if `lang` is Arabic (so the toggle is exercised). */
export async function open(page: Page, path: string, lang: UiLang): Promise<void> {
  await page.goto(path);
  if (lang === "ar") await page.getByRole("button", { name: "عربي" }).click();
}

export const SHOTS = "e2e/screenshots";
