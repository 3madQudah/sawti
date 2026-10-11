import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { MockApi } from "../../test/mockApi";
import { renderApp } from "../../test/render";

/**
 * The expected values are typed in from eval_results.md (2026-10-10), not read
 * from the mock, so these fail if the mock ever stops matching the recorded
 * 6.3 numbers. (The backend pins the mock to `GET /results` as well.)
 */
const MEMORY = {
  ar: { delta: "+0.021", noise: "noise 0.015 · Phase 4 +0.056", verdict: "Contradicts Phase 4", n: "60 / 60" },
  en: { delta: "+0.022", noise: "noise 0.035 · Phase 4 +0.005", verdict: "Inconclusive (within noise)", n: "30 / 30" },
  mixed: { delta: "+0.042", noise: "noise 0.009 · Phase 4 +0.000", verdict: "Confirms the hypothesis", n: "60 / 60" },
} as const;

/** Text without the bidi isolates the page wraps numbers in. */
const plain = (el: HTMLElement) => (el.textContent ?? "").replace(/[\u2066-\u2069]/g, "");

describe("Results", () => {
  it("shows the memory effect against noise and Phase 4, per language, from GET /results", async () => {
    const { api } = renderApp("/results");
    for (const [language, expected] of Object.entries(MEMORY)) {
      expect(await screen.findByTestId(`delta-${language}`)).toHaveTextContent(expected.delta);
      expect(plain(screen.getByTestId(`noise-${language}`))).toBe(expected.noise);
      expect(screen.getByTestId(`verdict-${language}`)).toHaveTextContent(expected.verdict);
      expect(screen.getByTestId(`memory-${language}`)).toHaveTextContent(`n = ${expected.n} paired calls per repeat`);
    }
    expect(screen.getByText("Memory on minus memory off · 600 calls · 2 repeats", { exact: false })).toBeInTheDocument();
    expect(plain(screen.getByTestId("memory-ar"))).toContain("Repeat 1: off 0.874 → on 0.899 (Δ +0.025)");
    expect(api.requests.map((r) => r.path)).toEqual(["/results"]);
  });

  it("shows the serving comparison for one language at a time, with n and hardware", async () => {
    renderApp("/results");
    const gemini = await screen.findByTestId("arm-gemini");
    // Default language: code-switched.
    expect(within(gemini).getByTestId("accuracy")).toHaveTextContent("0.857");
    expect(within(gemini).getByTestId("latency")).toHaveTextContent("13.1 / 56.4 s");
    expect(within(gemini).getByTestId("cost")).toHaveTextContent("$4.42");
    expect(within(gemini).getByTestId("throughput")).toHaveTextContent("2.946 calls/min");
    const c16 = screen.getByTestId("arm-vllm_adapter_c16");
    expect(within(c16).getByTestId("accuracy")).toHaveTextContent("0.633");
    expect(within(c16).getByTestId("cost")).toHaveTextContent("$5.95");
    expect(within(c16).getByTestId("schema-valid")).toHaveTextContent("15 / 15");
    expect(within(screen.getByTestId("arm-vllm_base_c16")).getByTestId("throughput")).toHaveTextContent("3.549 calls/min");
    expect(within(screen.getByTestId("arm-hf_adapter_c1")).getByTestId("cost")).toHaveTextContent("$110.33");
    expect(screen.getByTestId("tile-throughput")).toHaveTextContent("12.5×");
    expect(screen.getByTestId("tile-throughput")).toHaveTextContent("n = 45 and 15");

    await userEvent.click(screen.getByRole("button", { name: "en" }));
    expect(within(gemini).getByTestId("accuracy")).toHaveTextContent("0.926");
    expect(within(c16).getByTestId("schema-valid")).toHaveTextContent("13 / 15");
    expect(within(screen.getByTestId("arm-vllm_adapter_c1")).getByTestId("cost")).toHaveTextContent("$100.17");
    expect(screen.getByTestId("tile-latency")).toHaveTextContent("8.1 s");
    expect(screen.getByTestId("tile-latency")).toHaveTextContent("p95 69.4 s");
    expect(screen.getByTestId("tile-latency")).toHaveTextContent("n = 14, 1 sleep-affected call excluded");
    expect(screen.getByTestId("tile-failures")).toHaveTextContent("2 / 15");
    expect(screen.getByText(/not a fair comparison across providers/)).toBeInTheDocument();
  });

  it("renders in Arabic and RTL with the same numbers", async () => {
    renderApp("/results", { lang: "ar" });
    expect(await screen.findByRole("heading", { level: 1 })).toHaveTextContent("ماذا تقول القياسات");
    expect(document.documentElement.dir).toBe("rtl");
    expect(await screen.findByTestId("delta-ar")).toHaveTextContent("+0.021");
    expect(screen.getByTestId("verdict-ar")).toHaveTextContent("يناقض المرحلة 4");
    expect(screen.getByTestId("verdict-mixed")).toHaveTextContent("يؤكد الفرضية");
    expect(screen.getByTestId("verdict-en")).toHaveTextContent("غير حاسم (ضمن التذبذب)");
    expect(within(screen.getByTestId("arm-gemini")).getByTestId("accuracy")).toHaveTextContent("0.857");
  });

  it("says so when the results are not recorded", async () => {
    const api = new MockApi();
    api.resultsMissing = true;
    renderApp("/results", { api });
    expect(await screen.findByRole("alert")).toHaveTextContent("results not recorded yet");
  });
});
