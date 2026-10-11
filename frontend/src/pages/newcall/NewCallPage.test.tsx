import { File as NodeFile } from "node:buffer";

import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { MockApi } from "../../test/mockApi";
import { renderApp } from "../../test/render";

// The upload goes through Node's fetch (undici), which can only serialize its own FormData and Files:
// jsdom's File has no stream() and jsdom's FormData stringifies a Node File. So this test uses Node's
// classes for both. (The browser, and the Playwright e2e, have one consistent set and need nothing.)
const NodeFormData = (await new Response("a=1", { headers: { "content-type": "application/x-www-form-urlencoded" } }).formData()).constructor as typeof FormData;
beforeEach(() => {
  vi.stubGlobal("FormData", NodeFormData);
});
const wav = () => new NodeFile([new Uint8Array([0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x41, 0x56, 0x45])], "call_0067_mixed.wav", { type: "audio/wav" }) as unknown as File;

/** Short pipeline so the test doesn't wait on several polls. */
function fastApi() {
  const api = new MockApi();
  api.audioProgress = ["processing", "awaiting_review"];
  return api;
}

describe("New call", () => {
  it("needs a recording and a language before it uploads", async () => {
    renderApp("/new");
    const start = screen.getByRole("button", { name: "Upload and analyze" });
    expect(start).toBeDisabled();
    await userEvent.upload(screen.getByLabelText("Call recording"), wav());
    expect(start).toBeDisabled();
    await userEvent.click(screen.getByRole("radio", { name: /Code-switched/ }));
    expect(start).toBeEnabled();
  });

  it("refuses a file that is not wav, mp3 or m4a without calling the API", async () => {
    const { api } = renderApp("/new");
    await userEvent.upload(screen.getByLabelText("Call recording"), new File(["x"], "notes.txt", { type: "text/plain" }), { applyAccept: false });
    expect(screen.getByRole("alert")).toHaveTextContent("Only .wav, .mp3 or .m4a files are accepted.");
    expect(api.requests).toHaveLength(0);
  });

  it("uploads, follows the pipeline to the review queue, and links to the review", async () => {
    const api = fastApi();
    renderApp("/new", { api });
    await userEvent.upload(screen.getByLabelText("Call recording"), wav());
    await userEvent.click(screen.getByRole("radio", { name: /Code-switched/ }));
    await userEvent.type(screen.getByLabelText(/Agent id/), "agent-lina");
    await userEvent.click(screen.getByRole("button", { name: "Upload and analyze" }));

    const review = await screen.findByRole("link", { name: "Open review" }, { timeout: 5000 });
    expect(api.uploads).toEqual([{ fileName: "call_0067_mixed.wav", size: 12, language_hint: "mixed", agent_external_id: "agent-lina" }]);
    expect(review.getAttribute("href")).toMatch(/^\/calls\/5e1f0a00-/);
    expect(screen.getByText("Escalated: grounding coverage 0.67 is below 0.70. Waiting for a reviewer.")).toBeInTheDocument();
    expect(screen.getByTestId("step-1")).toHaveTextContent("3 PII spans redacted");
    expect(screen.getByTestId("step-3")).toHaveTextContent("8.1 s");
    expect(screen.getByTestId("step-4")).toHaveAttribute("data-state", "done");
    expect(screen.getByText("Awaiting review")).toBeInTheDocument(); // recent uploads
    expect(screen.getByRole("button", { name: "Analyze another call" })).toBeInTheDocument();
  });

  it("works in Arabic and RTL, and reports a failed pipeline", async () => {
    const api = new MockApi();
    api.audioProgress = ["failed"];
    renderApp("/new", { api, lang: "ar" });
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("تحليل مكالمة جديدة");
    expect(document.documentElement.dir).toBe("rtl");
    await userEvent.upload(screen.getByLabelText("تسجيل المكالمة"), wav());
    await userEvent.click(screen.getByRole("radio", { name: /عربي/ }));
    await userEvent.click(screen.getByRole("button", { name: "رفع وتحليل" }));
    expect(await screen.findByText("توقفت المعالجة: asr: no speech detected")).toBeInTheDocument();
    expect(screen.getByTestId("step-1")).toHaveAttribute("data-state", "failed");
    expect(api.uploads[0]!.language_hint).toBe("ar");
    expect(api.uploads[0]!.agent_external_id).toBeNull();
  });
});
