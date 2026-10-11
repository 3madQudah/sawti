import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { CONFLICT_IDS } from "../../test/fixtures";
import { renderApp } from "../../test/render";

describe("Memory", () => {
  it("shows per-language rule counts and the pending conflicts beside the rule they contradict", async () => {
    renderApp("/memory");
    await waitFor(() => expect(screen.getByTestId("rules-mixed")).toHaveTextContent("2"));
    expect(screen.getByTestId("rules-ar")).toHaveTextContent("1");
    expect(screen.getByTestId("fp-en")).toHaveTextContent("no resolutions yet");
    expect(screen.getByTestId("fp-mixed")).toHaveTextContent("2 of 5 resolutions");
    expect(screen.getByText(/4 distinct active rules/)).toBeInTheDocument();

    const list = await screen.findByRole("list", { name: "Pending conflicts" });
    expect(within(list).getAllByRole("button")).toHaveLength(2);
    const existing = screen.getByTestId("existing-rule");
    expect(existing).toHaveTextContent("Asking only for the line number does not count as identity verification.");
    expect(existing).toHaveTextContent("similarity 0.84");
    expect(existing).toHaveTextContent("From 4 corrections · used in 37 analyses");
    expect(screen.getByTestId("proposed-rule")).toHaveTextContent("Confirming the line number counts");
    expect(screen.getByTestId("proposed-rule")).toHaveTextContent("qa-lina");
  });

  it("resolves a conflict with the chosen action and shows what changed", async () => {
    const { api } = renderApp("/memory");
    await screen.findByTestId("existing-rule");
    await userEvent.click(screen.getByRole("radio", { name: /Keep new/ }));
    await userEvent.click(screen.getByRole("button", { name: "Resolve conflict" }));
    expect(await screen.findByRole("status")).toHaveTextContent("Resolved: Keep new. Consolidation queued on the worker.");
    expect(api.resolutions).toEqual([{ conflictId: CONFLICT_IDS[0], action: "keep_new", reviewer: "qa-test" }]);
    // The next conflict is selected; the resolved one is gone.
    expect(await screen.findByText('"بكرا" said after business hours means the next business day.', { selector: "[data-testid=proposed-rule] p" })).toBeInTheDocument();
    expect(within(screen.getByRole("list", { name: "Pending conflicts" })).getAllByRole("button")).toHaveLength(1);

    await userEvent.click(screen.getByRole("tab", { name: /Active rules/ }));
    expect(await screen.findByText("Confirming the line number counts as identity verification for billing questions.")).toBeInTheDocument();
    expect(screen.queryByText("Asking only for the line number does not count as identity verification.")).not.toBeInTheDocument();
  });

  it("lists active rules with languages, sources and usage", async () => {
    renderApp("/memory?tab=rules");
    const rows = await screen.findAllByRole("row");
    expect(rows).toHaveLength(5); // header + 4
    expect(screen.getByText("4 active rules.", { exact: false })).toBeInTheDocument();
  });

  it("works in Arabic and RTL", async () => {
    const { api } = renderApp("/memory", { lang: "ar" });
    expect(await screen.findByRole("heading", { level: 1 })).toHaveTextContent("ذاكرة القواعد");
    expect(document.documentElement.dir).toBe("rtl");
    await screen.findByTestId("existing-rule");
    expect(screen.getByTestId("existing-rule")).toHaveTextContent("من 4 تصحيحات · استُخدمت في 37 تحليلات");
    await userEvent.click(screen.getByRole("radio", { name: /ليس تعارضاً فعلياً/ }));
    await userEvent.click(screen.getByRole("button", { name: "حسم التعارض" }));
    expect(await screen.findByRole("status")).toHaveTextContent("حُسم: ليس تعارضاً فعلياً.");
    expect(api.resolutions[0]!.action).toBe("not_a_conflict");
  });
});
