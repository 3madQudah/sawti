import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { renderApp } from "../test/render";

describe("Queue", () => {
  it("lists every escalated call with its reason translated from the API's numbers", async () => {
    const { api } = renderApp("/");
    const rows = await screen.findAllByRole("row");
    expect(rows).toHaveLength(7); // header + 6
    expect(screen.getByText("3 of 9 claims had no verbatim evidence")).toBeInTheDocument();
    expect(screen.getByText("Extraction failed: no claims to check")).toBeInTheDocument();
    expect(screen.getByText("Threshold 0.70 ·", { exact: false })).toBeInTheDocument();
    expect(api.requests[0]).toMatchObject({ method: "GET", path: "/reviews" });
    expect(api.requests[0]!.headers["x-api-key"]).toBe("test-key");
    expect(api.requests[0]!.headers["x-reviewer-id"]).toBe("qa-test");
  });

  it("shows pending counts per language, never blended", async () => {
    renderApp("/");
    await waitFor(() => expect(screen.getByTestId("pending-ar")).toHaveTextContent("2"));
    expect(screen.getByTestId("pending-en")).toHaveTextContent("1");
    expect(screen.getByTestId("pending-mixed")).toHaveTextContent("3");
  });

  it("filters by language without changing the per-language counters", async () => {
    renderApp("/");
    await screen.findAllByRole("row");
    await userEvent.click(screen.getByRole("button", { name: /Code-switched/ }));
    const body = screen.getAllByRole("rowgroup")[1]!;
    expect(within(body).getAllByRole("row")).toHaveLength(3);
    expect(screen.getByTestId("pending-ar")).toHaveTextContent("2");
  });

  it("switches the whole UI to Arabic and RTL", async () => {
    renderApp("/");
    await screen.findAllByRole("row");
    await userEvent.click(screen.getByRole("button", { name: "عربي" }));
    expect(document.documentElement.dir).toBe("rtl");
    expect(document.documentElement.lang).toBe("ar");
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("بانتظار المراجعة البشرية");
    expect(screen.getByText("3 من 9 ادعاءات بلا دليل حرفي")).toBeInTheDocument();
    expect(localStorage.getItem("sawti.ui-language")).toBe("ar");
  });

  it("links each row to its review", async () => {
    renderApp("/");
    const links = await screen.findAllByRole("link", { name: "Review" });
    expect(links[0]).toHaveAttribute("href", "/calls/7f3a91c2-5d4e-4b8a-9c1f-2e6d8a0b3c47");
  });
});
