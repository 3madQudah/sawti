import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { CALL_ID } from "../../test/fixtures";
import { renderApp } from "../../test/render";

const path = `/calls/${CALL_ID}`;

async function loaded() {
  return screen.findByRole("region", { name: "Transcript" });
}

function card(claimId: string) {
  return within(screen.getByTestId(`claim-${claimId}`));
}

describe("Review", () => {
  it("shows the summary from the API: coverage against the call's own threshold, claims kept, PII count", async () => {
    renderApp(path);
    await loaded();
    expect(screen.getByText("6 of 9 extracted claims survived grounding")).toBeInTheDocument();
    expect(screen.getByText("2 PII spans redacted before analysis")).toBeInTheDocument();
    expect(screen.getByText("/ threshold \u20660.70\u2069")).toBeInTheDocument();
    expect(screen.getByText("Rejected by grounding")).toBeInTheDocument();
  });

  it("gives every transcript turn its own direction, whatever the UI language", async () => {
    renderApp(path, { lang: "ar" });
    await screen.findByRole("region", { name: "النص" });
    const text = (n: number) => screen.getByTestId(`turn-${n}`).querySelector("p")!;
    expect(document.documentElement.dir).toBe("rtl");
    expect(text(1)).toHaveAttribute("dir", "rtl");
    expect(text(2)).toHaveAttribute("dir", "auto");
    expect(text(8)).toHaveAttribute("dir", "ltr");
  });

  it("highlights the selected claim's evidence in the transcript", async () => {
    renderApp(path);
    await loaded();
    await userEvent.click(card("compliance_flags[0]").getByRole("button", { name: "Show this evidence in the transcript" }));
    const selected = document.querySelector('mark[data-selected="true"]');
    expect(selected).toHaveTextContent("ممكن رقم الخط لو سمحت؟");
    expect(screen.getByTestId("turn-3")).toContainElement(selected as HTMLElement);
  });

  it("keeps submit disabled until every claim has a verdict", async () => {
    renderApp(path);
    await loaded();
    const submit = screen.getByRole("button", { name: "Submit review and resume run" });
    expect(submit).toBeDisabled();
    for (const id of ["commitments[0]", "commitments[1]", "compliance_flags[0]", "rubric_scores[0]", "rubric_scores[1]"]) {
      await userEvent.click(card(id).getByRole("button", { name: "Confirm" }));
    }
    expect(submit).toBeDisabled();
    expect(screen.getByText("5 of 6 decided")).toBeInTheDocument();
    await userEvent.click(card("rubric_scores[2]").getByRole("button", { name: "Reject" }));
    expect(submit).toBeEnabled();
  });

  it("requires corrected evidence to be verbatim and shows the turn it matched", async () => {
    renderApp(path);
    await loaded();
    const refund = card("commitments[0]");
    await userEvent.click(refund.getByRole("button", { name: "Correct" }));
    const evidence = refund.getByLabelText("Corrected evidence (optional, must be verbatim)");
    await userEvent.type(evidence, "خلال 3 أيام");
    expect(refund.getByRole("status")).toHaveTextContent("Not found verbatim in the transcript");
    await userEvent.clear(evidence);
    await userEvent.type(evidence, "خلال ٣ أيام عمل");
    expect(refund.getByRole("status")).toHaveTextContent("✓ Found verbatim in turn 7");
  });

  it("submits the structured correction, with verified evidence in the note, and waits for the resume", async () => {
    const { api } = renderApp(path);
    await loaded();
    const refund = card("commitments[0]");
    await userEvent.click(refund.getByRole("button", { name: "Correct" }));
    await userEvent.type(refund.getByLabelText("Corrected value"), "Refund credited within 3 business days");
    await userEvent.type(refund.getByLabelText("Corrected evidence (optional, must be verbatim)"), "رسالة");
    const submit = screen.getByRole("button", { name: "Submit review and resume run" });
    for (const id of ["commitments[1]", "compliance_flags[0]", "rubric_scores[0]", "rubric_scores[1]", "rubric_scores[2]"]) {
      await userEvent.click(card(id).getByRole("button", { name: "Confirm" }));
    }
    expect(submit).toBeEnabled();
    await userEvent.click(submit);

    expect(await screen.findByText("Review submitted. Resuming the paused run…", { exact: false })).toBeInTheDocument();
    const [submission] = api.submissions;
    expect(submission!.body.verdicts[0]).toEqual({
      claim_id: "commitments[0]",
      verdict: "correct",
      changes: { description: "Refund credited within 3 business days" },
      note: expect.stringMatching(/^Corrected evidence \(verbatim, turn 9, chars \d+-\d+\): «رسالة»$/),
    });
    expect(submission!.body.verdicts.slice(1).every((v) => v.verdict === "confirm" && !("changes" in v))).toBe(true);
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("finished as human-reviewed"), { timeout: 5000 });
  });

  it("blocks a correction whose value is unchanged or whose evidence is not verbatim", async () => {
    renderApp(path);
    await loaded();
    for (const id of ["commitments[1]", "compliance_flags[0]", "rubric_scores[0]", "rubric_scores[1]", "rubric_scores[2]"]) {
      await userEvent.click(card(id).getByRole("button", { name: "Confirm" }));
    }
    const refund = card("commitments[0]");
    await userEvent.click(refund.getByRole("button", { name: "Correct" }));
    const submit = screen.getByRole("button", { name: "Submit review and resume run" });
    await userEvent.type(refund.getByLabelText("Corrected value"), "Refund credited within 3 days");
    expect(refund.getByText("The corrected value must differ from the original")).toBeInTheDocument();
    expect(submit).toBeDisabled();
    await userEvent.type(refund.getByLabelText("Corrected value"), " (business)");
    expect(submit).toBeEnabled();
    await userEvent.type(refund.getByLabelText("Corrected evidence (optional, must be verbatim)"), "not in the call");
    expect(submit).toBeDisabled();
  });
});
