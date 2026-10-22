import { describe, expect, it } from "vitest";
import type { PlanProposal } from "./client";
import { canConfirm, EXAMPLES, parsePlan } from "./PlanPanel";

const proposal = (ok: boolean, warnings: string[]): PlanProposal => ({
  id: "plan-1",
  ok,
  warnings,
  summary: "",
  assumptions: [],
  notes: [],
  steps: [],
  checks: [],
  preview: [],
  energy: null,
  plan: {},
});

describe("Txt2Fly plan panel", () => {
  it("parses plans and explains what is wrong with the rest", () => {
    expect(parsePlan('{"steps": [{"do": "return_home"}]}')).toEqual({ plan: { steps: [{ do: "return_home" }] } });
    expect(parsePlan('{"do": "return_home"}')).toEqual({ error: "a plan is an object with a list of steps" });
    expect("error" in parsePlan("{steps: [")).toBe(true);
  });

  it("confirms only plans whose hard checks pass, once every warning is acknowledged", () => {
    expect(canConfirm(null, {})).toBe(false);
    expect(canConfirm(proposal(false, []), {})).toBe(false);
    expect(canConfirm(proposal(true, []), {})).toBe(true);
    expect(canConfirm(proposal(true, ["K7"]), {})).toBe(false);
    expect(canConfirm(proposal(true, ["K7"]), { K7: true })).toBe(true);
  });

  it("offers example plans that parse", () => {
    for (const ex of Object.values(EXAMPLES).flat()) expect("plan" in parsePlan(JSON.stringify(ex.plan))).toBe(true);
  });
});
