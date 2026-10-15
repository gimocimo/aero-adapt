import { describe, expect, it } from "vitest";
import { FAULT_GROUPS, faultLabel } from "./FaultPanel";

describe("fault panel catalogue", () => {
  const kinds = FAULT_GROUPS.flatMap((g) => g.kinds);

  it("has one row per faultlab kind: 26 v1 kinds and the four M1 faults", () => {
    expect(kinds).toHaveLength(30);
    expect(new Set(kinds.map((k) => k.label)).size).toBe(30);
  });

  it("names an injected spec after its row", () => {
    for (const k of kinds) {
      const spec = { ...k.spec, magnitude: k.magnitude?.value ?? 0, trigger: { at_time_s: 1, until_s: null } };
      expect(faultLabel(spec)).toBe(k.label);
    }
    expect(faultLabel({ type: "servo_jam", channel: "rudder" })).toBe("Servo jam (rudder)");
  });
});
