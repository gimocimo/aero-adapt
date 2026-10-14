import { describe, expect, it } from "vitest";
import goldens from "./__goldens__/frames.json";
import { eulerDegToQuat, matVec, rEnuFromFlu, type Mat3, type Vec3 } from "./frames";

const close = (a: readonly number[], b: readonly number[], tol = 1e-12) =>
  a.forEach((x, i) => expect(Math.abs(x - (b[i] as number))).toBeLessThanOrEqual(tol));

describe("frames match the Python source of truth (goldens)", () => {
  for (const c of goldens.cases) {
    const [roll, pitch, heading] = c.euler_deg as [number, number, number];
    it(`roll ${roll.toFixed(1)} pitch ${pitch.toFixed(1)} heading ${heading.toFixed(1)}`, () => {
      const q = eulerDegToQuat(roll, pitch, heading);
      close(q, c.quat);
      const r = rEnuFromFlu(q);
      (c.r_enu_flu as number[][]).forEach((row, i) => close(r[i as 0 | 1 | 2], row));
    });
  }
});

describe("visual semantics in ENU (east, north, up)", () => {
  const fwd: Vec3 = [1, 0, 0]; // FLU
  const left: Vec3 = [0, 1, 0]; // FLU
  const enu = (m: Mat3, v: Vec3) => matVec(m, v);

  it("heading 0° points the nose north, 90° east", () => {
    close(enu(rEnuFromFlu(eulerDegToQuat(0, 0, 0)), fwd), [0, 1, 0]);
    close(enu(rEnuFromFlu(eulerDegToQuat(0, 0, 90)), fwd), [1, 0, 0]);
  });
  it("positive pitch raises the nose", () => {
    const v = enu(rEnuFromFlu(eulerDegToQuat(0, 30, 0)), fwd);
    expect(v[2]).toBeCloseTo(0.5, 12);
  });
  it("positive roll (right wing down) raises the left (port) wing", () => {
    const v = enu(rEnuFromFlu(eulerDegToQuat(45, 0, 0)), left);
    expect(v[2]).toBeCloseTo(Math.SQRT1_2, 12);
  });
});
