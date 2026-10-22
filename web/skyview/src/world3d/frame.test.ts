import { describe, expect, it } from "vitest";
import { eulerDegToQuat, matVec, type Vec3 } from "../frames";
import { flatWorld, Ground } from "./payload";
import { rWorldFromModel, toWorld } from "./frame";

const close = (a: Vec3, b: Vec3) => a.forEach((x, i) => expect(x).toBeCloseTo(b[i]!, 9));

describe("model attitude in the three.js world", () => {
  it("level, heading north: nose to −z (north), up to +y, left wing to −x (west)", () => {
    const r = rWorldFromModel(eulerDegToQuat(0, 0, 0));
    close(matVec(r, [0, 0, 1]), [0, 0, -1]);
    close(matVec(r, [0, 1, 0]), [0, 1, 0]);
    close(matVec(r, [1, 0, 0]), [-1, 0, 0]);
  });

  it("heading east: nose to +x", () => {
    close(matVec(rWorldFromModel(eulerDegToQuat(0, 0, 90)), [0, 0, 1]), [1, 0, 0]);
  });

  it("nose up 30° climbs; right wing down in a right bank", () => {
    const up = matVec(rWorldFromModel(eulerDegToQuat(0, 30, 0)), [0, 0, 1]);
    expect(up[1]).toBeCloseTo(0.5, 9);
    // Right wing is −X in the model frame; banked right (roll +30°) it points down.
    const wing = matVec(rWorldFromModel(eulerDegToQuat(30, 0, 0)), [-1, 0, 0]);
    expect(wing[1]).toBeCloseTo(-0.5, 9);
  });

  it("maps NE and height to (east, up, south)", () => {
    expect(toWorld(100, 20, 5)).toEqual([20, 5, -100]);
  });
});

describe("ground heights", () => {
  it("interpolates bilinearly and clamps at the edges", () => {
    const w = flatWorld("t", "t", [0, 0, 10, 10]);
    w.dtm = { n0: 0, e0: 0, cell: 10, rows: 2, cols: 2, h: Float32Array.from([0, 10, 20, 30]) };
    const g = new Ground(w.dtm);
    expect(g.at(0, 0)).toBeCloseTo(0, 6);
    expect(g.at(5, 5)).toBeCloseTo(15, 3);
    expect(g.at(10, 0)).toBeCloseTo(20, 2);
    expect(g.at(-50, -50)).toBeCloseTo(0, 6);
  });
});
