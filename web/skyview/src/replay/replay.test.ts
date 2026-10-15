import { tableToIPC, tableFromArrays } from "apache-arrow";
import { describe, expect, it } from "vitest";
import { eulerToQuat, type Quat } from "../frames";
import { lerpAngle, locate, slerp } from "./interp";
import { columnsFromArrow, parseEvents } from "./load";

describe("locate", () => {
  const t = Float32Array.from([0, 0.02, 0.04, 0.06]);
  it("finds the bracketing interval and fraction", () => {
    const { i, f } = locate(t, 0.03);
    expect(i).toBe(1);
    expect(f).toBeCloseTo(0.5, 5);
  });
  it("clamps before the start and after the end", () => {
    expect(locate(t, -1)).toEqual({ i: 0, f: 0 });
    expect(locate(t, 9)).toEqual({ i: 3, f: 0 });
  });
});

describe("slerp", () => {
  const a = eulerToQuat(0, 0, 0);
  const b = eulerToQuat(0, 0, Math.PI / 2);
  it("hits the end points and stays unit length", () => {
    const close = (p: Quat, q: Quat) => p.forEach((x, k) => expect(x).toBeCloseTo(q[k]!, 9));
    close(slerp(a, b, 0), a);
    close(slerp(a, b, 1), b);
    expect(Math.hypot(...slerp(a, b, 0.3))).toBeCloseTo(1, 12);
  });
  it("interpolates the rotation angle linearly", () => {
    const mid = slerp(a, b, 0.5);
    const yaw = 2 * Math.atan2(mid[3], mid[0]);
    expect(yaw).toBeCloseTo(Math.PI / 4, 9);
  });
  it("takes the shorter path when the quaternions have opposite signs", () => {
    const negB: Quat = [-b[0], -b[1], -b[2], -b[3]];
    const mid = slerp(a, negB, 0.5);
    expect(Math.abs(2 * Math.atan2(mid[3], mid[0]))).toBeCloseTo(Math.PI / 4, 9);
  });
});

describe("lerpAngle", () => {
  it("wraps across ±π", () => {
    expect(lerpAngle(3.0, -3.0, 0.5)).toBeCloseTo(Math.PI, 2);
  });
});

describe("Arrow loader", () => {
  it("reads float32 and int32 columns from IPC bytes", () => {
    const table = tableFromArrays({
      t: Float32Array.from([0, 0.02, 0.04]),
      "truth.tas": Float32Array.from([25, 25.5, 26]),
      mode: Int32Array.from([0, 0, 1]),
    });
    const { t, cols } = columnsFromArrow(tableToIPC(table, "file"));
    expect(Array.from(t)).toEqual([0, Math.fround(0.02), Math.fround(0.04)]);
    expect(cols["truth.tas"]).toBeInstanceOf(Float32Array);
    expect(Array.from(cols.mode!)).toEqual([0, 0, 1]);
  });
  it("parses events.jsonl", () => {
    const ev = parseEvents('{"time_us": 1, "type": "start", "payload": {}}\n\n');
    expect(ev).toEqual([{ time_us: 1, type: "start", payload: {} }]);
  });
});
