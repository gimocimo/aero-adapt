import { describe, expect, it } from "vitest";
import type { Manifest } from "../replay/types";
import { LiveBuffer, type WireColumn } from "./buffer";

function wire(values: number[], dtype: "f4" | "i4" = "f4"): WireColumn {
  const arr = dtype === "i4" ? Int32Array.from(values) : Float32Array.from(values);
  // Offset the payload inside a larger buffer, as a msgpack decoder may hand it over.
  const padded = new Uint8Array(arr.byteLength + 3);
  padded.set(new Uint8Array(arr.buffer), 3);
  return { dtype, data: padded.subarray(3) };
}

describe("LiveBuffer", () => {
  it("grows across appends and exposes a consistent replay", () => {
    const b = new LiveBuffer("live-1", {} as Manifest);
    expect(b.snapshot().n).toBe(0);
    let t = 0;
    for (let chunk = 0; chunk < 2000; chunk++) {
      const ts = [t, t + 0.02, t + 0.04, t + 0.06, t + 0.08];
      b.append({ t: wire(ts), mode: wire([1, 1, 1, 1, 2], "i4") }, 5);
      t += 0.1;
    }
    const r = b.snapshot();
    expect(r.n).toBe(10_000);
    expect(r.t.length).toBe(10_000);
    expect(r.cols.mode).toBeInstanceOf(Int32Array);
    expect(r.cols.mode![4]).toBe(2);
    expect(r.duration).toBeCloseTo(199.98, 3);
    for (let i = 1; i < r.n; i++) expect(r.t[i]! > r.t[i - 1]!).toBe(true);
  });

  it("shares events with its snapshots", () => {
    const b = new LiveBuffer("live-2", {} as Manifest);
    b.append({ t: wire([0, 0.02]) }, 2);
    b.events.push({ time_us: 10_000, type: "fault_onset", payload: { fault: "motor_cut" } });
    expect(b.snapshot().events).toHaveLength(1);
  });
});
