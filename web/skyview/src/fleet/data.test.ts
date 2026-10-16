import { describe, expect, it } from "vitest";
import { type Fleet, sampleFlight } from "./data";

function fleet(): Fleet {
  // Two flights, three 1 Hz samples, seven fields.
  const rows = [
    [0, 0, 100, 0, 0, 0, 0], [10, 0, 100, 0, 0, 0, 0], [20, 0, 100, 0, 0, 0, 1],
    [0, 5, 50, 0, 0, 0, 0], [0, 15, 40, 0, 0, 0, 2], [0, 15, 40, 0, 0, 0, 2],
  ];
  return {
    meta: { id: "t", description: "", site: { origin: { lat_deg: 0, lon_deg: 0, h_ellipsoid_m: 0 } },
      dt_s: 1, samples: 3, fields: ["n", "e", "alt", "roll", "pitch", "yaw", "state"], flights: [] },
    data: Float32Array.from(rows.flat()),
  };
}

describe("fleet sampling", () => {
  it("interpolates position and holds the earlier state", () => {
    const s = sampleFlight(fleet(), 0, 1.5, { n: 0, e: 0, alt: 0, state: 0 });
    expect(s.n).toBeCloseTo(15);
    expect(s.state).toBe(0);
    const t = sampleFlight(fleet(), 1, 0.5, { n: 0, e: 0, alt: 0, state: 0 });
    expect(t.e).toBeCloseTo(10);
    expect(t.alt).toBeCloseTo(45);
  });

  it("clamps beyond the last sample", () => {
    const s = sampleFlight(fleet(), 0, 99, { n: 0, e: 0, alt: 0, state: 0 });
    expect(s.n).toBeCloseTo(20);
    expect(s.state).toBe(1);
  });
});
