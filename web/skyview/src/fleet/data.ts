// Fleet-view data (`aeroadapt export-fleet`): every bench flight's 1 Hz trajectory.
import type { Origin } from "../geo";

export interface FleetFlight {
  scenario: string;
  baseline: string;
  airframe_class: string;
  faults: string;
  survived: boolean;
  touchdown: string | null;
  in_zone: boolean;
  planned_landing: boolean;
}

export interface FleetMeta {
  id: string;
  description: string;
  site: { origin: Origin };
  dt_s: number;
  samples: number;
  fields: string[]; // n, e, alt, roll, pitch, yaw, state
  flights: FleetFlight[];
}

export interface Fleet {
  meta: FleetMeta;
  data: Float32Array; // flights × samples × fields
}

export const STATE_NOMINAL = 0;
export const STATE_CONTINGENCY = 1;
export const STATE_ENDED = 2;

export async function loadFleet(id: string): Promise<Fleet> {
  const base = `/fleet/${encodeURIComponent(id)}`;
  const [metaRes, binRes] = await Promise.all([fetch(`${base}/fleet.json`), fetch(`${base}/fleet.bin`)]);
  if (!metaRes.ok || !binRes.ok) throw new Error(`no fleet "${id}" (run: uv run aeroadapt export-fleet runs/bench/${id})`);
  const meta = (await metaRes.json()) as FleetMeta;
  const data = new Float32Array(await binRes.arrayBuffer());
  if (data.length !== meta.flights.length * meta.samples * meta.fields.length) {
    throw new Error("fleet.bin does not match fleet.json");
  }
  return { meta, data };
}

export interface FlightSample {
  n: number;
  e: number;
  alt: number;
  state: number;
}

/** Linear interpolation of flight `i` at time `t` (s); the state is the earlier sample's. */
export function sampleFlight(fleet: Fleet, i: number, t: number, out: FlightSample): FlightSample {
  const { samples, fields, dt_s } = fleet.meta;
  const f = fields.length;
  const x = Math.min(Math.max(t / dt_s, 0), samples - 1);
  const k = Math.min(Math.floor(x), samples - 2);
  const u = x - k;
  const a = (i * samples + k) * f;
  const b = a + f;
  const d = fleet.data;
  out.n = d[a]! + u * (d[b]! - d[a]!);
  out.e = d[a + 1]! + u * (d[b + 1]! - d[a + 1]!);
  out.alt = d[a + 2]! + u * (d[b + 2]! - d[a + 2]!);
  out.state = d[(i * samples + Math.floor(x)) * f + 6]!; // the sample at or before t
  return out;
}
