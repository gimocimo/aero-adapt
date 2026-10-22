import type { Quat } from "../frames";
import type { Column, Replay } from "./types";

/** Index `i` and fraction `f` such that time lies in [t[i], t[i+1]] (clamped to the ends). */
export function locate(t: Float32Array, time: number): { i: number; f: number } {
  const n = t.length;
  if (n === 0) return { i: 0, f: 0 };
  if (time <= t[0]!) return { i: 0, f: 0 };
  if (time >= t[n - 1]!) return { i: n - 1, f: 0 };
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (t[mid]! <= time) lo = mid;
    else hi = mid;
  }
  const span = t[hi]! - t[lo]!;
  return { i: lo, f: span > 0 ? (time - t[lo]!) / span : 0 };
}

export function lerpAt(col: Column, i: number, f: number): number {
  const a = col[i]!;
  const b = col[Math.min(i + 1, col.length - 1)]!;
  return a + (b - a) * f;
}

export function nearestAt(col: Column, i: number, f: number): number {
  return col[f < 0.5 ? i : Math.min(i + 1, col.length - 1)]!;
}

/** Angle interpolation along the shorter arc (radians). */
export function lerpAngle(a: number, b: number, f: number): number {
  let d = b - a;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return a + d * f;
}

/** Spherical linear interpolation of unit quaternions (shorter path, normalised output). */
export function slerp(a: Quat, b: Quat, f: number): Quat {
  let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  let bb: Quat = b;
  if (dot < 0) {
    dot = -dot;
    bb = [-b[0], -b[1], -b[2], -b[3]];
  }
  let wa: number;
  let wb: number;
  if (dot > 0.9995) {
    wa = 1 - f;
    wb = f;
  } else {
    const th = Math.acos(Math.min(dot, 1));
    const s = Math.sin(th);
    wa = Math.sin((1 - f) * th) / s;
    wb = Math.sin(f * th) / s;
  }
  const q: [number, number, number, number] = [
    wa * a[0] + wb * bb[0],
    wa * a[1] + wb * bb[1],
    wa * a[2] + wb * bb[2],
    wa * a[3] + wb * bb[3],
  ];
  const norm = Math.hypot(...q);
  return [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm];
}

/** Everything the scene and the HUD need at one instant. */
export interface Sample {
  time: number;
  index: number;
  n: number;
  e: number;
  /** Height above the site datum (the ground at home). */
  altAgl: number;
  /** Height above the ground below, and above whatever stands there (world sites). */
  hagl: number;
  clearance: number;
  course: number;
  quat: Quat;
  roll: number;
  pitch: number;
  yaw: number;
  climb: number;
  groundspeed: number;
  tasTrue: number;
  tasSensor: number;
  ias: number;
  airspeedUsed: number;
  airspeedHealthy: boolean;
  alpha: number;
  beta: number;
  windN: number;
  windE: number;
  throttle: number;
  rpm: number;
  batteryV: number;
  soc: number;
  rollDem: number;
  pitchDem: number;
  altTarget: number;
  speedTarget: number;
  mode: number;
  wpIdx: number;
  failed: boolean;
  phase: number;
  zone: number;
  reachMask: number;
  faultBits: number;
}

export function sample(r: Replay, time: number): Sample {
  const { i, f } = locate(r.t, time);
  const c = r.cols;
  const L = (name: string) => (c[name] ? lerpAt(c[name], i, f) : Number.NaN);
  const N = (name: string) => (c[name] ? nearestAt(c[name], i, f) : 0);
  const j = Math.min(i + 1, r.n - 1);
  const q = (k: number): Quat => [
    c["truth.qw"]![k]!,
    c["truth.qx"]![k]!,
    c["truth.qy"]![k]!,
    c["truth.qz"]![k]!,
  ];
  return {
    time: Math.min(Math.max(time, r.t[0] ?? 0), r.duration),
    index: i,
    n: L("truth.pos_n"),
    e: L("truth.pos_e"),
    altAgl: L("truth.alt_agl"),
    hagl: c["truth.hagl"] ? L("truth.hagl") : L("truth.alt_agl"),
    clearance: c["truth.clearance"] ? L("truth.clearance") : L("truth.alt_agl"),
    course: c["truth.vel_n"] ? Math.atan2(L("truth.vel_e"), L("truth.vel_n")) : Number.NaN,
    quat: slerp(q(i), q(j), f),
    roll: L("truth.roll"),
    pitch: L("truth.pitch"),
    yaw: lerpAngle(c["truth.yaw"]![i]!, c["truth.yaw"]![j]!, f),
    climb: L("truth.climb_rate"),
    groundspeed: L("truth.groundspeed"),
    tasTrue: L("truth.tas"),
    tasSensor: L("sensors.tas"),
    ias: L("est.ias"),
    airspeedUsed: L("est.airspeed"),
    airspeedHealthy: N("est.airspeed_healthy") > 0,
    alpha: L("truth.alpha"),
    beta: L("truth.beta"),
    windN: L("truth.wind_n"),
    windE: L("truth.wind_e"),
    throttle: L("actual.throttle"),
    rpm: L("truth.prop_rpm"),
    batteryV: L("truth.battery_v"),
    soc: L("truth.soc"),
    rollDem: L("demands.roll"),
    pitchDem: L("demands.pitch"),
    altTarget: L("demands.alt_target"),
    speedTarget: L("demands.speed_target"),
    mode: N("mode"),
    wpIdx: N("wp_idx"),
    failed: N("cont.failed") > 0,
    phase: N("cont.phase"),
    zone: N("cont.zone"),
    reachMask: N("cont.reach_mask"),
    faultBits: N("labels.fault_bits"),
  };
}
