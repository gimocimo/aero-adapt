// Traffic for the cinematic look (docs/07 §15): cars, black cabs and red buses along the
// roads (keeping left), trains on the railways, and boats on the river. Drawn only: the simulator has none of them
// (the measured surface of a survey world has whatever stood on the road that day).
import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import type { Ground, WorldPayload } from "./payload";

interface Path {
  pts: [number, number][];
  cum: number[]; // distance along to each point
  length: number;
  width: number;
  hs?: number[]; // the height at each point (a railway's), else the ground's
}

interface Mover {
  path: number;
  s: number; // distance along
  dir: 1 | -1;
  speed: number;
  offset: number; // to the left of travel (metres)
  kind: number; // index into the meshes
  slot: number; // instance index in its mesh
}

function path(points: [number, number][], width: number, hs?: number[]): Path {
  const cum = [0];
  for (let k = 1; k < points.length; k++) {
    const [n1, e1] = points[k - 1]!;
    const [n2, e2] = points[k]!;
    cum.push(cum[k - 1]! + Math.hypot(n2 - n1, e2 - e1));
  }
  return { pts: points, cum, length: cum[cum.length - 1]!, width, hs };
}

/** Point, unit direction (n, e) and height (NaN: none given) at a distance along a path. */
function at(p: Path, s: number): [number, number, number, number, number] {
  let lo = 0;
  let hi = p.cum.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (p.cum[mid]! <= s) lo = mid;
    else hi = mid;
  }
  const [n1, e1] = p.pts[lo]!;
  const [n2, e2] = p.pts[hi]!;
  const seg = Math.max(p.cum[hi]! - p.cum[lo]!, 1e-6);
  const t = Math.min(Math.max((s - p.cum[lo]!) / seg, 0), 1);
  const h = p.hs ? p.hs[lo]! + (p.hs[hi]! - p.hs[lo]!) * t : NaN;
  return [n1 + (n2 - n1) * t, e1 + (e2 - e1) * t, (n2 - n1) / seg, (e2 - e1) / seg, h];
}

/** A box (x along, y up, z across), vertex-coloured. */
function box(l: number, h: number, w: number, x: number, y: number, z: number, grey: number): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(l, h, w).toNonIndexed();
  g.translate(x, y + h / 2, z);
  const c = new Float32Array(g.getAttribute("position").count * 3).fill(grey);
  g.setAttribute("color", new THREE.BufferAttribute(c, 3));
  return g;
}

function merged(parts: THREE.BufferGeometry[]): THREE.BufferGeometry {
  return mergeGeometries(parts)!;
}

// Body white (tinted per vehicle), glass dark, tyres black.
const CAR = () =>
  merged([
    box(4.4, 0.75, 1.8, 0, 0.3, 0, 1),
    box(2.3, 0.6, 1.65, -0.2, 1.05, 0, 0.12),
    box(2.0, 0.1, 1.55, -0.2, 1.62, 0, 1),
    box(0.7, 0.55, 1.9, 1.3, 0.0, 0, 0.05),
    box(0.7, 0.55, 1.9, -1.4, 0.0, 0, 0.05),
  ]);
const CAB = () =>
  merged([
    box(4.5, 0.8, 1.8, 0, 0.3, 0, 1),
    box(2.4, 0.75, 1.7, -0.3, 1.1, 0, 0.1),
    box(2.3, 0.12, 1.6, -0.3, 1.85, 0, 1),
    box(0.7, 0.55, 1.9, 1.35, 0.0, 0, 0.05),
    box(0.7, 0.55, 1.9, -1.45, 0.0, 0, 0.05),
  ]);
const BUS = () =>
  merged([
    box(10.8, 1.0, 2.5, 0, 0.3, 0, 1),
    box(10.6, 1.05, 2.45, 0, 1.3, 0, 0.1), // lower deck windows
    box(10.8, 0.45, 2.5, 0, 2.35, 0, 1),
    box(10.6, 1.0, 2.45, 0, 2.8, 0, 0.1), // upper deck windows
    box(10.8, 0.45, 2.5, 0, 3.8, 0, 1),
    box(1.0, 0.6, 2.6, 3.6, 0.0, 0, 0.05),
    box(1.0, 0.6, 2.6, -3.4, 0.0, 0, 0.05),
  ]);
const BOAT = () =>
  merged([
    box(32, 2.2, 8, 0, 0, 0, 1),
    box(20, 2.4, 6.5, -2, 2.2, 0, 0.18), // the saloon's windows
    box(19, 0.4, 6.3, -2, 4.6, 0, 0.92),
    box(3, 1.6, 3, 8, 4.6, 0, 0.85), // wheelhouse
  ]);
const BARGE = () => merged([box(55, 2.0, 10, 0, 0, 0, 1), box(6, 3, 8, -23, 2.0, 0, 0.8)]);
// A railway carriage: body tinted per train, a dark band of windows, a grey roof.
const CARRIAGE = () =>
  merged([
    box(19.6, 1.3, 2.8, 0, 0.5, 0, 1),
    box(19.4, 1.0, 2.82, 0, 1.8, 0, 0.12),
    box(19.6, 0.5, 2.8, 0, 2.8, 0, 1),
    box(19.0, 0.35, 2.5, 0, 3.3, 0, 0.55),
  ]);
const TRAIN_COLOURS = ["#e8eaec", "#d9dde0", "#f0f0ec", "#cfd5da"];

const CAR_COLOURS = ["#e8e8e6", "#1c1d20", "#8b8f94", "#bfc3c6", "#2d3e5c", "#6b1f22", "#3c3f44", "#d9d4c6", "#4d5a3c", "#a0a4a8"];

/** The water's level at a point, or null off the water. */
export type LevelAt = (n: number, e: number) => number | null;

export class Traffic {
  readonly group = new THREE.Group();
  private paths: Path[] = [];
  private movers: Mover[] = [];
  private meshes: THREE.InstancedMesh[] = [];
  private last: number | null = null;
  private readonly m = new THREE.Matrix4();
  private readonly q = new THREE.Quaternion();
  private readonly up = new THREE.Vector3(0, 1, 0);

  constructor(
    world: WorldPayload,
    private readonly ground: Ground,
    private readonly waterLevel: LevelAt | null,
  ) {
    let seed = 1234567;
    const rnd = () => {
      seed = (seed * 16807) % 2147483647;
      return seed / 2147483647;
    };
    const kinds = [CAR(), CAB(), BUS(), BOAT(), BARGE(), CARRIAGE()];
    const counts = [0, 0, 0, 0, 0, 0];
    const trainColour: number[] = []; // per carriage mover: the train's colour
    const plan: Omit<Mover, "slot">[] = [];
    // Roads: a vehicle every 35 m of main road and 60 m of street, both ways.
    for (const r of world.roads) {
      if (r.kind !== "road" && r.kind !== "street") continue;
      const p = path(r.points, r.width);
      if (p.length < 60) continue;
      const k = this.paths.push(p) - 1;
      const spacing = r.kind === "road" ? 35 : 60;
      const n = Math.floor(p.length / spacing);
      for (let i = 0; i < n; i++) {
        const dir = rnd() < 0.5 ? 1 : -1;
        const x = rnd();
        const kind = r.kind === "road" && x < 0.1 ? 2 : x < 0.3 ? 1 : 0;
        plan.push({
          path: k,
          s: rnd() * p.length,
          dir,
          speed: (r.kind === "road" ? 7 : 4.5) + rnd() * 4,
          offset: Math.max(Math.min(p.width / 4, 3.2), 1.4),
          kind,
        });
      }
    }
    // Trains: six carriages each, a train every 900 m of line, on lines of 400 m or more.
    for (const r of world.rails ?? []) {
      const p = path(r.points.map(([n, e]) => [n, e] as [number, number]), 3, r.points.map(([, , h]) => h));
      if (p.length < 400) continue;
      const k = this.paths.push(p) - 1;
      const trains = Math.max(1, Math.floor(p.length / 900));
      for (let t = 0; t < trains; t++) {
        const dir = rnd() < 0.5 ? 1 : -1;
        const s0 = rnd() * p.length;
        const speed = 7 + rnd() * 8;
        const colour = Math.floor(rnd() * TRAIN_COLOURS.length);
        for (let c = 0; c < 6; c++) {
          plan.push({ path: k, s: s0 - dir * c * 20.2, dir, speed, offset: 0, kind: 5 });
          trainColour.push(colour);
        }
      }
    }
    // Boats on the river, both ways, off its middle.
    const river = world.river;
    if (river && waterLevel) {
      const p = path(river.points, river.width);
      const k = this.paths.push(p) - 1;
      const n = Math.max(2, Math.round(p.length / 500));
      for (let i = 0; i < n; i++) {
        plan.push({
          path: k,
          s: rnd() * p.length,
          dir: i % 2 ? 1 : -1,
          speed: 3 + rnd() * 6,
          offset: river.width * (0.12 + 0.15 * rnd()),
          kind: i % 3 === 2 ? 4 : 3,
        });
      }
    }
    for (const mv of plan) counts[mv.kind]! += 1;
    const material = new THREE.MeshLambertMaterial({ vertexColors: true });
    this.meshes = kinds.map((g, k) => {
      const mesh = new THREE.InstancedMesh(g, material, Math.max(counts[k]!, 1));
      mesh.count = counts[k]!;
      mesh.castShadow = mesh.receiveShadow = true;
      mesh.frustumCulled = false;
      this.group.add(mesh);
      return mesh;
    });
    const used = [0, 0, 0, 0, 0, 0];
    const c = new THREE.Color();
    let carriage = 0;
    for (const mv of plan) {
      const slot = used[mv.kind]!++;
      this.movers.push({ ...mv, slot });
      const colour =
        mv.kind === 5
          ? TRAIN_COLOURS[trainColour[carriage++]!]!
          : mv.kind === 1 ? "#141414" : mv.kind === 2 ? "#c8102e" : mv.kind === 3 ? (rnd() < 0.5 ? "#f2f2f0" : "#22344f") : mv.kind === 4 ? "#3d3a36" : CAR_COLOURS[Math.floor(rnd() * CAR_COLOURS.length)]!;
      this.meshes[mv.kind]!.setColorAt(slot, c.set(colour));
    }
    this.update(0);
  }

  /** Move everything on to ``now`` (seconds; a jump of more than half a second is taken as
   * half a second). */
  update(now: number): void {
    const dt = this.last === null ? 0 : Math.min(Math.max(now - this.last, 0), 0.5);
    this.last = now;
    for (const mv of this.movers) {
      const p = this.paths[mv.path]!;
      mv.s = (((mv.s + mv.dir * mv.speed * dt) % p.length) + p.length) % p.length;
      const [n, e, dn, de, rh] = at(p, mv.s);
      // Keep left of the direction of travel.
      const [fn, fe] = [dn * mv.dir, de * mv.dir];
      const [ln, le] = [fe, -fn];
      const pn = n + ln * mv.offset;
      const pe = e + le * mv.offset;
      const y =
        mv.kind === 5
          ? rh
          : mv.kind >= 3
            ? (this.waterLevel?.(pn, pe) ?? this.ground.at(pn, pe)) - 0.6
            : this.ground.at(pn, pe);
      this.q.setFromAxisAngle(this.up, Math.atan2(fn, fe)); // local x along travel: (n, e) → (x = e, z = -n)
      this.m.compose(new THREE.Vector3(pe, y, -pn), this.q, new THREE.Vector3(1, 1, 1));
      this.meshes[mv.kind]!.setMatrixAt(mv.slot, this.m);
    }
    for (const mesh of this.meshes) mesh.instanceMatrix.needsUpdate = true;
  }

  dispose(): void {
    for (const mesh of this.meshes) mesh.geometry.dispose();
    (this.meshes[0]?.material as THREE.Material | undefined)?.dispose();
  }
}
