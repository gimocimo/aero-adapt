// Trees for the cinematic look (docs/07 §15): near the camera each tree is a model of its
// kind (a broadleaf with a trunk, boughs and a lumpy crown; a layered conifer; a weeping
// willow; a shrub), shaded darker towards the inside of its crown, casting shadows; further
// off it is a single simple shape. The near set follows the camera.
import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import type { WorldPayload } from "./payload";

const NEAR_M = 340; // full models within this of the camera
const REFRESH_M = 30; // re-sort when the camera has moved this far
const NEAR_MAX = 16000;
const BROADLEAF = 0;
const CONIFER = 1;
const WILLOW = 2;
const SHRUB = 3;

function hash(k: number): number {
  const s = Math.sin(k * 12.9898 + 78.233) * 43758.5453;
  return s - Math.floor(s);
}

/** Colour a geometry: ``f(y, radial) → [r, g, b]`` (y and radial distance in unit space). */
function paint(g: THREE.BufferGeometry, f: (y: number, r: number, i: number) => [number, number, number]): THREE.BufferGeometry {
  const g2 = g.index ? g.toNonIndexed() : g;
  const p = g2.getAttribute("position");
  const c = new Float32Array(p.count * 3);
  for (let i = 0; i < p.count; i++) {
    const [r, gg, b] = f(p.getY(i), Math.hypot(p.getX(i), p.getZ(i)), i);
    c[3 * i] = r;
    c[3 * i + 1] = gg;
    c[3 * i + 2] = b;
  }
  g2.setAttribute("color", new THREE.BufferAttribute(c, 3));
  g2.deleteAttribute("uv");
  return g2;
}

/** Wobble a blob's vertices (by direction, so shared vertices stay together). */
function lumpy(g: THREE.BufferGeometry, amount: number, seed: number): THREE.BufferGeometry {
  const p = g.getAttribute("position");
  const v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    const n = v.clone().normalize();
    const w = 1 + amount * (hash(seed + Math.round(n.x * 7) * 13 + Math.round(n.y * 7) * 7 + Math.round(n.z * 7)) - 0.5);
    p.setXYZ(i, v.x * w, v.y * w, v.z * w);
  }
  g.computeVertexNormals();
  return g;
}

const BARK: [number, number, number] = [0.29, 0.22, 0.16];

/** Models in unit space: height 1, crown radius 1 (scaled per tree by height and crown). */
function species(): THREE.BufferGeometry[] {
  const trunk = (r0: number, r1: number, h: number) => {
    const t = new THREE.CylinderGeometry(r0, r1, h, 6);
    t.translate(0, h / 2, 0);
    return paint(t, () => BARK);
  };
  const leaf = (base: [number, number, number]) => (y: number, r: number, i: number): [number, number, number] => {
    const shade = 0.62 + 0.38 * Math.min(1, y) * (0.7 + 0.3 * Math.min(1, r)); // darker low and inside
    const j = 0.92 + 0.16 * hash(i * 0.37);
    return [base[0] * shade * j, base[1] * shade * j, base[2] * shade * j];
  };
  // A broadleaf: trunk, two boughs, a crown of lobes.
  const lobes: THREE.BufferGeometry[] = [];
  const spots: [number, number, number, number][] = [
    [0, 0.68, 0, 0.42], [0.42, 0.6, 0.1, 0.34], [-0.38, 0.62, -0.2, 0.36], [0.1, 0.62, 0.45, 0.33],
    [-0.15, 0.6, -0.48, 0.32], [0.2, 0.86, -0.15, 0.3], [-0.25, 0.84, 0.22, 0.3], [0.48, 0.72, -0.35, 0.26],
  ]; // prettier-ignore
  spots.forEach(([x, y, z, r], k) => {
    const s = lumpy(new THREE.IcosahedronGeometry(1, 1), 0.28, k * 31);
    s.scale(r * 1.15, r * 0.95, r * 1.15);
    s.translate(x * 0.85, y, z * 0.85);
    lobes.push(s);
  });
  const crown = paint(mergeGeometries(lobes.map((g) => g.toNonIndexed()))!, leaf([0.17, 0.31, 0.09]));
  const bough1 = trunk(0.025, 0.04, 0.35);
  bough1.rotateZ(0.7);
  bough1.translate(0, 0.38, 0);
  const bough2 = trunk(0.025, 0.04, 0.32);
  bough2.rotateX(-0.75);
  bough2.translate(0, 0.42, 0);
  const broadleaf = mergeGeometries([trunk(0.045, 0.07, 0.55), bough1, bough2, crown])!;
  // A conifer: trunk and five tiers, narrowing upwards.
  const tiers: THREE.BufferGeometry[] = [trunk(0.04, 0.06, 0.35)];
  for (let k = 0; k < 5; k++) {
    const r = 1.0 - 0.17 * k;
    const c = new THREE.ConeGeometry(r, 0.34, 9, 1, true);
    c.translate(0, 0.27 + 0.15 * k + 0.17, 0);
    tiers.push(paint(lumpy(c, 0.18, 77 + k), leaf([0.08, 0.18, 0.09])));
  }
  const conifer = mergeGeometries(tiers.map((g) => (g.index ? g.toNonIndexed() : g)))!;
  // A willow: a short trunk and a wide, drooping crown.
  const wl = lumpy(new THREE.SphereGeometry(1, 12, 8, 0, Math.PI * 2, 0, Math.PI * 0.62), 0.22, 5);
  wl.scale(1.05, 0.72, 1.05);
  wl.translate(0, 0.42, 0);
  const willow = mergeGeometries([trunk(0.05, 0.08, 0.5), paint(wl.toNonIndexed(), leaf([0.25, 0.37, 0.14]))])!;
  // A shrub: a few lobes on the ground.
  const sl: THREE.BufferGeometry[] = [];
  [[0, 0.45, 0, 0.55], [0.4, 0.35, 0.15, 0.42], [-0.35, 0.36, -0.2, 0.44]].forEach(([x, y, z, r], k) => {
    const s = lumpy(new THREE.IcosahedronGeometry(1, 1), 0.3, 200 + k);
    s.scale(r!, r! * 0.9, r!);
    s.translate(x!, y!, z!);
    sl.push(s.toNonIndexed());
  });
  const shrub = paint(mergeGeometries(sl)!, leaf([0.18, 0.30, 0.11]));
  for (const g of [broadleaf, conifer, willow, shrub]) g.computeVertexNormals();
  return [broadleaf, conifer, willow, shrub];
}

/** The simple shapes far off: a lumpy ball, a cone, a squashed ball. */
function farShapes(): THREE.BufferGeometry[] {
  const ball = lumpy(new THREE.IcosahedronGeometry(1, 1), 0.2, 3);
  const cone = new THREE.ConeGeometry(1, 1, 7);
  cone.translate(0, 0.5, 0);
  return [ball, cone, ball.clone()];
}

export class TreeField {
  readonly group = new THREE.Group();
  private t: Float32Array;
  private count: number;
  private far: THREE.InstancedMesh[] = [];
  private slot: Int32Array; // index in its far mesh
  private model: Uint8Array; // which model, near
  private near: THREE.InstancedMesh[] = [];
  private shown: number[] = []; // trees drawn near now
  private at = new THREE.Vector3(Infinity, 0, 0);
  private grid = new Map<string, number[]>();

  constructor(w: WorldPayload) {
    this.t = w.trees;
    this.count = this.t.length / 6;
    const t = this.t;
    this.slot = new Int32Array(this.count);
    this.model = new Uint8Array(this.count);
    const shapes = farShapes();
    const byKind: number[][] = [[], [], []];
    for (let k = 0; k < this.count; k++) {
      const kind = Math.min(2, Math.max(0, Math.round(t[6 * k + 5]!)));
      this.slot[k] = byKind[kind]!.length;
      byKind[kind]!.push(k);
      const h = t[6 * k + 3]!;
      this.model[k] = kind === 1 ? CONIFER : kind === 2 ? WILLOW : h < 7 ? SHRUB : BROADLEAF;
      const key = `${Math.floor(t[6 * k]! / 100)},${Math.floor(t[6 * k + 1]! / 100)}`;
      const cell = this.grid.get(key);
      if (cell) cell.push(k);
      else this.grid.set(key, [k]);
    }
    const m = new THREE.Matrix4();
    const c = new THREE.Color();
    const tones = [[0.28, 0.48, 0.18], [0.36, 0.4, 0.13], [0.24, 0.42, 0.26]] as const; // foliage is darker than it looks
    byKind.forEach((ks, kind) => {
      const mesh = new THREE.InstancedMesh(shapes[kind]!, new THREE.MeshLambertMaterial({ flatShading: true }), Math.max(ks.length, 1));
      mesh.count = ks.length;
      ks.forEach((k, i) => {
        mesh.setMatrixAt(i, this.farMatrix(k, m));
        const j = hash(k);
        const [hue, sat, light] = tones[kind]!;
        mesh.setColorAt(i, c.setHSL(hue + 0.04 * (j - 0.5), sat, light - 0.05 + 0.1 * j));
      });
      mesh.receiveShadow = true;
      mesh.frustumCulled = false;
      this.far.push(mesh);
      this.group.add(mesh);
    });
    for (const g of species()) {
      const mesh = new THREE.InstancedMesh(g, new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true }), NEAR_MAX);
      mesh.count = 0;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.frustumCulled = false;
      this.near.push(mesh);
      this.group.add(mesh);
    }
  }

  private farMatrix(k: number, m: THREE.Matrix4): THREE.Matrix4 {
    const t = this.t;
    const [n, e, base, h, r, kind] = [t[6 * k]!, t[6 * k + 1]!, t[6 * k + 2]!, t[6 * k + 3]!, t[6 * k + 4]!, t[6 * k + 5]!];
    const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), hash(k) * 6.283);
    if (kind === 1) return m.compose(new THREE.Vector3(e, base + 0.15 * h, -n), q, new THREE.Vector3(r, 0.85 * h, r));
    if (kind === 2) return m.compose(new THREE.Vector3(e, base + 0.55 * h, -n), q, new THREE.Vector3(r, 0.42 * h, r));
    return m.compose(new THREE.Vector3(e, base + 0.62 * h, -n), q, new THREE.Vector3(r, 0.38 * h, r));
  }

  /** Bring the trees round ``eye`` (the chase camera) up to full models. */
  update(eye: THREE.Vector3): void {
    if (this.at.distanceTo(eye) < REFRESH_M) return;
    this.at.copy(eye);
    const t = this.t;
    const zero = new THREE.Matrix4().makeScale(0, 0, 0);
    const m = new THREE.Matrix4();
    // Give the far shapes back to the trees that leave the near set …
    const kindOf = (k: number) => Math.min(2, Math.max(0, Math.round(t[6 * k + 5]!)));
    for (const k of this.shown) this.far[kindOf(k)]!.setMatrixAt(this.slot[k]!, this.farMatrix(k, m));
    // … and gather the new near set.
    const n0 = -eye.z;
    const e0 = eye.x;
    const r2 = NEAR_M * NEAR_M;
    const near: number[] = [];
    const span = Math.ceil(NEAR_M / 100);
    const ci = Math.floor(n0 / 100);
    const cj = Math.floor(e0 / 100);
    for (let i = ci - span; i <= ci + span; i++) {
      for (let j = cj - span; j <= cj + span; j++) {
        for (const k of this.grid.get(`${i},${j}`) ?? []) {
          const dn = t[6 * k]! - n0;
          const de = t[6 * k + 1]! - e0;
          if (dn * dn + de * de < r2 && near.length < NEAR_MAX) near.push(k);
        }
      }
    }
    const counts = [0, 0, 0, 0];
    const q = new THREE.Quaternion();
    for (const k of near) {
      this.far[kindOf(k)]!.setMatrixAt(this.slot[k]!, zero);
      const [n, e, base, h, r] = [t[6 * k]!, t[6 * k + 1]!, t[6 * k + 2]!, t[6 * k + 3]!, t[6 * k + 4]!];
      const model = this.model[k]!;
      const mesh = this.near[model]!;
      q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), hash(k) * 6.283);
      const spread = model === CONIFER ? 1.0 : 1.0;
      m.compose(new THREE.Vector3(e, base - 0.2, -n), q, new THREE.Vector3(r * spread, h, r * spread));
      mesh.setMatrixAt(counts[model]!, m);
      counts[model]! += 1;
    }
    this.near.forEach((mesh, i) => {
      mesh.count = counts[i]!;
      mesh.instanceMatrix.needsUpdate = true;
    });
    for (const f of this.far) f.instanceMatrix.needsUpdate = true;
    this.shown = near;
  }
}
