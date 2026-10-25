// Trees for the cinematic look (docs/07 §15): each tree is a model of its kind (a broadleaf
// with a trunk and a lumpy crown; a tiered conifer; a weeping willow; a shrub), shaded darker
// low and towards the inside of its crown, casting shadows. Every tree is drawn as the same
// model wherever the camera is, so nothing changes as the aircraft comes near. The trees are
// grouped in tiles, each one instanced mesh per kind, so those out of view are not drawn.
import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import type { WorldPayload } from "./payload";

const TILE_M = 500; // trees are grouped in tiles this wide, each culled as a whole
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

/** Models in unit space: height 1, crown radius 1 (scaled per tree by height and crown); a
 * hundred triangles or fewer each, as a hundred thousand of them are drawn. */
function species(): THREE.BufferGeometry[] {
  const trunk = (r0: number, r1: number, h: number) => {
    const t = new THREE.CylinderGeometry(r0, r1, h, 5, 1, true);
    t.translate(0, h / 2, 0);
    return paint(t.toNonIndexed(), () => BARK);
  };
  const leaf = (base: [number, number, number]) => (y: number, r: number, i: number): [number, number, number] => {
    const shade = 0.62 + 0.38 * Math.min(1, y) * (0.7 + 0.3 * Math.min(1, r)); // darker low and inside
    const j = 0.92 + 0.16 * hash(i * 0.37);
    return [base[0] * shade * j, base[1] * shade * j, base[2] * shade * j];
  };
  const lobe = (detail: number, seed: number, [x, y, z]: [number, number, number], [sx, sy, sz]: [number, number, number]) => {
    const g = lumpy(new THREE.IcosahedronGeometry(1, detail), 0.26, seed);
    g.scale(sx, sy, sz);
    g.translate(x, y, z);
    return g.toNonIndexed();
  };
  // A broadleaf: a trunk, and a crown from a third of its height to its top (a main lobe and
  // three smaller ones round it).
  const crown = mergeGeometries([
    lobe(1, 31, [0, 0.66, 0], [0.82, 0.31, 0.82]),
    lobe(0, 62, [0.5, 0.58, -0.18], [0.5, 0.2, 0.5]),
    lobe(0, 93, [-0.45, 0.6, -0.3], [0.48, 0.2, 0.48]),
    lobe(0, 124, [0.05, 0.57, 0.52], [0.5, 0.2, 0.5]),
  ])!;
  const broadleaf = mergeGeometries([trunk(0.045, 0.07, 0.55), paint(crown, leaf([0.17, 0.31, 0.09]))])!;
  // A conifer: a trunk and three tiers, narrowing upwards to its top.
  const tiers: THREE.BufferGeometry[] = [trunk(0.04, 0.06, 0.3)];
  for (let k = 0; k < 3; k++) {
    const c = new THREE.ConeGeometry(1.0 - 0.25 * k, 0.44 - 0.04 * k, 8, 1, true);
    c.translate(0, 0.4 + 0.19 * k, 0);
    tiers.push(paint(lumpy(c.toNonIndexed(), 0.16, 77 + k), leaf([0.08, 0.18, 0.09])));
  }
  const conifer = mergeGeometries(tiers)!;
  // A willow: a short trunk and a wide, drooping crown.
  const wl = lumpy(new THREE.SphereGeometry(1, 9, 5, 0, Math.PI * 2, 0, Math.PI * 0.62), 0.22, 5);
  wl.scale(1.05, 0.55, 1.05);
  wl.translate(0, 0.42, 0);
  const willow = mergeGeometries([trunk(0.05, 0.08, 0.5), paint(wl.toNonIndexed(), leaf([0.25, 0.37, 0.14]))])!;
  // A shrub: one lumpy mound on the ground.
  const shrub = paint(lobe(1, 200, [0, 0.5, 0], [0.9, 0.5, 0.9]), leaf([0.18, 0.3, 0.11]));
  for (const g of [broadleaf, conifer, willow, shrub]) g.computeVertexNormals();
  return [broadleaf, conifer, willow, shrub];
}

export class TreeField {
  readonly group = new THREE.Group();


  constructor(w: WorldPayload) {
    const t = w.trees;
    const count = t.length / 6;
    const models = species();
    const material = new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true });
    // Trees by tile and kind.
    const byTile = new Map<string, number[][]>();
    for (let k = 0; k < count; k++) {
      const [kind, h] = [Math.min(2, Math.max(0, Math.round(t[6 * k + 5]!))), t[6 * k + 3]!];
      const model = kind === 1 ? CONIFER : kind === 2 ? WILLOW : h < 7 ? SHRUB : BROADLEAF;
      const key = `${Math.floor(t[6 * k]! / TILE_M)},${Math.floor(t[6 * k + 1]! / TILE_M)}`;
      let lists = byTile.get(key);
      if (!lists) byTile.set(key, (lists = [[], [], [], []]));
      lists[model]!.push(k);
    }
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const up = new THREE.Vector3(0, 1, 0);
    const c = new THREE.Color();
    for (const lists of byTile.values()) {
      lists.forEach((ks, model) => {
        if (!ks.length) return;
        const mesh = new THREE.InstancedMesh(models[model]!, material, ks.length);
        ks.forEach((k, slot) => {
          const [n, e, base, h, r] = [t[6 * k]!, t[6 * k + 1]!, t[6 * k + 2]!, t[6 * k + 3]!, t[6 * k + 4]!];
          q.setFromAxisAngle(up, hash(k) * 6.283);
          mesh.setMatrixAt(slot, m.compose(new THREE.Vector3(e, base - 0.2, -n), q, new THREE.Vector3(r, h, r)));
          const v = 0.9 + 0.2 * hash(k + 0.5); // each tree a little lighter or darker
          mesh.setColorAt(slot, c.setRGB(v, v * (0.97 + 0.06 * hash(k + 0.25)), v));
        });
        mesh.computeBoundingSphere(); // round its trees: the tile is culled as a whole
        mesh.castShadow = mesh.receiveShadow = true;
        this.group.add(mesh);
      });
    }
  }
}
