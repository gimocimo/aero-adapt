// Stylised meshes of a world (docs/07 §8): terrain draped with the painted land cover,
// hedges, trees, buildings and masts, each exactly where the simulator has them.
import * as THREE from "three";
import { Ground, type Building, type WorldPayload } from "./payload";

/** Terrain grid → mesh in the three.js world (x east, y up, z south), UVs over the bounds. */
export function terrainGeometry(w: WorldPayload, step = 1): THREE.BufferGeometry {
  const { n0, e0, cell, rows, cols, h } = w.dtm;
  const [bn0, be0, bn1, be1] = w.bounds;
  const ri = Array.from({ length: Math.ceil((rows - 1) / step) + 1 }, (_, k) => Math.min(k * step, rows - 1));
  const cj = Array.from({ length: Math.ceil((cols - 1) / step) + 1 }, (_, k) => Math.min(k * step, cols - 1));
  const pos = new Float32Array(ri.length * cj.length * 3);
  const uv = new Float32Array(ri.length * cj.length * 2);
  let k = 0;
  for (const i of ri) {
    for (const j of cj) {
      const n = n0 + i * cell;
      const e = e0 + j * cell;
      pos[3 * k] = e;
      pos[3 * k + 1] = h[i * cols + j]!;
      pos[3 * k + 2] = -n;
      uv[2 * k] = (e - be0) / (be1 - be0);
      uv[2 * k + 1] = (n - bn0) / (bn1 - bn0);
      k += 1;
    }
  }
  const idx: number[] = [];
  const W = cj.length;
  for (let a = 0; a < ri.length - 1; a++) {
    for (let b = 0; b < W - 1; b++) {
      const p = a * W + b;
      // Counter-clockwise seen from above (+y): (n, e) → (n, e+1) → (n+1, e).
      idx.push(p, p + 1, p + W, p + 1, p + W + 1, p + W);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

/** Open land round the world out to the horizon, at the world's edge height. */
export function skirt(w: WorldPayload, colour: string): THREE.Mesh {
  const [n0, e0, n1, e1] = w.bounds;
  let lo = Infinity;
  for (const v of w.dtm.h) lo = Math.min(lo, v);
  const outer = new THREE.Shape([
    new THREE.Vector2(-30000, -30000),
    new THREE.Vector2(30000, -30000),
    new THREE.Vector2(30000, 30000),
    new THREE.Vector2(-30000, 30000),
  ]);
  outer.holes.push(
    new THREE.Path([new THREE.Vector2(e0, n0), new THREE.Vector2(e0, n1), new THREE.Vector2(e1, n1), new THREE.Vector2(e1, n0)]),
  );
  const g = new THREE.ShapeGeometry(outer);
  g.rotateX(-Math.PI / 2); // shape (x=e, y=n) → (x=e, z=-n)
  const m = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ color: colour }));
  m.position.y = Math.min(lo, 0) - 0.5;
  return m;
}

/** Hedges: one box per segment, standing on the ground. */
export function hedges(w: WorldPayload, ground: Ground): THREE.InstancedMesh | null {
  const h = w.hedges;
  const n = h.length / 6;
  if (n === 0) return null;
  const box = new THREE.BoxGeometry(1, 1, 1);
  box.translate(0, 0.5, 0);
  const mesh = new THREE.InstancedMesh(box, new THREE.MeshLambertMaterial({ color: "#3d7535" }), n);
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const up = new THREE.Vector3(0, 1, 0);
  const c = new THREE.Color();
  for (let k = 0; k < n; k++) {
    const [na, ea, nb, eb, height, width] = [h[6 * k]!, h[6 * k + 1]!, h[6 * k + 2]!, h[6 * k + 3]!, h[6 * k + 4]!, h[6 * k + 5]!];
    const ga = ground.at(na, ea);
    const gb = ground.at(nb, eb);
    const len = Math.hypot(nb - na, eb - ea);
    const base = Math.min(ga, gb) - 0.4;
    q.setFromAxisAngle(up, Math.atan2(nb - na, eb - ea)); // box x along the segment
    m.compose(
      new THREE.Vector3((ea + eb) / 2, base, -(na + nb) / 2),
      q,
      new THREE.Vector3(len + width * 0.6, height + Math.abs(ga - gb) + 0.4, width),
    );
    mesh.setMatrixAt(k, m);
    mesh.setColorAt(k, c.setHSL(0.29 + 0.03 * Math.sin(k * 1.7), 0.45, 0.3 + 0.03 * Math.cos(k)));
  }
  return mesh;
}

/** Trees: crowns by kind (broadleaf, conifer, willow) and trunks, all instanced. ``thin``
 * keeps every n-th tree of the woods (with fuller crowns) for slow GPUs. */
export function trees(w: WorldPayload, thin = 1): THREE.Group {
  const t = w.trees;
  const group = new THREE.Group();
  const count = t.length / 6;
  const keep: number[] = [];
  for (let k = 0; k < count; k++) if (thin <= 1 || k % thin === 0 || t[6 * k + 5] === 2) keep.push(k);
  const crowns = [
    new THREE.IcosahedronGeometry(1, 1), // broadleaf: a lumpy ball
    new THREE.ConeGeometry(1, 1, 7), // conifer
    new THREE.IcosahedronGeometry(1, 1), // willow, squashed
  ];
  crowns[1]!.translate(0, 0.5, 0);
  const colours = [
    [0.27, 0.5, 0.3],
    [0.36, 0.45, 0.22],
    [0.22, 0.45, 0.4],
  ] as const;
  const byKind = [0, 1, 2].map((kind) => keep.filter((k) => Math.round(t[6 * k + 5]!) === kind));
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const c = new THREE.Color();
  const fuller = thin > 1 ? 1.35 : 1;
  byKind.forEach((ks, kind) => {
    if (ks.length === 0) return;
    const mesh = new THREE.InstancedMesh(crowns[kind]!, new THREE.MeshLambertMaterial({ flatShading: true }), ks.length);
    ks.forEach((k, i) => {
      const [n, e, base, h, r] = [t[6 * k]!, t[6 * k + 1]!, t[6 * k + 2]!, t[6 * k + 3]!, t[6 * k + 4]! * fuller];
      q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), (k * 2.399) % (2 * Math.PI));
      if (kind === 1) m.compose(new THREE.Vector3(e, base + 0.15 * h, -n), q, new THREE.Vector3(r, 0.85 * h, r));
      else if (kind === 2) m.compose(new THREE.Vector3(e, base + 0.55 * h, -n), q, new THREE.Vector3(r, 0.42 * h, r));
      else m.compose(new THREE.Vector3(e, base + 0.62 * h, -n), q, new THREE.Vector3(r, 0.38 * h, r));
      mesh.setMatrixAt(i, m);
      const [hue, sat, light] = colours[kind]!;
      const jitter = Math.sin(k * 12.9898) * 0.5 + 0.5;
      mesh.setColorAt(i, c.setHSL(hue + 0.04 * (jitter - 0.5), sat, light - 0.06 + 0.1 * jitter));
    });
    group.add(mesh);
  });
  // Trunks, except deep in the woods where nobody sees them.
  const outside = keep.filter((k) => t[6 * k + 3]! < 16 || k % 4 === 0);
  const trunk = new THREE.CylinderGeometry(0.22, 0.32, 1, 5);
  trunk.translate(0, 0.5, 0);
  const trunks = new THREE.InstancedMesh(trunk, new THREE.MeshLambertMaterial({ color: "#6a4a2c" }), outside.length);
  outside.forEach((k, i) => {
    const [n, e, base, h] = [t[6 * k]!, t[6 * k + 1]!, t[6 * k + 2]!, t[6 * k + 3]!];
    m.compose(new THREE.Vector3(e, base - 0.3, -n), q.identity(), new THREE.Vector3(1, 0.45 * h + 0.3, 1));
    trunks.setMatrixAt(i, m);
  });
  group.add(trunks);
  return group;
}

// Wall and roof colours by building kind and variant.
const WALLS = ["#b8664c", "#e6dcc3", "#cdbd9c", "#f1ede4", "#a85d45", "#d9cfb6", "#c7a67f", "#b9b3a6"];
const ROOFS = ["#4b5059", "#9d4c37", "#7b4b35", "#3f444c", "#5a5e66", "#8a4632", "#4b5059", "#6c4a3a"];
const KIND_WALL: Record<string, string> = { barn: "#7c8576", shed: "#8b9088", hangar: "#9aa1a8", nave: "#c9c0a8", tower: "#c4bba2" };
const KIND_ROOF: Record<string, string> = { barn: "#8a9296", shed: "#80878c", hangar: "#7c838a", nave: "#4a4f58", tower: "#7d8389" };

class Builder {
  pos: number[] = [];
  col: number[] = [];
  private c = new THREE.Color();

  tri(a: THREE.Vector3, b: THREE.Vector3, d: THREE.Vector3, colour: string): void {
    this.c.set(colour);
    for (const p of [a, b, d]) {
      this.pos.push(p.x, p.y, p.z);
      this.col.push(this.c.r, this.c.g, this.c.b);
    }
  }

  quad(a: THREE.Vector3, b: THREE.Vector3, c: THREE.Vector3, d: THREE.Vector3, colour: string): void {
    this.tri(a, b, c, colour);
    this.tri(a, c, d, colour);
  }

  geometry(): THREE.BufferGeometry {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(this.col, 3));
    g.computeVertexNormals();
    return g;
  }
}

function addBuilding(B: Builder, b: Building): void {
  const u = new THREE.Vector2(Math.cos(b.heading), Math.sin(b.heading)); // along (n, e)
  const v = new THREE.Vector2(-u.y, u.x); // across
  // Local (a along, c across, y up) → world.
  const P = (a: number, c: number, y: number) =>
    new THREE.Vector3(b.e + a * u.y + c * v.y, b.base + y, -(b.n + a * u.x + c * v.x));
  const L = b.length / 2;
  const W = b.width / 2;
  const wall = KIND_WALL[b.kind] ?? WALLS[b.variant % WALLS.length]!;
  const roof = KIND_ROOF[b.kind] ?? ROOFS[b.variant % ROOFS.length]!;
  const top = b.roof === "flat" ? b.ridge : b.eaves;
  const sink = -0.6; // walls start a little below the ground (slopes)
  const corners: [number, number][] = [
    [L, W],
    [-L, W],
    [-L, -W],
    [L, -W],
  ];
  for (let k = 0; k < 4; k++) {
    const [a1, c1] = corners[k]!;
    const [a2, c2] = corners[(k + 1) % 4]!;
    B.quad(P(a1, c1, sink), P(a2, c2, sink), P(a2, c2, top), P(a1, c1, top), wall);
  }
  if (b.roof === "flat") {
    B.quad(P(L, W, top), P(-L, W, top), P(-L, -W, top), P(L, -W, top), roof);
    return;
  }
  if (b.roof === "spire") {
    const apex = P(0, 0, b.ridge);
    for (let k = 0; k < 4; k++) {
      const [a1, c1] = corners[k]!;
      const [a2, c2] = corners[(k + 1) % 4]!;
      B.tri(P(a1 * 0.9, c1 * 0.9, top), P(a2 * 0.9, c2 * 0.9, top), apex, roof);
    }
    B.quad(P(L, W, top), P(-L, W, top), P(-L, -W, top), P(L, -W, top), wall);
    return;
  }
  // Gable: ridge along the length, a little overhang.
  const o = 0.35;
  const r1 = P(L + o, 0, b.ridge);
  const r2 = P(-L - o, 0, b.ridge);
  B.quad(P(L + o, W + o, top - 0.25), P(-L - o, W + o, top - 0.25), r2, r1, roof);
  B.quad(P(-L - o, -W - o, top - 0.25), P(L + o, -W - o, top - 0.25), r1, r2, roof);
  B.tri(P(L, W, top), P(L, 0, b.ridge), P(L, -W, top), wall);
  B.tri(P(-L, -W, top), P(-L, 0, b.ridge), P(-L, W, top), wall);
}

/** Every building in one mesh (vertex colours, flat shading). */
export function buildings(w: WorldPayload): THREE.Mesh | null {
  if (w.buildings.length === 0) return null;
  const B = new Builder();
  for (const b of w.buildings) addBuilding(B, b);
  return new THREE.Mesh(
    B.geometry(),
    new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true, side: THREE.DoubleSide }),
  );
}

/** Masts (a slender lattice with a warning light) and silos. */
export function structures(w: WorldPayload): THREE.Group {
  const g = new THREE.Group();
  for (const s of w.structures) {
    if (s.kind === "mast") {
      const mast = new THREE.Mesh(
        new THREE.CylinderGeometry(0.35, 0.9, s.height, 4, 8),
        new THREE.MeshLambertMaterial({ color: "#c9ccd0", wireframe: false }),
      );
      mast.position.set(s.e, s.base + s.height / 2, -s.n);
      const bands = new THREE.Mesh(
        new THREE.CylinderGeometry(0.42, 0.5, 3, 4),
        new THREE.MeshLambertMaterial({ color: "#d9473b" }),
      );
      bands.position.set(s.e, s.base + s.height - 1.5, -s.n);
      const light = new THREE.Mesh(new THREE.SphereGeometry(0.7, 8, 6), new THREE.MeshBasicMaterial({ color: "#ff3b30" }));
      light.position.set(s.e, s.base + s.height + 0.5, -s.n);
      g.add(mast, bands, light);
    } else {
      const silo = new THREE.Mesh(
        new THREE.CylinderGeometry(s.radius, s.radius, s.height - s.radius, 14),
        new THREE.MeshLambertMaterial({ color: "#b8bcbf" }),
      );
      silo.position.set(s.e, s.base + (s.height - s.radius) / 2, -s.n);
      const dome = new THREE.Mesh(
        new THREE.SphereGeometry(s.radius, 14, 6, 0, Math.PI * 2, 0, Math.PI / 2),
        new THREE.MeshLambertMaterial({ color: "#9aa0a4" }),
      );
      dome.position.set(s.e, s.base + s.height - s.radius, -s.n);
      g.add(silo, dome);
    }
  }
  return g;
}
