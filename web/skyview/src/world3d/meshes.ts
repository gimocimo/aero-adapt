// Stylised meshes of a world (docs/07 §8): terrain draped with the painted land cover,
// hedges, trees, buildings and masts, each exactly where the simulator has them.
import * as THREE from "three";
import { buildingMaterial, FACADE, roadMaterial, waterMaterial } from "./materials";
import { ROAD } from "./paint";
import { Ground, type Building, type Structure, type WorldPayload } from "./payload";

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

/** Round a city (London): the ground running on from the map's edge (no cliff at it), the
 * river carrying on beyond both ends, and an ordinary city filling the rest out to a few
 * kilometres, fading into the haze. Drawn only: none of it is in the simulator. */
export function outskirts(w: WorldPayload, ground: Ground, levelAt: ((n: number, e: number) => number | null) | null): THREE.Group {
  const group = new THREE.Group();
  const [n0, e0, n1, e1] = w.bounds;
  const N = 480;
  // A point at fraction t round a rectangle grown by d (anticlockwise from the south-west).
  const around = (t: number, d: number): [number, number] => {
    const [a0, b0, a1, b1] = [n0 - d, e0 - d, n1 + d, e1 + d];
    const [h, wd] = [a1 - a0, b1 - b0];
    let x = t * 2 * (h + wd);
    if (x < wd) return [a0, b0 + x];
    x -= wd;
    if (x < h) return [a0 + x, b1];
    x -= h;
    if (x < wd) return [a1, b1 - x];
    return [a1 - (x - wd), b0];
  };
  const inner = Array.from({ length: N }, (_, k) => around(k / N, 0));
  const edge = inner.map(([n, e]) => ground.at(n, e));
  const level = [...edge].sort((a, b) => a - b)[Math.floor(N / 2)]!;
  const rings: [number, number][] = [[0, 0], [150, 0.5], [600, 0.9], [2500, 1], [30000, 1]];
  const pos: number[] = [];
  const idx: number[] = [];
  rings.forEach(([d, wgt]) => {
    for (let k = 0; k < N; k++) {
      const [n, e] = d === 0 ? inner[k]! : around(k / N, d);
      pos.push(e, edge[k]! * (1 - wgt) + level * wgt - 0.3, -n);
    }
  });
  for (let r = 0; r + 1 < rings.length; r++) {
    for (let k = 0; k < N; k++) {
      const [a, b, c, d] = [r * N + k, r * N + ((k + 1) % N), (r + 1) * N + ((k + 1) % N), (r + 1) * N + k];
      idx.push(a, d, b, b, d, c);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  const land = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ color: "#a2a5a7", side: THREE.DoubleSide }));
  land.receiveShadow = true;
  group.add(land);

  // The river beyond both ends, straight on along its last reach.
  const river = w.river;
  const reaches: [[number, number], [number, number]][] = [];
  if (river && river.points.length > 4) {
    const pts = river.points;
    for (const [a, b] of [[pts.length - 4, pts.length - 1], [3, 0]] as const) {
      const [na, ea] = pts[a]!;
      const [nb, eb] = pts[b]!;
      const len = Math.hypot(nb - na, eb - ea) || 1;
      const [un, ue] = [(nb - na) / len, (eb - ea) / len];
      reaches.push([[nb - un * 40, eb - ue * 40], [nb + un * 4000, eb + ue * 4000]]);
    }
    const lv = levelAt?.(...pts[Math.floor(pts.length / 2)]!) ?? level - 3;
    const half = river.width / 2;
    const wpos: number[] = [];
    const wuv: number[] = [];
    const widx: number[] = [];
    for (const [[na, ea], [nb, eb]] of reaches) {
      const len = Math.hypot(nb - na, eb - ea);
      const [un, ue] = [(nb - na) / len, (eb - ea) / len];
      const [ln, le] = [-ue * half, un * half];
      const base = wpos.length / 3;
      for (const [n, e] of [[na + ln, ea + le], [na - ln, ea - le], [nb - ln, eb - le], [nb + ln, eb + le]] as const) {
        wpos.push(e, lv, -n);
        wuv.push(e, n);
      }
      widx.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }
    const wg = new THREE.BufferGeometry();
    wg.setAttribute("position", new THREE.Float32BufferAttribute(wpos, 3));
    wg.setAttribute("uv", new THREE.Float32BufferAttribute(wuv, 2));
    wg.setIndex(widx);
    wg.computeVertexNormals();
    const water = new THREE.Mesh(wg, waterMaterial());
    water.material.side = THREE.DoubleSide;
    water.name = "water-beyond";
    group.add(water);
  }
  const offRiver = (n: number, e: number) =>
    reaches.every(([[na, ea], [nb, eb]]) => {
      const [dn, de] = [nb - na, eb - ea];
      const t = Math.min(Math.max(((n - na) * dn + (e - ea) * de) / (dn * dn + de * de), 0), 1);
      return Math.hypot(n - na - t * dn, e - ea - t * de) > (river?.width ?? 0) / 2 + 25;
    });

  // An ordinary city: blocks of a few buildings between streets, now and then a tower.
  const rnd = seeded(20261023);
  const BLOCK = 85;
  const REACH = 3500;
  const boxes: [number, number, number, number, number, number][] = []; // n, e, length, width, height, heading
  for (let n = n0 - REACH; n < n1 + REACH; n += BLOCK) {
    for (let e = e0 - REACH; e < e1 + REACH; e += BLOCK) {
      const [cn, ce] = [n + BLOCK / 2 + (rnd() - 0.5) * 20, e + BLOCK / 2 + (rnd() - 0.5) * 20];
      if (cn > n0 - 60 && cn < n1 + 60 && ce > e0 - 60 && ce < e1 + 60) continue;
      if (!offRiver(cn, ce)) continue;
      const far = Math.max(n0 - cn, cn - n1, e0 - ce, ce - e1);
      const heading = (rnd() - 0.5) * 0.5;
      const parts = 2 + Math.floor(rnd() * 3);
      for (let k = 0; k < parts; k++) {
        const r = rnd();
        const height = r > 0.995 ? 90 + rnd() * 70 : r > 0.96 ? 35 + rnd() * 45 : 9 + 18 * rnd() * rnd() + 6 * rnd();
        const l = 18 + rnd() * 30;
        const wd = 14 + rnd() * 22;
        const [on, oe] = [(rnd() - 0.5) * (BLOCK - 18 - l * 0.5), (rnd() - 0.5) * (BLOCK - 18 - wd * 0.5)];
        boxes.push([cn + on, ce + oe, l, wd, height * (far > 2000 ? 0.85 : 1), heading]);
      }
    }
  }
  const unit = new THREE.BoxGeometry(1, 1, 1);
  unit.translate(0, 0.5, 0);
  const city = new THREE.InstancedMesh(unit, new THREE.MeshLambertMaterial(), boxes.length);
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const up = new THREE.Vector3(0, 1, 0);
  const c = new THREE.Color();
  const palette = ["#cfc9bc", "#b9b3a6", "#a5a29b", "#9a6b55", "#c6b89a", "#8f969d", "#d8d3c8", "#a9876b", "#7f8a96", "#bcb7ae"];
  boxes.forEach(([n, e, l, wd, h, hd], k) => {
    q.setFromAxisAngle(up, hd);
    m.compose(new THREE.Vector3(e, level - 0.3, -n), q, new THREE.Vector3(wd, h, l));
    city.setMatrixAt(k, m);
    city.setColorAt(k, c.set(palette[k % palette.length]!));
  });
  city.name = "outskirts";
  group.add(city);
  return group;
}

/** Hedges: a row of overlapping, irregular bushes along each segment (from the segment's
 * height and width) over a dense core that fills the dips between them, as the eye sees a
 * hedgerow: one lumpy mass, never quite the same. */
export function hedges(w: WorldPayload, ground: Ground): THREE.Group | null {
  const h = w.hedges;
  const n = h.length / 6;
  if (n === 0) return null;
  const SPACING = 2.6;
  const bushes: [number, number, number, number, number, number, number][] = []; // e, base, -n, radius, height, seed, yaw
  const cores: [number, number, number, number, number, number, number][] = []; // e, base, -n, length, height, width, yaw
  for (let k = 0; k < n; k++) {
    const [na, ea, nb, eb, height, width] = [h[6 * k]!, h[6 * k + 1]!, h[6 * k + 2]!, h[6 * k + 3]!, h[6 * k + 4]!, h[6 * k + 5]!];
    const len = Math.hypot(nb - na, eb - ea);
    const yaw = Math.atan2(nb - na, eb - ea); // the bush's long axis along the hedge
    const [mn, me] = [(na + nb) / 2, (ea + eb) / 2];
    const low = Math.min(ground.at(na, ea), ground.at(nb, eb), ground.at(mn, me));
    cores.push([me, low - 0.3, -mn, len, height * 0.55 + 0.3, width * 0.7, yaw]);
    const count = Math.max(1, Math.round(len / SPACING));
    for (let i = 0; i < count; i++) {
      const s = (i + 0.5) / count;
      const seed = Math.sin((k * 131 + i) * 12.9898) * 43758.5453;
      const j = seed - Math.floor(seed); // 0..1, fixed per bush
      const nn = na + (nb - na) * s + (j - 0.5) * 0.6;
      const ee = ea + (eb - ea) * s + (j - 0.5) * 0.6;
      bushes.push([ee, ground.at(nn, ee) - 0.3, -nn, (width / 2) * (0.85 + 0.5 * j), height * (0.8 + 0.35 * j), j, yaw]);
    }
  }
  const blob = new THREE.IcosahedronGeometry(1, 1);
  blob.translate(0, 0.55, 0);
  const mesh = new THREE.InstancedMesh(blob, new THREE.MeshLambertMaterial({ flatShading: true }), bushes.length);
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const up = new THREE.Vector3(0, 1, 0);
  const c = new THREE.Color();
  bushes.forEach(([x, y, z, r, ht, j, yaw], k) => {
    // Long along the hedge, so neighbours overlap into one lumpy hedge (not beads on a string).
    q.setFromAxisAngle(up, yaw + (j - 0.5) * 0.5);
    m.compose(new THREE.Vector3(x, y, z), q, new THREE.Vector3(Math.max(r * 1.9, 1.75), ht * 0.62, r * 1.1));
    mesh.setMatrixAt(k, m);
    mesh.setColorAt(k, c.setHSL(0.27 + 0.03 * (j - 0.5), 0.42, 0.3 + 0.035 * j));
  });
  // The core: a box per segment, a little lower and narrower than the bushes round it.
  const box = new THREE.BoxGeometry(1, 1, 1);
  box.translate(0, 0.5, 0);
  const shade = new THREE.Color().setHSL(0.27, 0.42, 0.27); // the bushes' green, in their shadow
  const core = new THREE.InstancedMesh(box, new THREE.MeshLambertMaterial({ color: shade }), cores.length);
  cores.forEach(([x, y, z, len, ht, wd, yaw], k) => {
    q.setFromAxisAngle(up, yaw);
    m.compose(new THREE.Vector3(x, y, z), q, new THREE.Vector3(len, ht, wd));
    core.setMatrixAt(k, m);
  });
  const group = new THREE.Group();
  group.add(mesh, core);
  return group;
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

type UV = [number, number];

class Builder {
  pos: number[] = [];
  col: number[] = [];
  fac: number[] = []; // per vertex: metres along, metres up, facade kind (materials.FACADE)
  private c = new THREE.Color();

  tri(a: THREE.Vector3, b: THREE.Vector3, d: THREE.Vector3, colour: string, uv?: [UV, UV, UV], kind = 0): void {
    this.c.set(colour);
    [a, b, d].forEach((p, k) => {
      this.pos.push(p.x, p.y, p.z);
      this.col.push(this.c.r, this.c.g, this.c.b);
      const [x, y] = uv?.[k] ?? [0, 0];
      this.fac.push(x, y, kind);
    });
  }

  quad(a: THREE.Vector3, b: THREE.Vector3, c: THREE.Vector3, d: THREE.Vector3, colour: string, uv?: [UV, UV, UV, UV], kind = 0): void {
    this.tri(a, b, c, colour, uv && [uv[0], uv[1], uv[2]], kind);
    this.tri(a, c, d, colour, uv && [uv[0], uv[2], uv[3]], kind);
  }

  geometry(): THREE.BufferGeometry {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(this.col, 3));
    g.setAttribute("aFacade", new THREE.Float32BufferAttribute(this.fac, 3));
    g.computeVertexNormals();
    return g;
  }
}

/** Which facade and roof a building has (windows by use; tiles on pitched roofs). */
function facades(b: Building): [number, number] {
  const roof = b.roof === "flat" ? FACADE.flat : FACADE.tiles;
  switch (b.kind) {
    case "tower":
      return b.roof === "spire" ? [FACADE.stone, roof] : [FACADE.glass, FACADE.flat]; // a church's, or a tall block
    case "modern":
    case "block":
    case "hall":
      return [FACADE.offices, roof];
    case "old":
    case "terrace":
    case "house":
    case "farmhouse":
    case "pub":
      return [FACADE.homes, roof];
    case "landmark":
    case "castle":
      return [FACADE.civic, roof];
    case "nave":
      return [FACADE.stone, roof];
    case "industrial":
    case "barn":
    case "shed":
    case "hangar":
    case "reactor":
      return [FACADE.industry, roof];
    default:
      return [FACADE.none, FACADE.none]; // masts, antennas, gantries
  }
}

/** A building on its own footprint: walls round every ring (up to the roof's edge), and a
 * flat, sloping (skillion), pyramidal or domed roof; a part clear of the ground is closed
 * underneath. */
function addFootprint(B: Builder, b: Building): void {
  const outer = b.outline!;
  const holes = b.holes ?? [];
  const P = (n: number, e: number, y: number) => new THREE.Vector3(e, b.base + y, -n);
  const wall = b.colours?.[0] ?? KIND_WALL[b.kind] ?? WALLS[b.variant % WALLS.length]!;
  const roofC = b.colours?.[1] ?? KIND_ROOF[b.kind] ?? ROOFS[b.variant % ROOFS.length]!;
  let [fw, fr] = facades(b);
  const bottom = b.bottom ?? 0;
  const foot = bottom > 0.5 ? bottom : -0.6; // walls start a little below the ground (slopes)
  const rise = b.ridge - b.eaves;
  // The roof's edge height at a point of the outline (above the base).
  let edge = (_n: number, _e: number) => (b.roof === "flat" ? b.ridge : b.eaves);
  if (b.roof === "skillion") {
    const [c, s] = [Math.cos(b.slope ?? 0), Math.sin(b.slope ?? 0)];
    const d = outer.map(([n, e]) => n * c + e * s);
    const [d0, d1] = [Math.min(...d), Math.max(...d)];
    edge = (n, e) => b.ridge - (rise * (n * c + e * s - d0)) / Math.max(d1 - d0, 1e-6);
    // A steep one is a face of the building (the Shard's glass), not a roof.
    if (rise > d1 - d0) fr = fw;
  }
  for (const ring of [outer, ...holes]) {
    let along = 0;
    ring.forEach(([n1, e1], k) => {
      const [n2, e2] = ring[(k + 1) % ring.length]!;
      const len = Math.hypot(n2 - n1, e2 - e1);
      const [t1, t2] = [edge(n1, e1), edge(n2, e2)];
      B.quad(P(n1, e1, foot), P(n2, e2, foot), P(n2, e2, t2), P(n1, e1, t1), wall,
             [[along, foot], [along + len, foot], [along + len, t2], [along, t1]], fw); // prettier-ignore
      along += len;
    });
  }
  const v2 = (r: [number, number][]) => r.map(([n, e]) => new THREE.Vector2(e, n));
  const flat = (y: (n: number, e: number) => number, colour: string, kind: number) => {
    const pts = [...outer, ...holes.flat()];
    const at = (k: number) => {
      const [n, e] = pts[k]!;
      return { p: P(n, e, y(n, e)), uv: [e, n] as UV };
    };
    for (const face of THREE.ShapeUtils.triangulateShape(v2(outer), holes.map(v2))) {
      const [a, c, d] = face.map(at) as [ReturnType<typeof at>, ReturnType<typeof at>, ReturnType<typeof at>];
      B.tri(a.p, c.p, d.p, colour, [a.uv, c.uv, d.uv], kind);
    }
  };
  if (bottom > 0.5) flat(() => bottom, wall, FACADE.none);
  if (b.roof === "flat" || b.roof === "skillion") {
    flat(edge, roofC, fr);
    return;
  }
  // A spire or a dome over the footprint, from its centroid.
  const cn = outer.reduce((t, [n]) => t + n, 0) / outer.length;
  const ce = outer.reduce((t, [, e]) => t + e, 0) / outer.length;
  const rings = b.roof === "spire" ? 1 : 6;
  for (let k = 0; k < rings; k++) {
    const [a1, a2] = [(k / rings) * (Math.PI / 2), ((k + 1) / rings) * (Math.PI / 2)];
    const [s1, s2] = b.roof === "spire" ? [1, 0] : [Math.cos(a1), Math.cos(a2)];
    const [h1, h2] = b.roof === "spire" ? [b.eaves, b.ridge] : [b.eaves + rise * Math.sin(a1), b.eaves + rise * Math.sin(a2)];
    outer.forEach(([n1, e1], j) => {
      const [n2, e2] = outer[(j + 1) % outer.length]!;
      const at = (n: number, e: number, sc: number, h: number) => P(cn + (n - cn) * sc, ce + (e - ce) * sc, h);
      const len = Math.hypot(n2 - n1, e2 - e1);
      B.quad(at(n1, e1, s1, h1), at(n2, e2, s1, h1), at(n2, e2, s2, h2), at(n1, e1, s2, h2), roofC,
             [[0, h1], [len, h1], [len * s2, h2], [0, h2]], fr); // prettier-ignore
    });
  }
}

function addBuilding(B: Builder, b: Building): void {
  if (b.outline) {
    addFootprint(B, b);
    return;
  }
  const u = new THREE.Vector2(Math.cos(b.heading), Math.sin(b.heading)); // along (n, e)
  const v = new THREE.Vector2(-u.y, u.x); // across
  // Local (a along, c across, y up) → world.
  const P = (a: number, c: number, y: number) =>
    new THREE.Vector3(b.e + a * u.y + c * v.y, b.base + y, -(b.n + a * u.x + c * v.x));
  const L = b.length / 2;
  const W = b.width / 2;
  const wall = b.colours?.[0] ?? KIND_WALL[b.kind] ?? WALLS[b.variant % WALLS.length]!;
  const roof = b.colours?.[1] ?? KIND_ROOF[b.kind] ?? ROOFS[b.variant % ROOFS.length]!;
  const [fw, fr] = facades(b);
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
    const len = Math.hypot(a2 - a1, c2 - c1);
    B.quad(P(a1, c1, sink), P(a2, c2, sink), P(a2, c2, top), P(a1, c1, top), wall,
           [[0, sink], [len, sink], [len, top], [0, top]], fw); // prettier-ignore
  }
  if (b.roof === "flat") {
    B.quad(P(L, W, top), P(-L, W, top), P(-L, -W, top), P(L, -W, top), roof, [[L, W], [-L, W], [-L, -W], [L, -W]], fr);
    return;
  }
  if (b.roof === "spire") {
    const apex = P(0, 0, b.ridge);
    const slope = Math.hypot(Math.max(L, W) * 0.9, b.ridge - top);
    for (let k = 0; k < 4; k++) {
      const [a1, c1] = corners[k]!;
      const [a2, c2] = corners[(k + 1) % 4]!;
      const len = Math.hypot(a2 - a1, c2 - c1) * 0.9;
      B.tri(P(a1 * 0.9, c1 * 0.9, top), P(a2 * 0.9, c2 * 0.9, top), apex, roof, [[0, 0], [len, 0], [len / 2, slope]], fr);
    }
    B.quad(P(L, W, top), P(-L, W, top), P(-L, -W, top), P(L, -W, top), wall);
    return;
  }
  // Gable: ridge along the length, a little overhang.
  const o = 0.35;
  const r1 = P(L + o, 0, b.ridge);
  const r2 = P(-L - o, 0, b.ridge);
  const slope = Math.hypot(W + o, b.ridge - top + 0.25);
  const tiles: [UV, UV, UV, UV] = [[L + o, 0], [-L - o, 0], [-L - o, slope], [L + o, slope]];
  B.quad(P(L + o, W + o, top - 0.25), P(-L - o, W + o, top - 0.25), r2, r1, roof, tiles, fr);
  B.quad(P(-L - o, -W - o, top - 0.25), P(L + o, -W - o, top - 0.25), r1, r2, roof, tiles, fr);
  B.tri(P(L, W, top), P(L, 0, b.ridge), P(L, -W, top), wall, [[0, top], [W, b.ridge], [2 * W, top]], fw);
  B.tri(P(-L, -W, top), P(-L, 0, b.ridge), P(-L, W, top), wall, [[0, top], [W, b.ridge], [2 * W, top]], fw);
}

/** A small seeded generator (the same roofs every time). */
function seeded(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function inside(ring: [number, number][], n: number, e: number): boolean {
  let hit = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [ni, ei] = ring[i]!;
    const [nj, ej] = ring[j]!;
    if (ei > e !== ej > e && n < ((nj - ni) * (e - ei)) / (ej - ei) + ni) hit = !hit;
  }
  return hit;
}

/** A box on a roof: plant room or air-handling unit. */
function addBox(B: Builder, n: number, e: number, l: number, w: number, heading: number, y0: number, h: number, colour: string): void {
  const u = [Math.cos(heading), Math.sin(heading)];
  const P = (a: number, c: number, y: number) =>
    new THREE.Vector3(e + a * u[1]! + c * u[0]!, y, -(n + a * u[0]! - c * u[1]!));
  const c4: [number, number][] = [[l / 2, w / 2], [-l / 2, w / 2], [-l / 2, -w / 2], [l / 2, -w / 2]];
  for (let k = 0; k < 4; k++) {
    const [a1, b1] = c4[k]!;
    const [a2, b2] = c4[(k + 1) % 4]!;
    B.quad(P(a1, b1, y0), P(a2, b2, y0), P(a2, b2, y0 + h), P(a1, b1, y0 + h), colour);
  }
  B.quad(P(l / 2, w / 2, y0 + h), P(-l / 2, w / 2, y0 + h), P(-l / 2, -w / 2, y0 + h), P(l / 2, -w / 2, y0 + h), colour,
         [[0, 0], [l, 0], [l, w], [0, w]], FACADE.flat); // prettier-ignore
}

/** What stands on a big flat roof (drawn only; the simulator has the measured roofs): a plant
 * room or two and air-handling units, placed inside the footprint. */
function rooftop(B: Builder, b: Building, seed: number): void {
  const ring = b.outline;
  if (!ring || b.roof !== "flat" || b.ridge < 12 || (b.bottom ?? 0) > 0.5 || b.kind === "bridge") return;
  let area = 0;
  ring.forEach(([n1, e1], k) => {
    const [n2, e2] = ring[(k + 1) % ring.length]!;
    area += n1 * e2 - n2 * e1;
  });
  area = Math.abs(area) / 2;
  if (area < 250) return;
  const rnd = seeded(seed);
  const ns = ring.map(([n]) => n);
  const es = ring.map(([, e]) => e);
  const [n0, n1, e0, e1] = [Math.min(...ns), Math.max(...ns), Math.min(...es), Math.max(...es)];
  const top = b.base + b.ridge;
  const fits = (n: number, e: number, l: number, w: number, hd: number) =>
    [[1, 1], [-1, 1], [-1, -1], [1, -1], [0, 0]].every(([sa, sb]) => {
      const a = (sa! * (l / 2 + 1.5)) as number;
      const c = (sb! * (w / 2 + 1.5)) as number;
      return inside(ring, n + a * Math.cos(hd) - c * Math.sin(hd), e + a * Math.sin(hd) + c * Math.cos(hd));
    });
  const place = (count: number, size: [number, number], height: [number, number], grey: string[]) => {
    for (let k = 0, tries = 0; k < count && tries < count * 12; tries++) {
      const n = n0 + rnd() * (n1 - n0);
      const e = e0 + rnd() * (e1 - e0);
      const l = size[0] + rnd() * (size[1] - size[0]);
      const w = size[0] + rnd() * (size[1] - size[0]) * 0.6;
      const hd = b.heading + (rnd() < 0.5 ? 0 : Math.PI / 2);
      if (!fits(n, e, l, w, hd)) continue;
      addBox(B, n, e, l, w, hd, top, height[0] + rnd() * (height[1] - height[0]), grey[Math.floor(rnd() * grey.length)]!);
      k++;
    }
  };
  place(Math.min(3, 1 + Math.floor(area / 1500)), [4, Math.min(16, Math.sqrt(area) * 0.3)], [2.6, 4.2], ["#a9abab", "#9b9e9f", "#b3b1aa"]);
  place(Math.min(10, Math.floor(area / 300)), [1.2, 2.6], [0.9, 1.6], ["#c3c5c6", "#8d9194", "#b9bbbd"]);
}

/** Every building in one mesh (vertex colours, flat shading); with ``cinematic``, facades
 * and roof tiles, plant on the flat roofs, and shadows. */
export function buildings(w: WorldPayload, cinematic = false): THREE.Mesh | null {
  if (w.buildings.length === 0) return null;
  const B = new Builder();
  w.buildings.forEach((b, k) => {
    addBuilding(B, b);
    if (cinematic) rooftop(B, b, k * 7919 + 13);
  });
  const mesh = new THREE.Mesh(
    B.geometry(),
    cinematic
      ? buildingMaterial()
      : new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true, side: THREE.DoubleSide }),
  );
  mesh.castShadow = mesh.receiveShadow = cinematic;
  return mesh;
}

/** Water as surfaces, rippled and reflecting the sky: each water cover (river, branch, lake)
 * filled on a grid, level at its lowest shore nearby, so it runs bank to bank and steps down
 * with the river; the cells reach a little under the banks, which hide the edge. */
export function waterSurfaces(w: WorldPayload, ground: Ground): THREE.Mesh | null {
  const water = w.plots.filter((p) => p.kind === "water");
  if (water.length === 0) return null;
  const CELL = 6;
  const pos: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];
  const levels: { outer: [number, number][]; level: (n: number, e: number) => number }[] = [];
  for (const p of water) {
    const rings = [p.outer, ...p.holes];
    const level = shoreLevel(rings, ground);
    levels.push({ outer: p.outer, level });
    // Rows of cells: where each row's lines cross the shore, inside is between pairs.
    const ns = p.outer.map(([n]) => n);
    const es = p.outer.map(([, e]) => e);
    const n0 = Math.floor(Math.min(...ns) / CELL) * CELL;
    const e0 = Math.floor(Math.min(...es) / CELL) * CELL;
    const n1 = Math.max(...ns);
    const vertex = new Map<number, number>();
    const at = (i: number, j: number) => {
      const key = i * 1_000_003 + j;
      let k = vertex.get(key);
      if (k === undefined) {
        const [n, e] = [n0 + i * CELL, e0 + j * CELL];
        pos.push(e, level(n, e), -n);
        uv.push(e, n);
        k = pos.length / 3 - 1;
        vertex.set(key, k);
      }
      return k;
    };
    for (let i = 0; n0 + i * CELL < n1; i++) {
      const spans = [0, 0.5, 1].flatMap((f) => insideSpans(rings, n0 + (i + f) * CELL));
      const cells = new Set<number>();
      for (const [a, b] of spans) {
        for (let j = Math.floor((a - 3 - e0) / CELL); e0 + j * CELL < b + 3; j++) cells.add(j);
      }
      for (const j of cells) {
        const [a, b, c, d] = [at(i, j), at(i, j + 1), at(i + 1, j + 1), at(i + 1, j)];
        idx.push(a, b, c, a, c, d); // facing up
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx);
  g.computeVertexNormals();
  const mesh = new THREE.Mesh(g, waterMaterial());
  mesh.receiveShadow = true;
  mesh.name = "water";
  // The water's level at a point (for what floats on it), or null off the water.
  mesh.userData.levelAt = (n: number, e: number): number | null => {
    for (const l of levels) if (inside(l.outer, n, e)) return l.level(n, e);
    return null;
  };
  return mesh;
}

/** Where a line of constant north crosses into and out of the rings: [from, to] east spans. */
function insideSpans(rings: [number, number][][], n: number): [number, number][] {
  const xs: number[] = [];
  for (const r of rings) {
    for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
      const [ni, ei] = r[i]!;
      const [nj, ej] = r[j]!;
      if (ni > n !== nj > n) xs.push(ei + ((n - ni) * (ej - ei)) / (nj - ni));
    }
  }
  xs.sort((a, b) => a - b);
  const spans: [number, number][] = [];
  for (let k = 0; k + 1 < xs.length; k += 2) spans.push([xs[k]!, xs[k + 1]!]);
  return spans;
}

/** The water's level at a point: a little below the lowest shore within ~40 m (further out
 * on a lake, until some shore is found). */
function shoreLevel(rings: [number, number][][], ground: Ground): (n: number, e: number) => number {
  const B = 40;
  const buckets = new Map<string, number[]>(); // (n, e, height) triples by 40 m square
  for (const r of rings) {
    r.forEach(([n, e], i) => {
      const [pn, pe] = r[(i + 1) % r.length]!;
      const steps = Math.max(1, Math.ceil(Math.hypot(pn - n, pe - e) / 5));
      for (let s = 0; s < steps; s++) {
        const [sn, se] = [n + ((pn - n) * s) / steps, e + ((pe - e) * s) / steps];
        const key = `${Math.floor(sn / B)},${Math.floor(se / B)}`;
        const list = buckets.get(key) ?? [];
        list.push(sn, se, ground.at(sn, se));
        buckets.set(key, list);
      }
    });
  }
  return (n, e) => {
    for (let reach = 1; reach < 64; reach *= 2) {
      let low = Infinity;
      const [bi, bj] = [Math.floor(n / B), Math.floor(e / B)];
      for (let di = -reach; di <= reach; di++) {
        for (let dj = -reach; dj <= reach; dj++) {
          const list = buckets.get(`${bi + di},${bj + dj}`);
          if (!list) continue;
          for (let k = 0; k < list.length; k += 3) {
            if (Math.hypot(list[k]! - n, list[k + 1]! - e) <= reach * B) low = Math.min(low, list[k + 2]!);
          }
        }
      }
      if (low < Infinity) return low - 0.3;
    }
    return ground.at(n, e);
  };
}

/** Roads as surfaces of their own (cinematic): sharp edges, tarmac grain and centre lines
 * close up, where the painted ground is too coarse; they fade into the painted roads with
 * distance. ``aRoad``: metres along, metres across, half width, kind (0 plain, 1 centre
 * line, 2 track). */
export function roadSurfaces(w: WorldPayload, ground: Ground): THREE.Mesh | null {
  if (w.roads.length === 0) return null;
  const pos: number[] = [];
  const col: number[] = [];
  const attr: number[] = [];
  const idx: number[] = [];
  const c = new THREE.Color();
  const LIFT = 0.25;
  const vertex = (n: number, e: number, along: number, across: number, hw: number, kind: number) => {
    pos.push(e, ground.at(n, e) + LIFT, -n);
    col.push(c.r, c.g, c.b);
    attr.push(along, across, hw, kind);
    return pos.length / 3 - 1;
  };
  for (const rd of w.roads) {
    const hw = rd.width / 2;
    const kind = rd.kind === "road" ? 1 : rd.kind === "track" ? 2 : 0;
    c.set(ROAD[rd.kind]);
    // Resample every 4 m so the surface follows the ground.
    const pts: [number, number][] = [];
    rd.points.forEach(([n, e], i) => {
      if (i === 0) return void pts.push([n, e]);
      const [pn, pe] = rd.points[i - 1]!;
      const steps = Math.max(1, Math.ceil(Math.hypot(n - pn, e - pe) / 4));
      for (let s = 1; s <= steps; s++) pts.push([pn + ((n - pn) * s) / steps, pe + ((e - pe) * s) / steps]);
    });
    if (pts.length < 2) continue;
    let along = 0;
    let prev = -1;
    pts.forEach(([n, e], i) => {
      if (i > 0) along += Math.hypot(n - pts[i - 1]![0], e - pts[i - 1]![1]);
      const [an, ae] = pts[Math.max(i - 1, 0)]!;
      const [bn, be] = pts[Math.min(i + 1, pts.length - 1)]!;
      const len = Math.hypot(bn - an, be - ae) || 1;
      const [pn, pe] = [-(be - ae) / len, (bn - an) / len]; // across, to the left
      const l = vertex(n + pn * hw, e + pe * hw, along, -hw, hw, kind);
      vertex(n - pn * hw, e - pe * hw, along, hw, hw, kind);
      if (prev >= 0) idx.push(prev, l, prev + 1, prev + 1, l, l + 1); // facing up
      prev = l;
    });
    // Round ends, so roads that meet join without notches.
    for (const [i, total] of [
      [0, 0],
      [pts.length - 1, along],
    ] as const) {
      const [n, e] = pts[i]!;
      const centre = vertex(n, e, total, 0, hw, kind);
      const SIDES = 12;
      for (let s = 0; s <= SIDES; s++) {
        const a = (s / SIDES) * 2 * Math.PI;
        const v = vertex(n + Math.cos(a) * hw, e + Math.sin(a) * hw, total, Math.sin(a) * hw, hw, kind);
        if (s > 0) idx.push(centre, v, v - 1);
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  g.setAttribute("aRoad", new THREE.Float32BufferAttribute(attr, 4));
  g.setIndex(idx);
  g.computeVertexNormals();
  const mesh = new THREE.Mesh(g, roadMaterial());
  mesh.receiveShadow = true;
  mesh.name = "roads";
  return mesh;
}

/** A wind turbine: a tapering white tower, the nacelle, the hub and three blades, the rotor
 * facing the wind (``heading``). */
function turbine(s: Structure): THREE.Group {
  const g = new THREE.Group();
  const white = new THREE.MeshLambertMaterial({ color: "#eef0f2" });
  const hub = s.hub ?? s.height * 0.65;
  const r = (s.rotor ?? 90) / 2;
  const tower = new THREE.Mesh(new THREE.CylinderGeometry(1.4, 2.3, hub, 12), white);
  tower.position.set(0, hub / 2, 0);
  const facing = s.heading ?? 0;
  const f = new THREE.Vector3(Math.sin(facing), 0, -Math.cos(facing)); // (n, e) → (x = e, z = -n)
  const nacelle = new THREE.Mesh(new THREE.BoxGeometry(3.4, 3.6, 11), white);
  nacelle.position.set(-f.x * 2, hub, -f.z * 2);
  nacelle.lookAt(new THREE.Vector3(f.x * 10, hub, f.z * 10));
  const rotor = new THREE.Group();
  rotor.position.set(f.x * 4.5, hub, f.z * 4.5);
  rotor.lookAt(new THREE.Vector3(f.x * 20, hub, f.z * 20));
  const outline = [[-1.6, 1.5], [1.6, 1.5], [0.4, r], [-0.4, r]].map(([x, y]) => new THREE.Vector2(x, y));
  const blade = new THREE.ShapeGeometry(new THREE.Shape(outline));
  const bladeMat = new THREE.MeshLambertMaterial({ color: "#f6f7f8", side: THREE.DoubleSide });
  const phase = ((s.n * 7.13 + s.e * 3.71) % 120) * (Math.PI / 180);
  const spinner = new THREE.Group(); // the view turns it (about 15 rpm)
  spinner.userData.spin = phase;
  for (let k = 0; k < 3; k++) {
    const b = new THREE.Mesh(blade, bladeMat);
    b.rotation.z = (k * 2 * Math.PI) / 3;
    b.castShadow = true;
    spinner.add(b);
  }
  spinner.rotation.z = phase;
  rotor.add(spinner, new THREE.Mesh(new THREE.SphereGeometry(1.6, 10, 8), white));
  tower.castShadow = nacelle.castShadow = true;
  g.add(tower, nacelle, rotor);
  g.position.set(s.e, s.base, -s.n);
  return g;
}

/** A lattice pylon: a tapering four-sided tower with three cross-arms a side and a peak. */
function pylon(s: Structure): THREE.Group {
  const g = new THREE.Group();
  const steel = new THREE.MeshLambertMaterial({ color: "#8e939a", wireframe: true });
  const solid = new THREE.MeshLambertMaterial({ color: "#8e939a" });
  const body = new THREE.Mesh(new THREE.CylinderGeometry(1.6 * Math.SQRT2, s.radius * Math.SQRT2, 45, 4, 6, true), steel);
  body.rotation.y = Math.PI / 4;
  body.position.y = 22.5;
  const peak = new THREE.Mesh(new THREE.ConeGeometry(1.6 * Math.SQRT2, s.height - 45, 4, 1, true), steel);
  peak.rotation.y = Math.PI / 4;
  peak.position.y = 45 + (s.height - 45) / 2;
  g.add(body, peak);
  for (const [z, half] of [[29, 10.5], [37, 11.5], [45, 9.5]] as const) {
    const arm = new THREE.Mesh(new THREE.BoxGeometry(2 * half, 0.6, 1.2), solid);
    arm.position.y = z;
    g.add(arm);
  }
  g.rotation.y = -(s.heading ?? 0) + Math.PI / 2; // arms across the line
  g.position.set(s.e, s.base, -s.n);
  return g;
}

/** A natural-draught cooling tower: a concrete hyperboloid shell, open at the top. */
function coolingTower(s: Structure): THREE.Mesh {
  const [H, rb] = [s.height, s.radius];
  const [rt, zt] = [0.63 * rb, 0.78 * H];
  const c = zt / Math.sqrt((rb / rt) ** 2 - 1);
  const pts = Array.from({ length: 25 }, (_, k) => {
    const z = (k / 24) * H;
    return new THREE.Vector2(rt * Math.sqrt(1 + ((z - zt) / c) ** 2), z);
  });
  const m = new THREE.Mesh(new THREE.LatheGeometry(pts, 40), new THREE.MeshLambertMaterial({ color: "#d9d6cf", side: THREE.DoubleSide }));
  m.position.set(s.e, s.base, -s.n);
  return m;
}

/** Every wire (conductors and mast stays) as thin lines. */
export function wires(w: WorldPayload): THREE.LineSegments | null {
  const ws = w.wires ?? [];
  if (ws.length === 0) return null;
  const pos: number[] = [];
  for (const x of ws) {
    for (let k = 0; k + 1 < x.points.length; k++) {
      const [na, ea, za] = x.points[k]!;
      const [nb, eb, zb] = x.points[k + 1]!;
      pos.push(ea, za, -na, eb, zb, -nb);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: "#3d4146" }));
}

/** Masts (a slender lattice with a warning light), silos, wind turbines, pylons and cooling
 * towers. */
export function structures(w: WorldPayload): THREE.Group {
  const g = new THREE.Group();
  for (const s of w.structures) {
    if (s.kind === "turbine") {
      g.add(turbine(s));
    } else if (s.kind === "pylon") {
      g.add(pylon(s));
    } else if (s.kind === "cooling_tower") {
      g.add(coolingTower(s));
    } else if (s.kind === "mast") {
      const mast = new THREE.Mesh(
        new THREE.CylinderGeometry(0.35, 0.9, s.height, 4, 8),
        // A tall guyed mast is a red and white lattice; a farm mast a plain pole.
        new THREE.MeshLambertMaterial({ color: s.height > 100 ? "#d9473b" : "#c9ccd0", wireframe: s.height > 100 }),
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
