// The stylised 3D view (docs/07 §8): one WebGL canvas behind the page, drawn twice per frame
// into two panels: a chase camera behind the aircraft, and a fixed top view of the whole
// map. Everything the simulator can touch (terrain, hedges, trees, buildings) is drawn
// exactly where it is, so what the views show is what the aircraft flies through.
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { Line2 } from "three/examples/jsm/lines/Line2.js";
import { LineGeometry } from "three/examples/jsm/lines/LineGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { MapControls } from "three/examples/jsm/controls/MapControls.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { PlanReport, SiteLayers } from "../live/client";
import { sample, type Sample } from "../replay/interp";
import type { Replay } from "../replay/types";
import { CATEGORIES, categoryOf } from "../scene/categories";
import type { Layers } from "../state/store";
import { rWorldFromModel } from "./frame";
import { HAZE, Look } from "./look";
import { Post } from "./post";
import { type LevelAt, Traffic } from "./traffic";
import { groundMaterial } from "./materials";
import { buildings, hedges, outskirts, railways, roadSurfaces, skirt, structures, terrainGeometry, trees, waterSurfaces, wires } from "./meshes";
import { type OverlayState, paintClasses, paintGround, paintOverlay } from "./paint";
import { TreeField } from "./trees3d";
import { flatWorld, Ground, type WorldPayload } from "./payload";

export type ChaseMode = "chase" | "orbit" | "free";

const MODEL_URI = "/models/calibration_uav.glb";
const TRAIL_S = 120;
const TRAIL_STRIDE = 5; // 50 Hz → 10 Hz
const SKY = new THREE.Color("#bcd7ee");
const GAP = new THREE.Color("#0b0e12"); // between the panels
// Places named on the map (the rest stay in the gazetteer, for plans).
const MAP_LABEL_KINDS = new Set([
  ...["home", "village", "wood", "farm", "church", "mast", "airstrip", "river", "campus"],
  ...["town", "landmark", "power station", "solar farm", "wind farm"], // baked worlds (docs/07 §14)
]);
const LAYER_CHASE = 1; // the aircraft at its real size: chase view only
const LAYER_TOP = 2; // symbols sized in pixels: top view only

interface Prepared {
  replay: Replay;
  pts: number[]; // x, y, z every TRAIL_STRIDE samples
  cat: number[];
}

interface Label {
  text: string;
  cls: string;
  pos: THREE.Vector3;
  views: ("chase" | "top")[];
  grid?: boolean; // a grid coordinate: shown with the grid layer
  els: Partial<Record<"chase" | "top", HTMLDivElement>>;
}

function line(colour: string, width: number, dashed = false): Line2 {
  const mat = new LineMaterial({ color: colour, linewidth: width, dashed, dashSize: 14, gapSize: 9, worldUnits: false });
  mat.toneMapped = false; // symbols keep their colours under the cinematic look's grading
  const l = new Line2(new LineGeometry(), mat);
  l.frustumCulled = false;
  l.visible = false;
  return l;
}

function setLine(l: Line2, pts: number[], colours?: number[]): void {
  if (pts.length < 6) {
    l.visible = false;
    return;
  }
  const g = new LineGeometry();
  g.setPositions(pts);
  if (colours) g.setColors(colours);
  l.geometry.dispose();
  l.geometry = g;
  l.computeLineDistances();
  l.visible = true;
}

function onLayer(o: THREE.Object3D, layer: number): void {
  o.traverse((c) => c.layers.set(layer));
}

/** The extent to show for a site without a world. */
export function extentOf(replay: Replay | null, site: SiteLayers | null): [number, number, number, number] {
  const pts: [number, number][] = [[0, 0]];
  site?.geofence.forEach((p) => pts.push(p));
  replay?.manifest.site.safe_zones.forEach((z) => z.polygon_ne_m.forEach((p) => pts.push(p)));
  replay?.manifest.mission.waypoints_ne_alt_m.forEach(([n, e]) => pts.push([n, e]));
  const ns = pts.map((p) => p[0]);
  const es = pts.map((p) => p[1]);
  const cn = (Math.min(...ns) + Math.max(...ns)) / 2;
  const ce = (Math.min(...es) + Math.max(...es)) / 2;
  const half = Math.max(Math.max(...ns) - Math.min(...ns), Math.max(...es) - Math.min(...es)) / 2 + 600;
  return [cn - half, ce - half, cn + half, ce + half];
}

export class WorldView {
  readonly renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  readonly chaseCam = new THREE.PerspectiveCamera(55, 1, 0.5, 30000);
  readonly topCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 10, 20000);
  private worldGroup = new THREE.Group();
  private world: WorldPayload | null = null;
  private ground: Ground | null = null;
  private overlay: { canvas: HTMLCanvasElement; tex: THREE.CanvasTexture; key: string } | null = null;
  private aircraft = new THREE.Group();
  private ghostCraft = new THREE.Group();
  private marker: THREE.Mesh;
  private ghostMarker: THREE.Mesh;
  private trail = line("#ffffff", 3);
  private ghostTrail = line("#ffffff", 2);
  private mission = line("#ffffff", 1.5, true);
  private proposal = line("#4cc9f0", 3, true);
  private active = line("#ffffff", 2);
  private failing = new THREE.Points(
    new THREE.BufferGeometry(),
    new THREE.PointsMaterial({ size: 6, sizeAttenuation: false, vertexColors: true, depthTest: false }),
  );
  private pin = new THREE.Group();
  private main: Prepared | null = null;
  private ghost: Prepared | null = null;
  private site: SiteLayers | null = null;
  private plans: { proposal: PlanReport | null; active: PlanReport | null } = { proposal: null, active: null };
  private labels: Label[] = [];
  private hosts: Partial<Record<"chase" | "top", HTMLElement>> = {};
  private trailKey = "";
  private trailAt = -1;
  private smoothYaw: number | null = null;
  private freeControls: OrbitControls | null = null;
  private freeOffset = new THREE.Vector3(-60, 40, 60);
  private topControls: MapControls | null = null;
  private fitKey = "";
  private loader = new GLTFLoader();
  private modelUri = "";
  private worldKey = "";
  readonly thin: number;
  /** The cinematic look (docs/07 §15): sky, sunlight and shadows, haze, detailed trees,
   * facades, water. Off for slow machines (``?quality=low``). */
  readonly cinematic: boolean;
  /** Film mode (``&film=1``): the picture alone, no map overlay, labels or route lines. */
  readonly film: boolean;
  private look: Look | null = null;
  private post: Post | null = null;
  private treeField: TreeField | null = null;
  private spinners: THREE.Object3D[] = [];
  private water: THREE.Mesh | null = null;
  private shot: { eye: THREE.Vector3; at: THREE.Vector3 } | null = null;
  private traffic: Traffic | null = null;
  private clock: number | null = null;
  last: Sample | null = null;
  private showGrid = true;

  constructor(canvas: HTMLCanvasElement, opts: { thin?: number; film?: boolean } = {}) {
    this.thin = opts.thin ?? 1;
    this.film = opts.film ?? false;
    this.cinematic = this.thin <= 1;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, this.thin > 1 ? 1 : 2));
    this.renderer.setClearColor(SKY);
    this.scene.background = SKY;
    if (this.cinematic) {
      this.look = new Look(this.renderer, this.scene);
      this.post = new Post(this.renderer, this.scene, this.chaseCam);
    } else {
      this.scene.add(new THREE.HemisphereLight("#eaf4ff", "#5d6e45", 1.15));
      const sun = new THREE.DirectionalLight("#fff4df", 1.6);
      sun.position.set(-0.6, 1.0, 0.45).multiplyScalar(1000);
      this.scene.add(sun);
    }
    this.scene.add(this.worldGroup);
    this.chaseCam.layers.enable(LAYER_CHASE);
    this.topCam.layers.enable(LAYER_TOP);
    this.topCam.up.set(0, 0, -1); // north up

    // Aircraft symbols for the top view: arrows of a fixed pixel size.
    const arrow = new THREE.Shape([
      new THREE.Vector2(0, 1),
      new THREE.Vector2(0.62, -0.7),
      new THREE.Vector2(0, -0.35),
      new THREE.Vector2(-0.62, -0.7),
    ]);
    const ag = new THREE.ShapeGeometry(arrow);
    ag.rotateX(-Math.PI / 2); // in the ground plane, pointing north (−z)
    this.marker = new THREE.Mesh(ag, new THREE.MeshBasicMaterial({ color: "#ffb24c", depthTest: false, toneMapped: false }));
    const outline = new THREE.LineLoop(
      new THREE.BufferGeometry().setFromPoints(arrow.getPoints().map((p) => new THREE.Vector3(p.x, 0.01, -p.y))),
      new THREE.LineBasicMaterial({ color: "#000000", depthTest: false }),
    );
    this.marker.add(outline);
    this.marker.renderOrder = 10;
    this.ghostMarker = new THREE.Mesh(
      ag,
      new THREE.MeshBasicMaterial({ color: "#ffffff", transparent: true, opacity: 0.6, depthTest: false }),
    );
    onLayer(this.marker, LAYER_TOP);
    onLayer(this.ghostMarker, LAYER_TOP);
    this.ghostMarker.renderOrder = 9;

    const pinMat = new THREE.MeshBasicMaterial({ color: "#ffffff" });
    const stick = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.25, 12, 6), pinMat);
    stick.position.y = 6;
    const head = new THREE.Mesh(new THREE.SphereGeometry(1.4, 12, 8), new THREE.MeshBasicMaterial({ color: "#ff5c5c" }));
    head.position.y = 12.5;
    this.pin.add(stick, head);
    this.pin.visible = false;
    this.failing.renderOrder = 11;
    this.scene.add(
      this.aircraft, this.ghostCraft, this.marker, this.ghostMarker, this.trail, this.ghostTrail,
      this.mission, this.proposal, this.active, this.failing, this.pin,
    ); // prettier-ignore
    (this.ghostTrail.material as LineMaterial).opacity = 0.4;
    (this.ghostTrail.material as LineMaterial).transparent = true;
    this.setModel(null);
  }

  /** Where the two views' labels live (and the mouse controls for the free and top views). */
  setHosts(chase: HTMLElement | null, top: HTMLElement | null): void {
    this.hosts = { chase: chase ?? undefined, top: top ?? undefined };
    this.freeControls?.dispose();
    this.topControls?.dispose();
    this.freeControls = chase ? new OrbitControls(this.chaseCam, chase) : null;
    if (this.freeControls) {
      this.freeControls.enabled = false;
      this.freeControls.enableDamping = true;
    }
    if (top) {
      const mc = new MapControls(this.topCam, top);
      mc.enableRotate = false;
      mc.screenSpacePanning = true;
      mc.zoomToCursor = true;
      top.addEventListener("dblclick", () => (this.fitKey = ""));
      this.topControls = mc;
    }
    this.rebuildLabels();
  }

  dispose(): void {
    this.freeControls?.dispose();
    this.topControls?.dispose();
    this.clearWorld();
    this.look?.dispose();
    this.post?.dispose();
    for (const l of this.labels) for (const el of Object.values(l.els)) el?.remove();
    this.renderer.dispose();
  }

  // ------------------------------------------------------------------ world
  private clearWorld(): void {
    this.worldGroup.traverse((o) => {
      const m = o as THREE.Mesh;
      m.geometry?.dispose();
      const mats = Array.isArray(m.material) ? m.material : m.material ? [m.material] : [];
      for (const mat of mats) {
        (mat as THREE.MeshLambertMaterial).map?.dispose();
        mat.dispose();
      }
    });
    this.worldGroup.clear();
    this.overlay = null;
    this.treeField = null;
    this.traffic = null; // its meshes went with the group
    this.spinners = [];
    this.water = null;
  }

  /** The world to draw; ``null`` with an extent draws open ground (a site without a world). */
  setWorld(w: WorldPayload | null, extent: [number, number, number, number], name = "site"): void {
    const world = w ?? flatWorld("flat", name, extent);
    const key = w ? `world:${w.id}` : `flat:${extent.map(Math.round).join(",")}`;
    if (key === this.worldKey) return;
    this.worldKey = key;
    this.clearWorld();
    this.world = world;
    this.ground = new Ground(world.dtm);
    const size = w ? 4096 : 2048;
    // With water surfaces, the painted water is only seen at the edges: as shallows.
    const ground = paintGround(world, size, this.cinematic ? "#46696f" : undefined);
    if (!w) this.paintGrid(ground, world);
    const tex = new THREE.CanvasTexture(ground);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
    const geo = terrainGeometry(world);
    let groundMat: THREE.Material = new THREE.MeshLambertMaterial({ map: tex });
    if (this.cinematic && w) {
      const classes = new THREE.CanvasTexture(paintClasses(world, 2048));
      classes.magFilter = classes.minFilter = THREE.NearestFilter;
      classes.generateMipmaps = false;
      groundMat = groundMaterial(tex, classes);
    }
    const terrain = new THREE.Mesh(geo, groundMat);
    terrain.receiveShadow = this.cinematic;
    this.worldGroup.add(terrain);
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 2048;
    const otex = new THREE.CanvasTexture(canvas);
    otex.colorSpace = THREE.SRGBColorSpace;
    const over = new THREE.Mesh(
      geo,
      new THREE.MeshBasicMaterial({
        map: otex,
        transparent: true,
        toneMapped: false,
        depthWrite: false,
        polygonOffset: true,
        polygonOffsetFactor: -4,
        polygonOffsetUnits: -4,
      }),
    );
    over.renderOrder = 1;
    over.userData.overlay = true;
    this.worldGroup.add(over);
    this.overlay = { canvas, tex: otex, key: "" };
    // The ground past the map's edge (a city: the outskirts, once the water is known).
    const city = world.plots[0]?.kind === "settlement";
    if (!(city && this.cinematic)) this.worldGroup.add(skirt(world, city ? "#a2a5a7" : "#9bc078"));
    const h = hedges(world, this.ground);
    if (h) {
      h.traverse((o) => (o.receiveShadow = this.cinematic));
      this.worldGroup.add(h);
    }
    if (world.trees.length) {
      if (this.cinematic) {
        this.treeField = new TreeField(world);
        this.worldGroup.add(this.treeField.group);
      } else this.worldGroup.add(trees(world, this.thin));
    }
    const b = buildings(world, this.cinematic);
    if (b) this.worldGroup.add(b);
    const st = structures(world);
    st.traverse((o) => {
      if (this.cinematic && (o as THREE.Mesh).isMesh) o.castShadow = o.receiveShadow = true;
      if (o.userData.spin !== undefined) this.spinners.push(o);
    });
    this.worldGroup.add(st);
    const lines = wires(world);
    if (lines) this.worldGroup.add(lines);
    if (this.cinematic) {
      this.water = waterSurfaces(world, this.ground);
      if (this.water) this.worldGroup.add(this.water);
      const roads = roadSurfaces(world, this.ground);
      if (roads) this.worldGroup.add(roads);
      const rails = railways(world, this.ground);
      if (rails) this.worldGroup.add(rails);
      const levelAt = (this.water?.userData.levelAt as LevelAt | undefined) ?? null;
      if (city) this.worldGroup.add(outskirts(world, this.ground, levelAt));
      this.traffic = new Traffic(world, this.ground, levelAt);
      this.worldGroup.add(this.traffic.group);
    }
    this.scene.fog = null;
    this.fitKey = "";
    this.rebuildLabels();
  }

  private paintGrid(canvas: HTMLCanvasElement, w: WorldPayload): void {
    const ctx = canvas.getContext("2d")!;
    const [n0, e0, n1, e1] = w.bounds;
    const px = canvas.width / (e1 - e0);
    ctx.strokeStyle = "rgba(255,255,255,0.25)";
    ctx.lineWidth = 1;
    for (let e = Math.ceil(e0 / 100) * 100; e < e1; e += 100) {
      ctx.beginPath();
      ctx.moveTo((e - e0) * px, 0);
      ctx.lineTo((e - e0) * px, canvas.height);
      ctx.stroke();
    }
    for (let n = Math.ceil(n0 / 100) * 100; n < n1; n += 100) {
      ctx.beginPath();
      ctx.moveTo(0, (n1 - n) * px);
      ctx.lineTo(canvas.width, (n1 - n) * px);
      ctx.stroke();
    }
  }

  groundAt(n: number, e: number): number {
    return this.ground?.at(n, e) ?? 0;
  }

  // ------------------------------------------------------------------ aircraft
  setModel(uri: string | null): void {
    const want = uri ?? MODEL_URI;
    if (want === this.modelUri) return;
    this.modelUri = want;
    this.loader.load(
      want,
      (gltf) => {
        if (this.modelUri !== want) return;
        const make = (ghost: boolean) => {
          const m = gltf.scene.clone(true);
          m.traverse((o) => {
            const mesh = o as THREE.Mesh;
            if (!mesh.isMesh) return;
            const mat = (mesh.material as THREE.MeshStandardMaterial).clone();
            if (ghost) {
              mat.transparent = true;
              mat.opacity = 0.45;
            } else mat.color?.lerp(new THREE.Color("#ffb24c"), 0.35);
            mesh.material = mat;
          });
          onLayer(m, LAYER_CHASE);
          return m;
        };
        this.aircraft.clear();
        this.ghostCraft.clear();
        this.aircraft.add(make(false));
        this.ghostCraft.add(make(true));
      },
      undefined,
      () => undefined,
    );
  }

  private prepare(replay: Replay, from?: Prepared): Prepared {
    const c = replay.cols;
    const { mode: modes, phase: phases } = replay.manifest.enums;
    const pts = from ? from.pts : [];
    const cat = from ? from.cat : [];
    for (let i = (pts.length / 3) * TRAIL_STRIDE; i < replay.n; i += TRAIL_STRIDE) {
      pts.push(c["truth.pos_e"]![i]!, c["truth.alt_agl"]![i]!, -c["truth.pos_n"]![i]!);
      cat.push(categoryOf(phases[c["cont.phase"]![i]!] ?? "", modes[c.mode![i]!] ?? ""));
    }
    return { replay, pts, cat };
  }

  setReplays(main: Replay | null, ghost: Replay | null): void {
    if (main && this.main && this.main.replay !== main && this.main.replay.id === main.id) {
      this.main = this.prepare(main, this.main);
    } else if (main !== this.main?.replay) {
      this.main = main ? this.prepare(main) : null;
      this.smoothYaw = null;
      if (main) this.buildMission(main);
    }
    this.ghost = ghost ? this.prepare(ghost) : null;
    this.trailAt = -1;
    this.rebuildLabels();
  }

  private buildMission(r: Replay): void {
    const wps = r.manifest.mission.waypoints_ne_alt_m;
    const pts = wps.flatMap(([n, e, h]) => [e, h, -n]);
    if (r.manifest.mission.on_complete === "loop" && wps.length > 1) pts.push(...pts.slice(0, 3));
    setLine(this.mission, pts);
  }

  setSite(site: SiteLayers | null): void {
    if (site === this.site) return;
    this.site = site;
    if (this.overlay) this.overlay.key = "";
    this.rebuildLabels();
  }

  setPlans(proposal: PlanReport | null, active: PlanReport | null): void {
    if (proposal === this.plans.proposal && active === this.plans.active) return;
    this.plans = { proposal, active };
    const track = (r: PlanReport | null) => (r ? r.preview.flatMap(([n, e, h]) => [e, h, -n]) : []);
    setLine(this.active, track(active));
    setLine(this.proposal, track(proposal));
    const pos: number[] = [];
    const col: number[] = [];
    for (const c of proposal?.checks ?? []) {
      if (c.ok) continue;
      const colour = new THREE.Color(c.hard ? "#e5484d" : "#ffc93c");
      for (const [n, e] of c.where) {
        pos.push(e, this.groundAt(n, e) + 2, -n);
        col.push(colour.r, colour.g, colour.b);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    this.failing.geometry.dispose();
    this.failing.geometry = g;
    if (this.overlay) this.overlay.key = "";
  }

  // ------------------------------------------------------------------ labels
  private rebuildLabels(): void {
    for (const l of this.labels) for (const el of Object.values(l.els)) el?.remove();
    this.labels = [];
    const add = (text: string, cls: string, n: number, e: number, h: number, views: Label["views"]) =>
      this.labels.push({ text, cls, pos: new THREE.Vector3(e, h, -n), views, els: {} });
    const zones = this.main?.replay.manifest.site.safe_zones ?? [];
    for (const z of zones) {
      const [n, e] = z.polygon_ne_m.reduce(([a, b], [x, y]) => [a + x, b + y], [0, 0]).map((v) => v / z.polygon_ne_m.length) as [number, number];
      add(z.id, "lbl-zone", n, e, this.groundAt(n, e) + 3, ["chase", "top"]);
    }
    for (const p of this.site?.places ?? []) {
      if (!MAP_LABEL_KINDS.has(p.kind)) continue;
      const [n, e] = p.point;
      add(p.kind === "home" ? "⌂ home" : p.name, `lbl-place lbl-${p.kind}`, n, e, this.groundAt(n, e) + 25, p.kind === "home" ? ["chase", "top"] : ["top"]);
    }
    if (!this.site && this.world) {
      // A replay has no gazetteer: name the world's own features.
      const centre = (ring: [number, number][]) =>
        ring.reduce(([a, b], [n, e]) => [a + n / ring.length, b + e / ring.length], [0, 0]) as [number, number];
      const named: [string, [number, number]][] = [
        ...this.world.woods.filter((w) => w.name).map((w) => [w.name, centre(w.outer)] as [string, [number, number]]),
        ...this.world.plots
          .filter((p) => p.name && (p.kind === "settlement" || p.kind === "yard" || p.kind === "airstrip"))
          .map((p) => [p.name!, centre(p.outer)] as [string, [number, number]]),
        ...this.world.structures.filter((x) => x.kind === "mast").map((x) => [x.name, [x.n, x.e]] as [string, [number, number]]),
      ];
      for (const [name, [n, e]] of named) add(name, "lbl-place", n, e, this.groundAt(n, e) + 25, ["top"]);
    }
    if (this.world) {
      // Grid coordinates every 500 m along the map's west and south edges.
      const [n0, e0, n1, e1] = this.world.bounds;
      const fmt = (v: number, pos: string, neg: string) => (v === 0 ? "0" : `${Math.abs(v)} m ${v > 0 ? pos : neg}`);
      for (let v = Math.ceil(n0 / 500) * 500; v < n1; v += 500) {
        if (v > n0) this.labels.push({ text: fmt(v, "N", "S"), cls: "lbl-grid", pos: new THREE.Vector3(e0 + 15, 0, -v), views: ["top"], els: {}, grid: true });
      }
      for (let v = Math.ceil(e0 / 500) * 500; v < e1; v += 500) {
        if (v > e0) this.labels.push({ text: fmt(v, "E", "W"), cls: "lbl-grid lbl-grid-e", pos: new THREE.Vector3(v, 0, -n0 - 15), views: ["top"], els: {}, grid: true });
      }
    }
    const named = new Set<string>();
    for (const a of this.site?.no_fly ?? []) {
      const ring = a.polygons[0];
      if (!ring || named.has(a.name)) continue; // one label for a group (the wind turbines)
      named.add(a.name);
      const n = Math.max(...ring.map((p) => p[0]));
      const e = ring.reduce((s, p) => s + p[1], 0) / ring.length;
      const below = a.ceiling_m == null ? "" : ` below ${a.ceiling_m} m${a.ceiling_from === "centre" ? " over its foot" : ""}`;
      add(`${a.name} · no-fly${below}`, a.ceiling_m == null ? "lbl-nofly" : "lbl-nofly lbl-nofly-low", n, e, this.groundAt(n, e) + 30, ["top"]);
    }
    for (const view of ["chase", "top"] as const) {
      const host = this.hosts[view];
      if (!host) continue;
      for (const l of this.labels) {
        if (!l.views.includes(view)) continue;
        const el = document.createElement("div");
        el.className = `lbl ${l.cls}`;
        el.textContent = l.text;
        host.appendChild(el);
        l.els[view] = el;
      }
    }
  }

  private placeLabels(view: "chase" | "top", cam: THREE.Camera, w: number, h: number): void {
    const v = new THREE.Vector3();
    for (const l of this.labels) {
      const el = l.els[view];
      if (!el) continue;
      v.copy(l.pos).project(cam);
      const visible = v.z < 1 && Math.abs(v.x) < 1.05 && Math.abs(v.y) < 1.05 && (!l.grid || this.showGrid);
      el.style.display = visible ? "" : "none";
      if (visible) el.style.transform = `translate(${((v.x + 1) / 2) * w}px, ${((1 - v.y) / 2) * h}px) translate(-50%, -100%)`;
    }
  }

  // ------------------------------------------------------------------ per frame
  private place(obj: THREE.Object3D, mark: THREE.Object3D, s: Sample): THREE.Vector3 {
    const p = new THREE.Vector3(s.e, s.altAgl, -s.n);
    obj.position.copy(p);
    const r = rWorldFromModel(s.quat);
    obj.quaternion.setFromRotationMatrix(
      new THREE.Matrix4().set(...r[0], 0, ...r[1], 0, ...r[2], 0, 0, 0, 0, 1),
    );
    mark.position.set(p.x, p.y + 2, p.z);
    mark.rotation.set(0, -s.yaw, 0);
    return p;
  }

  private repaintOverlay(s: Sample | null, layers: Layers): void {
    if (!this.overlay || !this.world) return;
    const zones = this.main?.replay.manifest.site.safe_zones ?? [];
    this.showGrid = layers.grid;
    const key = `${layers.zones}|${layers.grid}|${s?.reachMask ?? 0}|${s?.zone ?? -1}|${s?.failed ?? false}`;
    if (key === this.overlay.key) return;
    this.overlay.key = key;
    const st: OverlayState = {
      showZones: layers.zones,
      grid: layers.grid,
      zones: zones.map((z, k) => ({
        id: z.id,
        polygon: z.polygon_ne_m,
        state: s && s.zone === k ? "selected" : s && (s.reachMask & (1 << k)) !== 0 ? "reachable" : "unreachable",
      })),
      noFly: (this.site?.no_fly ?? []).flatMap((a) => a.polygons.map((ring) => ({ ring, limited: a.ceiling_m != null }))),
      geofence: this.site?.geofence ?? null,
      proposal: this.plans.proposal?.preview.map(([n, e]) => [n, e] as [number, number]) ?? null,
      active: this.plans.active?.preview.map(([n, e]) => [n, e] as [number, number]) ?? null,
    };
    paintOverlay(this.overlay.canvas, this.world, st);
    this.overlay.tex.needsUpdate = true;
  }

  private rebuildTrails(time: number, layers: Layers): void {
    const build = (p: Prepared, l: Line2) => {
      const hz = p.replay.manifest.rates.viz_hz;
      const end = Math.min(Math.floor((time * hz) / TRAIL_STRIDE), p.pts.length / 3 - 1);
      const start = Math.max(0, end - Math.round((TRAIL_S * hz) / TRAIL_STRIDE));
      const pts = p.pts.slice(3 * start, 3 * (end + 1));
      const cols: number[] = [];
      const c = new THREE.Color();
      for (let i = start; i <= end; i++) {
        c.set(CATEGORIES[p.cat[i] ?? 0]!.css);
        cols.push(c.r, c.g, c.b);
      }
      setLine(l, pts, cols);
    };
    const key = `${layers.trail}|${layers.ghost}|${this.ghost ? 1 : 0}`;
    if (key === this.trailKey && Math.abs(time - this.trailAt) < 0.2) return;
    this.trailKey = key;
    this.trailAt = time;
    (this.trail.material as LineMaterial).vertexColors = true;
    (this.ghostTrail.material as LineMaterial).vertexColors = true;
    if (layers.trail && this.main) build(this.main, this.trail);
    else this.trail.visible = false;
    if (layers.trail && layers.ghost && this.ghost) build(this.ghost, this.ghostTrail);
    else this.ghostTrail.visible = false;
  }

  /** Called once per animation frame, before ``render``. */
  update(time: number, layers: Layers, mode: ChaseMode): Sample | null {
    const m = this.main;
    this.mission.visible = !!m && layers.mission && !this.plans.active;
    if (!m || m.replay.n === 0) {
      this.aircraft.visible = this.marker.visible = false;
      this.repaintOverlay(null, layers);
      return null;
    }
    const s = sample(m.replay, time);
    this.last = s;
    const pos = this.place(this.aircraft, this.marker, s);
    this.aircraft.visible = this.marker.visible = true;
    const g = this.ghost && layers.ghost ? sample(this.ghost.replay, time) : null;
    this.ghostCraft.visible = this.ghostMarker.visible = g !== null;
    if (g) this.place(this.ghostCraft, this.ghostMarker, g);
    this.repaintOverlay(s, layers);
    this.rebuildTrails(time, layers);

    // The end of the flight: a pin where it touched down, or what it hit.
    const end = m.replay.events.find((e) => e.type === "touchdown" || e.type === "collision");
    this.pin.visible = !!end && time >= end.time_us * 1e-6;
    if (end && this.pin.visible) {
      const [n, e] = end.payload.pos_ne_m as [number, number];
      this.pin.position.set(e, this.groundAt(n, e), -n);
      (this.pin.children[1] as THREE.Mesh).material = new THREE.MeshBasicMaterial({
        color: end.type === "collision" ? "#ff3b30" : "#38d16a",
      });
    }
    this.chaseCamera(pos, s, mode, time);
    return s;
  }

  /** The clock for what moves on its own (traffic, boats, turbines, water), in seconds:
   * ``null`` follows the wall clock; a recording sets it frame by frame. */
  setClock(t: number | null): void {
    this.clock = t;
  }

  /** A set shot for stills and video: the camera at ``eye`` (n, e, height above the datum)
   * looking at ``at``; ``null`` gives the follow view back. */
  setShot(eye: [number, number, number] | null, at?: [number, number, number]): void {
    this.shot = eye && at ? { eye: new THREE.Vector3(eye[1], eye[2], -eye[0]), at: new THREE.Vector3(at[1], at[2], -at[0]) } : null;
  }

  private chaseCamera(pos: THREE.Vector3, s: Sample, mode: ChaseMode, time: number): void {
    const cam = this.chaseCam;
    if (this.shot) {
      cam.position.copy(this.shot.eye);
      cam.lookAt(this.shot.at);
      return;
    }
    if (this.freeControls) this.freeControls.enabled = mode === "free";
    if (mode === "free" && this.freeControls) {
      // The mouse orbits the aircraft; the camera follows it.
      this.freeControls.target.copy(pos);
      cam.position.copy(pos).add(this.freeOffset);
      this.freeControls.update();
      this.freeOffset.copy(cam.position).sub(pos);
      return;
    }
    if (this.smoothYaw === null) this.smoothYaw = s.yaw;
    let d = s.yaw - this.smoothYaw;
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    this.smoothYaw += 0.05 * d;
    const yaw = mode === "orbit" ? 0.12 * time : this.smoothYaw;
    const back = mode === "orbit" ? 170 : 13;
    const up = mode === "orbit" ? 70 : 3.2;
    const fwd = new THREE.Vector3(Math.sin(yaw), 0, -Math.cos(yaw));
    cam.position.copy(pos).addScaledVector(fwd, -back);
    cam.position.y += up;
    const floor = this.groundAt(-cam.position.z, cam.position.x) + 2;
    if (cam.position.y < floor) cam.position.y = floor;
    cam.lookAt(pos.clone().addScaledVector(fwd, mode === "orbit" ? 0 : 30).add(new THREE.Vector3(0, -2, 0)));
  }

  private fitTop(w: number, h: number): void {
    if (!this.world) return;
    const key = `${w}x${h}|${this.worldKey}|${this.site ? 1 : 0}`;
    if (key === this.fitKey) return;
    this.fitKey = key;
    const pts = this.site?.geofence ?? [];
    let [n0, e0, n1, e1] = this.world.bounds;
    if (pts.length) {
      n0 = Math.min(...pts.map((p) => p[0])) - 120;
      n1 = Math.max(...pts.map((p) => p[0])) + 120;
      e0 = Math.min(...pts.map((p) => p[1])) - 120;
      e1 = Math.max(...pts.map((p) => p[1])) + 120;
    }
    const cn = (n0 + n1) / 2;
    const ce = (e0 + e1) / 2;
    const aspect = w / Math.max(h, 1);
    const halfH = Math.max((n1 - n0) / 2, (e1 - e0) / 2 / aspect);
    const cam = this.topCam;
    cam.left = -halfH * aspect;
    cam.right = halfH * aspect;
    cam.top = halfH;
    cam.bottom = -halfH;
    cam.zoom = 1;
    cam.position.set(ce, 4000, -cn);
    cam.lookAt(ce, 0, -cn);
    cam.updateProjectionMatrix();
    if (this.topControls) {
      this.topControls.target.set(ce, 0, -cn);
      this.topControls.update();
    }
  }

  /** Draw both views into their panels' rectangles (CSS pixels). */
  render(chase: DOMRect | null, top: DOMRect | null): void {
    const r = this.renderer;
    const canvas = r.domElement;
    const W = canvas.clientWidth;
    const H = canvas.clientHeight;
    if (canvas.width !== Math.floor(W * r.getPixelRatio()) || canvas.height !== Math.floor(H * r.getPixelRatio())) {
      r.setSize(W, H, false);
    }
    const box = canvas.getBoundingClientRect();
    r.setScissorTest(false);
    r.setClearColor(GAP);
    r.clear();
    if (!this.world) return;
    r.setScissorTest(true);
    r.setClearColor(SKY);
    const lines = [this.trail, this.ghostTrail, this.mission, this.proposal, this.active];
    if (this.film) {
      for (const l of lines) l.visible = false;
      this.worldGroup.children.forEach((o) => {
        if (o.userData.overlay) o.visible = false;
      });
    }
    const view = (rect: DOMRect, cam: THREE.Camera, name: "chase" | "top") => {
      const x = rect.left - box.left;
      const y = box.bottom - rect.bottom;
      r.setViewport(x, y, rect.width, rect.height);
      r.setScissor(x, y, rect.width, rect.height);
      r.clear();
      for (const l of lines) (l.material as LineMaterial).resolution.set(rect.width, rect.height);
      if (!this.film) this.placeLabels(name, cam, rect.width, rect.height);
      if (name === "chase" && this.post) this.post.render(rect.width, rect.height);
      else r.render(this.scene, cam);
    };
    // Moving parts: the turbines turn (about 15 rpm), the water flows, the traffic drives.
    const now = this.clock ?? performance.now() / 1000;
    this.traffic?.update(now);
    for (const sp of this.spinners) sp.rotation.z = (sp.userData.spin as number) - 1.6 * now;
    const flow = (this.water?.material as THREE.MeshStandardMaterial | undefined)?.userData.flow as THREE.Texture | undefined;
    if (flow) flow.offset.set(0.004 * now, 0.0025 * now);
    if (chase && chase.width > 0) {
      this.chaseCam.aspect = chase.width / chase.height;
      this.chaseCam.updateProjectionMatrix();
      if (this.look) {
        const target = this.shot ? this.shot.at : this.freeControls?.enabled ? this.freeControls.target : this.aircraft.position;
        this.look.follow(target, this.chaseCam.position);
        this.treeField?.update(this.chaseCam.position);
        r.shadowMap.autoUpdate = true;
        // Aerial perspective: a few per cent at a kilometre, a quarter at three, half at five.
        this.scene.fog = new THREE.FogExp2(HAZE, 0.00017);
      } else this.scene.fog = new THREE.Fog(SKY, 1800, 9000);
      view(chase, this.chaseCam, "chase");
    }
    if (top && top.width > 0) {
      this.fitTop(top.width, top.height);
      this.topControls?.update();
      // Pixel-sized symbols: the arrow is 24 px long whatever the zoom.
      const perPx = (this.topCam.right - this.topCam.left) / this.topCam.zoom / top.width;
      this.marker.scale.setScalar(12 * perPx);
      this.ghostMarker.scale.setScalar(12 * perPx);
      this.scene.fog = null;
      r.shadowMap.autoUpdate = false; // the chase view's shadows serve here too
      view(top, this.topCam, "top");
    }
    r.setScissorTest(false);
  }
}
