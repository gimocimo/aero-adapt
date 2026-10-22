// The ground's look, painted on canvases draped over the terrain (docs/07 §8): land cover
// (fields, woods, village, roads, river) once, and an overlay (safe zones, no-fly areas,
// geofence, plan tracks) repainted whenever its state changes.
import type { Ring, WorldPayload } from "./payload";

export const CROP: Record<string, string> = {
  wheat: "#d7c06c",
  barley: "#e2d595",
  oilseed: "#ecdc4a",
  pasture: "#88c25c",
  meadow: "#a8d374",
  ploughed: "#9c7a56",
  stubble: "#d9cb94",
  rough: "#9bc078",
  settlement: "#c9c7b4",
  yard: "#b3ab9c",
  green: "#8ccf68",
  airstrip: "#79bd58",
};
const ARABLE = new Set(["wheat", "barley", "oilseed", "ploughed", "stubble"]);
const WOOD_FLOOR = { broadleaf: "#3f6f37", conifer: "#2f5a35" };
const ROAD = { road: "#5c5f64", lane: "#686b70", street: "#6c6e72", track: "#b39f7a" };
const WATER = "#4389cf";
const BANK = "#6f9a52";
const HEDGE = "#356a2f";

/** Pixel mapping of a world's bounds onto a square canvas: north up. */
export class Mapping {
  constructor(
    readonly bounds: [number, number, number, number],
    readonly size: number,
  ) {}

  x(e: number): number {
    const [, e0, , e1] = this.bounds;
    return ((e - e0) / (e1 - e0)) * this.size;
  }

  y(n: number): number {
    const [n0, , n1] = this.bounds;
    return ((n1 - n) / (n1 - n0)) * this.size;
  }

  /** Pixels per metre (east). */
  get scale(): number {
    const [, e0, , e1] = this.bounds;
    return this.size / (e1 - e0);
  }
}

function path(m: Mapping, ring: Ring | [number, number][]): Path2D {
  const p = new Path2D();
  const rings = Array.isArray(ring) ? [ring] : [ring.outer, ...ring.holes];
  for (const r of rings) {
    r.forEach(([n, e], i) => (i === 0 ? p.moveTo(m.x(e), m.y(n)) : p.lineTo(m.x(e), m.y(n))));
    p.closePath();
  }
  return p;
}

function line(m: Mapping, pts: [number, number][]): Path2D {
  const p = new Path2D();
  pts.forEach(([n, e], i) => (i === 0 ? p.moveTo(m.x(e), m.y(n)) : p.lineTo(m.x(e), m.y(n))));
  return p;
}

/** Long-axis angle (canvas radians) of an outline, for crop rows. */
function axisAngle(m: Mapping, pts: [number, number][]): number {
  let cx = 0;
  let cy = 0;
  for (const [n, e] of pts) {
    cx += m.x(e);
    cy += m.y(n);
  }
  cx /= pts.length;
  cy /= pts.length;
  let sxx = 0;
  let syy = 0;
  let sxy = 0;
  for (const [n, e] of pts) {
    const dx = m.x(e) - cx;
    const dy = m.y(n) - cy;
    sxx += dx * dx;
    syy += dy * dy;
    sxy += dx * dy;
  }
  return 0.5 * Math.atan2(2 * sxy, sxx - syy);
}

function shade(hex: string, f: number): string {
  const v = parseInt(hex.slice(1), 16);
  const c = (s: number) => Math.max(0, Math.min(255, Math.round(((v >> s) & 255) * f)));
  return `rgb(${c(16)},${c(8)},${c(0)})`;
}

function rows(ctx: CanvasRenderingContext2D, colour: string, spacing: number): CanvasPattern | null {
  const s = Math.max(Math.round(spacing), 3);
  const c = document.createElement("canvas");
  c.width = s;
  c.height = s;
  const g = c.getContext("2d")!;
  g.fillStyle = colour;
  g.fillRect(0, 0, s, s);
  g.fillStyle = shade(colour, 0.9);
  g.fillRect(0, 0, s, Math.max(1, Math.round(s / 3)));
  return ctx.createPattern(c, "repeat");
}

/** Land cover, once per world. */
export function paintGround(w: WorldPayload, size: number): HTMLCanvasElement {
  const m = new Mapping(w.bounds, size);
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  // Open ground, tinted by height (lowland green to upland khaki); fields paint over it.
  const tint = elevationTint(w);
  if (tint) {
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(tint, 0, 0, size, size);
  } else {
    ctx.fillStyle = CROP.rough!;
    ctx.fillRect(0, 0, size, size);
  }
  const px = m.scale;

  for (const p of w.plots) {
    const colour = CROP[p.cover] ?? CROP.rough!;
    const shape = path(m, p);
    if (ARABLE.has(p.cover)) {
      const pat = rows(ctx, colour, 6 * px);
      if (pat) pat.setTransform(new DOMMatrix().rotate((axisAngle(m, p.outer) * 180) / Math.PI));
      ctx.fillStyle = pat ?? colour;
    } else ctx.fillStyle = colour;
    ctx.fill(shape, "evenodd");
    if (p.kind === "airstrip") {
      // Mown stripes and the runway's centre line.
      ctx.save();
      ctx.clip(shape);
      const pat = rows(ctx, CROP.airstrip!, 12 * px);
      if (pat) pat.setTransform(new DOMMatrix().rotate((axisAngle(m, p.outer) * 180) / Math.PI + 90));
      ctx.fillStyle = pat ?? CROP.airstrip!;
      ctx.fill(shape);
      ctx.restore();
    }
  }
  // Village gardens: a few greens round the houses.
  for (const p of w.plots.filter((q) => q.kind === "green")) {
    ctx.fillStyle = CROP.green!;
    ctx.fill(path(m, p));
  }
  for (const wd of w.woods) {
    ctx.fillStyle = WOOD_FLOOR[wd.kind];
    ctx.fill(path(m, wd), "evenodd");
  }
  if (w.river) {
    const r = line(m, w.river.points);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.strokeStyle = BANK;
    ctx.lineWidth = (w.river.width + 10) * px;
    ctx.stroke(r);
    ctx.strokeStyle = WATER;
    ctx.lineWidth = w.river.width * px;
    ctx.stroke(r);
  }
  for (const rd of w.roads) {
    const p = line(m, rd.points);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.strokeStyle = rd.kind === "track" ? "#c9b48c" : "#a7a493";
    ctx.lineWidth = (rd.width + 2.5) * px;
    ctx.stroke(p);
    ctx.strokeStyle = ROAD[rd.kind];
    ctx.lineWidth = rd.width * px;
    ctx.stroke(p);
    if (rd.kind === "road") {
      ctx.setLineDash([4 * px, 6 * px]);
      ctx.strokeStyle = "rgba(255,255,255,0.75)";
      ctx.lineWidth = Math.max(0.18 * px, 0.8);
      ctx.stroke(p);
      ctx.setLineDash([]);
    }
  }
  // Hedge footprints (the 3D hedges stand on them).
  const h = w.hedges;
  ctx.strokeStyle = HEDGE;
  ctx.lineCap = "round";
  for (let k = 0; k < h.length; k += 6) {
    ctx.lineWidth = Math.max(h[k + 5]! * px, 1);
    ctx.beginPath();
    ctx.moveTo(m.x(h[k + 1]!), m.y(h[k]!));
    ctx.lineTo(m.x(h[k + 3]!), m.y(h[k + 2]!));
    ctx.stroke();
  }
  // Shaded relief over everything, so slopes read even from straight above.
  const relief = hillshade(w);
  if (relief) {
    ctx.save();
    ctx.globalCompositeOperation = "soft-light";
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(relief, 0, 0, size, size);
    ctx.restore();
  }
  return canvas;
}

/** Shaded relief of the terrain grid, as on maps: lit from the north-west, 45° up; mid-grey
 * where the ground is level (soft-light leaves those colours as they are). Null for flat
 * ground. */
export function hillshade(w: WorldPayload): HTMLCanvasElement | null {
  const { rows, cols, cell, h } = w.dtm;
  if (rows < 3 || cols < 3) return null;
  const canvas = document.createElement("canvas");
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(cols, rows);
  const alt = Math.PI / 4;
  const az = (315 * Math.PI) / 180;
  const [ln, le, lu] = [Math.cos(alt) * Math.cos(az), Math.cos(alt) * Math.sin(az), Math.sin(alt)];
  const at = (i: number, j: number) => h[Math.min(Math.max(i, 0), rows - 1) * cols + Math.min(Math.max(j, 0), cols - 1)]!;
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const dn = (at(i + 1, j) - at(i - 1, j)) / (2 * cell);
      const de = (at(i, j + 1) - at(i, j - 1)) / (2 * cell);
      const norm = Math.hypot(dn, de, 1);
      const shade = (-dn * ln - de * le + lu) / norm;
      const v = 128 + 127 * Math.max(-1, Math.min(1, 5 * (shade - lu)));
      const k = 4 * ((rows - 1 - i) * cols + j); // canvas rows run from north to south
      img.data[k] = img.data[k + 1] = img.data[k + 2] = v;
      img.data[k + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

// Height tint for open ground: low → high over at least 20 m of relief.
const TINT_STOPS: [number, [number, number, number]][] = [
  [0.0, [150, 196, 112]],
  [0.35, [178, 202, 122]],
  [0.65, [204, 192, 132]],
  [1.0, [190, 160, 112]],
];

/** The terrain grid coloured by height (null for flat ground). */
export function elevationTint(w: WorldPayload): HTMLCanvasElement | null {
  const { rows, cols, h } = w.dtm;
  if (rows < 3 || cols < 3) return null;
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of h) {
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  const span = Math.max(hi - lo, 20);
  const canvas = document.createElement("canvas");
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(cols, rows);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const t = Math.min(Math.max((h[i * cols + j]! - lo) / span, 0), 1);
      let k = 1;
      while (k < TINT_STOPS.length - 1 && TINT_STOPS[k]![0] < t) k++;
      const [t0, c0] = TINT_STOPS[k - 1]!;
      const [t1, c1] = TINT_STOPS[k]!;
      const f = (t - t0) / (t1 - t0);
      const o = 4 * ((rows - 1 - i) * cols + j);
      for (let c = 0; c < 3; c++) img.data[o + c] = c0[c]! + (c1[c]! - c0[c]!) * f;
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

export interface OverlayZone {
  id: string;
  polygon: [number, number][];
  state: "reachable" | "unreachable" | "selected";
}

export interface OverlayState {
  zones: OverlayZone[];
  showZones: boolean;
  /** Reference grid: 100 m lines, every fifth stronger. */
  grid: boolean;
  noFly: [number, number][][];
  geofence: [number, number][] | null;
  /** Ground tracks under the plans (proposal dashed, active solid). */
  proposal: [number, number][] | null;
  active: [number, number][] | null;
}

const ZONE_COLOUR = { reachable: "56,209,106", unreachable: "138,148,158", selected: "255,201,60" };

/** Draped overlay: transparent except for what it shows. */
export function paintOverlay(canvas: HTMLCanvasElement, w: WorldPayload, s: OverlayState): void {
  const ctx = canvas.getContext("2d")!;
  const m = new Mapping(w.bounds, canvas.width);
  const px = m.scale;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineJoin = "round";
  if (s.grid) {
    const [n0, e0, n1, e1] = w.bounds;
    for (let k = Math.ceil(Math.min(n0, e0) / 100); k * 100 <= Math.max(n1, e1); k++) {
      const v = k * 100;
      ctx.strokeStyle = k % 5 === 0 ? "rgba(255,255,255,0.42)" : "rgba(255,255,255,0.16)";
      ctx.lineWidth = k % 5 === 0 ? Math.max(1.6 * px, 1.5) : 1;
      ctx.beginPath();
      if (v >= e0 && v <= e1) {
        ctx.moveTo(m.x(v), 0);
        ctx.lineTo(m.x(v), canvas.height);
      }
      if (v >= n0 && v <= n1) {
        ctx.moveTo(0, m.y(v));
        ctx.lineTo(canvas.width, m.y(v));
      }
      ctx.stroke();
    }
  }
  for (const poly of s.noFly) {
    const p = path(m, poly);
    ctx.fillStyle = "rgba(229,72,77,0.28)";
    ctx.fill(p);
    ctx.strokeStyle = "rgba(229,72,77,0.95)";
    ctx.lineWidth = Math.max(2.5 * px, 2);
    ctx.stroke(p);
  }
  if (s.geofence) {
    ctx.setLineDash([16 * px, 10 * px]);
    ctx.strokeStyle = "rgba(229,72,77,0.9)";
    ctx.lineWidth = Math.max(3 * px, 2);
    ctx.stroke(path(m, s.geofence));
    ctx.setLineDash([]);
  }
  if (s.showZones) {
    for (const z of s.zones) {
      const p = path(m, z.polygon);
      const c = ZONE_COLOUR[z.state];
      ctx.fillStyle = `rgba(${c},0.22)`;
      ctx.fill(p);
      ctx.strokeStyle = `rgba(${c},1)`;
      ctx.lineWidth = Math.max(3 * px, 2);
      ctx.stroke(p);
    }
  }
  const track = (pts: [number, number][], dash: boolean, colour: string) => {
    ctx.setLineDash(dash ? [12 * px, 8 * px] : []);
    ctx.strokeStyle = colour;
    ctx.lineWidth = Math.max(2 * px, 1.5);
    ctx.stroke(line(m, pts));
    ctx.setLineDash([]);
  };
  if (s.active && s.active.length > 1) track(s.active, false, "rgba(255,255,255,0.45)");
  if (s.proposal && s.proposal.length > 1) track(s.proposal, true, "rgba(76,201,240,0.55)");
}
