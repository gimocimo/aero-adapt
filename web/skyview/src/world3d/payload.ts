// The world as the server sends it (aeroadapt.world.export, docs/07 §8), and the frame
// conventions of the stylised view.
//
// three.js world frame: x = east, y = up, z = south (right-handed). Site NE metres and
// heights above the site datum (the ground at home) map to (e, h, -n).
import { decode } from "@msgpack/msgpack";

export interface Ring {
  outer: [number, number][];
  holes: [number, number][][];
}

export interface Plot extends Ring {
  kind: "field" | "rough" | "settlement" | "yard" | "green" | "airstrip";
  cover: string;
  zone: string | null;
  name: string | null;
}

export interface Wood extends Ring {
  id: string;
  name: string;
  kind: "broadleaf" | "conifer";
  canopy: number;
}

export interface Road {
  id: string;
  name: string | null;
  kind: "road" | "lane" | "street" | "track";
  width: number;
  points: [number, number][];
}

export interface Building {
  kind: string;
  n: number;
  e: number;
  length: number;
  width: number;
  heading: number;
  base: number;
  eaves: number;
  ridge: number;
  roof: "gable" | "flat" | "spire";
  variant: number;
  name: string | null;
}

export interface Structure {
  kind: "mast" | "silo";
  name: string;
  n: number;
  e: number;
  base: number;
  height: number;
  radius: number;
}

export interface WorldPayload {
  version: number;
  id: string;
  name: string;
  /** [n0, e0, n1, e1] */
  bounds: [number, number, number, number];
  dtm: { n0: number; e0: number; cell: number; rows: number; cols: number; h: Float32Array };
  plots: Plot[];
  woods: Wood[];
  river: { name: string; width: number; points: [number, number][] } | null;
  roads: Road[];
  /** (n_a, e_a, n_b, e_b, height, width) per hedge segment. */
  hedges: Float32Array;
  /** (n, e, ground, height, crown radius, kind) per tree; kind 0 broadleaf, 1 conifer, 2 willow. */
  trees: Float32Array;
  buildings: Building[];
  structures: Structure[];
}

interface Wire extends Omit<WorldPayload, "dtm" | "hedges" | "trees"> {
  dtm: Omit<WorldPayload["dtm"], "h"> & { h: Uint8Array };
  hedges: Uint8Array;
  trees: Uint8Array;
}

function f32(b: Uint8Array): Float32Array {
  // Copy into an aligned buffer: msgpack bin payloads may start at any byte offset.
  return new Float32Array(b.slice().buffer);
}

/** The decoded wire form (the hello message's ``world``, or a replay's world.msgpack). */
export function fromWire(w: unknown): WorldPayload {
  const x = w as Wire;
  return { ...x, dtm: { ...x.dtm, h: f32(x.dtm.h) }, hedges: f32(x.hedges), trees: f32(x.trees) };
}

export async function loadWorld(url: string): Promise<WorldPayload | null> {
  const res = await fetch(url);
  if (!res.ok || res.headers.get("content-type")?.includes("html")) return null;
  return fromWire(decode(new Uint8Array(await res.arrayBuffer())));
}

/** A flat stand-in for sites without a world: open ground over the given extent. */
export function flatWorld(id: string, name: string, bounds: [number, number, number, number]): WorldPayload {
  const [n0, e0, n1, e1] = bounds;
  return {
    version: 1,
    id,
    name,
    bounds,
    dtm: { n0, e0, cell: Math.max(n1 - n0, e1 - e0), rows: 2, cols: 2, h: new Float32Array(4) },
    plots: [],
    woods: [],
    river: null,
    roads: [],
    hedges: new Float32Array(0),
    trees: new Float32Array(0),
    buildings: [],
    structures: [],
  };
}

/** Ground height (above the datum) by bilinear interpolation, clamped at the edges. */
export class Ground {
  constructor(readonly d: WorldPayload["dtm"]) {}

  at(n: number, e: number): number {
    const { n0, e0, cell, rows, cols, h } = this.d;
    const x = Math.min(Math.max((n - n0) / cell, 0), rows - 1.0001);
    const y = Math.min(Math.max((e - e0) / cell, 0), cols - 1.0001);
    const i = Math.floor(x);
    const j = Math.floor(y);
    const fx = x - i;
    const fy = y - j;
    const v = (a: number, b: number) => h[a * cols + b]!;
    return (
      v(i, j) * (1 - fx) * (1 - fy) + v(i + 1, j) * fx * (1 - fy) + v(i, j + 1) * (1 - fx) * fy + v(i + 1, j + 1) * fx * fy
    );
  }
}
