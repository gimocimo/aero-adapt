import { tableFromIPC } from "apache-arrow";
import type { Column, Manifest, Replay, ReplayEvent } from "./types";

/** Parse the Arrow IPC (Feather v2) viz table into typed columns. */
export function columnsFromArrow(bytes: Uint8Array): { t: Float32Array; cols: Record<string, Column> } {
  const table = tableFromIPC(bytes);
  const cols: Record<string, Column> = {};
  for (const field of table.schema.fields) {
    const vec = table.getChild(field.name);
    if (!vec) continue;
    const arr = vec.toArray() as unknown;
    if (arr instanceof Float32Array || arr instanceof Int32Array) cols[field.name] = arr;
    else if (arr instanceof Float64Array) cols[field.name] = Float32Array.from(arr);
    else cols[field.name] = Int32Array.from(arr as ArrayLike<number>);
  }
  const t = cols.t;
  if (!(t instanceof Float32Array)) throw new Error("viz.arrow: missing float32 time column 't'");
  return { t, cols };
}

export function parseEvents(text: string): ReplayEvent[] {
  return text
    .split("\n")
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line) as ReplayEvent);
}

/** Load a replay bundle served under `base` (e.g. "/replays/m1-motorcut-classical"). */
export async function loadReplay(base: string): Promise<Replay> {
  const get = async (name: string) => {
    const res = await fetch(`${base}/${name}`);
    if (!res.ok) throw new Error(`${base}/${name}: HTTP ${res.status}`);
    return res;
  };
  const [manifest, events, arrow] = await Promise.all([
    get("manifest.json").then((r) => r.json() as Promise<Manifest>),
    get("events.jsonl").then((r) => r.text()),
    get("viz.arrow").then((r) => r.arrayBuffer()),
  ]);
  const { t, cols } = columnsFromArrow(new Uint8Array(arrow));
  return {
    id: base.split("/").filter(Boolean).pop() ?? base,
    manifest,
    events: parseEvents(events),
    t,
    cols,
    n: t.length,
    duration: t[t.length - 1] ?? 0,
  };
}

/** Replay ids listed by `aeroadapt export-viz` in /replays/index.json. */
export async function listReplays(): Promise<string[]> {
  const res = await fetch("/replays/index.json");
  // Before any export the file is missing, and the dev server answers with index.html (SPA fallback).
  if (!res.ok || !res.headers.get("content-type")?.includes("json")) return [];
  return (await res.json()) as string[];
}
