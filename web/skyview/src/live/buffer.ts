// A growing replay for live sessions: typed columns with doubling capacity, exposed as an
// ordinary Replay (subarray views), so the scene, HUD and panels have one code path.
import type { Column, Manifest, Replay, ReplayEvent } from "../replay/types";

export interface WireColumn {
  dtype: "f4" | "i4";
  data: Uint8Array;
}

function decode(c: WireColumn): Column {
  // Copy into an aligned buffer: msgpack bin payloads may start at any byte offset.
  const bytes = c.data.slice();
  return c.dtype === "i4" ? new Int32Array(bytes.buffer) : new Float32Array(bytes.buffer);
}

export class LiveBuffer {
  private cols: Record<string, Column> = {};
  private n = 0;
  private capacity = 0;
  readonly events: ReplayEvent[] = [];

  constructor(
    readonly id: string,
    readonly manifest: Manifest,
  ) {}

  get length(): number {
    return this.n;
  }

  append(wire: Record<string, WireColumn>, rows: number): void {
    if (rows === 0) return;
    if (this.n + rows > this.capacity) this.grow(Math.max(2 * this.capacity, this.n + rows, 4096));
    for (const [name, w] of Object.entries(wire)) {
      const src = decode(w);
      let dst = this.cols[name];
      if (!dst) {
        dst = w.dtype === "i4" ? new Int32Array(this.capacity) : new Float32Array(this.capacity);
        this.cols[name] = dst;
      }
      dst.set(src.subarray(0, rows), this.n);
    }
    this.n += rows;
  }

  private grow(capacity: number): void {
    for (const [name, old] of Object.entries(this.cols)) {
      const next = old instanceof Int32Array ? new Int32Array(capacity) : new Float32Array(capacity);
      next.set(old.subarray(0, this.n));
      this.cols[name] = next;
    }
    this.capacity = capacity;
  }

  /** The current contents as a Replay (views, no copies). */
  snapshot(): Replay {
    const cols: Record<string, Column> = {};
    for (const [name, c] of Object.entries(this.cols)) cols[name] = c.subarray(0, this.n);
    const t = (cols.t as Float32Array | undefined) ?? new Float32Array(0);
    return {
      id: this.id,
      manifest: this.manifest,
      events: this.events,
      t,
      cols,
      n: this.n,
      duration: this.n > 0 ? t[this.n - 1]! : 0,
    };
  }
}
