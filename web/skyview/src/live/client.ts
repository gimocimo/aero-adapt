// Live session client (M2 spec §9): msgpack over WebSocket to `aeroadapt serve`.
import { decode, encode } from "@msgpack/msgpack";
import type { Manifest, ReplayEvent } from "../replay/types";
import { LiveBuffer, type WireColumn } from "./buffer";

/** A fault of the session; `index` is what `clear_fault` takes. */
export interface LiveFault {
  index: number;
  spec: Record<string, unknown> & { type: string; trigger: { at_time_s: number; until_s: number | null } };
  clearable: boolean;
}

export interface LiveAirframe {
  id: string;
  v_cruise_mps: number;
}

export interface LiveStatus {
  connected: boolean;
  timeS: number;
  speed: number;
  paused: boolean;
  done: boolean;
  error: string | null;
  faults: LiveFault[];
  scenario: string;
  scenarios: string[];
  /** The fleet airframe flying, or null for the scenario's own. */
  airframe: string | null;
  airframes: LiveAirframe[];
  /** Object URL of the flying fleet airframe's glTF (null: the default model). */
  modelUrl: string | null;
}

export const OFFLINE: LiveStatus = {
  connected: false,
  timeS: 0,
  speed: 1,
  paused: false,
  done: false,
  error: null,
  faults: [],
  scenario: "",
  scenarios: [],
  airframe: null,
  airframes: [],
  modelUrl: null,
};

type Message =
  | {
      type: "hello";
      manifest: Manifest;
      scenario: string;
      scenarios: string[];
      airframe: string | null;
      airframes: LiveAirframe[];
      model: Uint8Array | null;
    }
  | { type: "frames"; n: number; cols: Record<string, WireColumn> }
  | { type: "events"; events: ReplayEvent[] }
  | {
      type: "status";
      time_s: number;
      speed: number;
      paused: boolean;
      done: boolean;
      error: string | null;
      faults: LiveFault[];
    };

export interface LiveHandlers {
  onBuffer: (b: LiveBuffer) => void; // a new session (hello)
  onData: (b: LiveBuffer) => void; // new rows or events
  onStatus: (s: LiveStatus) => void;
}

export class LiveClient {
  private ws: WebSocket | null = null;
  private buffer: LiveBuffer | null = null;
  private status: LiveStatus = OFFLINE;
  private sessions = 0;
  private closed = false;
  private retry: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private readonly url: string,
    private readonly h: LiveHandlers,
  ) {
    this.connect();
  }

  /** Connect, and keep retrying every second while the server is not (yet) up. */
  private connect(): void {
    const ws = new WebSocket(this.url);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => this.setStatus({ connected: true });
    ws.onclose = () => {
      this.setStatus({ connected: false });
      if (!this.closed) this.retry = setTimeout(() => this.connect(), 1000);
    };
    ws.onmessage = (ev: MessageEvent<ArrayBuffer>) => this.handle(decode(new Uint8Array(ev.data)) as Message);
    this.ws = ws;
  }

  send(msg: Record<string, unknown>): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(encode(msg));
  }

  close(): void {
    this.closed = true;
    if (this.retry) clearTimeout(this.retry);
    this.ws?.close();
    if (this.status.modelUrl) URL.revokeObjectURL(this.status.modelUrl);
  }

  private setStatus(p: Partial<LiveStatus>): void {
    this.status = { ...this.status, ...p };
    this.h.onStatus(this.status);
  }

  private handle(m: Message): void {
    if (m.type === "hello") {
      this.sessions += 1;
      this.buffer = new LiveBuffer(`live-${this.sessions}`, m.manifest);
      if (this.status.modelUrl) URL.revokeObjectURL(this.status.modelUrl);
      const modelUrl = m.model
        ? URL.createObjectURL(new Blob([m.model as Uint8Array<ArrayBuffer>], { type: "model/gltf-binary" }))
        : null;
      this.setStatus({
        scenario: m.scenario,
        scenarios: m.scenarios,
        airframe: m.airframe,
        airframes: m.airframes,
        modelUrl,
        faults: [],
      });
      this.h.onBuffer(this.buffer);
    } else if (m.type === "frames" && this.buffer) {
      this.buffer.append(m.cols, m.n);
      this.h.onData(this.buffer);
    } else if (m.type === "events" && this.buffer) {
      this.buffer.events.push(...m.events);
      this.h.onData(this.buffer);
    } else if (m.type === "status") {
      this.setStatus({
        timeS: m.time_s,
        speed: m.speed,
        paused: m.paused,
        done: m.done,
        error: m.error,
        faults: m.faults,
      });
    }
  }
}
