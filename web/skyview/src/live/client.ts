// Live session client (M2 spec §9): msgpack over WebSocket to `aeroadapt serve`.
import { decode, encode } from "@msgpack/msgpack";
import type { Manifest, ReplayEvent } from "../replay/types";
import { LiveBuffer, type WireColumn } from "./buffer";

export interface LiveStatus {
  connected: boolean;
  timeS: number;
  speed: number;
  paused: boolean;
  done: boolean;
  error: string | null;
}

type Message =
  | { type: "hello"; manifest: Manifest }
  | { type: "frames"; n: number; cols: Record<string, WireColumn> }
  | { type: "events"; events: ReplayEvent[] }
  | { type: "status"; time_s: number; speed: number; paused: boolean; done: boolean; error: string | null };

export interface LiveHandlers {
  onBuffer: (b: LiveBuffer) => void; // a new session (hello)
  onData: (b: LiveBuffer) => void; // new rows or events
  onStatus: (s: LiveStatus) => void;
}

export class LiveClient {
  private ws: WebSocket | null = null;
  private buffer: LiveBuffer | null = null;
  private status: LiveStatus = { connected: false, timeS: 0, speed: 1, paused: false, done: false, error: null };
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
  }

  private setStatus(p: Partial<LiveStatus>): void {
    this.status = { ...this.status, ...p };
    this.h.onStatus(this.status);
  }

  private handle(m: Message): void {
    if (m.type === "hello") {
      this.sessions += 1;
      this.buffer = new LiveBuffer(`live-${this.sessions}`, m.manifest);
      this.h.onBuffer(this.buffer);
    } else if (m.type === "frames" && this.buffer) {
      this.buffer.append(m.cols, m.n);
      this.h.onData(this.buffer);
    } else if (m.type === "events" && this.buffer) {
      this.buffer.events.push(...m.events);
      this.h.onData(this.buffer);
    } else if (m.type === "status") {
      this.setStatus({ timeS: m.time_s, speed: m.speed, paused: m.paused, done: m.done, error: m.error });
    }
  }
}
