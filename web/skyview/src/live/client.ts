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

/** The site's operating area for planned missions (Txt2Fly, docs/06 §5); NE metres. */
export interface SiteLayers {
  geofence: [number, number][];
  home: [number, number];
  no_fly: { id: string; name: string; reason: string; polygons: [number, number][][] }[];
  places: { id: string; name: string; kind: string; point: [number, number] }[];
  attribution: string | null;
}

export interface PlanCheck {
  rule: string;
  title: string;
  ok: boolean;
  hard: boolean;
  reason: string;
  suggestion: string | null;
  /** NE points to show on the map (where the check fails). */
  where: [number, number][];
}

/** A plan built and checked from the aircraft's state (the server's ``report_json``). */
export interface PlanReport {
  ok: boolean;
  /** Soft rules that failed: the operator must acknowledge them to confirm. */
  warnings: string[];
  summary: string;
  assumptions: string[];
  notes: string[];
  steps: { label: string; alt: number | null; radius: number | null; speed: number | null }[];
  checks: PlanCheck[];
  /** The expected track, north, east, height, about every second. */
  preview: [number, number, number][];
  energy: { needed_mah: number | null; available_mah: number; minutes: number | null; laps: number | null } | null;
  plan: Record<string, unknown>;
}

export interface PlanProposal extends PlanReport {
  id: string;
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
  /** The mean wind set (it blends in over a few seconds): speed and the direction it blows from. */
  wind: LiveWind | null;
  /** Txt2Fly: the site's operating area (null: the site has none). */
  site: SiteLayers | null;
  /** The latest proposal, until it is confirmed or cancelled. */
  proposal: PlanProposal | null;
  /** The confirmed plan being flown. */
  activePlan: PlanProposal | null;
}

export interface LiveWind {
  speed_mps: number;
  from_deg: number;
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
  wind: null,
  site: null,
  proposal: null,
  activePlan: null,
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
      site: SiteLayers | null;
      plan: PlanProposal | null;
    }
  | { type: "plan_proposal"; proposal: PlanProposal }
  | { type: "plan_active"; plan: PlanProposal }
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
      wind: LiveWind;
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
    // A cancelled proposal disappears at once (the server keeps no reply for it).
    if (msg.cmd === "cancel_plan") this.setStatus({ proposal: null });
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
        site: m.site,
        proposal: null,
        activePlan: m.plan,
      });
      this.h.onBuffer(this.buffer);
    } else if (m.type === "frames" && this.buffer) {
      this.buffer.append(m.cols, m.n);
      this.h.onData(this.buffer);
    } else if (m.type === "events" && this.buffer) {
      this.buffer.events.push(...m.events);
      this.h.onData(this.buffer);
    } else if (m.type === "plan_proposal") {
      this.setStatus({ proposal: m.proposal });
    } else if (m.type === "plan_active") {
      this.setStatus({ activePlan: m.plan, proposal: null });
    } else if (m.type === "status") {
      this.setStatus({
        timeS: m.time_s,
        speed: m.speed,
        paused: m.paused,
        done: m.done,
        error: m.error,
        faults: m.faults,
        wind: m.wind,
      });
    }
  }
}
