// Live session client (M2 spec §9): msgpack over WebSocket to `aeroadapt serve`.
import { decode, encode } from "@msgpack/msgpack";
import type { Manifest, ReplayEvent } from "../replay/types";
import { fromWire, type WorldPayload } from "../world3d/payload";
import { LiveBuffer, type WireColumn } from "./buffer";

/** A fault of the session; `index` is what `clear_fault` takes. */
export interface LiveFault {
  index: number;
  spec: Record<string, unknown> & { type: string; trigger: { at_time_s: number; until_s: number | null } };
  clearable: boolean;
}

/** The site's operating area for planned missions (Txt2Fly, docs/06 §5); NE metres. */
export interface SiteLayers {
  id: string;
  geofence: [number, number][];
  home: [number, number];
  no_fly: {
    id: string;
    name: string;
    reason: string;
    polygons: [number, number][][];
    /** How high it reaches above the ground (from the ground at its centre: a flat top); null: all heights. */
    ceiling_m?: number | null;
    ceiling_from?: "ground" | "centre";
  }[];
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

/** One exchange with the translator (docs/06 §10): the operator's words and its answer. */
export interface AskEntry {
  text: string;
  /** null while the translator is working. */
  kind: "plan" | "clarify" | "refuse" | "error" | null;
  question?: string;
  options?: string[];
  reason?: string;
  message?: string;
  summary?: string;
  latency_s?: number;
}

export interface LiveTranslator {
  name: string | null;
  note: string;
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
  /** A world site's world (docs/07 §8); null on a flat site. */
  world: WorldPayload | null;
  /** The latest proposal, until it is confirmed or cancelled. */
  proposal: PlanProposal | null;
  /** The confirmed plan being flown. */
  activePlan: PlanProposal | null;
  /** Txt2Fly's language layer: which translator answers, and the conversation so far. */
  translator: LiveTranslator | null;
  conversation: AskEntry[];
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
  world: null,
  proposal: null,
  activePlan: null,
  translator: null,
  conversation: [],
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
      world: unknown;
      plan: PlanProposal | null;
      translator?: LiveTranslator;
    }
  | ({ type: "ask_reply" } & AskEntry)
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
    // The question shows at once, answered when the translator replies.
    if (msg.cmd === "ask" && typeof msg.text === "string") {
      this.setStatus({ conversation: [...this.status.conversation, { text: msg.text, kind: null }].slice(-12) });
    }
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
        world: m.world ? fromWire(m.world) : null,
        proposal: null,
        activePlan: m.plan,
        translator: m.translator ?? null,
        conversation: [],
      });
      this.h.onBuffer(this.buffer);
    } else if (m.type === "frames" && this.buffer) {
      this.buffer.append(m.cols, m.n);
      this.h.onData(this.buffer);
    } else if (m.type === "events" && this.buffer) {
      this.buffer.events.push(...m.events);
      this.h.onData(this.buffer);
    } else if (m.type === "ask_reply") {
      const { type: _type, ...entry } = m;
      const conv = [...this.status.conversation];
      const i = conv.findIndex((e) => e.kind === null && e.text === entry.text);
      if (i >= 0) conv[i] = entry;
      else conv.push(entry);
      this.setStatus({ conversation: conv.slice(-12) });
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
