import { create } from "zustand";
import type { LiveStatus } from "../live/client";
import type { Replay } from "../replay/types";

export type CameraMode = "chase" | "orbit" | "top" | "free";
export const SPEEDS = [0.25, 0.5, 1, 2, 4, 8] as const;

export interface Layers {
  trail: boolean;
  mission: boolean;
  zones: boolean;
  ghost: boolean;
  /** The reference grid on the ground: 100 m lines, labelled every 500 m on the map. */
  grid: boolean;
}

interface State {
  replays: string[];
  replay: Replay | null;
  ghost: Replay | null;
  time: number;
  playing: boolean;
  speed: number;
  camera: CameraMode;
  layers: Layers;
  status: string;
  /** Live session (null when replaying a bundle). */
  live: LiveStatus | null;
  liveSend: ((msg: Record<string, unknown>) => void) | null;
  setReplays: (ids: string[]) => void;
  setReplay: (r: Replay | null) => void;
  setGhost: (r: Replay | null) => void;
  setTime: (t: number) => void;
  advance: (dtWall: number) => void;
  setPlaying: (p: boolean) => void;
  setSpeed: (s: number) => void;
  setCamera: (c: CameraMode) => void;
  toggleLayer: (k: keyof Layers) => void;
  setStatus: (s: string) => void;
  setLive: (live: LiveStatus | null, send?: ((msg: Record<string, unknown>) => void) | null) => void;
  /** Swap in a grown live replay without moving the playhead. */
  setLiveReplay: (r: Replay) => void;
}

/** Live playback trails the newest sample slightly, so interpolation always has data. */
export const LIVE_DELAY_S = 0.1;

export const useStore = create<State>((set, get) => ({
  replays: [],
  replay: null,
  ghost: null,
  time: 0,
  playing: false,
  speed: 1,
  camera: "chase",
  layers: { trail: true, mission: true, zones: true, ghost: true, grid: true },
  status: "",
  live: null,
  liveSend: null,
  setReplays: (replays) => set({ replays }),
  setReplay: (replay) => set({ replay, time: replay?.t[0] ?? 0, playing: false }),
  setGhost: (ghost) => set({ ghost }),
  setTime: (t) => {
    const r = get().replay;
    set({ time: r ? Math.min(Math.max(t, 0), r.duration) : 0 });
  },
  advance: (dtWall) => {
    const { replay, playing, speed, time, live } = get();
    if (live && replay) {
      // A finished session shows its last instant (the touchdown).
      set({ time: Math.max(replay.duration - (live.done ? 0 : LIVE_DELAY_S), 0) });
      return;
    }
    if (!replay || !playing) return;
    const next = time + dtWall * speed;
    if (next >= replay.duration) set({ time: replay.duration, playing: false });
    else set({ time: next });
  },
  setPlaying: (playing) => {
    const { replay, time } = get();
    // Pressing play at the end restarts.
    if (playing && replay && time >= replay.duration) set({ time: 0 });
    set({ playing });
  },
  setSpeed: (speed) => set({ speed }),
  setCamera: (camera) => set({ camera }),
  toggleLayer: (k) => set({ layers: { ...get().layers, [k]: !get().layers[k] } }),
  setStatus: (status) => set({ status }),
  setLive: (live, send) => set({ live, liveSend: send === undefined ? get().liveSend : send }),
  setLiveReplay: (replay) => set({ replay }),
}));
