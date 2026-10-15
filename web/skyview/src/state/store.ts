import { create } from "zustand";
import type { Replay } from "../replay/types";

export type CameraMode = "chase" | "orbit" | "top" | "free";
export const SPEEDS = [0.25, 0.5, 1, 2, 4, 8] as const;

export interface Layers {
  trail: boolean;
  mission: boolean;
  zones: boolean;
  wind: boolean;
  ghost: boolean;
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
}

export const useStore = create<State>((set, get) => ({
  replays: [],
  replay: null,
  ghost: null,
  time: 0,
  playing: false,
  speed: 1,
  camera: "chase",
  layers: { trail: true, mission: true, zones: true, wind: true, ghost: true },
  status: "",
  setReplays: (replays) => set({ replays }),
  setReplay: (replay) => set({ replay, time: replay?.t[0] ?? 0, playing: false }),
  setGhost: (ghost) => set({ ghost }),
  setTime: (t) => {
    const r = get().replay;
    set({ time: r ? Math.min(Math.max(t, 0), r.duration) : 0 });
  },
  advance: (dtWall) => {
    const { replay, playing, speed, time } = get();
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
}));
