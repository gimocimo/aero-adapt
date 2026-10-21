import { useEffect, useRef, useState } from "react";
import { Hud } from "./hud/Hud";
import { LiveClient, OFFLINE } from "./live/client";
import { FaultPanel, LiveBar } from "./live/FaultPanel";
import { Controls, EventLog, Timeline } from "./panels/Panels";
import { sample } from "./replay/interp";
import { listReplays, loadReplay } from "./replay/load";
import { SceneController } from "./scene/SceneController";
import { SPEEDS, useStore } from "./state/store";
import { createViewer } from "./viewer";

function params() {
  const q = new URLSearchParams(window.location.search);
  return {
    replay: q.get("replay"),
    ghost: q.get("ghost"),
    live: q.get("live"),
    tiles: q.get("tiles") !== "0",
  };
}

/** "/replays/<id>" for ids from the index; a path containing "/" is used as given. */
function bundleUrl(id: string): string {
  return id.includes("/") ? `/${id.replace(/^\/+/, "")}` : `/replays/${id}`;
}

function HudContainer() {
  const replay = useStore((s) => s.replay);
  const time = useStore((s) => s.time);
  if (!replay) return null;
  return <Hud s={sample(replay, time)} replay={replay} />;
}

export default function Player() {
  const container = useRef<HTMLDivElement>(null);
  const replay = useStore((s) => s.replay);
  const live = useStore((s) => s.live);
  const [error, setError] = useState<string | null>(null);

  // Viewer, scene controller and the animation loop.
  useEffect(() => {
    if (!container.current) return;
    const store = useStore.getState();
    const { viewer, destroy } = createViewer(container.current, params().tiles, store.setStatus);
    const controller = new SceneController(viewer);
    if (import.meta.env.DEV) {
      // Dev hooks for console debugging (and stepping frames while the tab is hidden).
      const w = window as unknown as Record<string, unknown>;
      w.__controller = controller;
      w.__store = useStore;
    }

    let raf = 0;
    let last = performance.now();
    let shown: unknown = null;
    const loop = (now: number) => {
      const st = useStore.getState();
      st.advance(Math.min((now - last) / 1000, 0.1));
      last = now;
      if (st.replay !== shown) {
        controller.setReplays(st.replay, st.ghost);
        shown = st.replay;
      }
      const s = useStore.getState();
      controller.update(s.time, s.layers, s.camera);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    const unsubGhost = useStore.subscribe((s, prev) => {
      if (s.ghost !== prev.ghost) controller.setReplays(s.replay, s.ghost);
      if (s.live?.modelUrl !== prev.live?.modelUrl) controller.setModel(s.live?.modelUrl ?? null);
      // Txt2Fly: the operating area, the proposed plan and the plan flown.
      if (s.live !== prev.live || s.replay !== prev.replay) {
        controller.setSite(s.live?.site ?? null);
        controller.setPlans(s.live?.proposal ?? null, s.live?.activePlan ?? null);
      }
      // A replay of a fleet airframe brings its own model.
      if (!s.live && s.replay !== prev.replay) {
        const m = s.replay?.manifest.viz_model;
        controller.setModel(s.replay && m ? `${bundleUrl(s.replay.id)}/${m}` : null);
      }
    });
    return () => {
      cancelAnimationFrame(raf);
      unsubGhost();
      controller.destroy();
      destroy();
    };
  }, []);

  // Live mode (?live=ws://localhost:8765): a growing replay streamed by `aeroadapt serve`.
  // `&ghost=<bundle>` flies a recorded replay of the same scenario alongside, for comparison.
  useEffect(() => {
    const url = params().live;
    if (!url) return;
    const store = useStore.getState();
    const ghostId = params().ghost;
    if (ghostId) {
      loadReplay(bundleUrl(ghostId))
        .then((g) => store.setGhost(g))
        .catch((e: unknown) => setError(String(e)));
    }
    const client = new LiveClient(url, {
      // A new session: show it once its first rows arrive (an empty replay cannot be sampled).
      onBuffer: () => store.setReplay(null),
      onData: (b) => {
        if (b.length === 0) return;
        const shown = useStore.getState().replay;
        const snap = b.snapshot();
        if (shown?.id === snap.id) store.setLiveReplay(snap);
        else store.setReplay(snap);
      },
      onStatus: (s) => store.setLive(s),
    });
    store.setLive(OFFLINE, (msg) => client.send(msg));
    if (import.meta.env.DEV) (window as unknown as Record<string, unknown>).__live = client;
    return () => client.close();
  }, []);

  // Replays: the index, then the requested replay (and ghost).
  useEffect(() => {
    if (params().live) return;
    const store = useStore.getState();
    let cancelled = false;
    (async () => {
      const ids = await listReplays();
      if (cancelled) return;
      store.setReplays(ids);
      const p = params();
      const id = p.replay ?? ids[0];
      if (!id) {
        setError("No replays found. Run: uv run aeroadapt run <scenario> && uv run aeroadapt export-viz runs/<id>");
        return;
      }
      const [main, ghost] = await Promise.all([
        loadReplay(bundleUrl(id)),
        p.ghost ? loadReplay(bundleUrl(p.ghost)) : Promise.resolve(null),
      ]);
      if (cancelled) return;
      store.setGhost(ghost);
      store.setReplay(main);
      store.setPlaying(true);
    })().catch((e: unknown) => setError(String(e)));
    return () => {
      cancelled = true;
    };
  }, []);

  // Keyboard: space play/pause, ←/→ seek 5 s, [ / ] speed.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const st = useStore.getState();
      const t = e.target;
      if (t instanceof HTMLInputElement || t instanceof HTMLSelectElement || t instanceof HTMLTextAreaElement) return;
      if (e.key === " ") {
        st.setPlaying(!st.playing);
        e.preventDefault();
      } else if (e.key === "ArrowRight") st.setTime(st.time + 5);
      else if (e.key === "ArrowLeft") st.setTime(st.time - 5);
      else if (e.key === "]" || e.key === "[") {
        const k = SPEEDS.indexOf(st.speed as (typeof SPEEDS)[number]);
        const next = SPEEDS[Math.min(Math.max(k + (e.key === "]" ? 1 : -1), 0), SPEEDS.length - 1)];
        if (next !== undefined) st.setSpeed(next);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <>
      <div ref={container} className="viewer" />
      {error && <div className="error">{error}</div>}
      {replay && (
        <>
          <Controls replay={replay} />
          <EventLog replay={replay} />
          <HudContainer />
          {live ? <LiveBar /> : <Timeline replay={replay} />}
          <FaultPanel />
        </>
      )}
    </>
  );
}
