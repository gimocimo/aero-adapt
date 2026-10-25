// Skyview's player (docs/07 §8): the follow view fills the window, the map seen from above is
// a minimap in a circle (top right), the instruments sit bottom left and the controls float
// bottom right. Both 3D views show the stylised world the simulator flies in. ?layout=quad
// keeps the four panels; ?map=real keeps the Google 3D-tiles player.
import { useEffect, useRef, useState } from "react";
import { LiveClient, OFFLINE } from "../live/client";
import { sample } from "../replay/interp";
import { listReplays, loadReplay } from "../replay/load";
import { SPEEDS, useStore } from "../state/store";
import { loadWorld, type WorldPayload } from "../world3d/payload";
import { type ChaseMode, extentOf, WorldView } from "../world3d/WorldView";
import { Avionics } from "./Avionics";
import { ControlDeck } from "./ControlDeck";
import { Instruments } from "./Instruments";

function params() {
  const q = new URLSearchParams(window.location.search);
  const t = Number(q.get("t"));
  return {
    replay: q.get("replay"),
    ghost: q.get("ghost"),
    live: q.get("live"),
    quality: q.get("quality"),
    /** &film=1: the follow view alone, filling the window (for video). */
    film: q.get("film") === "1",
    /** &layout=quad: the four panels (follow view, map, instruments, controls) side by side. */
    quad: q.get("layout") === "quad",
    /** &t=<s>: open a replay paused at that moment (links and stills). */
    t: q.has("t") && Number.isFinite(t) ? t : null,
  };
}

/** "/replays/<id>" for ids from the index; a path containing "/" is used as given. */
function bundleUrl(id: string): string {
  return id.includes("/") ? `/${id.replace(/^\/+/, "")}` : `/replays/${id}`;
}

/** The instruments: compact avionics over the full-screen picture, or the four-panel set. */
function InstrumentsContainer({ compact = false }: { compact?: boolean }) {
  const replay = useStore((s) => s.replay);
  const time = useStore((s) => s.time);
  if (!replay || replay.n === 0) return null;
  const s = sample(replay, time);
  return compact ? <Avionics s={s} replay={replay} /> : <Instruments s={s} replay={replay} />;
}

/** Wind as seen in a view: an arrow (blowing towards) and its speed. */
function WindBadge({ view }: { view: "chase" | "top" }) {
  const replay = useStore((s) => s.replay);
  const time = useStore((s) => s.time);
  if (!replay || replay.n === 0) return null;
  const s = sample(replay, time);
  const speed = Math.hypot(s.windN, s.windE);
  const towards = (Math.atan2(s.windE, s.windN) * 180) / Math.PI; // from north, clockwise
  const angle = view === "top" ? towards : towards - (s.yaw * 180) / Math.PI;
  return (
    <div className="windbadge" title={view === "top" ? "Wind (north up)" : "Wind relative to the nose"}>
      <span className="arrow" style={{ transform: `rotate(${angle}deg)` }}>
        ↑
      </span>
      {speed.toFixed(0)} m/s
    </div>
  );
}

export default function QuadPlayer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chaseRef = useRef<HTMLElement>(null);
  const topRef = useRef<HTMLElement>(null);
  const viewRef = useRef<WorldView | null>(null);
  const liveRef = useRef<LiveClient | null>(null);
  const readyFor = useRef(0); // the session whose world was last reported drawn
  const replay = useStore((s) => s.replay);
  const camera = useStore((s) => s.camera);
  const [error, setError] = useState<string | null>(null);
  const [world, setWorld] = useState<WorldPayload | null>(null);

  // The renderer and the animation loop.
  useEffect(() => {
    if (!canvasRef.current) return;
    const p = params();
    const view = new WorldView(canvasRef.current, { thin: p.quality === "low" ? 3 : 1, film: p.film, minimap: !p.quad && !p.film });
    view.setHosts(chaseRef.current, topRef.current);
    viewRef.current = view;
    if (import.meta.env.DEV) {
      const w = window as unknown as Record<string, unknown>;
      w.__view = view;
      w.__store = useStore;
    }
    let raf = 0;
    let last = performance.now();
    let shown: unknown = null;
    let ghostShown: unknown = null;
    const loop = (now: number) => {
      const st = useStore.getState();
      st.advance(Math.min((now - last) / 1000, 0.1));
      last = now;
      if (st.replay !== shown || st.ghost !== ghostShown) {
        view.setReplays(st.replay, st.ghost);
        shown = st.replay;
        ghostShown = st.ghost;
      }
      const s = useStore.getState();
      const mode: ChaseMode = s.camera === "orbit" || s.camera === "free" ? s.camera : "chase";
      view.update(s.time, s.layers, mode);
      const top = params().film ? null : (topRef.current?.getBoundingClientRect() ?? null);
      view.render(chaseRef.current?.getBoundingClientRect() ?? null, top);
      // Live: the session's world is on screen, so its clock may start (once per hello).
      const client = liveRef.current;
      if (s.live?.loading && client && readyFor.current !== client.sessions && view.showing(s.live.world)) {
        readyFor.current = client.sessions;
        client.ready();
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    const unsub = useStore.subscribe((s, prev) => {
      if (s.live?.modelUrl !== prev.live?.modelUrl) view.setModel(s.live?.modelUrl ?? null);
      if (s.live !== prev.live || s.replay !== prev.replay) {
        view.setSite(s.live?.site ?? null);
        view.setPlans(s.live?.proposal ?? null, s.live?.activePlan ?? null);
      }
      if (!s.live && s.replay !== prev.replay) {
        const m = s.replay?.manifest.viz_model;
        view.setModel(s.replay && m ? `${bundleUrl(s.replay.id)}/${m}` : null);
      }
    });
    return () => {
      cancelAnimationFrame(raf);
      unsub();
      view.dispose();
      viewRef.current = null;
    };
  }, []);

  // The world: from the live server's hello, or next to a replay.
  const live = useStore((s) => s.live);
  useEffect(() => {
    if (live) {
      setWorld(live.world);
      return;
    }
    const w = replay?.manifest.site.world;
    if (!replay || !w) {
      setWorld(null);
      return;
    }
    let cancelled = false;
    loadWorld(`${bundleUrl(replay.id)}/world.msgpack`)
      .then((p) => !cancelled && setWorld(p))
      .catch(() => !cancelled && setWorld(null));
    return () => {
      cancelled = true;
    };
  }, [live?.world, replay?.manifest.site.world, replay?.id, live === null]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const view = viewRef.current;
    if (!view || (!replay && !live?.site)) return;
    view.setWorld(world, extentOf(replay, live?.site ?? null), replay?.manifest.site.name);
    view.setReplays(replay, useStore.getState().ghost);
  }, [world, replay?.manifest.site.id, live?.site]); // eslint-disable-line react-hooks/exhaustive-deps

  // Live mode (?live=ws://localhost:8765): a growing replay streamed by `aeroadapt serve`.
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
    const hold = true; // the server holds the clock until the world is drawn
    const client = new LiveClient(url, {
      onBuffer: () => store.setReplay(null),
      onData: (b) => {
        if (b.length === 0) return;
        const shown = useStore.getState().replay;
        const snap = b.snapshot();
        if (shown?.id === snap.id) store.setLiveReplay(snap);
        else store.setReplay(snap);
      },
      onStatus: (s) => store.setLive(s),
    }, hold);
    liveRef.current = client;
    store.setLive(OFFLINE, (msg) => client.send(msg));
    if (import.meta.env.DEV) (window as unknown as Record<string, unknown>).__live = client;
    return () => {
      liveRef.current = null;
      client.close();
    };
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
      if (p.t !== null) store.setTime(p.t);
      else store.setPlaying(true);
    })().catch((e: unknown) => setError(String(e)));
    return () => {
      cancelled = true;
    };
  }, []);

  // Keyboard: space play/pause, ←/→ seek 5 s, [ / ] speed, c cycles the follow camera.
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
      else if (e.key === "c") st.setCamera(st.camera === "chase" ? "orbit" : st.camera === "orbit" ? "free" : "chase");
      else if (e.key === "]" || e.key === "[") {
        const k = SPEEDS.indexOf(st.speed as (typeof SPEEDS)[number]);
        const next = SPEEDS[Math.min(Math.max(k + (e.key === "]" ? 1 : -1), 0), SPEEDS.length - 1)];
        if (next !== undefined) st.setSpeed(next);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const cameraName = camera === "orbit" ? "orbit" : camera === "free" ? "free" : "chase";
  const [collapsed, setCollapsed] = useState(false);
  // Live: what the page is waiting for before the flight starts.
  const loading = live?.connected && live.loading ? `Loading ${live.world?.name || live.scenario || "the site"}…` : null;
  if (!params().quad) {
    const mapName = world ? world.name : (replay?.manifest.site.name ?? "Map");
    return (
      <div className={params().film ? "fly film" : "fly"}>
        <canvas ref={canvasRef} className="gl" />
        <section ref={chaseRef} className="f-chase" data-testid="view-follow">
          {loading && <div className="loading">{loading}</div>}
          <div className="f-badges">
            <span className="f-title">{cameraName === "free" ? "Free camera" : cameraName === "orbit" ? "Orbit" : "Chase"}</span>
            <WindBadge view="chase" />
          </div>
        </section>
        <div className="f-mapbox" data-testid="view-map">
          <section ref={topRef} className="f-map" title="Scroll to zoom · double-click: the whole map, or follow">
            <span className="f-north">N</span>
          </section>
          <div className="f-caption">
            <span>{mapName}</span>
            <WindBadge view="top" />
          </div>
          {!world && replay && <p className="f-note">Open ground: no world model for this site.</p>}
        </div>
        <section className="f-inst">
          <InstrumentsContainer compact />
        </section>
        <section className={collapsed ? "f-deck collapsed" : "f-deck"}>
          <button className="f-collapse" onClick={() => setCollapsed(!collapsed)} aria-label={collapsed ? "Show the controls" : "Hide the controls"}>
            {collapsed ? "▴" : "▾"}
          </button>
          {replay ? <ControlDeck replay={replay} /> : <p className="q-note">Waiting for data…</p>}
        </section>
        {error && <div className="error">{error}</div>}
      </div>
    );
  }
  return (
    <div className={params().film ? "quad film" : "quad"}>
      <canvas ref={canvasRef} className="gl" />
      <section ref={chaseRef} className="q q-chase" data-testid="view-follow">
        {loading && <div className="loading">{loading}</div>}
        <h2 className="q-title">Follow · {cameraName}</h2>
        <WindBadge view="chase" />
      </section>
      <section ref={topRef} className="q q-top" data-testid="view-map">
        <h2 className="q-title">{world ? world.name : (replay?.manifest.site.name ?? "Map")}</h2>
        <WindBadge view="top" />
        {!world && replay && <p className="q-note">No world model for this site: open ground. ?map=real for the real map.</p>}
      </section>
      <section className="q q-inst">
        <InstrumentsContainer />
      </section>
      <section className="q q-deck">{replay ? <ControlDeck replay={replay} /> : <p className="q-note">Waiting for data…</p>}</section>
      {error && <div className="error">{error}</div>}
    </div>
  );
}
