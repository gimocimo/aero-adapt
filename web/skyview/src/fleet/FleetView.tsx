// Fleet view (?fleet=<bench id>): every flight of a bench run at once. B1 is the stock stack,
// B2 adds the classical contingency; the scenarios (and their noise) are the same.
import { useEffect, useRef, useState } from "react";
import { createViewer } from "../viewer";
import { type Fleet, loadFleet } from "./data";
import { FLEET_COLOURS, type FleetCounts, FleetScene, type FleetState } from "./FleetScene";

const SPEEDS = [1, 4, 16, 64];
const LABELS: Record<FleetState, string> = {
  flying: "flying the mission",
  contingency: "in the contingency",
  landed: "landed safely",
  lost: "lost (crash or loss of control)",
};

function fmt(t: number): string {
  const m = Math.floor(t / 60);
  return `${m}:${Math.floor(t - 60 * m).toString().padStart(2, "0")}`;
}

export default function FleetView() {
  const container = useRef<HTMLDivElement>(null);
  const id = new URLSearchParams(window.location.search).get("fleet") ?? "";
  const tiles = new URLSearchParams(window.location.search).get("tiles") !== "0";
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [fleet, setFleet] = useState<Fleet | null>(null);
  const [ui, setUi] = useState({ time: 0, counts: null as FleetCounts | null });
  const [baseline, setBaseline] = useState("B2");
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(16);
  const ctl = useRef({ time: 0, baseline: "B2", playing: true, speed: 16, seek: null as number | null });

  useEffect(() => {
    ctl.current.baseline = baseline;
    ctl.current.playing = playing;
    ctl.current.speed = speed;
  }, [baseline, playing, speed]);

  useEffect(() => {
    if (!container.current) return;
    const { viewer, destroy } = createViewer(container.current, tiles, setStatus);
    let scene: FleetScene | null = null;
    let raf = 0;
    let cancelled = false;
    loadFleet(id)
      .then((f) => {
        if (cancelled) return;
        setFleet(f);
        scene = new FleetScene(viewer, f);
        scene.home();
        if (import.meta.env.DEV) (window as unknown as Record<string, unknown>).__fleet = { scene, ctl };
        let last = performance.now();
        let lastUi = 0;
        const duration = (f.meta.samples - 1) * f.meta.dt_s;
        const loop = (now: number) => {
          const c = ctl.current;
          if (c.seek !== null) {
            c.time = c.seek;
            c.seek = null;
          } else if (c.playing) {
            c.time = Math.min(c.time + Math.min((now - last) / 1000, 0.1) * c.speed, duration);
          }
          last = now;
          const counts = scene!.update(c.time, c.baseline);
          if (now - lastUi > 200) {
            setUi({ time: c.time, counts });
            lastUi = now;
          }
          raf = requestAnimationFrame(loop);
        };
        raf = requestAnimationFrame(loop);
      })
      .catch((e: unknown) => setError(String(e)));
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      scene?.destroy();
      destroy();
    };
  }, [id, tiles]);

  const duration = fleet ? (fleet.meta.samples - 1) * fleet.meta.dt_s : 0;
  const shown = fleet?.meta.flights.filter((f) => f.baseline === baseline).length ?? 0;
  return (
    <>
      <div ref={container} className="viewer" />
      {error && <div className="error">{error}</div>}
      {fleet && (
        <div className="controls fleetpanel" data-testid="fleet-panel">
          <h1>Fleet view</h1>
          <p className="status">{status}</p>
          <p className="note">
            <b>{fleet.meta.id}</b> · {shown} flights · {fleet.meta.description}
          </p>
          <div className="cams">
            {["B1", "B2"].map((b) => (
              <button key={b} className={b === baseline ? "on" : ""} onClick={() => setBaseline(b)}>
                {b === "B1" ? "B1 stock" : "B2 + contingency"}
              </button>
            ))}
          </div>
          <ul className="fleetcounts" data-testid="fleet-counts">
            {(Object.keys(LABELS) as FleetState[]).map((k) => (
              <li key={k}>
                <i style={{ background: FLEET_COLOURS[k] }} />
                <b>{ui.counts?.[k] ?? 0}</b> {LABELS[k]}
              </li>
            ))}
          </ul>
        </div>
      )}
      {fleet && (
        <div className="timeline live">
          <button onClick={() => setPlaying(!playing)}>{playing ? "❚❚" : "▶"}</button>
          <span className="clock" data-testid="clock">
            {fmt(ui.time)} / {fmt(duration)}
          </span>
          <input
            type="range"
            min={0}
            max={duration}
            step={1}
            value={ui.time}
            onChange={(e) => (ctl.current.seek = Number(e.target.value))}
            aria-label="time"
          />
          {SPEEDS.map((x) => (
            <button key={x} className={x === speed ? "on" : ""} onClick={() => setSpeed(x)}>
              {x}×
            </button>
          ))}
        </div>
      )}
    </>
  );
}
