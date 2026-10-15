import { CATEGORIES } from "../scene/categories";
import { type CameraMode, type Layers, SPEEDS, useStore } from "../state/store";
import type { Replay, ReplayEvent } from "../replay/types";

const IMPORTANT = new Set([
  "fault_onset",
  "failure_detected",
  "zone_selected",
  "contingency_phase",
  "airspeed_health",
  "touchdown",
  "terminated",
  "mode",
]);

export function describe(e: ReplayEvent): string {
  const p = e.payload;
  switch (e.type) {
    case "fault_onset":
      return `Fault: ${String(p.fault).replaceAll("_", " ")}`;
    case "failure_detected":
      return "Motor failure detected";
    case "zone_selected":
      return p.zone ? `Safe zone ${String(p.zone)} selected` : "No zone reachable: straight in";
    case "contingency_phase":
      return `Phase: ${String(p.phase)}`;
    case "airspeed_health":
      return p.healthy ? "Airspeed sensor healthy" : "Airspeed sensor flagged unreliable";
    case "touchdown":
      return `Touchdown: ${String(p.class)}${p.zone ? `, zone ${String(p.zone)} ${p.in_zone ? "(inside)" : "(outside)"}` : ""}`;
    case "terminated":
      return `Terminated: ${String(p.reason)}`;
    case "mode":
      return `Mode ${String(p.mode)}`;
    case "waypoint":
      return `Waypoint ${String(p.index)}`;
    default:
      return e.type;
  }
}

function fmtTime(t: number): string {
  const m = Math.floor(t / 60);
  const s = t - 60 * m;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

export function Timeline({ replay }: { replay: Replay }) {
  const time = useStore((s) => s.time);
  const playing = useStore((s) => s.playing);
  const speed = useStore((s) => s.speed);
  const { setTime, setPlaying, setSpeed } = useStore.getState();
  const events = replay.events.filter((e) => IMPORTANT.has(e.type));
  return (
    <div className="timeline">
      <button className="play" onClick={() => setPlaying(!playing)} data-testid="play">
        {playing ? "❚❚" : "▶"}
      </button>
      <span className="clock" data-testid="clock">
        {fmtTime(time)} / {fmtTime(replay.duration)}
      </span>
      <div className="scrub">
        <input
          type="range"
          min={0}
          max={replay.duration}
          step={0.02}
          value={time}
          onChange={(e) => setTime(Number(e.target.value))}
          aria-label="timeline"
        />
        {events.map((e, k) => (
          <button
            key={`${e.time_us}-${k}`}
            className={`tick tick-${e.type}`}
            style={{ left: `${(100 * e.time_us * 1e-6) / replay.duration}%` }}
            title={`${(e.time_us * 1e-6).toFixed(1)} s · ${describe(e)}`}
            onClick={() => setTime(e.time_us * 1e-6)}
          />
        ))}
      </div>
      <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="speed">
        {SPEEDS.map((s) => (
          <option key={s} value={s}>
            {s}×
          </option>
        ))}
      </select>
    </div>
  );
}

export function EventLog({ replay }: { replay: Replay }) {
  const time = useStore((s) => s.time);
  const { setTime } = useStore.getState();
  const events = replay.events.filter((e) => IMPORTANT.has(e.type) || e.type === "waypoint");
  return (
    <div className="eventlog">
      <h2>Events</h2>
      <ol>
        {events.map((e, k) => (
          <li
            key={`${e.time_us}-${k}`}
            className={e.time_us * 1e-6 <= time ? `past ev-${e.type}` : `future ev-${e.type}`}
            onClick={() => setTime(e.time_us * 1e-6)}
          >
            <span className="ev-time">{(e.time_us * 1e-6).toFixed(1)} s</span> {describe(e)}
          </li>
        ))}
      </ol>
    </div>
  );
}

const LAYER_LABELS: Record<keyof Layers, string> = {
  trail: "Trail",
  mission: "Mission",
  zones: "Safe zones",
  wind: "Wind",
  ghost: "Ghost",
};

export function Controls({ replay }: { replay: Replay }) {
  const replays = useStore((s) => s.replays);
  const ghost = useStore((s) => s.ghost);
  const camera = useStore((s) => s.camera);
  const layers = useStore((s) => s.layers);
  const status = useStore((s) => s.status);
  const { setCamera, toggleLayer } = useStore.getState();
  const m = replay.manifest;
  const select = (key: "replay" | "ghost", value: string) => {
    const url = new URL(window.location.href);
    if (value) url.searchParams.set(key, value);
    else url.searchParams.delete(key);
    window.location.href = url.toString();
  };
  return (
    <div className="controls">
      <h1>Skyview</h1>
      <p className="status">{status}</p>
      <label className="row">
        <span>Replay</span>
        <select value={replay.id} onChange={(e) => select("replay", e.target.value)} aria-label="replay">
          {[...new Set([replay.id, ...replays])].map((id) => (
            <option key={id}>{id}</option>
          ))}
        </select>
      </label>
      <label className="row">
        <span>Ghost</span>
        <select value={ghost?.id ?? ""} onChange={(e) => select("ghost", e.target.value)} aria-label="ghost">
          <option value="">none</option>
          {replays
            .filter((id) => id !== replay.id)
            .map((id) => (
              <option key={id}>{id}</option>
            ))}
        </select>
      </label>
      <p className="note">{m.scenario.description}</p>
      <p className="note">
        {m.site.name} · wind {m.scenario.environment.wind.speed_mps.toFixed(0)} m/s from{" "}
        {m.scenario.environment.wind.from_deg.toFixed(0)}° · turbulence{" "}
        {m.scenario.environment.turbulence.intensity} · contingency {m.scenario.stack.contingency}
      </p>
      <div className="cams">
        {(["chase", "orbit", "top", "free"] as CameraMode[]).map((c) => (
          <button key={c} className={c === camera ? "on" : ""} onClick={() => setCamera(c)}>
            {c}
          </button>
        ))}
      </div>
      <div className="layers">
        {(Object.keys(LAYER_LABELS) as (keyof Layers)[]).map((k) => (
          <label key={k}>
            <input type="checkbox" checked={layers[k]} onChange={() => toggleLayer(k)} /> {LAYER_LABELS[k]}
          </label>
        ))}
      </div>
      <div className="legend">
        {CATEGORIES.map((c) => (
          <span key={c.key}>
            <i style={{ background: c.css }} /> {c.label}
          </span>
        ))}
        <span>
          <i style={{ background: "#38d16a" }} /> reachable
        </span>
        <span>
          <i style={{ background: "#ffc93c" }} /> selected
        </span>
      </div>
    </div>
  );
}
