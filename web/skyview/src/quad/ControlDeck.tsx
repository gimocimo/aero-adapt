// The control deck (bottom right): the timeline or live bar, then tabs for the session, the
// plan, faults, events and what the views show.
import { useState } from "react";
import { FaultControls, LiveBar, SessionControls } from "../live/FaultPanel";
import { PlanPanel } from "../live/PlanPanel";
import { describe, EventLog, Timeline } from "../panels/Panels";
import type { Replay } from "../replay/types";
import { CATEGORIES } from "../scene/categories";
import { type CameraMode, type Layers, useStore } from "../state/store";

type Tab = "session" | "replay" | "plan" | "faults" | "events" | "view";

const LAYER_LABELS: Record<keyof Layers, string> = {
  trail: "Trail",
  mission: "Mission",
  zones: "Safe zones",
  ghost: "Ghost",
  grid: "Grid (100 m)",
};

const IMPORTANT = new Set([
  "fault_onset",
  "fault_cleared",
  "failure_detected",
  "zone_selected",
  "contingency_phase",
  "airspeed_health",
  "touchdown",
  "collision",
  "terminated",
  "mode",
]);

/** The latest few events, always in view (whichever tab is open). */
export function Ticker({ replay }: { replay: Replay }) {
  const time = useStore((s) => s.time);
  const recent = replay.events.filter((e) => IMPORTANT.has(e.type) && e.time_us * 1e-6 <= time + 1e-6).slice(-4);
  return (
    <ol className="ticker" data-testid="ticker">
      {recent.map((e, k) => (
        <li key={`${e.time_us}-${k}`} className={`ev-${e.type}`}>
          <span className="ev-time">{(e.time_us * 1e-6).toFixed(1)} s</span> {describe(e)}
        </li>
      ))}
    </ol>
  );
}

function ReplayControls({ replay }: { replay: Replay }) {
  const replays = useStore((s) => s.replays);
  const ghost = useStore((s) => s.ghost);
  const select = (key: "replay" | "ghost", value: string) => {
    const url = new URL(window.location.href);
    if (value) url.searchParams.set(key, value);
    else url.searchParams.delete(key);
    window.location.href = url.toString();
  };
  return (
    <div className="deck-form">
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
    </div>
  );
}

function About({ replay }: { replay: Replay }) {
  const live = useStore((s) => s.live);
  const status = useStore((s) => s.status);
  const m = replay.manifest;
  const wind = live?.wind ?? m.scenario.environment.wind;
  return (
    <>
      <p className="note">{m.scenario.description}</p>
      <p className="note">
        {m.site.name} · wind {wind.speed_mps.toFixed(0)} m/s from {wind.from_deg.toFixed(0)}° · turbulence{" "}
        {m.scenario.environment.turbulence.intensity} · contingency {m.scenario.stack.contingency}
      </p>
      {status && <p className="note">{status}</p>}
    </>
  );
}

function ViewControls() {
  const camera = useStore((s) => s.camera);
  const layers = useStore((s) => s.layers);
  const { setCamera, toggleLayer } = useStore.getState();
  const real = new URL(window.location.href);
  real.searchParams.set("map", "real");
  return (
    <div className="deck-form">
      <h2>Follow view</h2>
      <div className="cams">
        {(["chase", "orbit", "free"] as CameraMode[]).map((c) => (
          <button key={c} className={c === camera ? "on" : ""} onClick={() => setCamera(c)} data-testid={`cam-${c}`}>
            {c}
          </button>
        ))}
      </div>
      <p className="note">Free: drag to orbit the aircraft, scroll to zoom. Map: scroll to zoom, drag to pan, double-click to fit.</p>
      <h2>Layers</h2>
      <div className="layers">
        {(Object.keys(LAYER_LABELS) as (keyof Layers)[]).map((k) => (
          <label key={k}>
            <input type="checkbox" checked={layers[k]} onChange={() => toggleLayer(k)} /> {LAYER_LABELS[k]}
          </label>
        ))}
      </div>
      <h2>Trail</h2>
      <div className="legend">
        {CATEGORIES.map((c) => (
          <span key={c.key}>
            <i style={{ background: c.css }} /> {c.label}
          </span>
        ))}
        <span>
          <i style={{ background: "#38d16a" }} /> zone in reach
        </span>
        <span>
          <i style={{ background: "#ffc93c" }} /> zone chosen
        </span>
        <span>
          <i style={{ background: "#e5484d" }} /> no-fly, geofence
        </span>
      </div>
      <p className="note">
        <a href={real.toString()}>Real map (Google 3D tiles)</a>: for real sites; the terrain there is not what the
        simulator flies.
      </p>
    </div>
  );
}

export function ControlDeck({ replay }: { replay: Replay }) {
  const live = useStore((s) => s.live);
  const [tab, setTab] = useState<Tab>(live ? "session" : "replay");
  const tabs: [Tab, string][] = live
    ? [["session", "Session"], ...(live.site ? ([["plan", "Plan"]] as [Tab, string][]) : []), ["faults", "Faults"], ["events", "Events"], ["view", "View"]]
    : [["replay", "Replay"], ["events", "Events"], ["view", "View"]];
  const current = tabs.some(([t]) => t === tab) ? tab : tabs[0]![0];
  return (
    <div className="deck" data-testid="control-deck">
      <div className="deck-bar">{live ? <LiveBar /> : <Timeline replay={replay} />}</div>
      <div className="tabs" role="tablist">
        {tabs.map(([t, label]) => (
          <button
            key={t}
            role="tab"
            aria-selected={current === t}
            className={current === t ? "on" : ""}
            onClick={() => setTab(t)}
            data-testid={`${t}-tab`}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="deck-body">
        {current === "session" && (
          <div className="faultpanel docked">
            <SessionControls />
            <About replay={replay} />
          </div>
        )}
        {current === "replay" && (
          <>
            <ReplayControls replay={replay} />
            <About replay={replay} />
          </>
        )}
        {current === "plan" && (
          <div className="faultpanel docked">
            <PlanPanel />
          </div>
        )}
        {current === "faults" && (
          <div className="faultpanel docked" data-testid="fault-panel">
            <FaultControls />
          </div>
        )}
        {current === "events" && <EventLog replay={replay} />}
        {current === "view" && <ViewControls />}
      </div>
      <Ticker replay={replay} />
    </div>
  );
}
