// Live-mode controls: inject faults while the aircraft flies, and pace the server.
import { useStore } from "../state/store";

export interface FaultPreset {
  label: string;
  spec: Record<string, unknown>;
}

/** One click, one fault: faultlab kinds with representative magnitudes (M2 spec §7). */
export const FAULT_GROUPS: { title: string; presets: FaultPreset[] }[] = [
  {
    title: "Propulsion",
    presets: [
      { label: "Motor cut", spec: { type: "motor_cut" } },
      { label: "Thrust −50 %", spec: { type: "propulsion_fault", kind: "partial_loss", magnitude: 0.5 } },
      { label: "Prop damage", spec: { type: "propulsion_fault", kind: "prop_damage", magnitude: 0.3 } },
      { label: "Motor overheat", spec: { type: "propulsion_fault", kind: "thermal_derate", magnitude: 0.6, ramp_s: 15 } },
    ],
  },
  {
    title: "Control surfaces",
    presets: [
      { label: "Aileron jam", spec: { type: "servo_jam", channel: "aileron" } },
      { label: "Elevator hard-over", spec: { type: "actuator_fault", kind: "hard_over", channel: "elevator", magnitude: -1 } },
      { label: "Rudder floating", spec: { type: "actuator_fault", kind: "floating", channel: "rudder" } },
      { label: "Aileron reversed", spec: { type: "actuator_fault", kind: "reversed", channel: "aileron" } },
      { label: "Elevator slow", spec: { type: "actuator_fault", kind: "slow", channel: "elevator", magnitude: 0.1 } },
    ],
  },
  {
    title: "Air data",
    presets: [
      { label: "Pitot + drain blocked", spec: { type: "pitot_blockage", variant: "drain_blocked" } },
      { label: "Airspeed +5 m/s", spec: { type: "sensor_fault", kind: "airspeed_bias", magnitude: 5 } },
      { label: "Airspeed stuck", spec: { type: "sensor_fault", kind: "airspeed_stuck" } },
    ],
  },
  {
    title: "Navigation",
    presets: [
      { label: "GNSS dropout", spec: { type: "sensor_fault", kind: "gnss_dropout", magnitude: 0 } },
      { label: "GNSS spoofing", spec: { type: "sensor_fault", kind: "gnss_spoof", magnitude: 0.3, axis: 1 } },
      { label: "Gyro bias", spec: { type: "sensor_fault", kind: "gyro_bias", magnitude: 0.05, axis: 2 } },
      { label: "Compass interference", spec: { type: "sensor_fault", kind: "mag_interference", magnitude: 0.3, axis: 0 } },
    ],
  },
  {
    title: "Battery and airframe",
    presets: [
      { label: "Cell failure", spec: { type: "energy_fault", kind: "cell_failure" } },
      { label: "Battery sag ×5", spec: { type: "energy_fault", kind: "abnormal_sag", magnitude: 5 } },
      { label: "CG shift aft", spec: { type: "airframe_fault", kind: "cg_shift", magnitude: 0.03 } },
      { label: "Payload drop", spec: { type: "airframe_fault", kind: "payload_drop", magnitude: 0.25 } },
      { label: "Wing damage (right)", spec: { type: "airframe_fault", kind: "wing_damage", magnitude: 0.15, side: 1 } },
    ],
  },
];

const SPEEDS = [0.25, 0.5, 1, 2, 4];

export function FaultPanel() {
  const live = useStore((s) => s.live);
  const send = useStore((s) => s.liveSend);
  if (!live || !send) return null;
  return (
    <div className="faultpanel" data-testid="fault-panel">
      <h2>Inject a fault</h2>
      {FAULT_GROUPS.map((g) => (
        <div key={g.title} className="faultgroup">
          <h3>{g.title}</h3>
          {g.presets.map((p) => (
            <button key={p.label} disabled={live.done} onClick={() => send({ cmd: "inject_fault", spec: p.spec })}>
              {p.label}
            </button>
          ))}
        </div>
      ))}
      {live.error && <p className="faulterror">{live.error}</p>}
    </div>
  );
}

function fmt(t: number): string {
  const m = Math.floor(t / 60);
  return `${m}:${(t - 60 * m).toFixed(1).padStart(4, "0")}`;
}

/** The live counterpart of the replay timeline. */
export function LiveBar() {
  const live = useStore((s) => s.live);
  const send = useStore((s) => s.liveSend);
  const latest = useStore((s) => s.replay?.duration ?? 0);
  if (!live || !send) return null;
  return (
    <div className="timeline live">
      <span className={`livedot ${live.connected && !live.paused && !live.done ? "on" : ""}`}>●</span>
      <span className="clock" data-testid="clock">
        {live.connected ? (live.done ? "ENDED" : live.paused ? "PAUSED" : "LIVE") : "OFFLINE"} {fmt(latest)}
      </span>
      <button onClick={() => send({ cmd: live.paused ? "resume" : "pause" })}>{live.paused ? "▶" : "❚❚"}</button>
      {SPEEDS.map((x) => (
        <button key={x} className={x === live.speed ? "on" : ""} onClick={() => send({ cmd: "set_speed", x })}>
          {x}×
        </button>
      ))}
      <button onClick={() => send({ cmd: "reset" })} title="Restart the scenario (the session so far is recorded)">
        ↺ restart
      </button>
    </div>
  );
}
