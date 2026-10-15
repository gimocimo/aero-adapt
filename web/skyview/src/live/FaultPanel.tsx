// Live-mode controls: inject (and clear) faults while the aircraft flies, pick the scenario
// and airframe, and pace the server.
import { useState } from "react";
import { useStore } from "../state/store";
import type { LiveFault } from "./client";

type Choice = "channel" | "body_axis" | "ned_axis" | "side";

export interface FaultKind {
  label: string;
  spec: Record<string, unknown>; // type, kind, fixed fields
  magnitude?: { value: number; unit: string; step: number };
  choice?: Choice;
}

const CHOICES: Record<Choice, { field: string; options: [string, string | number][] }> = {
  channel: { field: "channel", options: [["aileron", "aileron"], ["elevator", "elevator"], ["rudder", "rudder"]] },
  body_axis: { field: "axis", options: [["x", 0], ["y", 1], ["z", 2]] },
  ned_axis: { field: "axis", options: [["N", 0], ["E", 1], ["D", 2]] },
  side: { field: "side", options: [["right", 1], ["left", -1]] },
};

const m = (value: number, unit: string, step: number) => ({ value, unit, step });

/** One row per faultlab kind (M2 spec §7), with its magnitude in the kind's own units. */
export const FAULT_GROUPS: { title: string; kinds: FaultKind[] }[] = [
  {
    title: "Propulsion",
    kinds: [
      { label: "Motor cut", spec: { type: "motor_cut" } },
      { label: "Thrust loss", spec: { type: "propulsion_fault", kind: "partial_loss" }, magnitude: m(0.5, "fraction", 0.1) },
      { label: "Prop damage", spec: { type: "propulsion_fault", kind: "prop_damage" }, magnitude: m(0.3, "fraction", 0.1) },
      { label: "Motor overheat", spec: { type: "propulsion_fault", kind: "thermal_derate", ramp_s: 15 }, magnitude: m(0.6, "fraction", 0.1) },
    ],
  },
  {
    title: "Control surfaces",
    kinds: [
      { label: "Servo jam", spec: { type: "servo_jam" }, choice: "channel" },
      { label: "Hard-over", spec: { type: "actuator_fault", kind: "hard_over" }, magnitude: m(-1, "side ±1", 2), choice: "channel" },
      { label: "Floating", spec: { type: "actuator_fault", kind: "floating" }, choice: "channel" },
      { label: "Slow servo", spec: { type: "actuator_fault", kind: "slow" }, magnitude: m(0.1, "× rate", 0.1), choice: "channel" },
      { label: "Reduced throw", spec: { type: "actuator_fault", kind: "reduced_throw" }, magnitude: m(0.3, "× travel", 0.1), choice: "channel" },
      { label: "Backlash", spec: { type: "actuator_fault", kind: "backlash" }, magnitude: m(3, "°", 1), choice: "channel" },
      { label: "Reversed", spec: { type: "actuator_fault", kind: "reversed" }, choice: "channel" },
    ],
  },
  {
    title: "Air data",
    kinds: [
      { label: "Pitot blocked (drain open)", spec: { type: "pitot_blockage", variant: "drain_open" } },
      { label: "Pitot blocked (drain blocked)", spec: { type: "pitot_blockage", variant: "drain_blocked" } },
      { label: "Airspeed bias", spec: { type: "sensor_fault", kind: "airspeed_bias" }, magnitude: m(5, "m/s", 1) },
      { label: "Airspeed stuck", spec: { type: "sensor_fault", kind: "airspeed_stuck" } },
      { label: "Airspeed drift", spec: { type: "sensor_fault", kind: "airspeed_drift" }, magnitude: m(0.5, "m/s per s", 0.1) },
      { label: "Baro drift", spec: { type: "sensor_fault", kind: "baro_drift" }, magnitude: m(20, "Pa/s", 5) },
    ],
  },
  {
    title: "Inertial and compass",
    kinds: [
      { label: "Gyro bias", spec: { type: "sensor_fault", kind: "gyro_bias" }, magnitude: m(0.05, "rad/s", 0.01), choice: "body_axis" },
      { label: "Accel bias", spec: { type: "sensor_fault", kind: "accel_bias" }, magnitude: m(1, "m/s²", 0.5), choice: "body_axis" },
      { label: "IMU vibration", spec: { type: "sensor_fault", kind: "imu_vibration" }, magnitude: m(8, "m/s² RMS", 1) },
      { label: "Compass interference", spec: { type: "sensor_fault", kind: "mag_interference" }, magnitude: m(0.3, "gauss", 0.1), choice: "body_axis" },
    ],
  },
  {
    title: "GNSS",
    kinds: [
      { label: "GNSS dropout", spec: { type: "sensor_fault", kind: "gnss_dropout" }, magnitude: m(0, "s (0: until cleared)", 5) },
      { label: "GNSS drift", spec: { type: "sensor_fault", kind: "gnss_drift" }, magnitude: m(2, "m/s", 0.5), choice: "ned_axis" },
      { label: "GNSS spoofing", spec: { type: "sensor_fault", kind: "gnss_spoof" }, magnitude: m(0.3, "m/s²", 0.1), choice: "ned_axis" },
    ],
  },
  {
    title: "Battery",
    kinds: [
      { label: "Cell failure", spec: { type: "energy_fault", kind: "cell_failure" } },
      { label: "Abnormal sag", spec: { type: "energy_fault", kind: "abnormal_sag" }, magnitude: m(5, "× resistance", 1) },
      { label: "Capacity loss", spec: { type: "energy_fault", kind: "capacity_loss" }, magnitude: m(0.6, "fraction", 0.1) },
    ],
  },
  {
    title: "Airframe",
    kinds: [
      { label: "CG shift", spec: { type: "airframe_fault", kind: "cg_shift" }, magnitude: m(0.03, "m aft", 0.01) },
      { label: "Payload drop", spec: { type: "airframe_fault", kind: "payload_drop" }, magnitude: m(0.25, "fraction", 0.05) },
      { label: "Wing damage", spec: { type: "airframe_fault", kind: "wing_damage" }, magnitude: m(0.15, "fraction", 0.05), choice: "side" },
    ],
  },
];

const ALL_KINDS = FAULT_GROUPS.flatMap((g) => g.kinds);

/** The display name of an injected fault spec. */
export function faultLabel(spec: Record<string, unknown>): string {
  const k = ALL_KINDS.find((f) => Object.entries(f.spec).every(([key, v]) => key === "ramp_s" || spec[key] === v));
  const base = k?.label ?? String(spec.kind ?? spec.type);
  return spec.channel ? `${base} (${String(spec.channel)})` : base;
}

function FaultRow({ k, disabled, onInject }: { k: FaultKind; disabled: boolean; onInject: (spec: Record<string, unknown>) => void }) {
  const [mag, setMag] = useState(k.magnitude?.value ?? 0);
  const choice = k.choice ? CHOICES[k.choice] : null;
  const [pick, setPick] = useState<string | number>(choice?.options[0]?.[1] ?? 0);
  const inject = () =>
    onInject({ ...k.spec, ...(k.magnitude ? { magnitude: mag } : {}), ...(choice ? { [choice.field]: pick } : {}) });
  return (
    <div className="faultrow">
      <button disabled={disabled} onClick={inject}>
        {k.label}
      </button>
      {k.magnitude && (
        <label title={k.magnitude.unit}>
          <input
            type="number"
            value={mag}
            step={k.magnitude.step}
            onChange={(e) => setMag(Number(e.target.value))}
            aria-label={`${k.label} magnitude (${k.magnitude.unit})`}
          />
          <span className="unit">{k.magnitude.unit}</span>
        </label>
      )}
      {choice && (
        <select
          value={String(pick)}
          onChange={(e) => {
            const opt = choice.options.find(([, v]) => String(v) === e.target.value);
            if (opt) setPick(opt[1]);
          }}
          aria-label={`${k.label} ${choice.field}`}
        >
          {choice.options.map(([name, v]) => (
            <option key={name} value={String(v)}>
              {name}
            </option>
          ))}
        </select>
      )}
    </div>
  );
}

function ActiveFaults({ faults, send }: { faults: LiveFault[]; send: (m: Record<string, unknown>) => void }) {
  if (faults.length === 0) return null;
  return (
    <ul className="activefaults" data-testid="active-faults">
      {faults.map((f) => (
        <li key={f.index}>
          <span>
            {faultLabel(f.spec)} @ {f.spec.trigger.at_time_s.toFixed(1)} s
            {f.spec.trigger.until_s !== null && ` → cleared ${f.spec.trigger.until_s.toFixed(1)} s`}
          </span>
          {f.clearable && <button onClick={() => send({ cmd: "clear_fault", index: f.index })}>clear</button>}
        </li>
      ))}
    </ul>
  );
}

export function FaultPanel() {
  const live = useStore((s) => s.live);
  const send = useStore((s) => s.liveSend);
  if (!live || !send) return null;
  const reset = (p: { scenario?: string; airframe?: string | null }) =>
    send({ cmd: "reset", scenario: p.scenario ?? live.scenario, airframe: p.airframe === undefined ? live.airframe : p.airframe });
  return (
    <div className="faultpanel" data-testid="fault-panel">
      <div className="pickers">
        <label>
          Scenario
          <select value={live.scenario} onChange={(e) => reset({ scenario: e.target.value })} data-testid="scenario-picker">
            {live.scenarios.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
        <label>
          Airframe
          <select
            value={live.airframe ?? ""}
            onChange={(e) => reset({ airframe: e.target.value || null })}
            data-testid="airframe-picker"
          >
            <option value="">scenario airframe</option>
            {live.airframes.map((a) => (
              <option key={a.id} value={a.id}>
                {a.id} ({a.v_cruise_mps.toFixed(0)} m/s)
              </option>
            ))}
          </select>
        </label>
      </div>
      <ActiveFaults faults={live.faults} send={send} />
      {live.error && <p className="faulterror">{live.error}</p>}
      <h2>Inject a fault</h2>
      {FAULT_GROUPS.map((g) => (
        <details key={g.title} className="faultgroup" open={g.title === "Propulsion"}>
          <summary>{g.title}</summary>
          {g.kinds.map((k) => (
            <FaultRow key={k.label} k={k} disabled={live.done} onInject={(spec) => send({ cmd: "inject_fault", spec })} />
          ))}
        </details>
      ))}
    </div>
  );
}

function fmt(t: number): string {
  const min = Math.floor(t / 60);
  return `${min}:${(t - 60 * min).toFixed(1).padStart(4, "0")}`;
}

const SPEEDS = [0.25, 0.5, 1, 2, 4];

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
