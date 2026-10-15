import type { Sample } from "../replay/interp";
import { faultName } from "../replay/names";
import type { Replay } from "../replay/types";

const DEG = 180 / Math.PI;
const W = 440;
const H = 300;
const CX = W / 2;
const CY = 135;
const PX_PER_DEG = 4;

function fmt(x: number, digits = 1): string {
  return Number.isFinite(x) ? x.toFixed(digits) : "—";
}

/** Vertical tape: `value` at the centre, ticks every `step`, ±`span` visible. */
function Tape(props: {
  x: number;
  value: number;
  span: number;
  step: number;
  label: string;
  unit: string;
  markers: { value: number; color: string; title: string; shape: "bug" | "caret" }[];
  side: "left" | "right";
}) {
  const { x, value, span, step, label, unit, markers, side } = props;
  const h = 180;
  const top = CY - h / 2;
  const scale = h / (2 * span);
  const y = (v: number) => CY - (v - value) * scale;
  const ticks: number[] = [];
  for (let v = Math.ceil((value - span) / step) * step; v <= value + span; v += step) ticks.push(v);
  const w = 62;
  const textX = side === "left" ? x + w - 8 : x + 8;
  const anchor = side === "left" ? "end" : "start";
  return (
    <g>
      <rect x={x} y={top} width={w} height={h} className="hud-tape" />
      <clipPath id={`clip-${label}`}>
        <rect x={x} y={top} width={w} height={h} />
      </clipPath>
      <g clipPath={`url(#clip-${label})`}>
        {ticks.map((v) => (
          <g key={v}>
            <line
              x1={side === "left" ? x + w - 6 : x}
              x2={side === "left" ? x + w : x + 6}
              y1={y(v)}
              y2={y(v)}
              className="hud-tick"
            />
            {Math.round(v / step) % 2 === 0 && (
              <text x={textX - (side === "left" ? 8 : -8)} y={y(v) + 4} textAnchor={anchor} className="hud-small">
                {v.toFixed(0)}
              </text>
            )}
          </g>
        ))}
        {markers.map((m) =>
          m.shape === "bug" ? (
            <rect
              key={m.title}
              x={side === "left" ? x + w - 8 : x}
              y={y(m.value) - 3}
              width={8}
              height={6}
              fill={m.color}
            >
              <title>{m.title}</title>
            </rect>
          ) : (
            <polygon
              key={m.title}
              points={
                side === "left"
                  ? `${x + w},${y(m.value)} ${x + w - 12},${y(m.value) - 7} ${x + w - 12},${y(m.value) + 7}`
                  : `${x},${y(m.value)} ${x + 12},${y(m.value) - 7} ${x + 12},${y(m.value) + 7}`
              }
              fill={m.color}
            >
              <title>{m.title}</title>
            </polygon>
          ),
        )}
      </g>
      <rect x={x + 4} y={CY - 13} width={w - 8} height={26} className="hud-readout" />
      <text x={x + w / 2} y={CY + 6} textAnchor="middle" className="hud-big" data-testid={`hud-${label}`}>
        {fmt(value)}
      </text>
      <text x={x + w / 2} y={top - 6} textAnchor="middle" className="hud-small">
        {label} {unit}
      </text>
    </g>
  );
}

function Attitude({ roll, pitch, rollDem, pitchDem }: { roll: number; pitch: number; rollDem: number; pitchDem: number }) {
  const r = roll * DEG;
  const p = pitch * DEG;
  const ladder = [-20, -10, 10, 20];
  return (
    <g>
      <clipPath id="att-clip">
        <circle cx={CX} cy={CY} r={86} />
      </clipPath>
      <g clipPath="url(#att-clip)">
        <g transform={`rotate(${-r} ${CX} ${CY}) translate(0 ${p * PX_PER_DEG})`}>
          <rect x={CX - 300} y={CY - 600} width={600} height={600} className="hud-sky" />
          <rect x={CX - 300} y={CY} width={600} height={600} className="hud-ground" />
          <line x1={CX - 300} x2={CX + 300} y1={CY} y2={CY} className="hud-horizon" />
          {ladder.map((d) => (
            <g key={d}>
              <line x1={CX - 22} x2={CX + 22} y1={CY - d * PX_PER_DEG} y2={CY - d * PX_PER_DEG} className="hud-ladder" />
              <text x={CX + 28} y={CY - d * PX_PER_DEG + 4} className="hud-small">
                {d}
              </text>
            </g>
          ))}
        </g>
        {/* flight-director bars: demanded roll and pitch, relative to the aircraft */}
        <g transform={`rotate(${-(rollDem - roll) * DEG} ${CX} ${CY}) translate(0 ${(p - pitchDem * DEG) * PX_PER_DEG})`}>
          <line x1={CX - 40} x2={CX + 40} y1={CY} y2={CY} className="hud-director" />
        </g>
      </g>
      <circle cx={CX} cy={CY} r={86} className="hud-ring" />
      <path d={`M ${CX - 46} ${CY} h 30 l 8 8 l 8 -8 h 30`} className="hud-aircraft" />
      <circle cx={CX} cy={CY} r={2.5} fill="#ffd84c" />
      <polygon
        points={`${CX},${CY - 84} ${CX - 6},${CY - 74} ${CX + 6},${CY - 74}`}
        transform={`rotate(${-r} ${CX} ${CY})`}
        fill="#ffd84c"
      />
    </g>
  );
}

export function Hud({ s, replay }: { s: Sample; replay: Replay }) {
  const { mode: modes, phase: phases, fault_bits } = replay.manifest.enums;
  const heading = ((s.yaw * DEG) % 360 + 360) % 360;
  const phase = phases[s.phase] ?? "";
  const faults = Object.entries(fault_bits)
    .filter(([, bit]) => (s.faultBits & bit) !== 0)
    .map(([name]) => {
      // The latest onset: a cleared sensor fault may be injected again.
      const ev = replay.events
        .filter((e) => e.type === "fault_onset" && e.payload.fault === name && e.time_us * 1e-6 <= s.time)
        .at(-1);
      return `${faultName(name)} @ ${ev ? (ev.time_us * 1e-6).toFixed(1) : "?"} s`;
    });
  const td = replay.events.find((e) => e.type === "touchdown");
  const ended = td && s.time >= td.time_us * 1e-6;
  const speedErr = s.ias - s.tasTrue;
  return (
    <div className="hud" data-testid="hud">
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H}>
        <Attitude roll={s.roll} pitch={s.pitch} rollDem={s.rollDem} pitchDem={s.pitchDem} />
        <Tape
          x={14}
          value={s.ias}
          span={10}
          step={1}
          label="IAS"
          unit="m/s"
          side="left"
          markers={[
            { value: s.tasTrue, color: "#7dff7a", title: "true airspeed (truth)", shape: "caret" },
            { value: s.speedTarget, color: "#e070ff", title: "speed target", shape: "bug" },
          ]}
        />
        <Tape
          x={W - 76}
          value={s.altAgl}
          span={40}
          step={5}
          label="ALT"
          unit="m AGL"
          side="right"
          markers={[{ value: s.altTarget, color: "#e070ff", title: "altitude target", shape: "bug" }]}
        />
        <text x={14 + 31} y={CY + 112} textAnchor="middle" className="hud-truth">
          TAS {fmt(s.tasTrue)}
        </text>
        <text x={14 + 31} y={CY + 126} textAnchor="middle" className="hud-small">
          (truth)
        </text>
        <text x={W - 45} y={CY + 112} textAnchor="middle" className="hud-small">
          VS {fmt(s.climb)} m/s
        </text>
        <text x={CX} y={H - 28} textAnchor="middle" className="hud-big">
          HDG {heading.toFixed(0).padStart(3, "0")}°
        </text>
        <text x={CX} y={H - 10} textAnchor="middle" className="hud-small">
          α {fmt(s.alpha * DEG)}°  β {fmt(s.beta * DEG)}°  GS {fmt(s.groundspeed)} m/s
        </text>
        <rect x={W - 18} y={CY - 60} width={8} height={120} className="hud-tape" />
        <rect x={W - 18} y={CY + 60 - 120 * s.throttle} width={8} height={120 * s.throttle} fill="#4cc2ff" />
        <text x={W - 14} y={CY + 74} textAnchor="middle" className="hud-small">
          THR
        </text>
      </svg>
      <div className="hud-badges">
        <span className="badge">{modes[s.mode] ?? "?"}</span>
        {s.failed && <span className="badge badge-warn">{phase}</span>}
        {!s.airspeedHealthy && <span className="badge badge-alert">AIRSPEED UNRELIABLE</span>}
        {Math.abs(speedErr) > 3 && <span className="badge badge-alert">IAS−TAS {fmt(speedErr)}</span>}
        {faults.map((f) => (
          <span key={f} className="badge badge-fault">
            FAULT: {f}
          </span>
        ))}
      </div>
      {ended && td && (
        <div className="touchdown-card" data-testid="touchdown">
          <strong>Touchdown: {String(td.payload.class)}</strong>
          <span>
            sink {fmt(Number(td.payload.sink_rate_mps))} m/s · GS {fmt(Number(td.payload.groundspeed_mps))} m/s
          </span>
          <span>
            {td.payload.zone ? `zone ${String(td.payload.zone)} · ${td.payload.in_zone ? "inside" : "outside"}` : "no zone"}
          </span>
        </div>
      )}
    </div>
  );
}
