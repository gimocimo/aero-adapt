// The full-screen layout's avionics (docs/07 §8, bottom left): an attitude ball under a heading
// arc, speed and height either side with the autopilot's targets, and four gauges. The
// four-panel layout keeps the larger ``Instruments``.
import type { ReactElement } from "react";
import { alerts, EndCard } from "../hud/Hud";
import type { Sample } from "../replay/interp";
import type { Replay } from "../replay/types";
import { fmt, navTarget, wrap360 } from "./Instruments";

const DEG = 180 / Math.PI;
const W = 330;
const H = 176;
const CX = W / 2;
const CY = 96;
const BALL = 52; // the attitude ball's radius
const ARC = 66; // the heading arc's radius
const SPAN = 50; // the arc shows ±50° of heading
const K = 2; // px per degree of pitch

const clamp = (x: number, lo: number, hi: number) => Math.min(Math.max(x, lo), hi);
/** A point ``deg`` clockwise from straight up, ``r`` from the ball's centre. */
const polar = (deg: number, r: number): [number, number] => [CX + r * Math.sin(deg / DEG), CY - r * Math.cos(deg / DEG)];
/** ``deg`` relative to the heading, in [-180, 180). */
const rel = (deg: number, hdg: number) => ((((deg - hdg + 180) % 360) + 360) % 360) - 180;

/** Heading-up arc: ticks every 10°, a heading box, the ground track (green) and the bearing to
 * home or the chosen zone (magenta), pinned to the arc's ends when out of view. */
function HeadingArc({ s, brg }: { s: Sample; brg: number }) {
  const hdg = wrap360(s.yaw * DEG);
  const marks: ReactElement[] = [];
  for (let d = Math.ceil((hdg - SPAN) / 10) * 10; d <= hdg + SPAN; d += 10) {
    const a = d - hdg;
    const h = wrap360(d);
    const major = h % 30 === 0;
    const [x1, y1] = polar(a, ARC);
    const [x2, y2] = polar(a, ARC - (major ? 7 : 4));
    marks.push(<line key={d} x1={x1} y1={y1} x2={x2} y2={y2} className="av-tick" />);
    if (major && Math.abs(a) > 21 && Math.abs(a) < 45) {
      // clear of the heading box
      const [x, y] = polar(a, ARC + 12);
      marks.push(
        <text key={`n${d}`} x={x} y={y + 4} textAnchor="middle" className={h === 0 ? "av-scale av-north" : "av-scale"}>
          {h % 90 === 0 ? "NESW"[h / 90] : h / 10}
        </text>,
      );
    }
  }
  const [x0, y0] = polar(-SPAN, ARC);
  const [x1, y1] = polar(SPAN, ARC);
  const track = rel(s.course * DEG, hdg);
  return (
    <g>
      <path d={`M ${x0} ${y0} A ${ARC} ${ARC} 0 0 1 ${x1} ${y1}`} className="av-arc" />
      {marks}
      {Number.isFinite(track) && (
        <path
          d={`M ${CX - 3.5} ${CY - ARC - 8} h 7 l -3.5 6 z`}
          transform={`rotate(${clamp(track, -SPAN, SPAN)} ${CX} ${CY})`}
          className="av-track"
        />
      )}
      <path
        d={`M ${CX} ${CY - ARC - 4} l 4 4 -4 4 -4 -4 z`}
        transform={`rotate(${clamp(rel(brg, hdg), -SPAN, SPAN)} ${CX} ${CY})`}
        className="av-bearing"
      />
      <rect x={CX - 22} y={4} width={44} height={18} rx={4} className="av-hdgbox" />
      <text x={CX} y={17.5} textAnchor="middle" className="av-hdg">
        {fmt(hdg).padStart(3, "0")}°
      </text>
      <path d={`M ${CX} ${CY - ARC + 3} l -4 -6 h 8 z`} className="av-caret" />
    </g>
  );
}

/** The attitude ball: sky and ground, a pitch ladder, the roll scale and pointer, and the
 * flight director's bars (the demanded pitch and roll, relative to the aircraft). */
function Ball({ s }: { s: Sample }) {
  const r = s.roll * DEG;
  const p = s.pitch * DEG;
  const fd = Number.isFinite(s.rollDem) && Number.isFinite(s.pitchDem);
  const fdY = CY + clamp((p - s.pitchDem * DEG) * K, -40, 40);
  const fdX = CX + clamp((s.rollDem * DEG - r) * 1.2, -40, 40);
  return (
    <g>
      <clipPath id="av-ball">
        <circle cx={CX} cy={CY} r={BALL} />
      </clipPath>
      <g clipPath="url(#av-ball)">
        <g transform={`rotate(${-r} ${CX} ${CY}) translate(0 ${p * K})`}>
          <rect x={CX - 160} y={CY - 400} width={320} height={400} className="av-sky" />
          <rect x={CX - 160} y={CY} width={320} height={400} className="av-ground" />
          <line x1={CX - 160} x2={CX + 160} y1={CY} y2={CY} className="av-horizon" />
          {[-20, -15, -10, -5, 5, 10, 15, 20].map((d) => {
            const w = d % 10 === 0 ? 12 : 6;
            return <line key={d} x1={CX - w} x2={CX + w} y1={CY - d * K} y2={CY - d * K} className="av-ladder" />;
          })}
        </g>
        {fd && <path d={`M ${CX - 24} ${fdY} H ${CX + 24} M ${fdX} ${CY - 24} V ${CY + 24}`} className="av-director" />}
      </g>
      {[-45, -30, -20, -10, 10, 20, 30, 45].map((a) => {
        const [x1, y1] = polar(a, BALL);
        const [x2, y2] = polar(a, BALL - (Math.abs(a) === 30 ? 8 : 6));
        return <line key={a} x1={x1} y1={y1} x2={x2} y2={y2} className="av-roll" />;
      })}
      <path d={`M ${CX} ${CY - BALL} l -3.5 -5 h 7 z`} className="av-index" />
      <path d={`M ${CX} ${CY - BALL + 6} l -4 7 h 8 z`} transform={`rotate(${-r} ${CX} ${CY})`} className="av-pointer" />
      <path d={`M ${CX - 34} ${CY} h 18 M ${CX + 16} ${CY} h 18`} className="av-aircraft" />
      <circle cx={CX} cy={CY} r={2.5} className="av-dot" />
    </g>
  );
}

/** ``x`` with its unit, or a dash when the flight did not record it. */
const val = (x: number, unit: string, digits = 0) => (Number.isFinite(x) ? `${x.toFixed(digits)}${unit}` : "—");

function Cell(props: { label: string; value: string; frac: number; tone: string; testid?: string }) {
  const f = Number.isFinite(props.frac) ? clamp(props.frac, 0, 1) : 0;
  return (
    <div className={`av-cell ${props.tone}`} data-testid={props.testid}>
      <span className="av-k">{props.label}</span> <span className="av-v">{props.value}</span>
      <span className="bar">
        <i style={{ width: `${100 * f}%` }} />
      </span>
    </div>
  );
}

export function Avionics({ s, replay }: { s: Sample; replay: Replay }) {
  const { mode: modes, phase: phases } = replay.manifest.enums;
  const world = !!replay.manifest.site.world;
  const target = navTarget(s, replay);
  const dist = target.dist >= 1000 ? `${(target.dist / 1000).toFixed(1)} km` : `${fmt(target.dist)} m`;
  const vs = clamp(s.climb * 4, -20, 20);
  const clr = s.clearance;
  const al = alerts(s, replay);
  return (
    <div className="avionics" data-testid="instruments">
      <div className="av-head">
        <span className="av-mode">{modes[s.mode] ?? "?"}</span>
        <span className={s.failed ? "av-phase warn" : "av-phase"}>{s.failed ? phases[s.phase] : "monitoring"}</span>
      </div>
      {al.length > 0 && (
        <div className="av-alerts">
          {al.map((a) => (
            <span key={a.text} className={`badge ${a.cls}`}>
              {a.text}
            </span>
          ))}
        </div>
      )}
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} className="av-main" data-testid="hud">
        <HeadingArc s={s} brg={target.brg} />
        <Ball s={s} />
        <text x={14} y={76} className="av-lbl">
          IAS m/s
        </text>
        <text x={14} y={106} className="av-big" data-testid="hud-IAS">
          {fmt(s.ias, 1)}
        </text>
        {Number.isFinite(s.speedTarget) && (
          <text x={14} y={124} className="av-tgt">
            → {fmt(s.speedTarget, 1)}
          </text>
        )}
        <text x={14} y={140} className="av-sub">
          TAS {fmt(s.tasTrue, 1)} · GS {fmt(s.groundspeed, 1)}
        </text>
        <line x1={236} x2={236} y1={CY - 20} y2={CY + 20} className="av-vsi" />
        <line x1={232} x2={240} y1={CY} y2={CY} className="av-vsi" />
        <rect x={234} y={Math.min(CY, CY - vs)} width={4} height={Math.abs(vs)} className="av-vs" />
        <text x={W - 14} y={76} textAnchor="end" className="av-lbl">
          ALT {world ? "m (home)" : "m AGL"}
        </text>
        <text x={W - 14} y={106} textAnchor="end" className="av-big">
          {fmt(s.altAgl)}
        </text>
        {Number.isFinite(s.altTarget) && (
          <text x={W - 14} y={124} textAnchor="end" className="av-tgt">
            → {fmt(s.altTarget)}
          </text>
        )}
        <text x={W - 14} y={140} textAnchor="end" className="av-sub">
          {s.climb < 0 ? "▼" : "▲"} {fmt(Math.abs(s.climb), 1)} m/s
        </text>
        <text x={CX} y={168} textAnchor="middle" className="av-home">
          {target.name} {dist}
        </text>
      </svg>
      <div className="av-cells">
        <Cell label="THR" value={val(100 * s.throttle, "%")} frac={s.throttle} tone="neutral" />
        <Cell
          label="SoC"
          value={val(100 * s.soc, "%")}
          frac={s.soc}
          tone={s.soc < 0.15 ? "alert" : s.soc < 0.3 ? "warn" : "ok"}
        />
        <Cell label="AGL" value={val(s.hagl, " m")} frac={s.hagl / 150} tone={s.hagl < 40 ? "warn" : "ok"} testid="gauge-agl" />
        <Cell label="CLR" value={val(clr, " m")} frac={clr / 150} tone={clr < 10 ? "alert" : clr < 30 ? "warn" : "ok"} testid="gauge-clearance" />
      </div>
      <EndCard s={s} replay={replay} />
    </div>
  );
}
