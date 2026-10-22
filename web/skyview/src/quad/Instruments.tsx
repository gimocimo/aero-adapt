// The instrument panel (bottom left): the primary flight display, a navigation display, the
// energy and terrain gauges and the contingency's state. Truth values are marked as such.
import { EndCard, PfdSvg, StatusBadges } from "../hud/Hud";
import type { Sample } from "../replay/interp";
import type { Replay } from "../replay/types";

const DEG = 180 / Math.PI;

function fmt(x: number, digits = 0): string {
  return Number.isFinite(x) ? x.toFixed(digits) : "—";
}

function wrap360(deg: number): number {
  return ((deg % 360) + 360) % 360;
}

/** Heading-up navigation display: compass card, ground track, wind, bearing to the zone the
 * contingency chose (or home). */
function NavDisplay({ s, replay }: { s: Sample; replay: Replay }) {
  const R = 92;
  const C = 110;
  const hdg = s.yaw * DEG;
  const rot = (deg: number) => deg - hdg; // card angle on screen (0 = up)
  const zone = s.zone >= 0 ? replay.manifest.site.safe_zones[s.zone] : undefined;
  let target: { name: string; n: number; e: number } = { name: "home", n: 0, e: 0 };
  if (zone) {
    const [n, e] = zone.polygon_ne_m.reduce(([a, b], [x, y]) => [a + x, b + y], [0, 0]).map((v) => v / zone.polygon_ne_m.length) as [number, number];
    target = { name: `zone ${zone.id}`, n, e };
  }
  const brg = Math.atan2(target.e - s.e, target.n - s.n) * DEG;
  const dist = Math.hypot(target.n - s.n, target.e - s.e);
  const windFrom = Math.atan2(-s.windE, -s.windN) * DEG;
  const windSpeed = Math.hypot(s.windN, s.windE);
  const ticks = Array.from({ length: 36 }, (_, k) => k * 10);
  const pt = (deg: number, r: number) => {
    const a = (rot(deg) * Math.PI) / 180;
    return [C + r * Math.sin(a), C - r * Math.cos(a)] as const;
  };
  return (
    <svg viewBox="0 0 220 220" className="nav-svg" data-testid="nav-display">
      <circle cx={C} cy={C} r={R + 6} className="nav-bezel" />
      {ticks.map((d) => {
        const [x1, y1] = pt(d, R);
        const [x2, y2] = pt(d, d % 30 === 0 ? R - 10 : R - 5);
        return <line key={d} x1={x1} y1={y1} x2={x2} y2={y2} className="nav-tick" />;
      })}
      {[0, 90, 180, 270].map((d) => {
        const [x, y] = pt(d, R - 20);
        return (
          <text key={d} x={x} y={y + 4} textAnchor="middle" className={d === 0 ? "nav-north" : "nav-card"}>
            {["N", "E", "S", "W"][d / 90]}
          </text>
        );
      })}
      {[30, 60, 120, 150, 210, 240, 300, 330].map((d) => {
        const [x, y] = pt(d, R - 20);
        return (
          <text key={d} x={x} y={y + 4} textAnchor="middle" className="nav-num">
            {d / 10}
          </text>
        );
      })}
      {/* bearing to the target */}
      {(() => {
        const [x1, y1] = pt(brg, R - 2);
        const [x2, y2] = pt(brg + 180, R - 2);
        return <line x1={x1} y1={y1} x2={x2} y2={y2} className="nav-bearing" />;
      })()}
      {/* ground track */}
      {Number.isFinite(s.course) &&
        (() => {
          const [x, y] = pt(s.course * DEG, R - 30);
          return <circle cx={x} cy={y} r={4} className="nav-track" />;
        })()}
      {/* wind: an arrow blowing from its direction */}
      {windSpeed > 0.5 &&
        (() => {
          const [x1, y1] = pt(windFrom, R - 34);
          const [x2, y2] = pt(windFrom, R - 62);
          return <line x1={x1} y1={y1} x2={x2} y2={y2} className="nav-wind" markerEnd="url(#nav-arrow)" />;
        })()}
      <defs>
        <marker id="nav-arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="#7fd0ff" />
        </marker>
      </defs>
      <path d={`M ${C} ${C - 14} L ${C + 9} ${C + 10} L ${C} ${C + 5} L ${C - 9} ${C + 10} Z`} className="nav-plane" />
      <line x1={C} y1={C - R - 8} x2={C} y2={C - R + 6} className="nav-lubber" />
      <text x={6} y={14} className="nav-text">
        HDG {fmt(wrap360(hdg)).padStart(3, "0")}°
      </text>
      <text x={214} y={14} textAnchor="end" className="nav-text">
        TRK {Number.isFinite(s.course) ? fmt(wrap360(s.course * DEG)).padStart(3, "0") + "°" : "—"}
      </text>
      <text x={6} y={214} className="nav-wind-text">
        wind {fmt(windSpeed)} m/s from {fmt(wrap360(windFrom)).padStart(3, "0")}°
      </text>
      <text x={214} y={214} textAnchor="end" className="nav-target">
        {target.name} {dist >= 1000 ? `${(dist / 1000).toFixed(1)} km` : `${fmt(dist)} m`}
      </text>
    </svg>
  );
}

function Gauge(props: { label: string; value: number; unit: string; frac: number; tone?: "ok" | "warn" | "alert"; digits?: number; testid?: string }) {
  const f = Number.isFinite(props.frac) ? Math.min(Math.max(props.frac, 0), 1) : 0;
  return (
    <div className={`gauge ${props.tone ?? ""}`} data-testid={props.testid}>
      <span className="g-label">{props.label}</span>
      <span className="g-bar">
        <i style={{ width: `${100 * f}%` }} />
      </span>
      <span className="g-value">
        {fmt(props.value, props.digits ?? 0)} <small>{props.unit}</small>
      </span>
    </div>
  );
}

function ZoneChips({ s, replay }: { s: Sample; replay: Replay }) {
  const zones = replay.manifest.site.safe_zones;
  if (zones.length === 0) return null;
  return (
    <div className="zonechips" title="Safe zones: green within glide, amber chosen">
      {zones.map((z, k) => {
        const cls = s.zone === k ? "sel" : (s.reachMask & (1 << k)) !== 0 ? "reach" : "";
        return (
          <span key={z.id} className={`zchip ${cls}`}>
            {z.id}
          </span>
        );
      })}
    </div>
  );
}

export function Instruments({ s, replay }: { s: Sample; replay: Replay }) {
  const { phase: phases } = replay.manifest.enums;
  const world = !!replay.manifest.site.world;
  const clr = s.clearance;
  const clrTone = clr < 10 ? "alert" : clr < 30 ? "warn" : "ok";
  const agl = s.hagl;
  return (
    <div className="instruments" data-testid="instruments">
      <div className="inst-row">
        <div className="inst pfd" data-testid="hud">
          <PfdSvg s={s} width={330} altLabel={world ? "m (home)" : "m AGL"} />
        </div>
        <div className="inst nav">
          <NavDisplay s={s} replay={replay} />
        </div>
      </div>
      <div className="inst-row">
        <div className="inst gauges">
          <h3>Energy</h3>
          <Gauge label="THR" value={100 * s.throttle} unit="%" frac={s.throttle} />
          <Gauge label="RPM" value={s.rpm} unit="rpm" frac={s.rpm / 8000} />
          <Gauge label="BATT" value={s.batteryV} unit="V" frac={s.soc} digits={1} />
          <Gauge label="SoC" value={100 * s.soc} unit="%" frac={s.soc} tone={s.soc < 0.3 ? "warn" : "ok"} />
        </div>
        <div className="inst gauges">
          <h3>Height (truth)</h3>
          <Gauge label="AGL" value={agl} unit="m" frac={agl / 150} tone={agl < 40 ? "warn" : "ok"} digits={1} testid="gauge-agl" />
          <Gauge label="CLR" value={clr} unit="m" frac={clr / 150} tone={clrTone} digits={1} testid="gauge-clearance" />
          <Gauge label="GS" value={s.groundspeed} unit="m/s" frac={s.groundspeed / 35} digits={1} />
          <Gauge label="VS" value={s.climb} unit="m/s" frac={0.5 + s.climb / 10} digits={1} />
        </div>
        <div className="inst cont">
          <h3>Contingency</h3>
          <p className="cont-phase">{s.failed ? phases[s.phase] : "monitoring"}</p>
          <ZoneChips s={s} replay={replay} />
        </div>
      </div>
      <StatusBadges s={s} replay={replay} />
      <EndCard s={s} replay={replay} />
    </div>
  );
}
