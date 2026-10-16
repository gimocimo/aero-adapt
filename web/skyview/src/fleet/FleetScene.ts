// Every bench flight as a point over the site, coloured by what is happening to it.
import {
  Cartesian3,
  Color,
  HeadingPitchRange,
  Math as CesiumMath,
  NearFarScalar,
  type PointPrimitive,
  PointPrimitiveCollection,
  type Viewer,
} from "cesium";
import { SiteFrame } from "../geo";
import { type Fleet, type FlightSample, sampleFlight, STATE_CONTINGENCY, STATE_ENDED } from "./data";

export type FleetState = "flying" | "contingency" | "landed" | "lost";

export const FLEET_COLOURS: Record<FleetState, string> = {
  flying: "#e8f1ff",
  contingency: "#ffb24c",
  landed: "#38d16a",
  lost: "#ff4d4d",
};

const COLOURS = Object.fromEntries(
  Object.entries(FLEET_COLOURS).map(([k, v]) => [k, Color.fromCssColorString(v)]),
) as Record<FleetState, Color>;

export type FleetCounts = Record<FleetState, number>;

export class FleetScene {
  private readonly frame: SiteFrame;
  private readonly points: PointPrimitiveCollection;
  private readonly prims: PointPrimitive[] = [];
  private readonly s: FlightSample = { n: 0, e: 0, alt: 0, state: 0 };
  private readonly scratch = new Cartesian3();

  constructor(
    private readonly viewer: Viewer,
    private readonly fleet: Fleet,
  ) {
    this.frame = new SiteFrame(fleet.meta.site.origin);
    this.points = viewer.scene.primitives.add(new PointPrimitiveCollection()) as PointPrimitiveCollection;
    for (let i = 0; i < fleet.meta.flights.length; i++) {
      this.prims.push(
        this.points.add({
          pixelSize: 6,
          color: COLOURS.flying,
          outlineColor: Color.BLACK.withAlpha(0.6),
          outlineWidth: 1,
          scaleByDistance: new NearFarScalar(500, 1.6, 8000, 0.8),
        }),
      );
    }
  }

  /** The baseline's flights at time `t` (s); returns how many are in each state. */
  update(t: number, baseline: string): FleetCounts {
    const counts: FleetCounts = { flying: 0, contingency: 0, landed: 0, lost: 0 };
    const flights = this.fleet.meta.flights;
    for (let i = 0; i < flights.length; i++) {
      const p = this.prims[i]!;
      const f = flights[i]!;
      p.show = f.baseline === baseline;
      if (!p.show) continue;
      const s = sampleFlight(this.fleet, i, t, this.s);
      p.position = this.frame.toCartesian(s.n, s.e, Math.max(s.alt, 0.5), this.scratch);
      const state: FleetState =
        s.state >= STATE_ENDED ? (f.survived ? "landed" : "lost") : s.state >= STATE_CONTINGENCY ? "contingency" : "flying";
      p.color = COLOURS[state];
      counts[state]++;
    }
    return counts;
  }

  /** Look down over the site from the south. */
  home(): void {
    this.viewer.camera.setView({
      destination: this.frame.toCartesian(-3200, 0, 2600),
      orientation: { heading: 0, pitch: CesiumMath.toRadians(-42), roll: 0 },
    });
  }

  /** Orbit the site (demo capture): heading and pitch in degrees, range in metres. */
  orbit(headingDeg: number, pitchDeg: number, range: number): void {
    const centre = this.frame.toCartesian(-200, 0, 0);
    this.viewer.camera.lookAt(
      centre,
      new HeadingPitchRange(CesiumMath.toRadians(headingDeg), CesiumMath.toRadians(pitchDeg), range),
    );
  }

  destroy(): void {
    this.viewer.scene.primitives.remove(this.points);
  }
}
