import {
  Cartesian2,
  Cartesian3,
  ClassificationType,
  Color,
  ColorBlendMode,
  ColorMaterialProperty,
  ConstantPositionProperty,
  ConstantProperty,
  type Entity,
  HeadingPitchRange,
  HorizontalOrigin,
  LabelStyle,
  Material,
  Matrix4,
  PolygonHierarchy,
  type PolylineCollection,
  PolylineArrowMaterialProperty,
  PolylineCollection as PolylineCollectionCtor,
  PolylineDashMaterialProperty,
  VerticalOrigin,
  type Viewer,
} from "cesium";
import { orientationFromNedQuat } from "../cesiumOrientation";
import { SiteFrame } from "../geo";
import { sample, type Sample } from "../replay/interp";
import type { Replay } from "../replay/types";
import type { CameraMode, Layers } from "../state/store";
import { CATEGORIES, categoryOf } from "./categories";

const MODEL_URI = "/models/calibration_uav.glb";
const TRAIL_S = 60;
const TRAIL_STRIDE = 5; // 50 Hz → 10 Hz trail points
const DEG = Math.PI / 180;

const ZONE_REACHABLE = Color.fromCssColorString("#38d16a");
const ZONE_UNREACHABLE = Color.fromCssColorString("#8a949e");
const ZONE_SELECTED = Color.fromCssColorString("#ffc93c");

interface Prepared {
  replay: Replay;
  positions: Cartesian3[]; // every TRAIL_STRIDE-th sample
  category: Uint8Array;
}

/**
 * Owns every Cesium object of a replay and updates them imperatively once per animation
 * frame (no React re-render in the 60 fps path).
 */
export class SceneController {
  private frame: SiteFrame | null = null;
  private main: Prepared | null = null;
  private ghost: Prepared | null = null;
  private aircraft: Entity;
  private ghostAircraft: Entity;
  private wind: Entity;
  private touchdown: Entity;
  private mission: Entity | null = null;
  private zones: { fill: Entity; border: Entity; label: Entity }[] = [];
  private zoneState = "";
  private trail: PolylineCollection;
  private trailBuiltAt = -1;
  private trailKey = "";
  private smoothHeading: number | null = null;
  private lastCamera: CameraMode | null = null;

  constructor(private readonly viewer: Viewer) {
    const ents = viewer.entities;
    this.aircraft = ents.add({
      show: false,
      model: { uri: MODEL_URI, minimumPixelSize: 56, maximumScale: 25 },
    });
    this.ghostAircraft = ents.add({
      show: false,
      model: {
        uri: MODEL_URI,
        minimumPixelSize: 56,
        maximumScale: 25,
        color: Color.WHITE.withAlpha(0.45),
        colorBlendMode: ColorBlendMode.MIX,
        colorBlendAmount: 0.7,
      },
    });
    this.wind = ents.add({
      show: false,
      polyline: {
        width: 12,
        material: new PolylineArrowMaterialProperty(Color.WHITE.withAlpha(0.85)),
      },
    });
    this.touchdown = ents.add({
      show: false,
      point: { pixelSize: 12, color: Color.WHITE, outlineColor: Color.BLACK, outlineWidth: 2 },
      label: {
        text: "",
        font: "13px sans-serif",
        style: LabelStyle.FILL_AND_OUTLINE,
        outlineWidth: 3,
        outlineColor: Color.BLACK,
        verticalOrigin: VerticalOrigin.BOTTOM,
        pixelOffset: new Cartesian2(0, -14),
      },
    });
    this.trail = viewer.scene.primitives.add(new PolylineCollectionCtor()) as PolylineCollection;
  }

  destroy(): void {
    this.viewer.scene.primitives.remove(this.trail);
  }

  private prepare(replay: Replay): Prepared {
    const frame = this.frame!;
    const c = replay.cols;
    const { mode: modes, phase: phases } = replay.manifest.enums;
    const positions: Cartesian3[] = [];
    const category: number[] = [];
    for (let i = 0; i < replay.n; i += TRAIL_STRIDE) {
      positions.push(frame.toCartesian(c["truth.pos_n"]![i]!, c["truth.pos_e"]![i]!, c["truth.alt_agl"]![i]!));
      category.push(categoryOf(phases[c["cont.phase"]![i]!] ?? "", modes[c.mode![i]!] ?? ""));
    }
    return { replay, positions, category: Uint8Array.from(category) };
  }

  setReplays(main: Replay | null, ghost: Replay | null): void {
    // Site-dependent objects are rebuilt when the main replay changes.
    if (main && (!this.main || this.main.replay !== main)) {
      this.frame = new SiteFrame(main.manifest.site.origin);
      this.main = this.prepare(main);
      this.buildMission(main);
      this.buildZones(main);
      this.trailBuiltAt = -1;
      this.smoothHeading = null;
    }
    if (!main) this.main = null;
    this.ghost = ghost && this.frame ? this.prepare(ghost) : null;
    this.trailBuiltAt = -1;
  }

  private buildMission(r: Replay): void {
    if (this.mission) this.viewer.entities.remove(this.mission);
    const wps = r.manifest.mission.waypoints_ne_alt_m;
    const pts = wps.map(([n, e, h]) => this.frame!.toCartesian(n, e, h));
    if (r.manifest.mission.on_complete === "loop" && pts.length > 1) pts.push(pts[0]!);
    this.mission = this.viewer.entities.add({
      polyline: {
        positions: pts,
        width: 2,
        material: new PolylineDashMaterialProperty({
          color: Color.WHITE.withAlpha(0.7),
          dashLength: 24,
        }),
      },
    });
  }

  private buildZones(r: Replay): void {
    for (const z of this.zones) {
      this.viewer.entities.remove(z.fill);
      this.viewer.entities.remove(z.border);
      this.viewer.entities.remove(z.label);
    }
    this.zones = r.manifest.site.safe_zones.map((zone) => {
      const deg = zone.polygon_ne_m.map(([n, e]) => this.frame!.toDegrees(n, e));
      const ring = deg.flat();
      const closed = [...ring, ...deg[0]!];
      const [cn, ce] = zone.polygon_ne_m
        .reduce(([a, b], [n, e]) => [a + n, b + e], [0, 0])
        .map((v) => v / zone.polygon_ne_m.length) as [number, number];
      return {
        fill: this.viewer.entities.add({
          polygon: {
            hierarchy: new PolygonHierarchy(Cartesian3.fromDegreesArray(ring)),
            material: new ColorMaterialProperty(ZONE_UNREACHABLE.withAlpha(0.3)),
            classificationType: ClassificationType.BOTH,
          },
        }),
        border: this.viewer.entities.add({
          polyline: {
            positions: Cartesian3.fromDegreesArray(closed),
            clampToGround: true,
            width: 3,
            material: new ColorMaterialProperty(ZONE_UNREACHABLE),
            classificationType: ClassificationType.BOTH,
          },
        }),
        label: this.viewer.entities.add({
          position: this.frame!.toCartesian(cn, ce, 25),
          label: {
            text: `Zone ${zone.id}`,
            font: "bold 14px sans-serif",
            style: LabelStyle.FILL_AND_OUTLINE,
            outlineWidth: 3,
            outlineColor: Color.BLACK,
            horizontalOrigin: HorizontalOrigin.CENTER,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
        }),
      };
    });
    this.zoneState = "";
  }

  private updateZones(s: Sample, show: boolean): void {
    const key = `${show}|${s.reachMask}|${s.zone}|${s.failed}`;
    if (key === this.zoneState) return;
    this.zoneState = key;
    this.zones.forEach((z, k) => {
      const reachable = (s.reachMask & (1 << k)) !== 0;
      const color = s.zone === k ? ZONE_SELECTED : reachable ? ZONE_REACHABLE : ZONE_UNREACHABLE;
      z.fill.show = show;
      z.border.show = show;
      z.label.show = show;
      z.fill.polygon!.material = new ColorMaterialProperty(color.withAlpha(0.3));
      z.border.polyline!.material = new ColorMaterialProperty(color);
    });
  }

  private rebuildTrail(p: Prepared, time: number, alpha: number): void {
    const r = p.replay;
    const end = Math.min(Math.floor((time * r.manifest.rates.viz_hz) / TRAIL_STRIDE), p.positions.length - 1);
    const start = Math.max(0, end - Math.round((TRAIL_S * r.manifest.rates.viz_hz) / TRAIL_STRIDE));
    let runStart = start;
    for (let i = start + 1; i <= end + 1; i++) {
      if (i > end || p.category[i] !== p.category[runStart]) {
        const pts = p.positions.slice(runStart, Math.min(i + 1, end + 1));
        if (pts.length >= 2) {
          this.trail.add({
            positions: pts,
            width: 3,
            material: Material.fromType("Color", {
              color: Color.fromCssColorString(CATEGORIES[p.category[runStart]!]!.css).withAlpha(alpha),
            }),
          });
        }
        runStart = i;
      }
    }
  }

  private place(entity: Entity, s: Sample): Cartesian3 {
    const pos = this.frame!.toCartesian(s.n, s.e, s.altAgl);
    entity.position = new ConstantPositionProperty(pos);
    entity.orientation = new ConstantProperty(orientationFromNedQuat(pos, s.quat));
    return pos;
  }

  /** Called once per animation frame. */
  update(time: number, layers: Layers, camera: CameraMode): Sample | null {
    if (!this.main || !this.frame) {
      this.aircraft.show = false;
      return null;
    }
    const s = sample(this.main.replay, time);
    const pos = this.place(this.aircraft, s);
    this.aircraft.show = true;

    const g = this.ghost && layers.ghost ? sample(this.ghost.replay, time) : null;
    this.ghostAircraft.show = g !== null;
    if (g) this.place(this.ghostAircraft, g);

    this.updateZones(s, layers.zones);
    if (this.mission) this.mission.show = layers.mission;

    // Wind arrow: 10 m per m/s, drawn from the aircraft.
    const windSpeed = Math.hypot(s.windN, s.windE);
    this.wind.show = layers.wind && windSpeed > 0.5;
    if (this.wind.show) {
      const tip = this.frame.toCartesian(s.n + 10 * s.windN, s.e + 10 * s.windE, s.altAgl);
      this.wind.polyline!.positions = new ConstantProperty([pos, tip]) as never;
    }

    // Touchdown marker once the flight has ended on the ground.
    const td = this.main.replay.events.find((e) => e.type === "touchdown");
    const showTd = !!td && time >= td.time_us * 1e-6;
    this.touchdown.show = showTd;
    if (showTd && td) {
      const [n, e] = td.payload.pos_ne_m as [number, number];
      this.touchdown.position = new ConstantPositionProperty(this.frame.toCartesian(n, e, 0.5));
      const zone = td.payload.zone ? ` · zone ${String(td.payload.zone)}` : "";
      const inZone = td.payload.in_zone === true ? " (in zone)" : td.payload.in_zone === false ? " (outside)" : "";
      this.touchdown.label!.text = new ConstantProperty(`${String(td.payload.class)}${zone}${inZone}`);
    }

    // Trail: rebuilt at 5 Hz (or immediately after a seek or a layer change).
    const key = `${layers.trail}|${layers.ghost}|${this.ghost ? 1 : 0}`;
    if (key !== this.trailKey || Math.abs(time - this.trailBuiltAt) > 0.2) {
      this.trail.removeAll();
      if (layers.trail) {
        this.rebuildTrail(this.main, time, 0.95);
        if (this.ghost && layers.ghost) this.rebuildTrail(this.ghost, time, 0.35);
      }
      this.trailBuiltAt = time;
      this.trailKey = key;
    }

    this.updateCamera(pos, s, camera, time);
    return s;
  }

  private updateCamera(pos: Cartesian3, s: Sample, mode: CameraMode, time: number): void {
    const cam = this.viewer.camera;
    if (mode === "free") {
      if (this.lastCamera !== "free") cam.lookAtTransform(Matrix4.IDENTITY);
      this.lastCamera = mode;
      return;
    }
    this.lastCamera = mode;
    if (mode === "chase") {
      // Behind the aircraft, heading low-passed so the view stays steady in gusts.
      if (this.smoothHeading === null) this.smoothHeading = s.yaw;
      let d = s.yaw - this.smoothHeading;
      while (d > Math.PI) d -= 2 * Math.PI;
      while (d < -Math.PI) d += 2 * Math.PI;
      this.smoothHeading += 0.06 * d;
      cam.lookAt(pos, new HeadingPitchRange(this.smoothHeading, -14 * DEG, 70));
    } else if (mode === "orbit") {
      cam.lookAt(pos, new HeadingPitchRange(0.15 * time, -22 * DEG, 180));
    } else {
      cam.lookAt(pos, new HeadingPitchRange(0, -89.9 * DEG, 700));
    }
  }
}
