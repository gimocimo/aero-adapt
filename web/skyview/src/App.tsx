import {
  Cartesian3,
  Cartographic,
  type Cesium3DTileset,
  CesiumTerrainProvider,
  ConstantPositionProperty,
  ConstantProperty,
  type Entity,
  GridImageryProvider,
  HeadingPitchRange,
  ImageryLayer,
  Ion,
  Matrix4,
  sampleTerrainMostDetailed,
  type TerrainProvider,
  Viewer,
} from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";
import { useEffect, useRef, useState } from "react";
import { CANDIDATES } from "./candidates";
import { orientationFromNedQuat } from "./cesiumOrientation";
import { eulerDegToQuat } from "./frames";
import { loadGoogleTiles } from "./googleTiles";

const TOKEN = import.meta.env.VITE_CESIUM_ION_TOKEN?.trim();
const DEG = Math.PI / 180;

type CameraMode = "chase" | "top" | "side" | "free";

const CHECKLIST = [
  "Heading 0°: yellow nose points north (top-down view: screen up)",
  "Heading 90°: nose points east (top-down: screen right)",
  "Heading 180° / 270°: nose points south / west",
  "Pitch +30°: nose above the horizon (side view)",
  "Roll +45°: green (right) wingtip goes down",
  "Blue fin points up at zero roll and pitch",
  "Aircraft sits at the AGL height above the 3D ground (not buried or floating ~50 m)",
];

export default function App() {
  const container = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Viewer | null>(null);
  const aircraftRef = useRef<Entity | null>(null);
  const [tiles, setTiles] = useState<Cesium3DTileset | null>(null);
  const [terrain, setTerrain] = useState<TerrainProvider | null>(null);
  const [siteIdx, setSiteIdx] = useState(0);
  const [roll, setRoll] = useState(0);
  const [pitch, setPitch] = useState(0);
  const [heading, setHeading] = useState(0);
  const [agl, setAgl] = useState(120);
  // Bare-earth terrain height (Cesium World Terrain DTM): the authoritative site ground height.
  const [groundH, setGroundH] = useState<number | null>(null);
  // Height of the Google 3D Tiles surface (includes trees/buildings): a visual cross-check only.
  const [surfaceH, setSurfaceH] = useState<number | null>(null);
  const [camera, setCamera] = useState<CameraMode>("chase");
  const [status, setStatus] = useState("starting…");
  const site = CANDIDATES[siteIdx] ?? CANDIDATES[0]!;

  // Viewer lifecycle.
  useEffect(() => {
    if (!container.current) return;
    if (TOKEN) Ion.defaultAccessToken = TOKEN;
    const viewer = new Viewer(container.current, {
      timeline: false,
      animation: false,
      baseLayerPicker: false,
      geocoder: false, // Google 3D Tiles may only be paired with Google's geocoder: use none.
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      fullscreenButton: false,
      infoBox: false,
      selectionIndicator: false,
      baseLayer: TOKEN ? false : new ImageryLayer(new GridImageryProvider({})),
    });
    viewerRef.current = viewer;
    aircraftRef.current = viewer.entities.add({
      show: false, // hidden until the ground height is known
      model: { uri: "/models/calibration_uav.glb", minimumPixelSize: 48, maximumScale: 20 },
    });
    let cancelled = false;
    if (TOKEN) {
      viewer.scene.globe.show = false;
      loadGoogleTiles()
        .then((tileset) => {
          if (cancelled) {
            tileset.destroy(); // this viewer was already torn down (e.g. StrictMode remount)
            return;
          }
          viewer.scene.primitives.add(tileset);
          setTiles(tileset);
          setStatus("Google Photorealistic 3D Tiles loaded via Cesium ion");
        })
        .catch((err: unknown) => setStatus(`3D tiles failed: ${String(err)}`));
      CesiumTerrainProvider.fromIonAssetId(1) // Cesium World Terrain (WGS84 ellipsoidal heights)
        .then((tp) => !cancelled && setTerrain(tp))
        .catch((err: unknown) => setStatus(`terrain failed: ${String(err)}`));
    } else {
      setStatus("No VITE_CESIUM_ION_TOKEN: ellipsoid + grid fallback (frames still testable)");
    }
    return () => {
      cancelled = true;
      viewer.destroy();
      viewerRef.current = null;
    };
  }, []);

  // Ground height at the site origin, as a WGS84 *ellipsoidal* height (the UK geoid is ~50 m
  // above the ellipsoid). The authoritative value comes from the bare-earth terrain model,
  // which is deterministic. The 3D-tiles surface is sampled only as a cross-check, and our own
  // aircraft is excluded from that ray: Cesium scales the model up (minimumPixelSize) when the
  // camera is far away, so it can otherwise intercept the ray.
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    setGroundH(null);
    setSurfaceH(null);
    viewer.camera.lookAtTransform(Matrix4.IDENTITY);
    viewer.camera.setView({ destination: Cartesian3.fromDegrees(site.lon, site.lat, 900) });
    if (!TOKEN) {
      setGroundH(0);
      return;
    }
    if (!terrain) return;
    let cancelled = false;
    const origin = () => Cartographic.fromDegrees(site.lon, site.lat);
    sampleTerrainMostDetailed(terrain, [origin()])
      .then(([c]) => !cancelled && setGroundH(c?.height ?? null))
      .catch(() => !cancelled && setStatus("terrain sampling failed"));
    const aircraft = aircraftRef.current;
    if (tiles && aircraft) {
      viewer.scene
        .sampleHeightMostDetailed([origin()], [aircraft])
        .then(([c]) => !cancelled && setSurfaceH(c?.height ?? null))
        .catch(() => undefined);
    }
    return () => {
      cancelled = true;
    };
  }, [site, terrain, tiles]);

  // Pose and camera.
  useEffect(() => {
    const viewer = viewerRef.current;
    const aircraft = aircraftRef.current;
    if (!viewer || !aircraft) return;
    aircraft.show = groundH !== null;
    if (groundH === null) return;
    const position = Cartesian3.fromDegrees(site.lon, site.lat, groundH + agl);
    const q = eulerDegToQuat(roll, pitch, heading);
    aircraft.position = new ConstantPositionProperty(position);
    aircraft.orientation = new ConstantProperty(orientationFromNedQuat(position, q));
    const views: Record<Exclude<CameraMode, "free">, HeadingPitchRange> = {
      chase: new HeadingPitchRange(heading * DEG, -15 * DEG, 18),
      top: new HeadingPitchRange(0, -89.9 * DEG, 25),
      side: new HeadingPitchRange(270 * DEG, -2 * DEG, 18), // camera east of the aircraft, looking west
    };
    if (camera === "free") viewer.camera.lookAtTransform(Matrix4.IDENTITY);
    else viewer.camera.lookAt(position, views[camera]);
  }, [site, groundH, agl, roll, pitch, heading, camera]);

  const slider = (
    label: string,
    value: number,
    set: (v: number) => void,
    min: number,
    max: number,
    step = 1,
  ) => (
    <label className="row">
      <span>{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => set(Number(e.target.value))}
      />
      <output>{value}</output>
    </label>
  );

  return (
    <>
      <div ref={container} className="viewer" />
      <aside className="panel">
        <h1>Skyview · week-1 spike</h1>
        <p className="status">{status}</p>
        <label className="row">
          <span>Site</span>
          <select value={siteIdx} onChange={(e) => setSiteIdx(Number(e.target.value))}>
            {CANDIDATES.map((c, i) => (
              <option key={c.id} value={i}>
                {c.name}
              </option>
            ))}
          </select>
        </label>
        <p className="note">{site.note}</p>
        <dl>
          <dt>Origin</dt>
          <dd>
            {site.lat.toFixed(5)}, {site.lon.toFixed(5)}
          </dd>
          <dt>Ground h (terrain, ellipsoidal)</dt>
          <dd>{groundH === null ? "sampling…" : `${groundH.toFixed(2)} m`}</dd>
          {TOKEN && (
            <>
              <dt>3D-tiles surface h</dt>
              <dd>
                {surfaceH === null
                  ? "sampling…"
                  : `${surfaceH.toFixed(2)} m` +
                    (groundH === null ? "" : ` (${(surfaceH - groundH).toFixed(1)} m vs terrain)`)}
              </dd>
            </>
          )}
        </dl>
        {slider("Heading °", heading, setHeading, 0, 359)}
        {slider("Pitch °", pitch, setPitch, -60, 60)}
        {slider("Roll °", roll, setRoll, -90, 90)}
        {slider("AGL m", agl, setAgl, 2, 400)}
        <div className="cams">
          {(["chase", "top", "side", "free"] as const).map((m) => (
            <button key={m} className={m === camera ? "on" : ""} onClick={() => setCamera(m)}>
              {m === "top" ? "top-down (N up)" : m === "side" ? "side (from E)" : m}
            </button>
          ))}
        </div>
        <h2>Calibration checklist</h2>
        <ul className="checklist">
          {CHECKLIST.map((item) => (
            <li key={item}>
              <label>
                <input type="checkbox" /> {item}
              </label>
            </li>
          ))}
        </ul>
        <p className="legend">
          Nav lights: <b className="red">red</b> = left/port, <b className="green">green</b> =
          right/starboard, <b className="yellow">yellow</b> = nose, <b className="blue">blue</b> =
          fin.
        </p>
      </aside>
    </>
  );
}
