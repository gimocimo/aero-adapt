import {
  Cartesian3,
  Cartographic,
  type Cesium3DTileset,
  ConstantPositionProperty,
  ConstantProperty,
  createGooglePhotorealistic3DTileset,
  type Entity,
  GridImageryProvider,
  HeadingPitchRange,
  ImageryLayer,
  Ion,
  Matrix4,
  Viewer,
} from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";
import { useEffect, useRef, useState } from "react";
import { CANDIDATES } from "./candidates";
import { orientationFromNedQuat } from "./cesiumOrientation";
import { eulerDegToQuat } from "./frames";

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
  const [siteIdx, setSiteIdx] = useState(0);
  const [roll, setRoll] = useState(0);
  const [pitch, setPitch] = useState(0);
  const [heading, setHeading] = useState(0);
  const [agl, setAgl] = useState(120);
  const [groundH, setGroundH] = useState<number | null>(null);
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
      model: { uri: "/models/calibration_uav.glb", minimumPixelSize: 64, maximumScale: 400 },
    });
    let cancelled = false;
    if (TOKEN) {
      viewer.scene.globe.show = false;
      createGooglePhotorealistic3DTileset({ onlyUsingWithGoogleGeocoder: true })
        .then((tileset) => {
          if (cancelled) return;
          viewer.scene.primitives.add(tileset);
          setTiles(tileset);
          setStatus("Google Photorealistic 3D Tiles loaded via Cesium ion");
        })
        .catch((err: unknown) => setStatus(`3D tiles failed: ${String(err)}`));
    } else {
      setStatus("No VITE_CESIUM_ION_TOKEN: ellipsoid + grid fallback (frames still testable)");
    }
    return () => {
      cancelled = true;
      viewer.destroy();
      viewerRef.current = null;
    };
  }, []);

  // Sample the ground's *ellipsoidal* height at the site origin (the UK geoid is ~50 m above).
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    setGroundH(null);
    if (!tiles) {
      setGroundH(0);
      return;
    }
    let cancelled = false;
    viewer.camera.flyTo({
      destination: Cartesian3.fromDegrees(site.lon, site.lat, 600),
      duration: 0,
    });
    viewer.scene
      .sampleHeightMostDetailed([Cartographic.fromDegrees(site.lon, site.lat)])
      .then(([c]) => {
        if (!cancelled) setGroundH(c?.height ?? null);
      })
      .catch(() => !cancelled && setGroundH(null));
    return () => {
      cancelled = true;
    };
  }, [site, tiles]);

  // Pose and camera.
  useEffect(() => {
    const viewer = viewerRef.current;
    const aircraft = aircraftRef.current;
    if (!viewer || !aircraft || groundH === null) return;
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
          <dt>Ground h (ellipsoidal)</dt>
          <dd>{groundH === null ? "sampling…" : `${groundH.toFixed(2)} m`}</dd>
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
