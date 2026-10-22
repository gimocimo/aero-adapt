import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";
import QuadPlayer from "./quad/QuadPlayer";

// The CesiumJS views load only when asked for (CesiumJS alone is ~4 MB).
const Calibration = lazy(() => import("./Calibration"));
const FleetView = lazy(() => import("./fleet/FleetView"));
const Player = lazy(() => import("./Player"));
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("missing #root");
// The four-panel player over the stylised world by default (docs/07 §8); ?map=real the
// Google 3D-tiles player; ?fleet=<bench> the fleet view; ?calibrate the week-1 frames
// calibration checklist.
const q = new URLSearchParams(window.location.search);
const view = q.has("calibrate") ? (
  <Calibration />
) : q.has("fleet") ? (
  <FleetView />
) : q.get("map") === "real" ? (
  <Player />
) : (
  <QuadPlayer />
);
createRoot(root).render(
  <StrictMode>
    <Suspense fallback={null}>{view}</Suspense>
  </StrictMode>,
);
