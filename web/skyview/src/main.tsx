import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import Calibration from "./Calibration";
import FleetView from "./fleet/FleetView";
import Player from "./Player";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("missing #root");
// The replay and live player by default; ?fleet=<bench> the fleet view; ?calibrate the week-1
// frames calibration checklist.
const q = new URLSearchParams(window.location.search);
const view = q.has("calibrate") ? <Calibration /> : q.has("fleet") ? <FleetView /> : <Player />;
createRoot(root).render(<StrictMode>{view}</StrictMode>);
