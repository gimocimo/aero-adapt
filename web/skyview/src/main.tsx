import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import Calibration from "./Calibration";
import Player from "./Player";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("missing #root");
// The replay player by default; the week-1 frames calibration checklist with ?calibrate.
const calibrate = new URLSearchParams(window.location.search).has("calibrate");
createRoot(root).render(<StrictMode>{calibrate ? <Calibration /> : <Player />}</StrictMode>);
