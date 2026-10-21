"""Build the M2 demo video (deliverable D11) from seeds, deterministically.

1. A live fault-injection session, re-flown from its command log (byte-identical to the live
   session, V23): a sample-fleet aircraft flies the M1 loop; GNSS spoofing pulls it off the route
   while the autopilot believes it is on it; the spoofing clears; then the motor cuts and the
   classical contingency lands it. Recorded and exported to Skyview with the aircraft's model.
2. The fleet view of the bench (``runs/bench/b0``, exported by ``aeroadapt export-fleet``):
   the same scenarios flown by the stock stack (B1), then with the contingency (B2).
3. Frames captured headlessly (each waits for the 3D tiles), encoded with ffmpeg to
   ``runs/demo/m2_demo.mp4``.

Usage: ``uv run --with imageio-ffmpeg python tools/make_demo_m2.py``. Needs the bench run and the
Cesium ion token in the git-ignored ``web/skyview/.env.local`` for the photorealistic tiles.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web" / "skyview"
OUT = REPO / "runs" / "demo"
SAMPLE = "configs/fleets/sample"
URL = "http://localhost:5173"
LIVE_ID = "m2-live-demo"
COMMANDS = [  # what a user clicked in the live panel (times: the command's step)
    {"time_s": 20.0, "inject_fault": {"type": "sensor_fault", "kind": "gnss_spoof",
                                      "magnitude": 0.3, "axis": 1}},
    {"time_s": 45.0, "clear_fault": 0},
    {"time_s": 60.0, "inject_fault": {"type": "motor_cut"}},
]  # fmt: skip


def _server_up() -> bool:
    try:
        with urllib.request.urlopen(URL, timeout=2):
            return True
    except OSError:
        return False


def live_session() -> tuple[str, float]:
    """Fly the live demo from its command log; returns the airframe flown and when it ended."""
    from aeroadapt.live.session import replay_commands
    from aeroadapt.scenarios.schema import Scenario, load_scenario
    from aeroadapt.stack.config import load_stack_cfg

    lines = (REPO / SAMPLE / "stack_cfgs.jsonl").read_text().splitlines()
    ids = [json.loads(line)["airframe"] for line in lines]
    airframe = next((i for i in ids if "twin_boom" in i), ids[0])
    v = load_stack_cfg(REPO / SAMPLE, airframe).tecs.airspeed_cruise_mps
    d = load_scenario("m1-nominal").model_dump(mode="json")
    d.update(id=LIVE_ID, airframe=airframe, fleet=SAMPLE, duration_s=240.0,
             description="M2 demo: GNSS spoofing, then a motor cut, injected live")  # fmt: skip
    d["initial"]["airspeed_mps"] = v
    d["mission"]["speed_mps"] = v
    d["stack"]["params"] = SAMPLE
    scenario = Scenario.model_validate(d)
    session = replay_commands(scenario, [dict(c) for c in COMMANDS])
    while session.time_s < scenario.duration_s - 1e-6 and not session.done:
        session.step()
    out = session.record(REPO / "runs" / LIVE_ID)
    subprocess.run(["uv", "run", "aeroadapt", "export-viz", str(out)], cwd=REPO, check=True)
    return airframe, session.time_s


def segments(airframe: str, t_end: float) -> list[dict]:
    cls = airframe.split("-")[1].replace("_", "-")
    live = {"replay": LIVE_ID, "fps": 30}
    orbit = {"fps": 30, "fleet": "b0"}
    return [
        {**live, "t0": 5, "t1": 20, "speed": 1, "camera": "chase",
         "caption": f"A generated {cls} from the 10,000-aircraft fleet, its own tuned autopilot"},
        {**live, "t0": 20, "t1": 46, "speed": 2, "camera": "top",
         "caption": "Live panel: GNSS spoofing. The autopilot thinks it is on the route (2×)"},
        {**live, "t0": 46, "t1": 60, "speed": 2, "camera": "top",
         "caption": "Spoofing cleared from the panel: back onto the route (2×)"},
        {**live, "t0": 60, "t1": 66, "speed": 1, "camera": "chase",
         "caption": "Motor cut: detected from ESC telemetry, a reachable field selected"},
        {**live, "t0": 66, "t1": round(t_end - 0.1, 1), "speed": 8, "camera": "top",
         "caption": "The classical contingency glides to the field (8×)"},
        {**orbit, "baseline": "B1", "t0": 0, "t1": 300, "speed": 15, "orbit": [0, 60, -40, 5200],
         "caption": "Bench v0: the same scenarios flown by the stock autopilot (B1)"},
        {**orbit, "baseline": "B2", "t0": 0, "t1": 300, "speed": 15, "orbit": [60, 120, -40, 5200],
         "caption": "… and with the classical contingency (B2): motor cuts end in landings"},
    ]  # fmt: skip


def main() -> None:
    airframe, t_end = live_session()
    if not (WEB / "public" / "fleet" / "b0" / "fleet.json").exists():
        subprocess.run(["uv", "run", "aeroadapt", "export-fleet", "runs/bench/b0"], cwd=REPO,
                       check=True)  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    seg_file = OUT / "m2_segments.json"
    seg_file.write_text(json.dumps(segments(airframe, t_end), indent=1))
    server = None
    if not _server_up():
        server = subprocess.Popen(
            ["npm", "run", "dev", "--", "--port", "5173", "--strictPort"], cwd=WEB
        )
        for _ in range(60):
            if _server_up():
                break
            time.sleep(1)
    frames = OUT / "m2_frames"
    shutil.rmtree(frames, ignore_errors=True)
    try:
        subprocess.run(["node", "scripts/capture-frames.mjs", str(frames), str(seg_file)],
                       cwd=WEB, check=True)  # fmt: skip
    finally:
        if server is not None:
            server.terminate()
    ffmpeg = importlib.import_module("imageio_ffmpeg").get_ffmpeg_exe()
    video = OUT / "m2_demo.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-framerate", "30", "-i", str(frames / "f%05d.png"),
         "-vf", "scale=1280:720:flags=lanczos", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "28", "-preset", "slow", "-movflags", "+faststart", str(video)],
        check=True,
    )  # fmt: skip
    shutil.rmtree(frames)  # ~2 MB a frame: the video is what stays (a rerun redraws them)
    print(video, f"{video.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
