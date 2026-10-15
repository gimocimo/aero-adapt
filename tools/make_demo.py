"""Build the M1 demo video (deliverable D7) from seeds, deterministically.

1. Fly the demo scenarios and export their replays into Skyview.
2. Start the Skyview dev server if it is not running (3D tiles need the Cesium ion token in
   the git-ignored ``web/skyview/.env.local``; without it the fallback globe is captured).
3. Capture every frame headlessly (each frame waits for the tiles to finish loading).
4. Encode ``runs/demo/m1_demo.mp4`` with ffmpeg (the imageio-ffmpeg wheel bundles one).

Usage: ``uv run --with imageio-ffmpeg python tools/make_demo.py`` (or ``make demo``).
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web" / "skyview"
OUT = REPO / "runs" / "demo"
SCENARIOS = ("m1-motorcut-classical", "m1-motorcut-stock", "m1-pitot-drain-blocked")
URL = "http://localhost:5173"


def _server_up() -> bool:
    try:
        with urllib.request.urlopen(URL, timeout=2):
            return True
    except OSError:
        return False


def main() -> None:
    for sid in SCENARIOS:
        subprocess.run(["uv", "run", "aeroadapt", "run", sid], cwd=REPO, check=True)
        subprocess.run(
            ["uv", "run", "aeroadapt", "export-viz", f"runs/{sid}"], cwd=REPO, check=True
        )
    server = None
    if not _server_up():
        server = subprocess.Popen(
            ["npm", "run", "dev", "--", "--port", "5173", "--strictPort"], cwd=WEB
        )
        for _ in range(60):
            if _server_up():
                break
            time.sleep(1)
    frames = OUT / "frames"
    shutil.rmtree(frames, ignore_errors=True)
    try:
        subprocess.run(
            ["node", "scripts/capture-frames.mjs", str(frames), "scripts/demo-segments.json"],
            cwd=WEB,
            check=True,
        )
    finally:
        if server is not None:
            server.terminate()
    # Optional tool dependency (not in the lockfile): ``uv run --with imageio-ffmpeg``.
    ffmpeg = importlib.import_module("imageio_ffmpeg").get_ffmpeg_exe()
    video = OUT / "m1_demo.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            "30",
            "-i",
            str(frames / "f%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "22",
            "-preset",
            "slow",
            "-movflags",
            "+faststart",
            str(video),
        ],
        check=True,
    )
    print(video, f"{video.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
