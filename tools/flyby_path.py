"""A camera path for a fly-by video of a world (docs/07 §15): keyframes in, one camera per
frame out, smoothly through them (Catmull-Rom), each checked clear of the ground and what
stands on it. ``web/skyview/scripts/flyby.mjs`` renders the frames.

Keyframes (JSON): ``[{"t": seconds, "eye": [n, e, h], "at": [n, e, h]}, ...]``, heights
above the world's datum.

Usage::

    uv run python tools/flyby_path.py london keyframes.json frames.json [--fps 30] [--clear 12]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aeroadapt.world import load_world


def catmull_rom(points: np.ndarray, times: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Centripetal-free (uniform in time) Catmull-Rom through ``points`` at ``times``."""
    p = np.vstack([points[0], points, points[-1]])
    tt = np.concatenate([[times[0] - 1.0], times, [times[-1] + 1.0]])
    out = np.empty((len(t), points.shape[1]))
    for k, x in enumerate(t):
        i = int(np.clip(np.searchsorted(times, x, side="right") - 1, 0, len(times) - 2)) + 1
        t0, t1 = tt[i], tt[i + 1]
        u = (x - t0) / (t1 - t0)
        m1 = (p[i + 1] - p[i - 1]) / (tt[i + 1] - tt[i - 1]) * (t1 - t0)
        m2 = (p[i + 2] - p[i]) / (tt[i + 2] - tt[i]) * (t1 - t0)
        h00, h10 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u
        h01, h11 = -2 * u**3 + 3 * u**2, u**3 - u**2
        out[k] = h00 * p[i] + h10 * m1 + h01 * p[i + 1] + h11 * m2
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("world")
    ap.add_argument("keyframes", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--clear", type=float, default=12.0, help="least clearance (m) of the eye")
    a = ap.parse_args()
    keys = json.loads(a.keyframes.read_text())
    times = np.array([k["t"] for k in keys], float)
    t = np.arange(times[0], times[-1], 1.0 / a.fps)
    eye = catmull_rom(np.array([k["eye"] for k in keys], float), times, t)
    at = catmull_rom(np.array([k["at"] for k in keys], float), times, t)
    w = load_world(a.world)
    obs = w.obstacles()
    # The highest thing within 6 m of the eye, frame by frame.
    top = np.full(len(t), -1e9)
    for dn in (-6.0, 0.0, 6.0):
        for de in (-6.0, 0.0, 6.0):
            n, e = eye[:, 0] + dn, eye[:, 1] + de
            top = np.maximum(top, w.dtm.at(n, e) + obs.at(n, e))
    clearance = eye[:, 2] - top
    k = int(np.argmin(clearance))
    print(f"{len(t)} frames, {t[-1] - t[0]:.1f} s; least clearance {clearance[k]:.1f} m at "
          f"{t[k]:.1f} s ({eye[k, 0]:.0f}, {eye[k, 1]:.0f}, {eye[k, 2]:.0f})")  # fmt: skip
    if clearance[k] < a.clear:
        raise SystemExit(f"the camera passes within {clearance[k]:.1f} m of something: move it")
    frames = [
        {"t": round(float(x), 4), "eye": np.round(p, 2).tolist(), "at": np.round(q, 2).tolist()}
        for x, p, q in zip(t, eye, at, strict=True)
    ]
    a.out.write_text(json.dumps(frames))


if __name__ == "__main__":
    main()
