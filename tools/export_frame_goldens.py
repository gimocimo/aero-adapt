"""Export golden frame-conversion cases from the Python source of truth for the viewer's tests.

The TypeScript viewer reimplements Euler→quaternion and the body→ENU (FLU) rotation it needs
for CesiumJS. ``web/skyview/src/frames.test.ts`` checks it against these vectors, so the two
languages can never silently disagree on conventions (ADR-001).

Usage: ``uv run python tools/export_frame_goldens.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np

from aeroadapt.core import quaternion as Q

jax.config.update("jax_enable_x64", True)

OUT = Path(__file__).resolve().parents[1] / "web/skyview/src/__goldens__/frames.json"

R_ENU_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
R_FRD_FLU = np.diag([1.0, -1.0, -1.0])  # maps FLU vectors to FRD


def main() -> None:
    rng = np.random.default_rng(2026)
    named = [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 90.0),
        (0.0, 0.0, 180.0),
        (0.0, 0.0, 270.0),
        (0.0, 30.0, 0.0),
        (0.0, -30.0, 0.0),
        (45.0, 0.0, 0.0),
        (-45.0, 0.0, 0.0),
    ]
    rand = [tuple(rng.uniform([-170, -85, -180], [170, 85, 180])) for _ in range(40)]
    cases = []
    for roll, pitch, heading in named + rand:
        q = np.asarray(Q.from_euler(np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(heading)))
        r_enu_flu = R_ENU_NED @ np.asarray(Q.to_dcm(q)) @ R_FRD_FLU
        cases.append(
            {
                "euler_deg": [float(roll), float(pitch), float(heading)],
                "quat": q.tolist(),
                "r_enu_flu": r_enu_flu.tolist(),
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"convention": "ADR-001", "cases": cases}, indent=1) + "\n")
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
