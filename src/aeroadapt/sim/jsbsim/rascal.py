"""The Rascal 110 JSBSim model, a held-out reference airframe (M2 spec §5, doc 03 §2.2).

It lives in the ArduPilot repository (GPL-3.0, derived from FlightGear's Rascal110). Following
ADR-006 it is downloaded at a pinned commit into a user cache and never committed or
redistributed here.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

ARDUPILOT_COMMIT = "b032bcbde31a8649d748baa5e5910208a76460f2"
_BASE = (
    "https://raw.githubusercontent.com/ArduPilot/ardupilot/"
    f"{ARDUPILOT_COMMIT}/Tools/autotest/aircraft/Rascal"
)
FILES = ("Rascal.xml", "Engines/18x8.xml", "Engines/Zenoah_G-26A.xml")
CACHE_ROOT = (
    Path(os.environ.get("AEROADAPT_CACHE", Path.home() / ".cache" / "aeroadapt")) / "jsbsim"
)


def ensure_rascal(root: Path = CACHE_ROOT) -> Path:
    """Download the model if needed; returns the JSBSim root directory that contains it."""
    target = root / "aircraft" / "Rascal"
    for name in FILES:
        path = target / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(f"{_BASE}/{name}", timeout=30) as r:
                path.write_bytes(r.read())
    return root
