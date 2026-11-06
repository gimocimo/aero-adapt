"""The Rascal 110 JSBSim model, a held-out reference airframe (M2 spec §5, doc 03 §2.2).

It lives in the ArduPilot repository (GPL-3.0, derived from FlightGear's Rascal110). Following
ADR-006 it is downloaded at a pinned commit into a user cache and never committed or
redistributed here.

``derived_model`` writes, next to it in the cache, the same aircraft with our layer in place
of its control system and engine, as ``JSBSimVehicle`` expects of every JSBSim aircraft:

- surfaces are set through ``fcs/da-rad``, ``fcs/de-rad`` and ``fcs/dr-rad`` in B&M signs
  (the Rascal's own signs: positive aileron rolls right, positive elevator pitches down,
  positive rudder yaws left), and copied to the angles and normalised angles that its
  aerodynamics read. Servo travel, lag, rate limits and deadband are ours, as on every
  airframe;
- thrust and propeller torque are applied as body-axis external reactions at its thruster's
  location, from our motor, propeller and battery models of its datasheet propulsion (2.2 kW,
  18×8 in). Motor faults, the ESC's rpm and the battery then behave as on every other
  airframe. The engine goes with its fuel tank (1.5 lb), so the model flies at its 13 lb empty
  weight, the datasheet's, with the empty-weight centre of gravity;
- its landing gear is removed: our layer ends a flight at ground contact, as for the
  Aerosonde in JSBSim. So is the network input that ArduPilot's SITL drives it through.

Everything else is the Rascal's own: geometry, empty mass and inertia, centre of gravity and
the aerodynamic model, flown by JSBSim's equations of motion. The derived file is GPL data
too, so it stays in the cache.
"""

from __future__ import annotations

import os
import urllib.request
import xml.etree.ElementTree as ET
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
DERIVED = "rascal_aeroadapt"
# Outputs of the Rascal's control system that its aerodynamics read: (property, our surface,
# gain). The normalised angles divide by the travel its control system normalises with.
_SURFACE_OUTPUTS = (
    ("fcs/left-aileron-pos-rad", "fcs/da-rad", 1.0),
    ("fcs/right-aileron-pos-rad", "fcs/da-rad", -1.0),
    ("fcs/left-aileron-pos-norm", "fcs/da-rad", 1.0 / 0.35),
    ("fcs/right-aileron-pos-norm", "fcs/da-rad", -1.0 / 0.35),
    ("fcs/elevator-pos-rad", "fcs/de-rad", 1.0),
    ("fcs/elevator-pos-norm", "fcs/de-rad", 1.0 / 0.3),
    ("fcs/rudder-pos-rad", "fcs/dr-rad", 1.0),
    ("fcs/rudder-pos-norm", "fcs/dr-rad", 1.0 / 0.35),
)
PROPELLER_IN = (1.0, 0.0, 0.0)  # its thruster's location (structural frame, absolute)
# Signs of its C_ℓ,δa, C_m,δe, C_n,δr (Clda +0.13, Cmde −0.5, Cndr −0.05 per rad): B&M's.
MODEL_SIGNS = (1.0, -1.0, -1.0)


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


def _element(tag: str, text: str | None = None, **attrib: str) -> ET.Element:
    e = ET.Element(tag, attrib)
    e.text = text
    return e


def _flight_control() -> ET.Element:
    fcs = _element("flight_control", name="aeroadapt: surfaces from our actuator models")
    for surface in ("fcs/da-rad", "fcs/de-rad", "fcs/dr-rad"):
        fcs.append(_element("property", surface, value="0"))
    channel = ET.SubElement(fcs, "channel", name="surfaces")
    for out, src, gain in _SURFACE_OUTPUTS:
        g = ET.SubElement(channel, "pure_gain", name=out.removeprefix("fcs/"))
        g.append(_element("input", src))
        g.append(_element("gain", repr(gain)))
        g.append(_element("output", out))
    return fcs


def _external_reactions() -> ET.Element:
    ext = _element("external_reactions")
    x, y, z = (repr(v) for v in PROPELLER_IN)
    thrust = ET.SubElement(ext, "force", name="thrust", frame="BODY", unit="LBS")
    loc = ET.SubElement(thrust, "location", unit="IN")
    for tag, v in zip("xyz", (x, y, z), strict=True):
        loc.append(_element(tag, v))
    direction = ET.SubElement(thrust, "direction")
    for tag, v in zip("xyz", ("1", "0", "0"), strict=True):
        direction.append(_element(tag, v))
    torque = ET.SubElement(ext, "moment", name="prop_torque", frame="BODY", unit="LBSFT")
    direction = ET.SubElement(torque, "direction")
    for tag, v in zip("xyz", ("1", "0", "0"), strict=True):
        direction.append(_element(tag, v))
    return ext


def derived_model(root: Path = CACHE_ROOT) -> Path:
    """Write ``aircraft/rascal_aeroadapt/rascal_aeroadapt.xml`` under ``root`` (fetching the
    Rascal first if needed); returns ``root``."""
    src = ensure_rascal(root) / "aircraft" / "Rascal" / "Rascal.xml"
    tree = ET.parse(src)
    model = tree.getroot()
    model.set("name", DERIVED)
    replaced = {
        "ground_reactions": [_element("ground_reactions")],
        "propulsion": [_element("propulsion"), _external_reactions()],
        "flight_control": [_flight_control()],
        "input": [],  # ArduPilot SITL's socket
    }
    children = []
    for child in model:
        children += replaced.get(child.tag, [child])
    model[:] = children
    out = root / "aircraft" / DERIVED / f"{DERIVED}.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree)
    header = f"<!-- Derived from {src.name} by aeroadapt.sim.jsbsim.rascal; GPL-3.0. -->\n"
    # Atomically: another process may be loading the model.
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text('<?xml version="1.0"?>\n' + header + ET.tostring(model, encoding="unicode"))
    os.replace(tmp, out)
    return root
