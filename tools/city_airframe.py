"""The city airframe (docs/08 §5): a small conventional tractor to fly London among its
buildings, about 1.1 m across and 1 kg, cruising near 12 m/s.

It is not sampled: one design is written down here, and the M2 machinery does the rest. The
procedural generator lays it out and computes its mass, aerodynamics and propulsion (M2 spec
§4); aerojax's trim and stability filters must pass; the stack is auto-configured and
auto-tuned, and V20 must accept it (M2 spec §6). The result is a one-airframe fleet bundle,
``configs/fleets/city`` (with its glTF model and tuned stack), that scenarios name with
``fleet: configs/fleets/city`` and ``airframe: city-trainer``.

Usage (the generator extra brings AeroSandbox and NeuralFoil)::

    uv run --extra generator python tools/city_airframe.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from aeroadapt.generator.build import Candidate, build
from aeroadapt.generator.design import Design
from aeroadapt.generator.filters import check
from aeroadapt.generator.fleet import _write_bundle
from aeroadapt.generator.gltf import airframe_glb
from aeroadapt.harness.autoconfig import autoconfigure
from aeroadapt.harness.autotune import autotune
from aeroadapt.harness.fleet_acceptance import run_v20
from aeroadapt.scenarios import load_site

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "fleets" / "city"
AIRFRAME_ID = "city-trainer"
NAME = "City trainer (1.1 m)"

# A small sport trainer: a straight, slightly tapered Clark Y wing with a little dihedral, a
# conventional tail with generous volumes, big control surfaces for authority at low speed, a
# tractor propeller, and enough power to climb steeply out of a street.
DESIGN = Design(
    cls="conventional",
    seed=0,
    span=1.1,
    aspect_ratio=6.0,
    taper=0.7,
    sweep_deg=2.0,
    dihedral_deg=4.0,
    wing_airfoil="clarky",
    wing_incidence_deg=2.0,
    tail_airfoil="naca0010",
    h_volume=0.5,
    v_volume=0.04,
    h_aspect_ratio=4.5,
    v_aspect_ratio=1.6,
    aileron_chord=0.25,
    aileron_span=0.5,
    elevator_chord=0.35,
    rudder_chord=0.35,
    fuselage_length_ratio=0.75,
    fuselage_diameter_ratio=0.1,
    tail_arm_ratio=0.6,
    wing_loading=50.0,  # N/m²: about 1 kg on 0.2 m²
    power_to_weight=200.0,  # W/kg
    endurance_min=30.0,
    static_margin=0.10,
    pusher=False,
    prop_diameter_ratio=0.18,
    pitch_ratio=0.55,
    structure_density=0.9,
)


def main() -> None:
    cand = build(DESIGN)
    af = cand.airframe.model_copy(
        update={
            "id": AIRFRAME_ID,
            "name": NAME,
            "provenance": cand.airframe.provenance.model_copy(
                update={
                    "notes": (cand.airframe.provenance.notes or "")
                    + " Designed for docs/08 §5 (tools/city_airframe.py), not sampled."
                }
            ),
        }
    )
    meta = {**cand.meta, "id": AIRFRAME_ID}
    m = meta
    print(
        f"{NAME}: {m['mass_kg']:.2f} kg, span {m['span_m']:.2f} m, "
        f"wing {m['wing_area_m2']:.3f} m², loading {m['wing_loading']:.0f} N/m², "
        f"stall {m['v_stall']:.1f} m/s, cruise {m['v_cruise']:.1f} m/s, "
        f"static margin {m['static_margin']:.1%}, {m['n_cells']}S, {m['p_max_w']:.0f} W"
    )
    filt = check([af], [meta])
    metrics = {k: float(v[0]) for k, v in filt.metrics.items()}
    print("filters:", "pass" if filt.accepted[0] else f"FAIL {filt.reasons[0]}")
    if not filt.accepted[0]:
        raise SystemExit(1)

    site = load_site("cambridge_west")  # V20 flies in open air over flat ground
    ac = autoconfigure([af], site)[0]
    if ac.cfg is None:
        raise SystemExit(f"auto-configuration: {ac.reason}")
    cfg = autotune([af], [ac.cfg], site)[0]
    result = run_v20([af], [cfg], site, seed=0)[0]
    for name, (value, limit, ok) in result.checks.items():
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {value:.3g} (limit {limit:.3g})")
    print("V20:", "passed" if result.passed else f"FAILED {result.failures()}")
    if not result.passed:
        raise SystemExit(1)

    # The bundle: one airframe, its design and checks, its model and its tuned stack.
    (OUT / "gltf").mkdir(parents=True, exist_ok=True)
    row = {"id": AIRFRAME_ID, **DESIGN.to_dict(), **meta, **metrics}
    row.update(stage="filter", accepted=True, reasons="")
    _write_bundle(OUT, "city", [row], [Candidate(DESIGN, af, meta)], ["conventional"])
    cg = np.array([meta["x_cg_m"], 0.0, 0.0])
    (OUT / "gltf" / f"{AIRFRAME_ID}.glb").write_bytes(airframe_glb(DESIGN, cg))
    (OUT / "stack_cfgs.jsonl").write_text(
        json.dumps({"airframe": AIRFRAME_ID, "stack": cfg.model_dump(mode="json")}) + "\n"
    )
    (OUT / "v20.json").write_text(
        json.dumps(
            {
                "airframe": AIRFRAME_ID,
                "site": "cambridge_west",
                "passed": result.passed,
                "checks": {
                    k: {"value": v, "limit": lim, "ok": ok}
                    for k, (v, lim, ok) in result.checks.items()
                },
            },
            indent=2,
        )
        + "\n"
    )
    v = cfg.tecs.airspeed_cruise_mps
    radius = v * v / 9.80665 / math.tan(math.radians(45))
    print(f"cruise {v:.1f} m/s; at 45° of bank it turns on {radius:.0f} m")
    print(OUT)


if __name__ == "__main__":
    main()
