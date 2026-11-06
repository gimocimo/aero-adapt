"""The Rascal 110 as the stock stack knows it (M2 spec §5, D4): an airframe from its datasheet.

The Rascal is held out: the stack must be configured without its JSBSim aerodynamic tables,
as a user configures ArduPlane for a kit from the box. So one design is written down here from
what the kit publishes (SIG Manufacturing's Rascal 110: 110 in span, 1,522 in² of wing,
about 75 in long, 13 lb at the light end; a 26 cc-class engine, here the model's 2.2 kW on
an 18×8 in propeller) and a three-view (tail areas and arm), and the M2 machinery does the
rest. The procedural
generator lays it out and estimates its mass distribution, aerodynamics and propulsion from
that geometry (M2 spec §4); the stack is auto-configured and auto-tuned on that estimate in
aerojax, and V20 must accept it there (M2 spec §6). JSBSim then flies the stack on the
Rascal's own model (``aeroadapt.sim.jsbsim.rascal``), which the stack has never seen.

The result is a one-airframe fleet bundle, ``configs/fleets/rascal`` (with its glTF model,
tuned stack and V20 checks).

Usage (the generator extra brings AeroSandbox and NeuralFoil)::

    uv run --extra generator python tools/rascal_airframe.py
"""

from __future__ import annotations

import json
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
OUT = ROOT / "configs" / "fleets" / "rascal"
AIRFRAME_ID = "rascal-110"
NAME = "Rascal 110 (datasheet)"

IN = 0.0254
SPAN = 110 * IN
AREA = 1522 * IN**2
CHORD = AREA / SPAN
LENGTH = 75.5 * IN
MASS = 13 * 0.453_592
S_H, S_V, TAIL_ARM = 1.69 * 0.3048**2, 1.06 * 0.3048**2, 3.28 * 0.3048  # three-view

# A high-wing sport trainer: a rectangular flat-bottomed wing, a conventional tail, a tractor
# propeller. What the kit does not publish is a choice: the airfoil (a flat-bottomed Clark Y),
# the surfaces' chords and the dihedral, 3°: at 2° the generator's spiral filter rejects it
# (time to double 9 s, the filter asks ≥ 10 s), at 3° it doubles in 14 s.
DESIGN = Design(
    cls="conventional",
    seed=0,
    span=SPAN,
    aspect_ratio=SPAN**2 / AREA,
    taper=1.0,
    sweep_deg=0.0,
    dihedral_deg=3.0,
    wing_airfoil="clarky",
    wing_incidence_deg=1.5,
    tail_airfoil="naca0010",
    h_volume=S_H * TAIL_ARM / (AREA * CHORD),
    v_volume=S_V * TAIL_ARM / (AREA * SPAN),
    h_aspect_ratio=4.0,
    v_aspect_ratio=1.5,
    aileron_chord=0.25,
    aileron_span=0.5,
    elevator_chord=0.35,
    rudder_chord=0.4,
    fuselage_length_ratio=LENGTH / SPAN,
    fuselage_diameter_ratio=0.09,
    tail_arm_ratio=TAIL_ARM / LENGTH,
    wing_loading=MASS * 9.806_65 / AREA,
    power_to_weight=2207.0 / MASS,
    endurance_min=30.0,
    static_margin=0.10,
    pusher=False,
    prop_diameter_ratio=18 * IN / SPAN,
    pitch_ratio=8 / 18,
    structure_density=1.0,
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
                    + " The Rascal 110 from its datasheet (tools/rascal_airframe.py), held out:"
                    " never its JSBSim tables."
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

    site = load_site("cambridge_west")
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

    (OUT / "gltf").mkdir(parents=True, exist_ok=True)
    row = {"id": AIRFRAME_ID, **DESIGN.to_dict(), **meta, **metrics}
    row.update(stage="filter", accepted=True, reasons="")
    _write_bundle(OUT, "rascal", [row], [Candidate(DESIGN, af, meta)], ["conventional"])
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
    print(f"cruise {cfg.tecs.airspeed_cruise_mps:.1f} m/s → {OUT}")


if __name__ == "__main__":
    main()
