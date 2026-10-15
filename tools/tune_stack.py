"""Derive the flight stack's trim parameters from the airframe's trim solutions.

Prints the values for ``configs/stack/<airframe>.yaml``:

* ``tecs.thr_cruise`` and ``tecs.pitch_trim_deg``: straight-and-level trim at cruise, and
  the speed schedules: throttle linear in V (``thr_slope``), pitch linear in (V_cruise/V)²
  (``pitch_trim_b_deg``);
* ``attitude.servo_trim``: aileron and rudder trims at cruise, and the elevator trim curve
  a + b·(V_cruise/V)², least-squares fitted over the speed range;
* the nominal glide polar ((L/D)max, V_bg) that the contingency uses.

This is design data from the nominal model (as a real integrator would get from a trim
flight); the stack never reads the simulator at run time (ADR-006).

Usage: ``uv run python tools/tune_stack.py [--airframe aerosonde_bm] [--site cambridge_west]``
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from aeroadapt.airframes import load_airframe
from aeroadapt.core import quaternion
from aeroadapt.scenarios import load_site
from aeroadapt.sim.aerojax import make_env
from aeroadapt.stack.config import glide_polar, load_stack_cfg
from aeroadapt.vehicle.aerojax_vehicle import trimmed_start
from aeroadapt.vehicle.protocol import InitialCondition


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--airframe", default="aerosonde_bm")
    ap.add_argument("--site", default="cambridge_west")
    ap.add_argument("--alt", type=float, default=120.0, help="trim height AGL (m)")
    args = ap.parse_args()

    af = load_airframe(args.airframe)
    site = load_site(args.site)
    cfg = load_stack_cfg(args.airframe)
    v_cruise = cfg.tecs.airspeed_cruise_mps
    env = make_env(
        alt_origin_msl_m=site.ground_msl_m, magnetic_ned_gauss=site.magnetic.field_ned_gauss
    )

    speeds = np.arange(cfg.tecs.airspeed_min_mps, cfg.tecs.airspeed_max_mps + 0.1, 1.0)
    rows = []
    print(f"{'V':>5} {'theta':>7} {'ail':>7} {'ele':>7} {'rud':>7} {'thr':>6}")
    for v in speeds:
        rb, cmd = trimmed_start(af, env, InitialCondition(alt_agl_m=args.alt, airspeed_mps=v))
        theta = math.degrees(float(quaternion.to_euler(rb.quat)[1]))
        rows.append((v, theta, *cmd))
        print(f"{v:5.1f} {theta:7.3f} {cmd[0]:7.4f} {cmd[1]:7.4f} {cmd[2]:7.4f} {cmd[3]:6.3f}")
    data = np.asarray(rows)

    x = (v_cruise / data[:, 0]) ** 2
    a, b = np.linalg.lstsq(np.stack([np.ones_like(x), x], 1), data[:, 3], rcond=None)[0]
    resid = data[:, 3] - (a + b * x)
    ones = np.ones_like(x)
    _, thr_slope = np.linalg.lstsq(
        np.stack([ones, data[:, 0] - v_cruise], 1), data[:, 5], rcond=None
    )[0]
    _, pitch_b = np.linalg.lstsq(np.stack([ones, x - 1.0], 1), data[:, 1], rcond=None)[0]
    rb, cmd = trimmed_start(af, env, InitialCondition(alt_agl_m=args.alt, airspeed_mps=v_cruise))
    theta = math.degrees(float(quaternion.to_euler(rb.quat)[1]))
    ld, v_bg = glide_polar(af)

    print(f"\nelevator fit a + b·(V_cruise/V)²: max residual {np.abs(resid).max():.4f}")
    print(f"glide polar: (L/D)max {ld:.2f} at V_bg {v_bg:.2f} m/s (ρ0, windmilling prop)\n")
    print("tecs:")
    print(f"  thr_cruise: {cmd[3]:.3f}")
    print(f"  thr_slope: {thr_slope:.4f}")
    print(f"  pitch_trim_deg: {theta:.2f}")
    print(f"  pitch_trim_b_deg: {pitch_b:.2f}")
    print("attitude:")
    print("  servo_trim:")
    print(f"    aileron: {cmd[0]:.4f}")
    print(f"    rudder: {cmd[2]:.4f}")
    print(f"    elevator_a: {a:.4f}")
    print(f"    elevator_b: {b:.4f}")


if __name__ == "__main__":
    main()
