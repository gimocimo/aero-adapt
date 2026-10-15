"""Validity filters, re-checked in aerojax (M2 spec §4.1 step 8).

AeroSandbox produced the coefficients; these checks use only the simulator's own model, so an
airframe is accepted for what aerojax will actually fly. Trim and linearisation are batched
over the fleet with ``vmap`` (one compilation for any number of airframes):

- level trim at 1.25 V_stall, cruise and 1.5 V_cruise, with throttle in [0, 1], elevator
  within 80 % of travel and α below the stall at the slow point;
- modes at cruise: short period ζ ≥ 0.25, Dutch roll ζ ≥ 0.05, roll subsidence stable,
  phugoid not faster than a 0.02 1/s divergence, spiral time to double ≥ 10 s;
- static margin 3-25 % from the tables; C_nβ ≥ 0.03/rad (the low end of Raymer's typical
  range: directionally weaker designs sideslip badly under any autopilot) and C_ℓβ < 0;
  steady roll rate ≥ 60°/s at cruise with full aileron; static thrust-to-weight ≥ 0.4;
  stall speed 5-20 m/s; wing loading 20-150 N/m².
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from aeroadapt.airframes import Airframe, AirframeParams, to_vehicle_params
from aeroadapt.sim.aerojax.aero_tables import table_lookup
from aeroadapt.sim.aerojax.environment import Environment
from aeroadapt.sim.aerojax.propulsion import propeller
from aeroadapt.sim.aerojax.trim import LAT, LON, _euler_model, trim_level

RHO0, G0 = 1.225, 9.806_65
CNB_MIN = 0.03  # 1/rad: the low end of Raymer's typical range (0.0005-0.0015 per degree)


@dataclass(frozen=True)
class FilterResult:
    accepted: np.ndarray  # (N,) bool
    reasons: list[list[str]]  # per airframe, empty when accepted
    metrics: dict[str, np.ndarray]  # per airframe


def stack_params(airframes: Sequence[Airframe]) -> tuple[AirframeParams, np.ndarray]:
    """Fleet axis on every leaf, plus the surface travel limits (N, 3) in rad."""
    vps = [to_vehicle_params(a) for a in airframes]
    params = jax.tree.map(lambda *x: np.stack([np.asarray(v, dtype=np.float64) for v in x]),
                          *[v.airframe for v in vps])  # fmt: skip
    travel = np.stack([v.actuators.max_deflection for v in vps])
    return params, travel


@jax.jit
def _trim_batch(p: AirframeParams, va: Array) -> tuple[Array, Array]:
    env = Environment(RHO0, G0)
    return jax.vmap(lambda pi, v: trim_level(pi, env, jnp.asarray(pi.V_max), v))(p, va)


@jax.jit
def _jacobian_batch(p: AirframeParams, x: Array, u: Array) -> Array:
    def one(pi: AirframeParams, xi: Array, ui: Array) -> Array:
        return jax.jacfwd(_euler_model(pi, Environment(RHO0, G0), None), 0)(xi, ui)

    return jax.vmap(one)(p, x, u)


def _euler_state(z: np.ndarray, va: float) -> np.ndarray:
    a, th = z[0], z[1]
    return np.array([0, 0, -100.0, va * math.cos(a), 0, va * math.sin(a), z[2], th, 0, 0, 0, 0])


def _modes(a: np.ndarray) -> dict[str, float]:
    lon = np.linalg.eigvals(a[np.ix_(LON[:4], LON[:4])])
    lat = np.linalg.eigvals(a[np.ix_(LAT[:4], LAT[:4])])
    cplx_lon = sorted([e for e in lon if e.imag > 1e-6], key=abs)
    zeta = lambda e: -e.real / abs(e)  # noqa: E731
    out = {"sp_zeta": np.nan, "ph_real": np.nan, "dr_zeta": np.nan, "roll_real": np.nan,
           "spiral_real": np.nan}  # fmt: skip
    if len(cplx_lon) >= 2:
        out["ph_real"], out["sp_zeta"] = cplx_lon[0].real, zeta(cplx_lon[-1])
    elif len(cplx_lon) == 1:  # short period split into two real roots: heavily damped
        out["ph_real"], out["sp_zeta"] = cplx_lon[0].real, 1.0
    cplx_lat = [e for e in lat if e.imag > 1e-6]
    real_lat = sorted([e.real for e in lat if abs(e.imag) <= 1e-6])
    if cplx_lat:
        out["dr_zeta"] = zeta(cplx_lat[0])
    if len(real_lat) >= 2:
        out["roll_real"], out["spiral_real"] = real_lat[0], real_lat[-1]
    return out


def check(airframes: Sequence[Airframe], meta: Sequence[dict]) -> FilterResult:
    n = len(airframes)
    with jax.enable_x64(True):
        p, travel = stack_params(airframes)
        v_stall = np.array([m["v_stall"] for m in meta])
        v_cruise = np.array([m["v_cruise"] for m in meta])
        speeds = {"slow": 1.25 * v_stall, "cruise": v_cruise, "fast": 1.5 * v_cruise}
        trims = {
            k: [np.asarray(x) for x in _trim_batch(p, jnp.asarray(v))] for k, v in speeds.items()
        }
        zc = trims["cruise"][0]
        x = np.stack([_euler_state(zc[i], v_cruise[i]) for i in range(n)])
        u = zc[:, 3:7]
        jac = np.asarray(_jacobian_batch(p, jnp.asarray(x), jnp.asarray(u)))
        # Static margin from the tables at cruise: −dC_m/dC_L.
        a_cr = jnp.asarray(zc[:, 0])
        tables = p.tables
        assert tables is not None  # generated airframes are tabulated
        dcl = jax.vmap(jax.grad(table_lookup, argnums=1))(tables.C_L, a_cr)
        dcm = jax.vmap(jax.grad(table_lookup, argnums=1))(tables.C_m, a_cr)
        sm = np.asarray(-dcm / dcl)
        static_t = np.asarray(
            jax.vmap(
                lambda pi: propeller(pi, RHO0, jnp.asarray(0.0), jnp.asarray(1.0), pi.V_max).thrust
            )(p)
        )
    mass, span = np.asarray(p.mass), np.asarray(p.b)
    roll_rate = np.degrees(
        -np.asarray(p.C_ell_delta_a) * np.sign(np.asarray(p.C_ell_delta_a)) * travel[:, 0]
        * 2.0 * v_cruise / (np.asarray(p.C_ell_p) * span)
    )  # fmt: skip
    tw = static_t / (mass * G0)
    wl = mass * G0 / np.asarray(p.S)
    alpha_stall = np.array([a.aero.stall.alpha0_rad for a in airframes])
    cnb, clb = np.asarray(p.C_n_beta), np.asarray(p.C_ell_beta)

    accepted = np.ones(n, bool)
    reasons: list[list[str]] = [[] for _ in range(n)]
    modes = [_modes(jac[i]) for i in range(n)]

    def fail(i: int, why: str) -> None:
        accepted[i] = False
        reasons[i].append(why)

    for i in range(n):
        for k, (z, res) in trims.items():
            if not (np.isfinite(res[i]) and res[i] < 1e-6):
                fail(i, f"trim {k}: no solution")
                continue
            if not 0.0 <= z[i, 6] <= 1.0:
                fail(i, f"trim {k}: throttle out of range")
            if abs(z[i, 4]) > 0.8 * travel[i, 1]:
                fail(i, f"trim {k}: elevator saturated")
        if trims["slow"][1][i] < 1e-6 and trims["slow"][0][i, 0] > alpha_stall[i] - math.radians(
            1.0
        ):
            fail(i, "trim slow: at the stall")
        m = modes[i]
        if not m["sp_zeta"] >= 0.25:
            fail(i, "short period damping")
        if not m["dr_zeta"] >= 0.05:
            fail(i, "Dutch roll damping")
        if not m["roll_real"] < 0.0:
            fail(i, "roll mode")
        if not m["ph_real"] <= 0.02:
            fail(i, "phugoid divergence")
        if m["spiral_real"] > 0 and math.log(2) / m["spiral_real"] < 10.0:
            fail(i, "spiral divergence")
        if not 0.03 <= sm[i] <= 0.25:
            fail(i, "static margin")
        if not cnb[i] >= CNB_MIN:
            fail(i, "directional stability (C_n,beta)")
        if not clb[i] < 0.0:
            fail(i, "dihedral effect (C_l,beta)")
        if roll_rate[i] < 60.0:
            fail(i, "roll authority")
        if tw[i] < 0.4:
            fail(i, "thrust-to-weight")
        if not 5.0 <= v_stall[i] <= 20.0:
            fail(i, "stall speed")
        if not 20.0 <= wl[i] <= 150.0:
            fail(i, "wing loading")

    metrics = {
        "static_margin_aerojax": sm,
        "roll_rate_dps": roll_rate,
        "thrust_to_weight": tw,
        "trim_alpha_cruise_deg": np.degrees(zc[:, 0]),
        "trim_throttle_cruise": zc[:, 6],
        "trim_throttle_fast": trims["fast"][0][:, 6],
        **{k: np.array([m[k] for m in modes]) for k in modes[0]},
    }
    return FilterResult(accepted, reasons, metrics)
