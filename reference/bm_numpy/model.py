"""Independent NumPy reference implementation of the Beard & McLain fixed-wing model.

Purpose: N-version validation (V3) of ``aeroadapt.sim.aerojax``. Rules of this module:

* It is written directly from the book: R. W. Beard & T. W. McLain, *Small Unmanned
  Aircraft: Theory and Practice* (2012), ch. 3-4, plus McLain's propeller/motor addendum.
* It must never import ``aeroadapt``, and ``aeroadapt`` must never import it. import-linter
  enforces both directions in CI.
* It is deliberately unoptimised, float64-only, and uses *different formulations* from the
  production code wherever the maths allows, so that agreement is meaningful:

  ======================  ===============================  ==============================
  quantity                this reference                   production (aerojax)
  ======================  ===============================  ==============================
  vector rotation         quaternion sandwich q⊗v⊗q*       direction-cosine matrix
  quaternion kinematics   ½ Ω(ω) q matrix form             ½ q ⊗ [0, ω] product form
  angular acceleration    solve(J, M − ω × Jω)             Γ-coefficient form (B&M 3.13)
  stall sigmoid σ(α)      direct B&M formula (4.10)        log-sum-exp form
  lift/drag → body        stability-axis rotation matrix   expanded scalar expressions
  parameters              raw YAML dict                    pydantic schema + NamedTuple
  ======================  ===============================  ==============================

State vector x (13): [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r].
Controls (4): [delta_a, delta_e, delta_r, delta_t] in radians (B&M signs) and [0, 1].
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

Params = dict[str, Any]


def with_environment(p: Params, rho: float, gravity: float) -> Params:
    """Attach environment constants (air density, gravity) to an airframe parameter dict."""
    return {**p, "rho": rho, "gravity": gravity}


def load_params(path: str | Path) -> Params:
    """Read an airframe YAML into a flat dict (no schema validation, by design).

    Environment constants are not part of an airframe; add them with :func:`with_environment`.
    """
    with Path(path).open() as f:
        raw = yaml.safe_load(f)
    mi, geo, aero, prop = raw["mass_inertia"], raw["geometry"], raw["aero"], raw["propulsion"]
    p: Params = {
        "mass": mi["mass_kg"],
        "J": np.array(
            [
                [mi["Jx_kgm2"], 0.0, -mi["Jxz_kgm2"]],
                [0.0, mi["Jy_kgm2"], 0.0],
                [-mi["Jxz_kgm2"], 0.0, mi["Jz_kgm2"]],
            ]
        ),
        "S": geo["wing_area_m2"],
        "b": geo["span_m"],
        "c": geo["chord_m"],
        "e": geo["oswald_e"],
    }
    p["AR"] = p["b"] ** 2 / p["S"]
    p.update(aero["longitudinal"])
    p.update(aero["lateral"])
    p["M"] = aero["stall"]["M"]
    p["alpha0"] = aero["stall"]["alpha0_rad"]
    p["D"] = prop["prop_diameter_m"]
    p["C_T0"], p["C_T1"], p["C_T2"] = prop["C_T"]
    p["C_Q0"], p["C_Q1"], p["C_Q2"] = prop["C_Q"]
    kv = (1.0 / prop["motor"]["KV_rpm_per_volt"]) * 60.0 / (2.0 * math.pi)
    p["KV"] = kv
    p["KQ"] = kv
    p["R_motor"] = prop["motor"]["R_ohm"]
    p["i0"] = prop["motor"]["i0_a"]
    p["V_max"] = prop["battery"]["n_cells"] * prop["battery"]["v_cell_v"]
    return p


# --------------------------------------------------------------------------- quaternions
def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return np.array(
        [
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
        ]
    )


def _qconj(a: np.ndarray) -> np.ndarray:
    return np.array([a[0], -a[1], -a[2], -a[3]])


def rotate_body_to_ned(e: np.ndarray, v_body: np.ndarray) -> np.ndarray:
    """v_ned = e ⊗ [0, v] ⊗ e*  (sandwich product; e is a unit quaternion)."""
    return _qmul(_qmul(e, np.array([0.0, *v_body])), _qconj(e))[1:]


def rotate_ned_to_body(e: np.ndarray, v_ned: np.ndarray) -> np.ndarray:
    return _qmul(_qmul(_qconj(e), np.array([0.0, *v_ned])), e)[1:]


# --------------------------------------------------------------------------- air data
def air_data(x: np.ndarray, wind_ned: np.ndarray) -> tuple[float, float, float]:
    """Airspeed, angle of attack, sideslip (B&M eq. 2.8 and 4.1-4.3)."""
    e = x[6:10]
    v_air = x[3:6] - rotate_ned_to_body(e, wind_ned)
    ur, vr, wr = v_air
    va = math.sqrt(ur * ur + vr * vr + wr * wr)
    va_safe = max(va, 1e-6)
    alpha = math.atan2(wr, ur)
    beta = math.asin(min(1.0, max(-1.0, vr / va_safe)))
    return va, alpha, beta


# --------------------------------------------------------------------------- aerodynamics
def sigma(p: Params, alpha: float) -> float:
    """Stall blending function, B&M eq. 4.10 (direct form)."""
    m, a0 = p["M"], p["alpha0"]
    num = 1.0 + math.exp(-m * (alpha - a0)) + math.exp(m * (alpha + a0))
    den = (1.0 + math.exp(-m * (alpha - a0))) * (1.0 + math.exp(m * (alpha + a0)))
    return num / den


def c_lift(p: Params, alpha: float) -> float:
    """Nonlinear lift coefficient, B&M eq. 4.9."""
    s = sigma(p, alpha)
    flat_plate = 2.0 * math.copysign(1.0, alpha) * math.sin(alpha) ** 2 * math.cos(alpha)
    return (1.0 - s) * (p["C_L_0"] + p["C_L_alpha"] * alpha) + s * flat_plate


def c_drag(p: Params, alpha: float) -> float:
    """Nonlinear drag coefficient, B&M eq. 4.11."""
    return p["C_D_p"] + (p["C_L_0"] + p["C_L_alpha"] * alpha) ** 2 / (math.pi * p["e"] * p["AR"])


def propulsion(p: Params, va: float, delta_t: float) -> tuple[float, float, float]:
    """Thrust, torque and propeller speed (McLain propeller/motor addendum).

    Returns (T_p [N], Q_p [N·m], Omega [rad/s]).
    """
    rho, d = p["rho"], p["D"]
    v_in = p["V_max"] * delta_t
    a = rho * d**5 * p["C_Q0"] / (2.0 * math.pi) ** 2
    b = rho * d**4 * p["C_Q1"] * va / (2.0 * math.pi) + p["KQ"] ** 2 / p["R_motor"]
    c = rho * d**3 * p["C_Q2"] * va**2 - p["KQ"] * v_in / p["R_motor"] + p["KQ"] * p["i0"]
    omega = (-b + math.sqrt(max(b * b - 4.0 * a * c, 0.0))) / (2.0 * a)
    j = 2.0 * math.pi * va / (omega * d)
    ct = p["C_T2"] * j**2 + p["C_T1"] * j + p["C_T0"]
    cq = p["C_Q2"] * j**2 + p["C_Q1"] * j + p["C_Q0"]
    n = omega / (2.0 * math.pi)  # rev/s
    return rho * n**2 * d**4 * ct, rho * n**2 * d**5 * cq, omega


def forces_moments(
    p: Params, x: np.ndarray, u: np.ndarray, wind_ned: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Total body-frame force [N] and moment [N·m] (B&M ch. 4, eqs. 4.15-4.20)."""
    rho, s_w, b, c = p["rho"], p["S"], p["b"], p["c"]
    da, de, dr, dt = u
    e = x[6:10]
    pr, qr, rr = x[10:13]
    va, alpha, beta = air_data(x, wind_ned)
    va_safe = max(va, 1e-6)
    qbar = 0.5 * rho * va**2

    # Gravity: rotate the NED weight vector into the body frame.
    f_grav = rotate_ned_to_body(e, np.array([0.0, 0.0, p["mass"] * p["gravity"]]))

    # Lift and drag in the stability frame, then rotated into the body frame.
    lift = (
        qbar
        * s_w
        * (c_lift(p, alpha) + p["C_L_q"] * c / (2 * va_safe) * qr + p["C_L_delta_e"] * de)
    )
    drag = (
        qbar
        * s_w
        * (c_drag(p, alpha) + p["C_D_q"] * c / (2 * va_safe) * qr + p["C_D_delta_e"] * de)
    )
    r_body_stab = np.array(
        [
            [math.cos(alpha), 0.0, -math.sin(alpha)],
            [0.0, 1.0, 0.0],
            [math.sin(alpha), 0.0, math.cos(alpha)],
        ]
    )
    f_lift_drag = r_body_stab @ np.array([-drag, 0.0, -lift])

    side = (
        qbar
        * s_w
        * (
            p["C_Y_0"]
            + p["C_Y_beta"] * beta
            + p["C_Y_p"] * b / (2 * va_safe) * pr
            + p["C_Y_r"] * b / (2 * va_safe) * rr
            + p["C_Y_delta_a"] * da
            + p["C_Y_delta_r"] * dr
        )
    )

    thrust, torque, _ = propulsion(p, va, dt)

    force = f_grav + f_lift_drag + np.array([thrust, side, 0.0])

    ell = (
        qbar
        * s_w
        * b
        * (
            p["C_ell_0"]
            + p["C_ell_beta"] * beta
            + p["C_ell_p"] * b / (2 * va_safe) * pr
            + p["C_ell_r"] * b / (2 * va_safe) * rr
            + p["C_ell_delta_a"] * da
            + p["C_ell_delta_r"] * dr
        )
    )
    m_pitch = (
        qbar
        * s_w
        * c
        * (
            p["C_m_0"]
            + p["C_m_alpha"] * alpha
            + p["C_m_q"] * c / (2 * va_safe) * qr
            + p["C_m_delta_e"] * de
        )
    )
    n_yaw = (
        qbar
        * s_w
        * b
        * (
            p["C_n_0"]
            + p["C_n_beta"] * beta
            + p["C_n_p"] * b / (2 * va_safe) * pr
            + p["C_n_r"] * b / (2 * va_safe) * rr
            + p["C_n_delta_a"] * da
            + p["C_n_delta_r"] * dr
        )
    )
    # Propeller reaction torque: the propeller spins positively about +x_body, so the
    # airframe feels -Q_p (convention documented in ADR-001).
    moment = np.array([ell - torque, m_pitch, n_yaw])
    return force, moment


# --------------------------------------------------------------------------- dynamics
def derivatives(p: Params, x: np.ndarray, u: np.ndarray, wind_ned: np.ndarray) -> np.ndarray:
    """ẋ for the 13-state rigid body (B&M eqs. 3.14-3.17 with quaternions, Appendix B)."""
    force, moment = forces_moments(p, x, u, wind_ned)
    vel = x[3:6]
    e = x[6:10]
    omega = x[10:13]
    pr, qr, rr = omega

    pos_dot = rotate_body_to_ned(e, vel)
    vel_dot = np.cross(vel, omega) + force / p["mass"]  # = -ω × v + F/m
    big_omega = np.array(
        [
            [0.0, -pr, -qr, -rr],
            [pr, 0.0, rr, -qr],
            [qr, -rr, 0.0, pr],
            [rr, qr, -pr, 0.0],
        ]
    )
    e_dot = 0.5 * big_omega @ e
    j = p["J"]
    omega_dot = np.linalg.solve(j, moment - np.cross(omega, j @ omega))
    return np.concatenate([pos_dot, vel_dot, e_dot, omega_dot])


def rk4_step(
    p: Params, x: np.ndarray, u: np.ndarray, wind_ned: np.ndarray, dt: float
) -> np.ndarray:
    """One RK4 step with zero-order-hold inputs, then quaternion renormalisation."""
    k1 = derivatives(p, x, u, wind_ned)
    k2 = derivatives(p, x + 0.5 * dt * k1, u, wind_ned)
    k3 = derivatives(p, x + 0.5 * dt * k2, u, wind_ned)
    k4 = derivatives(p, x + dt * k3, u, wind_ned)
    x_next = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    x_next[6:10] /= np.linalg.norm(x_next[6:10])
    return x_next


def simulate(
    p: Params, x0: np.ndarray, controls: np.ndarray, winds: np.ndarray, dt: float
) -> np.ndarray:
    """Integrate ``len(controls)`` steps; returns states of shape (N + 1, 13)."""
    xs = np.empty((len(controls) + 1, 13))
    xs[0] = x0
    for k in range(len(controls)):
        xs[k + 1] = rk4_step(p, xs[k], controls[k], winds[k], dt)
    return xs
