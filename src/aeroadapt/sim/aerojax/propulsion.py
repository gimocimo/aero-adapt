"""Electric propulsion: DC motor + fixed-pitch propeller (McLain's addendum to B&M ch. 4).

Validity: the quadratic for the propeller speed Ω has a positive root only while
``c < 0``, i.e. while the motor drives the propeller. That covers the whole nominal
envelope. A cut motor (``V_in = 0``) is outside the model's validity, so the motor-cut
fault (week 2) uses a separate windmilling-drag branch instead (spec §6.2).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from aeroadapt.airframes.params import AirframeParams


class PropulsionOutput(NamedTuple):
    thrust: Array  # N, along +x_body
    torque: Array  # N·m, aerodynamic torque on the propeller (airframe reaction is -torque)
    omega: Array  # rad/s, propeller speed


def propeller(p: AirframeParams, rho: float, va: Array, delta_t: Array) -> PropulsionOutput:
    v_in = p.V_max * delta_t
    d = p.D_prop
    two_pi = 2.0 * jnp.pi
    a = rho * d**5 * p.C_Q0 / two_pi**2
    b = rho * d**4 * p.C_Q1 * va / two_pi + p.KQ**2 / p.R_motor
    c = rho * d**3 * p.C_Q2 * va**2 - p.KQ * v_in / p.R_motor + p.KQ * p.i0
    omega = (-b + jnp.sqrt(jnp.maximum(b**2 - 4.0 * a * c, 0.0))) / (2.0 * a)
    j = two_pi * va / (omega * d)  # advance ratio
    c_t = p.C_T0 + p.C_T1 * j + p.C_T2 * j**2
    c_q = p.C_Q0 + p.C_Q1 * j + p.C_Q2 * j**2
    rev_per_s_sq = (omega / two_pi) ** 2
    return PropulsionOutput(
        thrust=rho * rev_per_s_sq * d**4 * c_t,
        torque=rho * rev_per_s_sq * d**5 * c_q,
        omega=omega,
    )
