"""Attitude controllers with ArduPlane's structure (AP_Roll/Pitch/YawController). Pure JAX.

Each of roll and pitch: angle error → desired rate (time constant), then a rate loop with
feed-forward, P and I. Outputs are scaled by the speed scaler² (V_cruise/V)², because
control power grows with dynamic pressure. Pitch adds turn compensation; yaw combines turn
coordination, a yaw damper and a sideslip (lateral-acceleration) integrator.

Servo trims (SERVOn_TRIM analogues) are added to the outputs. The elevator trim follows the
level-flight trim curve a + b·n·(V_cruise/V)², so the pitch integrator only has to absorb
model error rather than the whole trim change across the speed range.

Outputs are normalised commands with aircraft-response signs (ADR-001).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from aeroadapt.stack.config import AttParams

G = 9.80665


class AttState(NamedTuple):
    roll_i: Array
    pitch_i: Array
    yaw_i: Array


class AttOut(NamedTuple):
    aileron: Array
    elevator: Array
    rudder: Array
    roll_rate_dem: Array
    pitch_rate_dem: Array


def att_init(dtype=jnp.float32) -> AttState:
    z = jnp.zeros((), dtype)
    return AttState(z, z, z)


def attitude_step(
    p: AttParams,
    s: AttState,
    *,
    roll: Array,
    pitch: Array,
    omega: Array,
    accel_y: Array,
    airspeed: Array,
    roll_dem: Array,
    pitch_dem: Array,
    dt: float,
) -> tuple[AttState, AttOut]:
    v = jnp.maximum(airspeed, 5.0)
    scaler = jnp.clip(p.v_cruise / v, p.scaler_min, p.scaler_max)
    sc2 = scaler**2
    pr, qr, rr = omega[0], omega[1], omega[2]

    # Roll
    p_dem = jnp.clip((roll_dem - roll) / p.roll_tau, -p.roll_rate_max, p.roll_rate_max)
    p_err = p_dem - pr
    roll_i = jnp.clip(s.roll_i + p.roll_i * p_err * dt * sc2, -p.roll_imax, p.roll_imax)
    aileron = p.trim_ail + (p.roll_ff * p_dem + p.roll_p * p_err) * sc2 + roll_i

    # Pitch, with coordinated-turn compensation q = g·tanφ·sinφ / V
    roll_c = jnp.clip(roll, -1.2, 1.2)
    q_turn = G / v * jnp.tan(roll_c) * jnp.sin(roll_c) * jnp.cos(pitch)
    q_dem = (
        jnp.clip((pitch_dem - pitch) / p.pitch_tau, -p.pitch_rate_max, p.pitch_rate_max) + q_turn
    )
    q_err = q_dem - qr
    pitch_i = jnp.clip(s.pitch_i + p.pitch_i * q_err * dt * sc2, -p.pitch_imax, p.pitch_imax)
    load = jnp.minimum(1.0 / jnp.cos(roll_c), 2.0)
    ele_trim = p.trim_ele_a + p.trim_ele_b * load * (p.v_cruise / v) ** 2
    elevator = ele_trim + (p.pitch_ff * q_dem + p.pitch_p * q_err) * sc2 + pitch_i

    # Yaw: coordination feed-forward + damper + sideslip integrator (null lateral accel).
    r_dem = G / v * jnp.sin(roll_c) * jnp.cos(pitch)
    yaw_i = jnp.clip(s.yaw_i - p.yaw_slip_i * accel_y * dt, -p.yaw_imax, p.yaw_imax)
    rudder = p.trim_rud + (p.yaw_ff * r_dem + p.yaw_damp * (r_dem - rr)) * sc2 + yaw_i

    out = AttOut(
        jnp.clip(aileron, -1, 1), jnp.clip(elevator, -1, 1), jnp.clip(rudder, -1, 1), p_dem, q_dem
    )
    return AttState(roll_i, pitch_i, yaw_i), out
