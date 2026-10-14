"""Actuator pipeline (spec §6.3), advanced at the physics rate.

Per channel: command → ServoMap (trim, max deflection, sign, reversal) → transport delay
(ring buffer) → [servo-jam fault] → first-order lag with deadband, clipped by the rate limit
→ actual deflection. The throttle channel has delay and lag only.

Commands use aircraft-response semantics (ADR-001): aileron +1 = roll right, elevator +1 =
nose up, rudder +1 = nose right, throttle in [0, 1]. The map to Beard & McLain deflection
signs is taken from the signs of the primary control derivatives, so it cannot be
misconfigured: δa ∝ sign(C_ℓ,δa), δe ∝ sign(C_m,δe), δr ∝ sign(C_n,δr).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from aeroadapt.airframes.params import ActuatorParams, AirframeParams
from aeroadapt.faultlab.slots import ServoJamFault

DELAY_BUFFER = 32  # physics steps of history (160 ms at 200 Hz); schema caps delays at 150 ms


class ActuatorState(NamedTuple):
    surfaces: Array  # (3,) actual deflections, B&M signs, rad (aileron, elevator, rudder)
    throttle: Array  # () actual throttle in [0, 1]
    buffer: Array  # (DELAY_BUFFER, 4) targets written each physics step
    head: Array  # () int32, next write index
    jam_latched: Array  # (3,) bool: jam position captured
    jam_position: Array  # (3,) rad


def response_signs(af: AirframeParams, act: ActuatorParams) -> Array:
    base = jnp.sign(jnp.array([af.C_ell_delta_a, af.C_m_delta_e, af.C_n_delta_r]))
    return jnp.where(jnp.asarray(act.reversed), -base, base)


def command_to_target(af: AirframeParams, act: ActuatorParams, cmd: Array) -> Array:
    """Normalised command (4,) → target (4,): surface deflections in rad, then throttle."""
    surf = act.trim + response_signs(af, act) * act.max_deflection * jnp.clip(cmd[:3], -1, 1)
    return jnp.concatenate([surf, jnp.clip(cmd[3:4], 0.0, 1.0)])


def target_to_command(af: AirframeParams, act: ActuatorParams, target: Array) -> Array:
    """Inverse ServoMap (e.g. to command a trim solution); not clipped."""
    surf = (target[:3] - act.trim) / (response_signs(af, act) * act.max_deflection)
    return jnp.concatenate([surf, target[3:4]])


def init_actuators(af: AirframeParams, act: ActuatorParams, cmd: Array) -> ActuatorState:
    target = command_to_target(af, act, cmd)
    return ActuatorState(
        surfaces=target[:3],
        throttle=target[3],
        buffer=jnp.broadcast_to(target, (DELAY_BUFFER, 4)),
        head=jnp.zeros((), jnp.int32),
        jam_latched=jnp.zeros(3, bool),
        jam_position=jnp.zeros(3, target.dtype),
    )


def actuator_step(
    af: AirframeParams,
    act: ActuatorParams,
    jam: ServoJamFault,
    s: ActuatorState,
    cmd: Array,
    t: Array,
    dt: float,
) -> ActuatorState:
    target = command_to_target(af, act, cmd)
    buffer = s.buffer.at[s.head].set(target)
    lag_steps = jnp.round(jnp.asarray(act.delay) / dt).astype(jnp.int32)
    delayed = buffer[(s.head - lag_steps) % DELAY_BUFFER, jnp.arange(4)]

    # Servo jam (ADR-003): latch the jam position at onset, then hold that target.
    jam_now = jnp.logical_and(jam.active, t >= jam.onset_s)
    newly = jnp.logical_and(jam_now, ~s.jam_latched)
    jam_position = jnp.where(
        newly, jnp.where(jam.at_onset, s.surfaces, jam.jam_rad), s.jam_position
    )
    jam_latched = jnp.logical_or(s.jam_latched, jam_now)
    surf_target = jnp.where(jam_now, jam_position, delayed[:3])

    # Lag with deadband, clipped by the rate limit; then mechanical limits.
    err = surf_target - s.surfaces
    half_db = act.deadband / 2
    err_db = jnp.sign(err) * jnp.maximum(jnp.abs(err) - half_db, 0.0)
    alpha = 1.0 - jnp.exp(-dt / act.tau[:3])
    step = jnp.clip(alpha * err_db, -act.rate_limit * dt, act.rate_limit * dt)
    limit = act.max_deflection + jnp.abs(act.trim)
    surfaces = jnp.clip(s.surfaces + step, -limit, limit)

    throttle = s.throttle + (1.0 - jnp.exp(-dt / act.tau[3])) * (delayed[3] - s.throttle)
    return ActuatorState(
        surfaces=surfaces,
        throttle=jnp.clip(throttle, 0.0, 1.0),
        buffer=buffer,
        head=(s.head + 1) % DELAY_BUFFER,
        jam_latched=jam_latched,
        jam_position=jam_position,
    )
