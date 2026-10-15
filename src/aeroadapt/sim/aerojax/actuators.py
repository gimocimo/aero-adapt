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
from jax.typing import ArrayLike

from aeroadapt.airframes.params import ActuatorParams, AirframeParams
from aeroadapt.faultlab.slots import ACTUATOR_KINDS, ActuatorFaultSlots, ServoJamFault, active_kind

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


_K = {name: i for i, name in enumerate(ACTUATOR_KINDS)}
FLOAT_TRAIL = 0.6  # a floating surface trails at about −0.6 × the local flow angle


def actuator_step(
    af: AirframeParams,
    act: ActuatorParams,
    jam: ServoJamFault,
    s: ActuatorState,
    cmd: Array,
    t: Array,
    dt: float,
    faults: ActuatorFaultSlots | None = None,
    flow: Array | None = None,
) -> ActuatorState:
    """``faults`` (faultlab v1) and ``flow`` = (α, β, p·b/2V) are optional; without them the
    M1 pipeline is unchanged."""
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

    rate_limit = act.rate_limit
    limit = act.max_deflection + jnp.abs(act.trim)
    half_db = act.deadband / 2
    if faults is not None:
        surf_target, rate_limit, limit, half_db = _apply_actuator_faults(
            faults, act, surf_target, rate_limit, limit, half_db, t, flow
        )

    # Lag with deadband, clipped by the rate limit; then mechanical limits.
    err = surf_target - s.surfaces
    err_db = jnp.sign(err) * jnp.maximum(jnp.abs(err) - half_db, 0.0)
    alpha = 1.0 - jnp.exp(-dt / act.tau[:3])
    step = jnp.clip(alpha * err_db, -rate_limit * dt, rate_limit * dt)
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


def _apply_actuator_faults(
    f: ActuatorFaultSlots,
    act: ActuatorParams,
    target: ArrayLike,
    rate_limit: ArrayLike,
    limit: ArrayLike,
    half_db: ArrayLike,
    t: Array,
    flow: Array | None,
) -> tuple[Array, Array, Array, Array]:
    """faultlab v1 actuator faults, slot by slot (each acts on one surface)."""
    target, rate_limit = jnp.asarray(target), jnp.asarray(rate_limit)
    limit, half_db = jnp.asarray(limit), jnp.asarray(half_db)
    max_defl, trim = jnp.asarray(act.max_deflection), jnp.asarray(act.trim)
    flow = jnp.zeros(3) if flow is None else flow
    # Local flow angle seen by each surface: aileron ~ roll-rate helix, elevator ~ α, rudder ~ β.
    local = jnp.stack([flow[2], flow[0], flow[1]])
    for i in range(f.kind.shape[0]):
        kind = active_kind(f.kind[i], f.onset_s[i], t)
        on = jnp.arange(3) == f.surface[i]
        m = f.magnitude[i]

        def where(k: str, new: Array, old: Array, kind: Array = kind, on: Array = on) -> Array:
            return jnp.where(jnp.logical_and(kind == _K[k], on), new, old)

        target = where("hard_over", jnp.sign(m + 1e-12) * max_defl, target)
        target = where("floating", -FLOAT_TRAIL * local, target)
        target = where("reversed", 2.0 * trim - target, target)
        rate_limit = where("slow", rate_limit * jnp.clip(m, 0.02, 1.0), rate_limit)
        limit = where("reduced_throw", limit * jnp.clip(m, 0.0, 1.0), limit)
        # Backlash: a dead zone in the linkage (half the magnitude, in degrees, each side).
        half_db = where("backlash", half_db + jnp.deg2rad(jnp.abs(m)) / 2, half_db)
    return target, rate_limit, limit, half_db
