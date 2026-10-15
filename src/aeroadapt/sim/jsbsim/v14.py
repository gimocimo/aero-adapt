"""V14 (M1 spec §12, M2 spec §10): aerojax against JSBSim on the same airframe.

The Aerosonde is exported to JSBSim (``export.airframe_xml``), so only the aerodynamic
coefficients are shared. Each simulator then finds its own power-off glide trim at the same
airspeed, density and gravity, and flies the same elevator doublet open loop. JSBSim brings its
own equations of motion, integrators, WGS84 gravity and rotating Earth.

Criteria: trim α within 0.1° and γ within 0.05°; pitch-attitude response (deviation from each
simulator's own trim) within 0.3° RMS over 10 s.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.optimize import least_squares

from aeroadapt.airframes.params import AirframeParams
from aeroadapt.core import quaternion
from aeroadapt.sim.aerojax.environment import Environment
from aeroadapt.sim.aerojax.model import state_derivative, step
from aeroadapt.sim.aerojax.state import RigidBodyState, SurfaceControls
from aeroadapt.sim.jsbsim.fdm import JSBSimFDM

AIRSPEED = 22.5  # m/s, the contingency's planning glide speed
ALT_MSL = 300.0  # m
LAT_DEG = 52.2  # Cambridge
DT = 0.005
DURATION = 10.0
DOUBLET = (1.0, 2.0, math.radians(2.0))  # start (s), half-period (s), amplitude (rad)


class GlideTrim(NamedTuple):
    alpha: float
    delta_e: float
    gamma: float
    residual: float


class V14Result(NamedTuple):
    aerojax: GlideTrim
    jsbsim: GlideTrim
    t: np.ndarray
    # Euler-angle deviations from each simulator's own trim, (N + 1, 3) rad: an elevator
    # doublet (the V14 criterion) and an aileron doublet (lateral conventions, recorded).
    elevator_aerojax: np.ndarray
    elevator_jsbsim: np.ndarray
    aileron_aerojax: np.ndarray
    aileron_jsbsim: np.ndarray
    rho: float
    gravity: float

    @property
    def alpha_error_deg(self) -> float:
        return math.degrees(abs(self.aerojax.alpha - self.jsbsim.alpha))

    @property
    def gamma_error_deg(self) -> float:
        return math.degrees(abs(self.aerojax.gamma - self.jsbsim.gamma))

    @staticmethod
    def _rms_deg(a: np.ndarray, b: np.ndarray) -> float:
        return math.degrees(float(np.sqrt(np.mean((a - b) ** 2))))

    @property
    def pitch_rms_deg(self) -> float:
        return self._rms_deg(self.elevator_aerojax[:, 1], self.elevator_jsbsim[:, 1])

    @property
    def roll_rms_deg(self) -> float:
        return self._rms_deg(self.aileron_aerojax[:, 0], self.aileron_jsbsim[:, 0])

    @property
    def yaw_rms_deg(self) -> float:
        return self._rms_deg(self.aileron_aerojax[:, 2], self.aileron_jsbsim[:, 2])

    @property
    def passed(self) -> bool:
        return (
            self.alpha_error_deg <= 0.1
            and self.gamma_error_deg <= 0.05
            and self.pitch_rms_deg <= 0.3
        )


def _glide_trim(x: np.ndarray, residual: np.ndarray) -> GlideTrim:
    return GlideTrim(float(x[0]), float(x[1]), float(x[2]), float(np.max(np.abs(residual))))


def unpowered(p: AirframeParams) -> AirframeParams:
    """Motor cut with a zero-drag propeller: no propulsive force or torque at all."""
    return p._replace(C_windmill=0.0)


def _glide_state(alpha: Array, gamma: Array, va: float, h: float) -> RigidBodyState:
    vel = va * jnp.stack([jnp.cos(alpha), jnp.zeros_like(alpha), jnp.sin(alpha)])
    q = quaternion.from_euler(jnp.zeros_like(alpha), alpha + gamma, jnp.zeros_like(alpha))
    return RigidBodyState(jnp.array([0.0, 0.0, -h]), vel, q, jnp.zeros(3))


def aerojax_glide_trim(p: AirframeParams, env: Environment, va: float) -> GlideTrim:
    p = unpowered(p)
    with jax.enable_x64(True):

        def residual(z: Array) -> Array:
            s = _glide_state(z[0], z[2], va, 100.0)
            u = SurfaceControls(jnp.asarray(0.0), z[1], jnp.asarray(0.0), jnp.asarray(0.0))
            d = state_derivative(p, env, s, u, jnp.zeros(3), motor_cut=jnp.asarray(True))
            return jnp.stack([d.vel_body[0], d.vel_body[2], d.omega[1]])

        fun, jac = jax.jit(residual), jax.jit(jax.jacfwd(residual))
        sol = least_squares(
            lambda z: np.asarray(fun(jnp.asarray(z))),
            np.array([0.08, -0.1, -0.08]),
            jac=lambda z: np.asarray(jac(jnp.asarray(z))),  # pyright: ignore[reportArgumentType]
            method="lm",
            xtol=1e-15,
            ftol=1e-15,
        )
    return _glide_trim(sol.x, sol.fun)


def jsbsim_glide_trim(fdm: JSBSimFDM, va: float, alt_msl: float, lat_deg: float) -> GlideTrim:
    """Zero (u̇, ẇ, q̇) in JSBSim itself, by finite-difference least squares over (α, δe, γ)."""

    def residual(z: np.ndarray) -> np.ndarray:
        fdm.initialise(alt_msl, va, alpha=z[0], gamma=z[2], lat_deg=lat_deg, surfaces=(0, z[1], 0))
        return np.array(fdm.body_accel)

    sol = least_squares(
        residual, np.array([0.08, -0.1, -0.08]), method="lm", diff_step=1e-7, xtol=1e-14
    )
    return _glide_trim(sol.x, sol.fun)


def doublet_schedule(trim: tuple[float, float, float], channel: int, n: int) -> np.ndarray:
    """(n, 3) surface deflections: trim plus a doublet on one channel (0 aileron, 1 elevator)."""
    t = DT * np.arange(n)
    start, half, amp = DOUBLET
    u = np.tile(np.asarray(trim, dtype=float), (n, 1))
    u[(t >= start) & (t < start + half), channel] += amp
    u[(t >= start + half) & (t < start + 2 * half), channel] -= amp
    return u


def _jsbsim_response(fdm: JSBSimFDM, trim: GlideTrim, surfaces: np.ndarray) -> np.ndarray:
    fdm.initialise(ALT_MSL, AIRSPEED, alpha=trim.alpha, gamma=trim.gamma, lat_deg=LAT_DEG)
    out = [fdm.euler]
    for u in surfaces:
        fdm.set_surfaces(*map(float, u))
        fdm.step()
        out.append(fdm.euler)
    e = np.array(out)
    e[:, 2] = np.unwrap(e[:, 2])
    return e - np.array([0.0, trim.alpha + trim.gamma, 0.0])


def _aerojax_response(
    p: AirframeParams, env: Environment, trim: GlideTrim, surfaces: np.ndarray
) -> np.ndarray:
    pu = unpowered(p)
    with jax.enable_x64(True):
        s0 = _glide_state(jnp.asarray(trim.alpha), jnp.asarray(trim.gamma), AIRSPEED, ALT_MSL)

        def body(s: RigidBodyState, u: Array) -> tuple[RigidBodyState, Array]:
            c = SurfaceControls(u[0], u[1], u[2], jnp.asarray(0.0))
            nxt = step(pu, env, s, c, jnp.zeros(3), DT, motor_cut=jnp.asarray(True))
            return nxt, jnp.stack(quaternion.to_euler(nxt.quat))

        _, e = jax.jit(lambda s, u: jax.lax.scan(body, s, u))(s0, jnp.asarray(surfaces))
    e = np.concatenate([[[0.0, trim.alpha + trim.gamma, 0.0]], np.asarray(e)])
    e[:, 2] = np.unwrap(e[:, 2])
    return e - np.array([0.0, trim.alpha + trim.gamma, 0.0])


def run_v14(p: AirframeParams, model: str = "aerosonde_bm") -> V14Result:
    fdm = JSBSimFDM(model, dt=DT)
    fdm.initialise(ALT_MSL, AIRSPEED, lat_deg=LAT_DEG)
    env = Environment(fdm.rho, fdm.gravity)  # density and gravity matched

    aj = aerojax_glide_trim(p, env, AIRSPEED)
    js = jsbsim_glide_trim(fdm, AIRSPEED, ALT_MSL, LAT_DEG)
    n = round(DURATION / DT)

    def responses(channel: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            _aerojax_response(p, env, aj, doublet_schedule((0, aj.delta_e, 0), channel, n)),
            _jsbsim_response(fdm, js, doublet_schedule((0, js.delta_e, 0), channel, n)),
        )

    elev_aj, elev_js = responses(1)
    ail_aj, ail_js = responses(0)
    return V14Result(
        aerojax=aj,
        jsbsim=js,
        t=DT * np.arange(n + 1),
        elevator_aerojax=elev_aj,
        elevator_jsbsim=elev_js,
        aileron_aerojax=ail_aj,
        aileron_jsbsim=ail_js,
        rho=float(env.rho),
        gravity=float(env.gravity),
    )
