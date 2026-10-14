"""V2: rigid-body invariants (spec §12).

These checks are independent of any aircraft data. A torque-free body must conserve its
inertial angular momentum vector and rotational kinetic energy. Gravity-only free fall must
match ½gt². The quaternion must stay unit-norm.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from aeroadapt.airframes import AirframeParams
from aeroadapt.core import quaternion as Q
from aeroadapt.sim.aerojax import RigidBodyState, rigid_body
from aeroadapt.sim.aerojax.integrators import rk4

DT = 0.005


def _integrate(params: AirframeParams, s0: RigidBodyState, force_fn, n_steps: int):
    """RK4 + renormalisation with forces/moments given by ``force_fn(state)``."""

    def f(s: RigidBodyState) -> RigidBodyState:
        force, moment = force_fn(s)
        return rigid_body.derivatives(params, s, force, moment)

    def body(s: RigidBodyState, _):
        nxt = rk4(f, s, DT)
        nxt = nxt._replace(quat=Q.normalize(nxt.quat))
        return nxt, nxt

    _, traj = jax.lax.scan(body, s0, None, length=n_steps)
    return traj


def _inertia(p: AirframeParams) -> jnp.ndarray:
    return jnp.array([[p.Jx, 0.0, -p.Jxz], [0.0, p.Jy, 0.0], [-p.Jxz, 0.0, p.Jz]])


def test_torque_free_motion_conserves_momentum_and_energy(params: AirframeParams) -> None:
    s0 = RigidBodyState(
        pos_ned=jnp.zeros(3),
        vel_body=jnp.array([20.0, 1.0, -0.5]),
        quat=Q.from_euler(0.2, -0.1, 0.7),
        omega=jnp.array([0.6, -0.4, 0.5]),  # spins about all axes (incl. intermediate axis)
    )
    zero = jnp.zeros(3)
    traj = _integrate(params, s0, lambda s: (zero, zero), n_steps=20_000)  # 100 s

    j = _inertia(params)
    h_body = traj.omega @ j.T
    h_ned = Q.body_to_ned(traj.quat, h_body)  # inertial angular momentum vector
    h0 = Q.body_to_ned(s0.quat, j @ s0.omega)
    ke = 0.5 * jnp.einsum("ni,ij,nj->n", traj.omega, j, traj.omega)
    ke0 = 0.5 * s0.omega @ j @ s0.omega

    assert float(jnp.max(jnp.linalg.norm(h_ned - h0, axis=-1)) / jnp.linalg.norm(h0)) <= 1e-9
    assert float(jnp.max(jnp.abs(ke - ke0)) / ke0) <= 1e-9
    assert float(jnp.max(jnp.abs(jnp.linalg.norm(traj.quat, axis=-1) - 1.0))) <= 1e-12
    # Translational momentum is conserved too: ground velocity in NED is constant. (Relative,
    # like the other invariants: the body-frame velocity is coupled to the attitude, so RK4
    # truncation shows up at the ~1e-10 level.)
    v_ned = Q.body_to_ned(traj.quat, traj.vel_body)
    v0 = Q.body_to_ned(s0.quat, s0.vel_body)
    assert float(jnp.max(jnp.linalg.norm(v_ned - v0, axis=-1)) / jnp.linalg.norm(v0)) <= 1e-9


def test_gravity_only_free_fall(params: AirframeParams) -> None:
    g = 9.81
    s0 = RigidBodyState(jnp.zeros(3), jnp.zeros(3), Q.from_euler(0.3, 0.2, -1.0), jnp.zeros(3))

    def gravity(s: RigidBodyState):
        f = Q.ned_to_body(s.quat, jnp.array([0.0, 0.0, params.mass * g]))
        return f, jnp.zeros(3)

    n = 2_000  # 10 s
    traj = _integrate(params, s0, gravity, n_steps=n)
    t = DT * jnp.arange(1, n + 1)
    np.testing.assert_allclose(traj.pos_ned[:, 2], 0.5 * g * t**2, rtol=0, atol=1e-9)
    np.testing.assert_allclose(traj.pos_ned[:, :2], 0.0, atol=1e-9)
