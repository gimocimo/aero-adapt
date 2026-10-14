"""Fixed-step integrators over pytrees."""

from __future__ import annotations

from collections.abc import Callable

import jax


def rk4[T](f: Callable[[T], T], x: T, dt: float) -> T:
    """One classical Runge-Kutta 4 step of ``ẋ = f(x)``; ``x`` may be any pytree."""

    def axpy(a: float, k: T, y: T) -> T:
        return jax.tree_util.tree_map(lambda yi, ki: yi + a * ki, y, k)

    k1 = f(x)
    k2 = f(axpy(dt / 2, k1, x))
    k3 = f(axpy(dt / 2, k2, x))
    k4 = f(axpy(dt, k3, x))
    return jax.tree_util.tree_map(
        lambda xi, a, b, c, d: xi + dt / 6.0 * (a + 2.0 * b + 2.0 * c + d), x, k1, k2, k3, k4
    )
