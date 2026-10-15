"""Aero v2: static coefficients tabulated over 360° of angle of attack (M2 spec §3).

The grid is uniform (``ALPHA_GRID_N`` points from -π to π), so a lookup is index arithmetic
rather than a search. That keeps it cheap on a GPU and lets a heterogeneous fleet ``vmap``
with one table shape.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from aeroadapt.airframes.params import ALPHA_GRID_N

_PER_RAD = (ALPHA_GRID_N - 1) / (2.0 * math.pi)


def table_lookup(table: ArrayLike, alpha: Array) -> Array:
    """Linear interpolation of ``table`` (shape (ALPHA_GRID_N,)) at ``alpha`` in [-π, π]."""
    x = (alpha + math.pi) * _PER_RAD
    i = jnp.clip(jnp.floor(x), 0, ALPHA_GRID_N - 2)
    frac = x - i
    k = i.astype(jnp.int32)
    table = jnp.asarray(table)
    lo = jnp.take(table, k, axis=-1)
    hi = jnp.take(table, k + 1, axis=-1)
    return lo + frac * (hi - lo)
