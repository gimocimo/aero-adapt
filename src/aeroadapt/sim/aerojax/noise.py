"""Precision-independent noise.

Every random draw in the simulator is made in float32 and then cast to the state's dtype.
Under float64, ``jax.random.normal`` would otherwise consume 64-bit random bits and produce a
*different* realisation, not a more precise one, so float32 and float64 runs of the same
seed would fly different turbulence. With this helper they see the same noise, and V7
measures precision alone.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike


def normal(key: Array, shape: tuple[int, ...], dtype: DTypeLike) -> Array:
    return jax.random.normal(key, shape, jnp.float32).astype(dtype)
