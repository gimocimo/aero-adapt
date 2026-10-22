"""Terrain and obstacles (docs/07 §4): what the aircraft can touch.

Two grids, both in site NE metres:

* the **terrain**: ground height above the site datum, bilinear between grid points (10 m);
* the **obstacles**: the height of whatever stands on the ground (trees, hedges, buildings,
  masts, turbines, wires) above it, per 2 m cell, in quanta (uint16).

A flat site has a 2 × 2 grid of zeros and no obstacles: every lookup returns exactly 0, so
flat-site flights are unchanged to the bit.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

Grid = Array | np.ndarray  # host arrays until they meet jit


class Terrain(NamedTuple):
    dtm_origin: Grid  # (2,) NE of grid point (0, 0)
    dtm_cell: Grid  # () m
    dtm: Grid  # (rows, cols) ground height above the site datum, m
    obs_origin: Grid  # (2,) NE of the centre of cell (0, 0)
    obs_cell: Grid  # () m
    obs_quantum: Grid  # () m per unit
    obs: Grid  # (rows, cols) uint16 obstacle height above the ground, in quanta


def flat_terrain() -> Terrain:
    return Terrain(
        dtm_origin=np.zeros(2),
        dtm_cell=np.array(1.0),
        dtm=np.zeros((2, 2)),
        obs_origin=np.zeros(2),
        obs_cell=np.array(1.0),
        obs_quantum=np.array(0.25),
        obs=np.zeros((1, 1), np.uint16),
    )


def ground_height(t: Terrain, ne: Array) -> Array:
    """Ground height at NE points (..., 2), bilinear, clamped at the grid's edge."""
    rows, cols = t.dtm.shape
    x = jnp.clip((ne[..., 0] - t.dtm_origin[0]) / t.dtm_cell, 0.0, rows - 1.0001)
    y = jnp.clip((ne[..., 1] - t.dtm_origin[1]) / t.dtm_cell, 0.0, cols - 1.0001)
    i, j = jnp.floor(x).astype(jnp.int32), jnp.floor(y).astype(jnp.int32)
    fx, fy = x - i, y - j
    h = jnp.asarray(t.dtm)  # (a host array when the environment is closed over)
    z = (h[i, j] * (1 - fx) * (1 - fy) + h[i + 1, j] * fx * (1 - fy)
         + h[i, j + 1] * (1 - fx) * fy + h[i + 1, j + 1] * fx * fy)  # fmt: skip
    return z.astype(ne.dtype)


def obstacle_height(t: Terrain, ne: Array) -> Array:
    """Obstacle height above the ground in the cell holding each NE point (0 off the grid)."""
    rows, cols = t.obs.shape
    i = jnp.round((ne[..., 0] - t.obs_origin[0]) / t.obs_cell).astype(jnp.int32)
    j = jnp.round((ne[..., 1] - t.obs_origin[1]) / t.obs_cell).astype(jnp.int32)
    inside = (i >= 0) & (j >= 0) & (i < rows) & (j < cols)
    q = jnp.asarray(t.obs)[jnp.clip(i, 0, rows - 1), jnp.clip(j, 0, cols - 1)]
    return jnp.where(inside, q * t.obs_quantum, 0.0).astype(ne.dtype)
