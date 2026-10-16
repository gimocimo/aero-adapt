"""Intervals for the bench tables: Wilson for rates, percentile bootstrap for means."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Z95 = 1.959964


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    """(rate, low, high) for k successes in n trials (nan when n = 0)."""
    if n == 0:
        return math.nan, math.nan, math.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_mean(
    x: Sequence[float], seed: int = 0, n_boot: int = 2000
) -> tuple[float, float, float]:
    """(mean, low, high): 95 % percentile bootstrap interval (nan when empty)."""
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, (n_boot, a.size))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(a.mean()), float(lo), float(hi)
