"""Polygon geometry shared by the safe-zone survey (world) and the flight stack's zone table."""

from __future__ import annotations

import math

import numpy as np


def _chord(pts: np.ndarray, c: np.ndarray, u: np.ndarray) -> tuple[float, float]:
    """Parameters (t_min < 0 < t_max) where the line c + t·u leaves the convex polygon."""
    ts = []
    for a, b in zip(pts, np.roll(pts, -1, axis=0), strict=True):
        e = b - a
        den = u[0] * e[1] - u[1] * e[0]
        if abs(den) < 1e-12:
            continue
        w = a - c
        t = (w[0] * e[1] - w[1] * e[0]) / den  # along the line
        s = (w[0] * u[1] - w[1] * u[0]) / den  # along the edge
        if -1e-9 <= s <= 1 + 1e-9:
            ts.append(t)
    return min(ts), max(ts)


def principal_chords(polygon: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """A zone polygon → (centre, axis heading in rad, half length, half width): the long axis
    is the principal component of the vertices; the centre and half extents are those of the
    chords through the polygon along and across that axis (the run actually available on the
    centre line, not the longer vertex projection)."""
    pts = np.asarray(polygon, dtype=np.float64)
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    major, minor = vt[0], vt[1]
    t0, t1 = _chord(pts, c, major)
    c = c + 0.5 * (t0 + t1) * major
    w0, w1 = _chord(pts, c, minor)
    centre = c + 0.5 * (w0 + w1) * minor
    return centre, math.atan2(major[1], major[0]), 0.5 * (t1 - t0), 0.5 * (w1 - w0)
