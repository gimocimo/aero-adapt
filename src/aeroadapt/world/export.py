"""The world for the browser (docs/07 §8): everything Skyview draws, as one msgpack-able dict.

Bulk arrays (terrain heights, trees, hedge segments) travel as little-endian float32 bytes;
outlines as lists of [north, east] pairs (0.1 m). The live server sends it in ``hello``;
``export-viz`` writes it next to a replay as ``world.msgpack``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aeroadapt.world.generate import World

PAYLOAD_VERSION = 1


def _ring(coords) -> list[list[float]]:
    return np.round(np.asarray(coords)[:-1], 1).tolist()


def _poly(p) -> dict:
    return {"outer": _ring(p.exterior.coords), "holes": [_ring(h.coords) for h in p.interiors]}


def _f32(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(a, dtype="<f4").tobytes()


def viewer_payload(world: World) -> dict:
    d, t = world.dtm, world.trees
    hedges = np.array(
        [[*h.line.coords[0], *h.line.coords[-1], h.height, h.width] for h in world.hedges],
        dtype=float,
    ).reshape(-1, 6)
    return {
        "version": PAYLOAD_VERSION,
        "id": world.id,
        "name": world.spec.name,
        "bounds": list(world.bounds),
        "dtm": {
            "n0": d.n0,
            "e0": d.e0,
            "cell": d.cell,
            "rows": d.h.shape[0],
            "cols": d.h.shape[1],
            "h": _f32(d.h),
        },
        "plots": [
            {"kind": p.kind, "cover": p.cover, "zone": p.zone, "name": p.name, **_poly(p.polygon)}
            for p in world.plots
        ],
        "woods": [
            {"id": w.id, "name": w.name, "kind": w.kind, "canopy": w.canopy, **_poly(w.polygon)}
            for w in world.woods
        ],
        "river": None
        if world.river is None
        else {
            "name": world.river.name,
            "width": world.river.width,
            "points": np.round(np.asarray(world.river.line.coords), 1).tolist(),
        },
        "roads": [
            {
                "id": r.id,
                "name": r.name,
                "kind": r.kind,
                "width": r.width,
                "points": np.round(np.asarray(r.line.coords), 1).tolist(),
            }
            for r in world.roads
        ],
        # (n_a, e_a, n_b, e_b, height, width) per hedge segment
        "hedges": _f32(hedges),
        # (n, e, ground, height, crown radius, kind) per tree
        "trees": _f32(np.stack([t.n, t.e, t.base, t.height, t.crown, t.kind.astype(float)], 1)),
        "buildings": [
            {
                "kind": b.kind,
                "n": round(b.n, 2),
                "e": round(b.e, 2),
                "length": b.length,
                "width": b.width,
                "heading": round(b.heading, 5),
                "base": round(b.base, 2),
                "eaves": round(b.eaves, 2),
                "ridge": round(b.ridge, 2),
                "roof": b.roof,
                "variant": b.variant,
                "name": b.name,
                "colours": list(b.colours) if b.colours else None,
                **(
                    {
                        "outline": np.round(np.asarray(b.outline), 2).tolist(),
                        "holes": [np.round(np.asarray(h), 2).tolist() for h in b.holes],
                        "bottom": round(b.bottom, 2),
                        "slope": round(b.slope, 4),
                    }
                    if b.outline is not None
                    else {}
                ),
            }
            for b in world.buildings
        ],
        "structures": [
            {
                "kind": s.kind,
                "name": s.name,
                "n": s.n,
                "e": s.e,
                "base": round(s.base, 2),
                "height": s.height,
                "radius": s.radius,
                "heading": round(s.heading, 4),
                "hub": s.hub,
                "rotor": s.rotor,
            }
            for s in world.structures
        ],
        # Each wire's points (n, e, height above the datum).
        "wires": [{"kind": w.kind, "points": np.round(w.points, 2).tolist()} for w in world.wires],
        # Each railway's points (n, e, height of the rails) and what carries each one.
        "rails": [
            {
                "kind": r.kind,
                "points": np.round(r.points, 2).tolist(),
                "carried": r.carried.tolist(),
            }
            for r in world.rails
        ],
    }


def write_payload(world: World, path: Path) -> Path:
    import msgpack

    packed = msgpack.packb(viewer_payload(world), use_bin_type=True)
    assert isinstance(packed, bytes)
    path.write_bytes(packed)
    return path
