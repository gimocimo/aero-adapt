"""Baked worlds (docs/07 §14): a world built elsewhere (the owner's test map is drawn with the
sketch tools, ``tools/sketch_bake.py``) and saved whole, read back exactly.

The file is a NumPy archive: the bulk arrays (terrain, trees, hedges, building boxes) as
arrays, and everything else (land cover, woods, roads, structures, wires, landing grounds,
places) as gzipped JSON in ``meta``. Coordinates are site NE metres; heights are metres above
the site datum, the ground at the origin.

A world from survey data (London, docs/08) also has buildings with their own footprints
(``outline``, ``holes``, ``bottom`` and a skillion roof's ``slope`` in each building's meta)
and the obstacles measured on the ground (``measured``: uint16 heights above the terrain, in
quanta, on the obstacle grid given by ``meta["measured"]``: centre of cell (0, 0), cell and
quantum), and its railways (``meta["rails"]``: points with the height of the rails, and what
carries each).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Polygon

from aeroadapt.world.generate import (
    Building,
    Dtm,
    Hedge,
    Plot,
    Rail,
    River,
    Road,
    Structure,
    Trees,
    Wire,
    Wood,
    World,
)
from aeroadapt.world.schema import WORLD_DIR, PlaceSpec, WorldSpec, ZoneSpec

BAKED_VERSION = 1
ROOFS = ("gable", "flat", "spire", "dome", "skillion")


def _poly(d: dict) -> Polygon:
    return Polygon(d["outer"], d.get("holes", []))


def load_baked(spec: WorldSpec) -> World:
    assert spec.baked is not None
    path = Path(spec.baked) if Path(spec.baked).is_absolute() else WORLD_DIR / spec.baked
    with np.load(path) as z:
        meta = json.loads(gzip.decompress(z["meta"].tobytes()))
        if meta["version"] != BAKED_VERSION:
            raise ValueError(
                f"{path.name}: baked world version {meta['version']}, expected {BAKED_VERSION}"
            )
        h = z["dtm"].astype(np.float64)
        tr = z["trees"].astype(np.float64)
        hd = z["hedges"].astype(np.float64)
        bx = z["buildings"].astype(np.float64)
        measured = None
        if "measured" in meta:
            from aeroadapt.world.raster import Obstacles

            mn0, me0, mcell, mq = meta["measured"]
            measured = Obstacles(
                float(mn0), float(me0), float(mcell), float(mq), z["measured"].astype(np.uint16)
            )
    n0, e0, cell = meta["dtm"]
    dtm = Dtm(float(n0), float(e0), float(cell), h)
    spec = spec.model_copy(
        update={
            "zones": [ZoneSpec.model_validate(x) for x in meta["zones"]],
            "places": [PlaceSpec.model_validate(x) for x in meta["places"]],
        }
    )
    r = meta["river"]
    river = (
        None
        if r is None
        else River(r["id"], r["name"], r["width"], r["depth"], LineString(r["points"]))
    )
    roads = [
        Road(x["id"], x["name"], x["kind"], x["width"], LineString(x["points"]))
        for x in meta["roads"]
    ]
    woods = [Wood(x["id"], x["name"], x["kind"], x["canopy"], _poly(x)) for x in meta["woods"]]
    plots = [
        Plot(x["id"], x["kind"], x["cover"], _poly(x), x["zone"], x["name"]) for x in meta["plots"]
    ]
    hedges = [
        Hedge(LineString([(a, b), (c, d)]), height, width) for a, b, c, d, height, width in hd
    ]
    trees = Trees(
        tr[:, 0],
        tr[:, 1],
        dtm.at(tr[:, 0], tr[:, 1]),
        tr[:, 2],
        tr[:, 3],
        tr[:, 4].astype(np.int8),
        tr[:, 5].astype(np.int16),
    )
    buildings = [
        Building(
            b["kind"],
            float(n),
            float(e),
            float(length),
            float(width),
            float(heading),
            float(base),
            float(eaves),
            float(ridge),
            ROOFS[int(roof)],
            int(variant),
            b["name"],
            tuple(b["colours"]) if b["colours"] else None,
            tuple(map(tuple, b["outline"])) if b.get("outline") else None,
            tuple(tuple(map(tuple, h)) for h in b.get("holes", [])),
            float(b.get("bottom", 0.0)),
            float(b.get("slope", 0.0)),
        )
        for b, (n, e, length, width, heading, base, eaves, ridge, roof, variant) in zip(
            meta["buildings"], bx, strict=True
        )
    ]
    structures = [Structure(**x) for x in meta["structures"]]
    wires = [Wire(x["kind"], np.asarray(x["points"], float)) for x in meta["wires"]]
    rails = [
        Rail(x["kind"], np.asarray(x["points"], float), np.asarray(x["carried"], np.int8))
        for x in meta.get("rails", [])
    ]
    return World(
        spec,
        dtm,
        river,
        roads,
        woods,
        plots,
        hedges,
        trees,
        buildings,
        structures,
        wires,
        measured,
        rails,
    )


def save_baked(
    path: Path,
    *,
    dtm: Dtm,
    trees: np.ndarray,
    hedges: np.ndarray,
    buildings: np.ndarray,
    meta: dict,
    measured=None,  # raster.Obstacles | None
) -> Path:
    """Write a baked world (the inverse of :func:`load_baked`). ``trees``: (k, 6) n, e, height,
    crown, kind, wood; ``hedges``: (k, 6) n_a, e_a, n_b, e_b, height, width; ``buildings``:
    (k, 10) n, e, length, width, heading, base, eaves, ridge, roof (index into ROOFS),
    variant, with ``meta["buildings"]`` holding each one's kind, name and colours (and a
    footprint building's ``outline``, ``holes`` and ``bottom``)."""
    meta = {**meta, "version": BAKED_VERSION, "dtm": [dtm.n0, dtm.e0, dtm.cell]}
    arrays = {}
    if measured is not None:
        meta["measured"] = [measured.n0, measured.e0, measured.cell, measured.quantum]
        arrays["measured"] = measured.agl.astype(np.uint16)
    blob = np.frombuffer(
        gzip.compress(json.dumps(meta, separators=(",", ":")).encode(), 6), np.uint8
    )
    np.savez_compressed(
        path,
        meta=blob,
        dtm=dtm.h.astype(np.float32),
        trees=trees.astype(np.float32),
        hedges=hedges.astype(np.float32),
        buildings=buildings.astype(np.float64),
        **arrays,
    )
    return path
