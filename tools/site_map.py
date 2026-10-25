"""Render a site's map for Txt2Fly's language model (docs/06 §17): a picture the model reads
with the text list of places, so it sees sizes, shapes and what lies where.

Drawn in metres from home, north up, with a grid every 500 m labelled as the text gives
positions (N, E): the ground (fields, woods, settlements, water) on a world site, or the
OpenStreetMap areas on another; rivers and named roads; the safe zones; the no-fly areas
(red, hatched: closed at all heights; orange: closed below a height); the geofence (dashed);
the places by name; home. The picture is written next to the site file
(``configs/sites/<site>.map.png``) with a note of what it was drawn from
(``<site>.map.json``); the test suite checks the two still agree.

Usage::

    uv run python tools/site_map.py wrenfold_vale testmap london cambridge_west
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.collections import LineCollection, PolyCollection

from aeroadapt.scenarios.site import SITE_DIR, load_site
from aeroadapt.txt2fly import features
from aeroadapt.txt2fly.gazetteer import load_gazetteer
from aeroadapt.txt2fly.translate import prompt_places

PX = 1024
DPI = 128
VERSION = 2  # bump when the drawing changes
GROUND = {"field": "#e4eed2", "rough": "#dce6c4", "settlement": "#cfcac1", "yard": "#d6cfc2",
          "green": "#bcdba4", "airstrip": "#d0d0d0", "water": "#9ec9ea",
          "paved": "#d9d5cd"}  # fmt: skip
# Not labelled: the streets, and what the wind farm and the power-line corridors show.
LABEL_SKIP = ("road", "street", "lane", "home", "turbine", "pylon")


def map_hash(site_id: str) -> str:
    """What the picture is drawn from: the places, their shapes and the no-fly areas. The shapes
    count to the metre: the geometry library's last digits differ between machines."""
    site = load_site(site_id)
    gaz = load_gazetteer(site)
    h = hashlib.sha256(f"v{VERSION}|{site.model_dump_json()}".encode())
    for p in prompt_places(gaz):
        geom = features.shape(site, p.id)
        h.update(f"{p.id}|{p.name}|{p.kind}|".encode())
        if geom is not None:
            h.update(np.round(shapely.get_coordinates(geom)).astype(np.int64).tobytes())
    return h.hexdigest()[:16]


def _xy(ne: np.ndarray, home: np.ndarray) -> np.ndarray:
    """Site NE → plot x (east of home), y (north of home)."""
    d = np.asarray(ne, float) - home
    return np.stack([d[..., 1], d[..., 0]], -1)


def _rings(geom) -> list[np.ndarray]:
    polys = list(getattr(geom, "geoms", [geom]))
    return [np.asarray(p.exterior.coords) for p in polys if p.geom_type == "Polygon"]


def render(site_id: str) -> Path:
    site = load_site(site_id)
    gaz = load_gazetteer(site)
    home = gaz.home
    fence = _xy(gaz.geofence, home)
    lo, hi = fence.min(0), fence.max(0)
    pad = 0.04 * float((hi - lo).max())
    fig = plt.figure(figsize=(PX / DPI, PX / DPI), dpi=DPI)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.set_facecolor("#f4f1ea")
    ax.axis("off")
    halo = [pe.withStroke(linewidth=2.5, foreground="white")]

    # The ground.
    if site.world is not None:
        from aeroadapt.world import load_world

        w = load_world(site.world)
        for kind, colour in GROUND.items():
            polys = [_xy(np.asarray(p.polygon.exterior.coords), home) for p in w.plots
                     if p.kind == kind and p.polygon.geom_type == "Polygon"]  # fmt: skip
            if polys:
                ax.add_collection(PolyCollection(polys, facecolor=colour, edgecolor="none"))
        woods = [_xy(np.asarray(x.polygon.exterior.coords), home) for x in w.woods]
        ax.add_collection(PolyCollection(woods, facecolor="#8fbf78", edgecolor="#6f9f58", lw=0.5))
        if len(w.buildings) < 6000:
            fp = [_xy(np.asarray(b.footprint().exterior.coords), home) for b in w.buildings]
            ax.add_collection(PolyCollection(fp, facecolor="#9a948a", edgecolor="none"))
        streets = [_xy(np.asarray(r.line.coords), home) for r in w.roads if not r.name]
        ax.add_collection(LineCollection(streets, colors="#b8b2a8", linewidths=0.5))
        named = [_xy(np.asarray(r.line.coords), home) for r in w.roads if r.name]
        city = len(named) > 100  # a city's every street is named: draw them as streets
        ax.add_collection(LineCollection(named, colors="#9d968b" if city else "#6e675d",
                                         linewidths=0.6 if city else 1.4))  # fmt: skip
        if w.river is not None:
            ax.plot(*_xy(np.asarray(w.river.line.coords), home).T, color="#3d86c6", lw=3.0)
    else:
        for p in gaz.places.values():
            if p.polygon is not None and p.kind != "safe zone":
                colour = {"wood": "#8fbf78", "field": "#e4eed2", "park": "#bcdba4",
                          "village": "#cfcac1"}.get(p.kind, "#d9d5cd")  # fmt: skip
                ax.fill(*_xy(p.polygon, home).T, color=colour, lw=0)

    # Safe zones, no-fly areas, the geofence.
    for p in gaz.places.values():
        if p.kind == "safe zone" and p.polygon is not None:
            ax.fill(*_xy(p.polygon, home).T, facecolor="none", edgecolor="#2e9a4a", lw=1.2)
    labelled: set[str] = set()
    for a in gaz.no_fly:
        limited = a.ceiling_m is not None
        colour = "#f0a030" if limited else "#d83a3a"
        for ring in a.polygons:
            xy = _xy(ring, home)
            ax.fill(*xy.T, facecolor=colour, alpha=0.18, edgecolor=colour, lw=1.0,
                    hatch="" if limited else "///")  # fmt: skip
        # How high an area closed up to a height above the ground reaches (the flat-topped
        # turbines and masts are many and small: the text gives theirs).
        if limited and a.base_m is None and a.name not in labelled:
            labelled.add(a.name)
            ring = max(a.polygons, key=lambda r: float(np.ptp(r, axis=0).max()))
            x, y = _xy(ring, home).min(axis=0)
            ax.text(x, y, f"closed below {a.ceiling_m:g} m above ground", fontsize=6,
                    color="#a05a00", ha="left", va="top", path_effects=halo)  # fmt: skip
    ax.plot(*np.vstack([fence, fence[:1]]).T, color="black", lw=1.2, ls=(0, (6, 4)))

    # The grid, in metres from home, as the text gives positions.
    span = float((hi - lo).max())
    step = 500.0 if span <= 6000 else 1000.0
    for v in np.arange(np.ceil((lo[0] - pad) / step) * step, hi[0] + pad, step):
        ax.axvline(v, color="#00000022", lw=0.6)
        ax.text(v, hi[1] + pad * 0.55, f"E {v:.0f}", ha="center", va="center", fontsize=7,
                color="#444", path_effects=halo)  # fmt: skip
    for v in np.arange(np.ceil((lo[1] - pad) / step) * step, hi[1] + pad, step):
        ax.axhline(v, color="#00000022", lw=0.6)
        ax.text(lo[0] - pad * 0.55, v, f"N {v:.0f}", ha="center", va="center", fontsize=7,
                color="#444", rotation=90, path_effects=halo)  # fmt: skip

    # Places and home.
    for p in prompt_places(gaz):
        if p.kind in LABEL_SKIP:
            continue
        geom = features.shape(site, p.id)
        at = np.asarray(p.point, float)
        if geom is not None and geom.geom_type in ("Polygon", "MultiPolygon"):
            c = geom.representative_point()
            at = np.array([c.x, c.y])
        elif geom is not None and features.is_line(geom):
            c = geom.interpolate(0.5, normalized=True)
            at = np.array([c.x, c.y])
        x, y = _xy(at, home)
        ax.plot(x, y, "o", ms=2.5, color="#222")
        size = 7.5 if p.kind in ("town", "village", "safe zone", "river", "wood") else 6.5
        ax.text(x, y + span * 0.006, p.name, ha="center", va="bottom", fontsize=size,
                color="#111", path_effects=halo)  # fmt: skip
    ax.plot(0, 0, marker="*", ms=12, color="#1f5fbf", mec="white")
    ax.text(0, -span * 0.012, "home", ha="center", va="top", fontsize=8, color="#1f5fbf",
            fontweight="bold", path_effects=halo)  # fmt: skip
    ax.annotate("N", xy=(hi[0] + pad * 0.4, hi[1] - pad * 0.2),
                xytext=(hi[0] + pad * 0.4, hi[1] - pad * 1.6), ha="center", fontsize=10,
                arrowprops={"arrowstyle": "-|>", "color": "black"})  # fmt: skip
    out = SITE_DIR / f"{site_id}.map.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    meta = {"site": site_id, "hash": map_hash(site_id), "px": PX,
            "extent_e_m": [float(lo[0] - pad), float(hi[0] + pad)],
            "extent_n_m": [float(lo[1] - pad), float(hi[1] + pad)], "grid_m": step}  # fmt: skip
    (SITE_DIR / f"{site_id}.map.json").write_text(json.dumps(meta, indent=1) + "\n")
    return out


if __name__ == "__main__":
    for sid in sys.argv[1:] or ["wrenfold_vale", "testmap", "london", "cambridge_west"]:
        print(render(sid))
