"""Draw a world as a top-down map sheet (docs/07): shaded relief, contours, every feature,
the safe zones and, if a site is given, its no-fly areas and geofence; a 100 m grid labelled
every 500 m from home.

Usage: ``uv run python tools/world_map.py WORLD [--site SITE] [--out FILE]``
Example: ``uv run python tools/world_map.py testmap --site testmap``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.patches import Polygon as MPoly

from aeroadapt.scenarios.site import load_site
from aeroadapt.world import load_world
from aeroadapt.world.places import world_gazetteer

COVER = {
    "wheat": "#d7c06c", "barley": "#e2d595", "oilseed": "#ecdc4a", "pasture": "#88c25c",
    "meadow": "#a8d374", "ploughed": "#9c7a56", "stubble": "#d9cb94", "rough": "#9bc078",
    "settlement": "#c9c7b4", "yard": "#b3ab9c", "green": "#8ccf68", "airstrip": "#79bd58",
}  # fmt: skip
# Elevation tint for open ground: lowland green to upland khaki.
TINT = LinearSegmentedColormap.from_list("tint", ["#a9cf86", "#c9d38f", "#d9c58c", "#c7a77a"])


def draw(world_id: str, site_id: str | None, out: Path) -> Path:
    w = load_world(world_id)
    site = load_site(site_id) if site_id else None
    n0, e0, n1, e1 = w.bounds
    size = max(n1 - n0, e1 - e0) / 1000.0
    fig, ax = plt.subplots(figsize=(11, 11))
    extent = (w.dtm.e0, w.dtm.e0 + w.dtm.cell * (w.dtm.h.shape[1] - 1),
              w.dtm.n0, w.dtm.n0 + w.dtm.cell * (w.dtm.h.shape[0] - 1))  # fmt: skip
    h = w.dtm.h
    span = max(float(h.max() - h.min()), 1.0)
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(h, cmap=TINT, vert_exag=3.0, blend_mode="soft", dx=w.dtm.cell, dy=w.dtm.cell,
                   vmin=float(h.min()), vmax=float(h.min()) + max(span, 20.0))  # fmt: skip
    ax.imshow(rgb, origin="lower", extent=extent, zorder=0)
    for p in w.plots:
        x, y = p.polygon.exterior.xy
        ax.fill(y, x, color=COVER.get(p.cover, "#ccc"), lw=0, alpha=0.85, zorder=1)
    for wd in w.woods:
        x, y = wd.polygon.exterior.xy
        ax.fill(y, x, color="#3f6f37", lw=0, zorder=2)
    if w.river is not None:
        x, y = w.river.line.xy
        ax.plot(y, x, color="#4389cf", lw=4, zorder=3)
    for r in w.roads:
        x, y = r.line.xy
        ax.plot(y, x, color="#5c5f64" if r.kind != "track" else "#b39f7a",
                lw=max(r.width / 2.5, 1), zorder=3)  # fmt: skip
    for hg in w.hedges:
        x, y = hg.line.xy
        ax.plot(y, x, color="#2d6a2d", lw=0.6, zorder=3)
    for b in w.buildings:
        x, y = b.footprint().exterior.xy
        ax.fill(y, x, color="#a0523d", lw=0, zorder=4)
    for s in w.structures:
        ax.plot(s.e, s.n, "^", color="#d9473b", ms=8, zorder=5)
    step = 5 if span <= 60 else 10
    levels = np.arange(np.floor(h.min() / step) * step, h.max() + step, step)
    levels = levels[np.abs(levels - np.median(h)) > 2.0]  # not the plain's own ripples
    cs = ax.contour(np.linspace(extent[0], extent[1], h.shape[1]),
                    np.linspace(extent[2], extent[3], h.shape[0]), h, levels=levels,
                    colors="#5a4a32", linewidths=0.5, alpha=0.6, zorder=5)  # fmt: skip
    ax.clabel(cs, fontsize=7, fmt="%d m")
    if site is not None:
        for z in site.safe_zones:
            pts = np.array([*z.polygon_ne_m, z.polygon_ne_m[0]])
            ax.plot(pts[:, 1], pts[:, 0], color="#1f8f45", lw=2, zorder=6)
            c = pts[:-1].mean(0)
            ax.text(c[1], c[0], z.id, fontsize=12, fontweight="bold", ha="center", va="center",
                    color="#0d4d24", zorder=7)  # fmt: skip
        ops = site.operations
        if ops is not None:
            areas = {a["id"]: a for a in world_gazetteer(w)["areas"]}
            for nf in ops.no_fly:
                polys = (areas[nf.world_area]["polygons_ne_m"] if nf.world_area
                         else nf.polygons_ne_m)  # fmt: skip
                for poly in polys:
                    ax.add_patch(MPoly(np.array(poly)[:, ::-1], closed=True, zorder=6,
                                       fc=(0.9, 0.28, 0.3, 0.2), ec="#e5484d", lw=1.5))  # fmt: skip
            fence = np.array([*ops.geofence_ne_m, ops.geofence_ne_m[0]])
            ax.plot(fence[:, 1], fence[:, 0], "--", color="#e5484d", lw=1.5, zorder=6)
    ax.plot(0, 0, marker="*", ms=16, color="#d9473b", zorder=8)
    ax.text(40, 40, "home", color="#9b1c20", fontsize=9, zorder=8)
    ticks = np.arange(np.ceil(min(n0, e0) / 500) * 500, max(n1, e1) + 1, 500)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticks(np.arange(np.ceil(e0 / 100) * 100, e1 + 1, 100), minor=True)
    ax.set_yticks(np.arange(np.ceil(n0 / 100) * 100, n1 + 1, 100), minor=True)
    ax.grid(which="minor", color="white", lw=0.3, alpha=0.35)
    ax.grid(which="major", color="white", lw=0.8, alpha=0.7)
    ax.set_xlim(e0, e1)
    ax.set_ylim(n0, n1)
    ax.set_aspect("equal")
    ax.set_xlabel("east of home (m)")
    ax.set_ylabel("north of home (m)")
    ax.set_title(f"{w.spec.name}: {size:g} km square; ground {h.min():.0f} to {h.max():.0f} m "
                 "above home; contours every " f"{step} m", fontsize=12)  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw a world as a top-down map sheet.")
    ap.add_argument("world")
    ap.add_argument("--site")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path("runs/maps") / f"{a.world}.png"
    print(draw(a.world, a.site, out))


if __name__ == "__main__":
    main()
