# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportOperatorIssue=false, reportReturnType=false, reportOptionalMemberAccess=false
# pyright: reportGeneralTypeIssues=false
# (Drawing tools over shapely, whose set operations are typed as returning a BaseGeometry; the
# code relies on them being polygons or lines at run time, which the checker cannot narrow.)
"""A high-resolution isometric picture of a map sketch (8K: 7680 × 4320 by default), at true
heights and seen from the south-west: the whole map as a solid block, the ground drawn in
detail, every building and tree standing on it.

How it is drawn: the ground (relief, water, roads, tracks, the town's streets, yards, parks
and squares) is first drawn top down at about 0.6 m a pixel. Each column of the picture is
then a line of sight across the terrain; the terrain along it is marched from near to far,
and each row of the picture takes the ground where the line of sight first rises past it (a
heightfield ray cast, as in voxel-space renderers), so hills hide what lies behind them.
Below the near edges the block's sides are drawn. The buildings and trees go on top, far to
near, each standing on the ground under it; any wholly hidden behind a hill are left out.

Usage: ``uv run python tools/sketch_8k.py testmap [--width 7680 --height 4320]``
→ runs/maps/<name>_8k.png (and a .jpg copy, smaller to pass around)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sketch_city import MAIN_ROAD, building_faces, draw_ground, fill, shade_overlay
from sketch_infra import infra_items
from sketch_iso import LABEL_OFFSET, VIEW_ISO, Z_UP, _darker, _relief
from sketch_land import shadow_of, tree_shapes, turbine_faces, turbine_shadow
from sketch_map import CELL, band, prepare
from sketch_power import power_items

SX, SY = 0.7071, 0.3536  # page x = SX (east − north), page y = SY (north + east) + Z_UP height
EARTH = np.array([0.55, 0.43, 0.31])
SKY = np.array([0.965, 0.972, 0.98])


def ground_texture(prep, px: int) -> np.ndarray:
    """The ground drawn top down over the map, ``px`` pixels a side (rows south to north)."""
    sk, ns, es, h, channels, _anchors, city = prep
    (n0, e0), (n1, e1) = sk["bounds_ne_m"]
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.imshow(
        _relief(h), origin="lower", extent=(es[0], es[-1], ns[0], ns[-1]), interpolation="bilinear"
    )
    for c in channels:
        o = band(c.line, c.width)
        ax.fill(o[:, 1], o[:, 0], color="#3f86cf", lw=0, zorder=3)

    def tf(n, e):
        return e, n

    if city is not None:
        fill(
            ax,
            [ln.buffer(4.5) for _, ln in city.roads],
            tf,
            3.25,
            facecolors=MAIN_ROAD,
            edgecolors="none",
        )
        fill(
            ax,
            [ln.buffer(1.6) for k, ln, _ in city.streets if k == "track"],
            tf,
            3.25,
            facecolors="#b59f78",
            edgecolors="none",
        )
        draw_ground(ax, city, tf, detail=True, crowns=False)
        ax.imshow(
            shade_overlay(h, CELL),
            origin="lower",
            extent=(es[0], es[-1], ns[0], ns[-1]),
            interpolation="bilinear",
            zorder=3.25,
        )
    ax.set_xlim(e0, e1)
    ax.set_ylim(n0, n1)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img[::-1]


def render(
    prep, out: Path, width: int, height: int, tex_px: int = 8192, zones: bool = False
) -> Path:
    sk, ns, es, h, _channels, _anchors, city = prep
    (n0, e0), (n1, e1) = sk["bounds_ne_m"]
    t0 = time.time()
    tex = ground_texture(prep, tex_px)
    print(f"ground drawn ({tex_px} px a side) in {time.time() - t0:.0f} s")

    hc = h

    def ground(n, e):
        n, e = np.atleast_1d(np.asarray(n, float)), np.atleast_1d(np.asarray(e, float))
        return map_coordinates(h, [(n - ns[0]) / CELL, (e - es[0]) / CELL], order=1, mode="nearest")

    # The page: the block from its underside to the top of the tallest tower, with a margin.
    base = float(h.min()) - 60.0
    tallest = max((b.height for b in city.buildings), default=0.0) if city else 0.0
    x_lo, x_hi = SX * (e0 - n1), SX * (e1 - n0)
    y_lo = SY * (n0 + e0) + Z_UP * base
    y_hi = SY * (n1 + e1) + Z_UP * (float(h.max()) + 20)
    y_hi = max(y_hi, SY * (n1 + e1) / 2 + Z_UP * tallest)
    s = max((x_hi - x_lo) / (width * 0.97), (y_hi - y_lo) / (height * 0.95))  # metres a pixel
    x0 = (x_lo + x_hi) / 2 - width * s / 2
    y_top = (y_lo + y_hi) / 2 + height * s / 2
    rows_y = y_top - (np.arange(height) + 0.5) * s

    # What to stand on the ground: every building's faces, every turbine and every tree, with
    # where they are; and the shadows they cast.
    items, shadows = [], []
    trees = np.zeros((0, 6))
    if city is not None:
        for b in city.buildings:
            c = b.centre
            z0 = float(ground(c[0], c[1])[0])
            polys, cols = [], []
            for pts, col in building_faces(b, VIEW_ISO):
                px_ = SX * (pts[:, 1] - pts[:, 0])
                py_ = SY * (pts[:, 0] + pts[:, 1]) + Z_UP * (pts[:, 2] + z0)
                polys.append(np.column_stack([px_, py_]))
                cols.append(col)
            items.append([c[0] + c[1], polys, cols, c, z0, b.height])
            shadows += shadow_of(
                [(p.footprint, p.z0, p.z1 + p.rise) for p in b.parts], SX, SY, Z_UP, z0
            )
        for t in city.turbines:
            z0 = float(ground(t["n"], t["e"])[0])
            polys, cols = [], []
            for pts, col in turbine_faces(t, VIEW_ISO):
                polys.append(
                    np.column_stack(
                        [
                            SX * (pts[:, 1] - pts[:, 0]),
                            SY * (pts[:, 0] + pts[:, 1]) + Z_UP * (pts[:, 2] + z0),
                        ]
                    )
                )
                cols.append(col)
            items.append(
                [
                    t["n"] + t["e"],
                    polys,
                    cols,
                    np.array([t["n"], t["e"]]),
                    z0,
                    t["hub"] + t["rotor"] / 2,
                ]
            )
            shadows += turbine_shadow(t, SX, SY, Z_UP, z0)
        for fn in (power_items, infra_items):
            more, more_shadows = fn(
                city, VIEW_ISO, lambda n, e, z: (SX * (e - n), SY * (n + e) + Z_UP * z), ground
            )
            items += [list(it) for it in more]
            shadows += more_shadows
        trees = np.vstack(
            [
                np.column_stack(
                    [
                        city.trees[:, :3],
                        city.trees[:, 3] + 0.9 * city.trees[:, 2],
                        np.zeros(len(city.trees)),
                        np.arange(len(city.trees)) % 5,
                    ]
                ),
                city.forest,
            ]
        )
        z0s = ground(trees[:, 0], trees[:, 1])
        for (n, e, r, hgt, kind, shade), z0 in zip(trees, z0s, strict=True):
            sh, shapes = tree_shapes(n, e, r, hgt, int(kind), shade, float(z0), SX, SY, Z_UP)
            items.append(
                [
                    n + e,
                    [p for p, _ in shapes],
                    [c for _, c in shapes],
                    np.array([n, e]),
                    float(z0),
                    hgt,
                ]
            )
            shadows.append(sh)
    print(f"{len(items)} things to stand up ({len(trees) if city is not None else 0} trees)")
    by_column: dict[int, list[int]] = {}
    for i, it in enumerate(items):
        c = it[3]
        j = int((SX * (c[1] - c[0]) - x0) / s)
        by_column.setdefault(j, []).append(i)

    # The terrain, one column (line of sight) at a time.
    t0 = time.time()
    north = np.full((height, width), np.nan, np.float32)
    east = np.full((height, width), np.nan, np.float32)
    side = np.zeros((height, width), np.int8)  # 1: the south face, 2: the west face
    hidden = np.zeros(len(items), bool)
    step = 0.8  # along the line of sight, in (north + east) metres
    for j in range(width):
        k = (x0 + (j + 0.5) * s) / SX  # east − north along this line of sight
        d_lo, d_hi = max(2 * n0 + k, 2 * e0 - k), min(2 * n1 + k, 2 * e1 - k)
        if d_hi - d_lo < step:
            continue
        d = np.arange(d_lo, d_hi, step)
        nn, ee = (d - k) / 2, (d + k) / 2
        y = SY * d + Z_UP * map_coordinates(
            hc, [(nn - ns[0]) / CELL, (ee - es[0]) / CELL], order=1, mode="nearest"
        )
        top = np.maximum.accumulate(y)
        seen = (rows_y >= top[0]) & (rows_y <= top[-1])
        idx = np.searchsorted(top, rows_y[seen], side="left")
        north[seen, j] = nn[idx]
        east[seen, j] = ee[idx]
        below = (rows_y < top[0]) & (rows_y >= SY * d[0] + Z_UP * base)
        side[below, j] = 1 if 2 * n0 + k >= 2 * e0 - k else 2
        for i in by_column.get(j, ()):  # wholly behind a hill?
            it = items[i]
            dd = it[3][0] + it[3][1]
            kk = int(np.searchsorted(d, dd)) - 2
            if kk > 0 and top[kk] > SY * dd + Z_UP * (it[4] + it[5]) + 1.0:
                hidden[i] = True
    took = time.time() - t0
    print(
        f"terrain cast ({width} columns) in {took:.0f} s; {int(hidden.sum())} items hidden by hills"
    )

    img = np.empty((height, width, 3), np.float32)
    img[:] = SKY
    ok = ~np.isnan(north)
    ti = np.clip(
        ((north[ok] - n0) / (n1 - n0) * (tex.shape[0] - 1)).round().astype(int), 0, tex.shape[0] - 1
    )
    tj = np.clip(
        ((east[ok] - e0) / (e1 - e0) * (tex.shape[1] - 1)).round().astype(int), 0, tex.shape[1] - 1
    )
    img[ok] = tex[ti, tj] / 255.0
    del tex, north, east
    rr = np.repeat(rows_y[:, None], width, axis=1)
    for face, shade in ((1, 0.66), (2, 0.84)):
        m = side == face
        depth = np.clip((rr[m] - y_lo) / max(Z_UP * 120, 1.0), 0, 1)  # darker towards the underside
        img[m] = EARTH * shade * (0.8 + 0.2 * depth)[:, None]
    del rr, side

    # The picture: the terrain, then the buildings and trees, far to near.
    t0 = time.time()
    dpi = 300
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    extent = (x0, x0 + width * s, y_top - height * s, y_top)
    ax.imshow(img, extent=extent, interpolation="none", zorder=0)
    ax.add_collection(
        PolyCollection(shadows, facecolors=(0, 0, 0, 0.24), edgecolors="none", zorder=4)
    )
    front, zlabels = [], []
    if zones and city is not None and city.ops:
        from sketch_ops import FIELD, STRIP, zone_volumes

        page = lambda n, e, z: (SX * (e - n), SY * (n + e) + Z_UP * z)  # noqa: E731
        for kind, polys, col in (
            (FIELD, [f["polygon"] for f in city.ops["landable"]], 0.3),
            (STRIP, [city.ops["airstrip"]["field"]] if city.ops["airstrip"] else [], 0.25),
        ):
            on_ground = []
            for p in polys:
                q = np.asarray(p.exterior.coords)
                x, y = page(q[:, 0], q[:, 1], ground(q[:, 0], q[:, 1]))
                on_ground.append(np.column_stack([x, y]))
            rgb = tuple(int(kind[i : i + 2], 16) / 255 for i in (1, 3, 5))
            ax.add_collection(
                PolyCollection(
                    on_ground,
                    facecolors=(*rgb, col),
                    edgecolors=(*rgb, 0.95),
                    linewidths=0.6,
                    zorder=4.2,
                )
            )
        behind, front, zlabels = zone_volumes(city.ops, VIEW_ISO, page, ground)
        ax.add_collection(
            PolyCollection(
                [q for q, _, _ in behind],
                facecolors=[f for _, f, _ in behind],
                edgecolors=[e for _, _, e in behind],
                linewidths=0.25,
                zorder=4.5,
            )
        )
    order = sorted((i for i in range(len(items)) if not hidden[i]), key=lambda i: -items[i][0])
    polys, cols = [], []
    for i in order:
        polys += items[i][1]
        cols += items[i][2]
    edge = {c: _darker(c) for c in set(cols)}
    ax.add_collection(
        PolyCollection(
            polys, facecolors=cols, edgecolors=[edge[c] for c in cols], linewidths=0.08, zorder=5
        )
    )
    if front:
        ax.add_collection(
            PolyCollection(
                [q for q, _, _ in front],
                facecolors=[f for _, f, _ in front],
                edgecolors=[e for _, _, e in front],
                linewidths=0.25,
                zorder=6,
            )
        )
        for x, y, text, col in zlabels:
            ax.annotate(
                text,
                xy=(x, y),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                fontsize=5,
                color="#4a1a10",
                zorder=10,
                bbox={"fc": "white", "ec": col, "alpha": 0.85, "pad": 0.8, "lw": 0.4},
            )
        for f in city.ops["landable"]:
            c = f["polygon"].representative_point()
            x, y = SX * (c.y - c.x), SY * (c.x + c.y) + Z_UP * float(ground(c.x, c.y)[0])
            ax.text(
                x,
                y,
                f["id"],
                ha="center",
                va="center",
                fontsize=4.5,
                fontweight="bold",
                color="#0d4d24",
                zorder=10,
                bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 0.5},
            )
        from matplotlib.patches import Patch

        from sketch_ops import CLOSED, FENCE, FIELD, LIMITED, STRIP

        ax.legend(
            handles=[
                Patch(fc=CLOSED, alpha=0.4, ec=CLOSED, label="no-fly zone, at all heights"),
                Patch(fc=LIMITED, alpha=0.4, ec=LIMITED, label="no-fly zone, up to its height"),
                Patch(
                    fc=FIELD,
                    alpha=0.4,
                    ec=FIELD,
                    label=f"landing field ({len(city.ops['landable'])})",
                ),
                Patch(fc=STRIP, alpha=0.4, ec=STRIP, label="airstrip"),
                Patch(
                    fc=FENCE,
                    alpha=0.15,
                    ec=FENCE,
                    label=f"geofence and ceiling, {city.ops['ceiling_m']:.0f} m "
                    + (
                        "altitude (flat)"
                        if city.ops.get("ceiling_ref") == "datum"
                        else "above the ground"
                    ),
                ),
            ],
            loc="lower left",
            fontsize=6,
            framealpha=0.9,
            bbox_to_anchor=(0.01, 0.02),
        )
    if city is not None:
        for b in city.labels:
            c = b.centre
            z0 = float(ground(c[0], c[1])[0])
            x, y = SX * (c[1] - c[0]), SY * (c[0] + c[1]) + Z_UP * (z0 + b.height)
            ax.annotate(
                f"{b.name} {b.height:.0f} m",
                xy=(x, y),
                xytext=LABEL_OFFSET.get(b.kind, (0, 10)),
                textcoords="offset points",
                ha="center",
                fontsize=5,
                color="#1d2a44",
                zorder=9,
                bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 0.8},
                arrowprops={"arrowstyle": "-", "color": "#1d2a44", "lw": 0.3},
            )
        for name, c in city.settlements:
            z0 = float(ground(c[0], c[1])[0])
            x, y = SX * (c[1] - c[0]), SY * (c[0] + c[1]) + Z_UP * (z0 + 40)
            ax.annotate(
                name,
                xy=(x, y),
                xytext=(0, 40),
                textcoords="offset points",
                ha="center",
                fontsize=6,
                fontweight="bold",
                color="#3a2a1a",
                zorder=9,
                bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 1},
            )
    ax.text(
        0.012,
        0.985,
        f"{sk['name'].replace(' (sketch)', '')}: {(e1 - e0) / 1000:g} km square, isometric from "
        f"the south-west, true heights; {len(city.buildings) if city else 0} buildings, "
        f"{len(city.trees) + len(city.forest) if city else 0} trees, "
        f"{len(city.turbines) if city else 0} wind turbines, "
        f"{len(city.pylons) if city else 0} pylons",
        transform=ax.transAxes,
        fontsize=6,
        va="top",
        color="#33373d",
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    fig.savefig(out.with_suffix(".jpg"), dpi=dpi, pil_kwargs={"quality": 92})
    plt.close(fig)
    took = time.time() - t0
    print(f"buildings and trees drawn ({len(order)} items) in {took:.0f} s; {s:.2f} m a pixel")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="A high-resolution isometric picture of a map sketch.")
    ap.add_argument("name")
    ap.add_argument("--width", type=int, default=7680)
    ap.add_argument("--height", type=int, default=4320)
    ap.add_argument(
        "--zones", action="store_true", help="draw the no-fly zones, landing fields and airstrip"
    )
    a = ap.parse_args()
    out = Path("runs/maps") / f"{a.name}_8k{'_nofly' if a.zones else ''}.png"
    print(render(prepare(a.name), out, a.width, a.height, zones=a.zones))


if __name__ == "__main__":
    main()
