# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportOperatorIssue=false, reportReturnType=false, reportOptionalMemberAccess=false
# pyright: reportGeneralTypeIssues=false
# (Drawing tools over shapely, whose set operations are typed as returning a BaseGeometry; the
# code relies on them being polygons or lines at run time, which the checker cannot narrow.)
"""Isometric views of a map sketch: the whole map with the sketch draped over its terrain and
the city standing up, and the city itself drawn crisply (every building and tree), whole and
round its centre. Quick and abstract (matplotlib).

Usage: ``uv run python tools/sketch_iso.py testmap``
→ runs/maps/<name>_iso.png, <name>_iso_city.png, <name>_iso_centre.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LightSource
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sketch_city import City, building_faces, draw_city, draw_ground, shade_overlay
from sketch_infra import infra_items
from sketch_land import shadow_of, tree_shapes, turbine_faces, turbine_shadow
from sketch_map import CELL, TINT, band, prepare
from sketch_power import power_items, tower_faces

# Isometric drawing: the plan turned 45° (north-east up the page) with its depth halved;
# heights rise straight up the page. The viewer is to the south-west, about 33° up.
Z_UP = 0.82  # page units per metre of height
VIEW_ISO = np.array([-1.16, -1.16, 1.0]) / np.linalg.norm([-1.16, -1.16, 1.0])
ELEV, AZIM = 32.0, -125.0  # the whole-map view (matplotlib 3D)
VIEW_3D = np.array(
    [
        np.sin(np.radians(AZIM)) * np.cos(np.radians(ELEV)),
        np.cos(np.radians(AZIM)) * np.cos(np.radians(ELEV)),
        np.sin(np.radians(ELEV)),
    ]
)


# Labels that would sit on top of each other (the station's) fan out.
LABEL_OFFSET = {"cooling": (-70, 34), "reactor": (60, 30)}


def _screen(n, e, z=0.0):
    n, e = np.asarray(n, float), np.asarray(e, float)
    return 0.7071 * (e - n), 0.3536 * (n + e) + np.asarray(z) * Z_UP


def _relief(h: np.ndarray) -> np.ndarray:
    ls = LightSource(azdeg=315, altdeg=40)
    return ls.shade(
        h,
        cmap=TINT,
        vert_exag=1.5,
        blend_mode="soft",
        dx=CELL,
        dy=CELL,
        vmin=float(h.min()),
        vmax=float(h.min()) + max(float(np.ptp(h)), 20.0),
    )


def texture(ns, es, h, channels, city, window, px: int) -> np.ndarray:
    """The sketch, drawn flat over ``window`` (n0, n1, e0, e1) into an RGB image."""
    n0, n1, e0, e1 = window
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.imshow(
        _relief(h), origin="lower", extent=(es[0], es[-1], ns[0], ns[-1]), interpolation="bilinear"
    )
    for c in channels:
        o = band(c.line, c.width)
        ax.fill(o[:, 1], o[:, 0], color="#3f86cf", lw=0, zorder=3)
    if city is not None:
        draw_city(ax, city, detail=False, labels=False, buildings=False)
        ax.imshow(
            shade_overlay(h, CELL),
            origin="lower",
            extent=(es[0], es[-1], ns[0], ns[-1]),
            zorder=3.25,
        )
    ax.set_xlim(e0, e1)
    ax.set_ylim(n0, n1)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(float) / 255.0
    plt.close(fig)
    return img[::-1]  # rows from south to north


def iso(prep, window, cell: float, exag: float, out: Path, title: str) -> Path:
    """The whole map in 3D, heights exaggerated, the city's buildings standing on it."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    _sk, ns, es, h, channels, _anchors, city = prep
    n0, n1, e0, e1 = window
    gn = np.arange(n0, n1 + cell, cell)
    ge = np.arange(e0, e1 + cell, cell)
    gnn, gee = np.meshgrid(gn, ge, indexing="ij")
    z = map_coordinates(h, [(gnn - ns[0]) / CELL, (gee - es[0]) / CELL], order=1, mode="nearest")
    # Each face takes the mean colour of the sketch under it, so a river narrower than a face
    # still shows as a continuous line (sampling one point per face breaks it into dots).
    faces, k = len(gn) - 1, 4
    img = texture(ns, es, h, channels, city, window, faces * k)[: faces * k, : faces * k]
    mean = img.reshape(faces, k, faces, k, 3).mean(axis=(1, 3))
    colours = np.pad(mean, ((0, 1), (0, 1), (0, 0)), mode="edge")
    fig = plt.figure(figsize=(16, 11.5), dpi=200)
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.plot_surface(
        gee,
        gnn,
        z * exag,
        facecolors=colours,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    if city is not None:
        polys, cols = [], []
        for b in city.buildings:
            c = b.centre
            z0 = float(
                map_coordinates(h, [[(c[0] - ns[0]) / CELL], [(c[1] - es[0]) / CELL]], order=1)[0]
            )
            for pts, col in building_faces(b, VIEW_3D):
                polys.append(np.stack([pts[:, 1], pts[:, 0], (z0 + pts[:, 2]) * exag], 1))
                cols.append(col)
        for t in city.cooling_towers:
            z0 = float(
                map_coordinates(h, [[(t["n"] - ns[0]) / CELL], [(t["e"] - es[0]) / CELL]], order=1)[
                    0
                ]
            )
            for pts, col in tower_faces(t, VIEW_3D):
                polys.append(np.stack([pts[:, 1], pts[:, 0], (z0 + pts[:, 2]) * exag], 1))
                cols.append(col)
        for t in city.turbines:
            z0 = float(
                map_coordinates(h, [[(t["n"] - ns[0]) / CELL], [(t["e"] - es[0]) / CELL]], order=1)[
                    0
                ]
            )
            for pts, col in turbine_faces(t, VIEW_3D):
                polys.append(np.stack([pts[:, 1], pts[:, 0], (z0 + pts[:, 2]) * exag], 1))
                cols.append(col)
        ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors="none"))
    ax.view_init(elev=ELEV, azim=AZIM)  # from the south-west, looking north-east over the valley
    ax.set_box_aspect((e1 - e0, n1 - n0, max(float(np.ptp(z)) * exag, 1.0)), zoom=1.45)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    # A 3D axes keeps a wide empty margin; crop to the drawing and title it afterwards.
    ink = np.argwhere((rgb < 245).any(axis=2))
    (r0, c0), (r1, c1) = ink.min(0), ink.max(0)
    crop = rgb[max(r0 - 10, 0) : r1 + 10, max(c0 - 10, 0) : c1 + 10]
    fig = plt.figure(figsize=(crop.shape[1] / 100, crop.shape[0] / 100 + 0.4), dpi=100)
    ax = fig.add_axes((0, 0, 1, crop.shape[0] / (crop.shape[0] + 40)))
    ax.imshow(crop)
    ax.set_axis_off()
    fig.suptitle(title, fontsize=12, y=0.995)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def _items(city: City, view=None) -> tuple[list, list]:
    """Every building's visible faces, every turbine and every tree (in the woods and out),
    as (depth, polygons, colours, where), furthest first; and their shadows on the ground
    (page polygons). All on flat ground (this drawing has no relief)."""
    items, shadows = [], []
    for b in city.buildings:
        polys, cols = [], []
        for pts, col in building_faces(b, VIEW_ISO):
            x, y = _screen(pts[:, 0], pts[:, 1], pts[:, 2])
            polys.append(np.column_stack([x, y]))
            cols.append(col)
        c = b.centre
        items.append((c[0] + c[1], polys, cols, c))
        shadows += shadow_of(
            [(p.footprint, p.z0, p.z1 + p.rise) for p in b.parts], 0.7071, 0.3536, Z_UP, 0.0
        )
    for t in city.turbines:
        polys, cols = [], []
        for pts, col in turbine_faces(t, VIEW_ISO):
            x, y = _screen(pts[:, 0], pts[:, 1], pts[:, 2])
            polys.append(np.column_stack([x, y]))
            cols.append(col)
        items.append((t["n"] + t["e"], polys, cols, np.array([t["n"], t["e"]])))
        shadows += turbine_shadow(t, 0.7071, 0.3536, Z_UP, 0.0)
    for fn in (power_items, infra_items):
        more, more_shadows = fn(city, VIEW_ISO, lambda n, e, z: _screen(n, e, z), lambda n, e: 0.0)
        items += [(d, ps, cs, c) for d, ps, cs, c, _, _ in more]
        shadows += more_shadows
    trees = [(n, e, r, zc + 0.9 * r, 0, k % 5) for k, (n, e, r, zc) in enumerate(city.trees)]
    trees += [tuple(row) for row in city.forest]
    for n, e, r, hgt, kind, shade in trees:
        sh, shapes = tree_shapes(n, e, r, hgt, int(kind), shade, 0.0, 0.7071, 0.3536, Z_UP)
        items.append((n + e, [p for p, _ in shapes], [c for _, c in shapes], np.array([n, e])))
        shadows.append(sh)
    items.sort(key=lambda it: -it[0])
    return items, shadows


def _panel(ax, prep, items, window, title: str, crop: float = 0.18) -> None:
    """One isometric drawing of the plan square ``window`` (n0, n1, e0, e1)."""
    from matplotlib.transforms import Affine2D

    _sk, ns, es, h, channels, _anchors, city = prep
    n0, n1, e0, e1 = window
    # Image axes are (east, north); the page is (x, y) = (0.7071 (e − n), 0.3536 (n + e)).
    m = Affine2D.from_values(0.7071, 0.3536, -0.7071, 0.3536, 0.0, 0.0)
    ax.imshow(
        _relief(h),
        origin="lower",
        extent=(es[0], es[-1], ns[0], ns[-1]),
        interpolation="bilinear",
        transform=m + ax.transData,
        zorder=0,
    )
    for c in channels:
        o = band(c.line, c.width)
        ax.fill(*_screen(o[:, 0], o[:, 1]), color="#3f86cf", lw=0, zorder=3)
    draw_ground(ax, city, _screen, detail=True, squash=0.5, crowns=False)
    ax.imshow(
        shade_overlay(h, CELL),
        origin="lower",
        extent=(es[0], es[-1], ns[0], ns[-1]),
        interpolation="bilinear",
        transform=m + ax.transData,
        zorder=3.25,
    )
    xs, ys = _screen(np.array([n0, n0, n1, n1]), np.array([e0, e1, e0, e1]))
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min() + crop * np.ptp(ys), ys.max() - 0.05 * np.ptp(ys) + 60
    items, shadows = items
    near = [
        s for s in shadows if x0 - 300 <= s[0, 0] <= x1 + 300 and y0 - 300 <= s[0, 1] <= y1 + 300
    ]
    ax.add_collection(PolyCollection(near, facecolors=(0, 0, 0, 0.22), edgecolors="none", zorder=7))
    polys, cols = [], []
    for _, ps, cs, c in items:  # everything on the page, not only inside the plan's square
        x, y = _screen(c[0], c[1])
        if x0 - 60 <= x <= x1 + 60 and y0 - 250 <= y <= y1 + 60:
            polys += ps
            cols += cs
    ax.add_collection(
        PolyCollection(
            polys, facecolors=cols, edgecolors=[_darker(c) for c in cols], linewidths=0.18, zorder=8
        )
    )
    for b in city.labels:
        c = b.centre
        if not (n0 <= c[0] <= n1 and e0 <= c[1] <= e1):
            continue
        x, y = _screen(c[0], c[1], b.height)
        ax.annotate(
            f"{b.name} {b.height:.0f} m",
            xy=(float(x), float(y)),
            xytext=LABEL_OFFSET.get(b.kind, (0, 26)),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            zorder=9,
            color="#1d2a44",
            bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 1},
            arrowprops={"arrowstyle": "-", "color": "#1d2a44", "lw": 0.6},
        )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=13)


def iso_city(prep, windows, items=None) -> list[Path]:
    """The city as crisp isometric drawings (true heights), one per (window, file, title,
    size): the ground tilted, every building's visible faces and every tree, far to near."""
    items = items or _items(prep[-1])
    outs = []
    for window, out, title, size in windows:
        fig, ax = plt.subplots(figsize=size)
        _panel(ax, prep, items, window, title)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        outs.append(out)
    return outs


def _extent(city: City, c: np.ndarray) -> float:
    """How far a village reaches from its centre (most of its buildings)."""
    d = np.array([np.hypot(*(b.centre - c)) for b in city.buildings])
    return float(np.percentile(d[d < 600], 92)) if (d < 600).any() else 200.0


def iso_villages(prep, out: Path, items=None) -> Path | None:
    """Each village close up, side by side."""
    city = prep[-1]
    if not city.settlements:
        return None
    items = items or _items(city)
    k = len(city.settlements)
    fig, axes = plt.subplots(1, k, figsize=(8 * k, 5.6))
    for ax, (name, c) in zip(np.atleast_1d(axes), city.settlements, strict=True):
        half = min(max(_extent(city, c) + 50, 200.0), 420.0)
        count = sum(1 for b in city.buildings if np.hypot(*(b.centre - c)) < half)
        _panel(
            ax,
            prep,
            items,
            (c[0] - half, c[0] + half, c[1] - half, c[1] + half),
            f"{name} ({count} buildings)",
            crop=0.0,
        )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _darker(c: str) -> str:
    if len(c) == 9:  # see-through (steam): no outline
        return "none"
    rgb = [int(c[i : i + 2], 16) * 0.62 for i in (1, 3, 5)]
    return "#" + "".join(f"{int(x):02x}" for x in rgb)


def main() -> None:
    ap = argparse.ArgumentParser(description="Isometric views of a map sketch.")
    ap.add_argument("name")
    a = ap.parse_args()
    base = Path("runs/maps")
    prep = prepare(a.name)
    print(
        iso(
            prep,
            (-2500, 2500, -2500, 2500),
            12.5,
            1.5,
            base / f"{a.name}_iso.png",
            "Test map, isometric (heights × 1.5)",
        )
    )
    city = prep[-1]
    if city is None:
        return
    core = np.asarray(prep[0]["city"]["founded_ne_m"], float)
    items_all = _items(city)
    print(iso_villages(prep, base / f"{a.name}_iso_villages.png", items_all))
    for name, c in city.settlements:  # the bigger villages get a view of their own
        half = min(max(_extent(city, c) + 50, 200.0), 420.0)
        if half > 300:
            out = base / f"{a.name}_iso_{name.replace(' ', '_')}.png"
            fig, ax = plt.subplots(figsize=(18, 10))
            n_b = sum(1 for b in city.buildings if np.hypot(*(b.centre - c)) < half)
            _panel(
                ax,
                prep,
                items_all,
                (c[0] - half, c[0] + half, c[1] - half, c[1] + half),
                f"{name} ({n_b} buildings)",
                crop=0.02,
            )
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(out)
    for p in iso_city(
        items=items_all,
        prep=prep,
        windows=[
            (
                (-1250.0, 1250.0, -1250.0, 1250.0),
                base / f"{a.name}_iso_city.png",
                f"{city.name}, isometric (true heights; seen from the south-west)",
                (22, 13),
            ),
            (
                (core[0] - 420, core[0] + 420, core[1] - 420, core[1] + 420),
                base / f"{a.name}_iso_centre.png",
                f"{city.name}: the centre",
                (20, 12),
            ),
        ],
    ):
        print(p)


if __name__ == "__main__":
    main()
