# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportOperatorIssue=false, reportReturnType=false, reportOptionalMemberAccess=false
# pyright: reportGeneralTypeIssues=false
# (Drawing tools over shapely, whose set operations are typed as returning a BaseGeometry; the
# code relies on them being polygons or lines at run time, which the checker cannot narrow.)
"""Draw a map sketch (configs/sketches/<name>.yaml) as an abstract map sheet: shaded relief,
contours and the features drawn so far, on a grid labelled every 500 m from the centre.

A sketch is the paper stage of a world (docs/07): quick to edit and redraw, with nothing
generated or simulated. Usage: ``uv run python tools/sketch_map.py testmap [--out FILE]``
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LightSource, LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
from scipy.spatial import KDTree

sys_path = Path(__file__).resolve().parent
import sys  # noqa: E402

sys.path.insert(0, str(sys_path))
from sketch_city import City, build_city, draw_city, load_city, save_city, shade_overlay  # noqa: E402
from sketch_country import add_countryside, load_countryside, save_countryside  # noqa: E402
from sketch_infra import apply_infra, build_infra, load_infra, save_infra  # noqa: E402
from sketch_land import add_land, load_land, save_land  # noqa: E402
from sketch_ops import apply_ops, build_ops, draw_ops, load_ops, save_ops  # noqa: E402
from sketch_power import apply_power, build_power, load_power, save_power  # noqa: E402

SKETCHES = Path(__file__).resolve().parents[1] / "configs" / "sketches"
TINT = LinearSegmentedColormap.from_list(
    "tint", ["#9fc77a", "#bccd86", "#d3c58d", "#c4a27a", "#a98a6e", "#efe9e2"]
)
CELL = 10.0
LOCK = False  # --lock: save the main river's course to its course_file


@dataclass
class Channel:
    name: str
    line: np.ndarray  # (k, 2) north, east
    width: np.ndarray  # (k,) m


class Curvature:
    """Random curvature that drifts at two scales (Ornstein-Uhlenbeck processes): long
    loops and small wiggles, never repeating. Scales follow the channel's width."""

    def __init__(self, rng, irr: float) -> None:
        self.rng, self.irr = rng, irr
        self.k_long = self.k_short = 0.0

    def step(self, ds: float, w: float, calm: float = 1.0) -> float:
        r_long, r_short = 9.0 * w + 120.0, 2.5 * w + 25.0  # typical bend radii
        t_long, t_short = 1.2 * r_long, 1.5 * r_short  # how long a bend keeps turning
        for name, r, tau, amp in (
            ("k_long", r_long, t_long, 1.0),
            ("k_short", r_short, t_short, 0.35),
        ):
            k = getattr(self, name)
            sigma = amp * self.irr / r * math.sqrt(2.0 / tau)
            k += -k * ds / tau + sigma * math.sqrt(ds) * self.rng.standard_normal()
            setattr(self, name, k)
        k = calm * (self.k_long + self.k_short)
        k_max = 1.0 / (2.5 * w)  # no river bends tighter than a few channel widths
        return max(-k_max, min(k_max, k))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def course(
    start,
    target,
    width_at,
    rng,
    irr: float,
    *,
    slope=None,
    keep=None,
    calm=None,
    arrive: float = 0.0,
    stop=None,
    axis=None,
    reach_m: float = 350.0,
    ds: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """A channel from ``start`` towards ``target`` (or along ``axis``, a bearing): random
    curvature on a leash, steered back towards the wanted heading over ``reach_m``, downhill
    on slopes (``slope(p)`` → gradient), extra steering (``keep(p)`` → radians), and a stop
    condition. Returns (line, widths)."""
    pos = np.asarray(start, float).copy()
    target = np.asarray(target, float)
    to = target - pos
    heading = math.atan2(to[1], to[0])
    curv = Curvature(rng, irr)
    pts, widths = [pos.copy()], [width_at(pos)]
    total = float(np.linalg.norm(to))
    for _ in range(int(6 * total / ds)):
        to = target - pos
        dist = float(np.linalg.norm(to))
        if dist < arrive or (stop is not None and stop(pos)):
            break
        want = math.atan2(to[1], to[0]) if axis is None else axis
        steep = 0.0
        if slope is not None:
            g = slope(pos)
            steep = float(np.hypot(*g))
            down = math.atan2(-g[1], -g[0])
            # Water runs downhill where the ground slopes, on its way to the river; a hollow
            # fills and spills over towards it.
            if steep > 0.02 and abs(_wrap(down - want)) < 0.9:
                want = want + min(0.75, steep / 0.1) * _wrap(down - want)
        if keep is not None:
            want += keep(pos)
        w = width_at(pos)
        reach = max(min(reach_m, 0.45 * dist), 40.0)  # the leash shortens near the target
        # On slopes a stream runs down its gully, hardly meandering.
        still = (calm(pos) if calm is not None else 1.0) / (1.0 + 15.0 * steep)
        turn = curv.step(ds, w, still)
        heading += turn * ds + _wrap(want - heading) * ds / reach
        off = _wrap(heading - want)
        if abs(off) > 1.4:  # bends may swing far, but never double back
            heading = want + math.copysign(1.4, off)
        pos = pos + ds * np.array([math.cos(heading), math.sin(heading)])
        pts.append(pos.copy())
        widths.append(w)
    line = np.asarray(pts)
    line = np.stack([gaussian_filter1d(line[:, i], 3, mode="nearest") for i in (0, 1)], 1)
    return line, np.asarray(widths)


def valley_floor(hills: np.ndarray, bounds, p0: np.ndarray, p1: np.ndarray):
    """The flat floor between the ranges, across the valley's axis: (fraction along the
    axis → offset of the floor's middle to the left of the axis, its half-width)."""
    (n0, e0), (n1, e1) = bounds
    length = float(np.linalg.norm(p1 - p0))
    u = (p1 - p0) / length
    left = np.array([u[1], -u[0]])
    fr = np.linspace(0.0, 1.0, 241)
    lat = np.arange(-2000.0, 2001.0, 20.0)
    pts = p0 + fr[:, None, None] * (p1 - p0) + lat[None, :, None] * left
    z = map_coordinates(
        hills, [(pts[..., 0] - n0) / CELL, (pts[..., 1] - e0) / CELL], order=1, mode="nearest"
    )
    # The floor: hills under 4 m, on the map (beyond its edge nothing is known).
    on_map = (pts[..., 0] > n0) & (pts[..., 0] < n1) & (pts[..., 1] > e0) & (pts[..., 1] < e1)
    floor = (z < 4.0) & on_map
    floor[:, len(lat) // 2] = True  # the axis itself, where the river enters and leaves
    mid, half = np.zeros_like(fr), np.zeros_like(fr)
    for i, row in enumerate(floor):
        runs, j = [], 0
        while j < len(row):
            if row[j]:
                k = j
                while k + 1 < len(row) and row[k + 1]:
                    k += 1
                runs.append((lat[j], lat[k]))
                j = k + 1
            else:
                j += 1
        a, b = min(runs, key=lambda r: 0.0 if r[0] <= 0 <= r[1] else min(abs(r[0]), abs(r[1])))
        mid[i], half[i] = 0.5 * (a + b), 0.5 * (b - a)
    mid, half = gaussian_filter1d(mid, 6), gaussian_filter1d(half, 6)
    return (lambda f: float(np.interp(f, fr, mid))), (lambda f: float(np.interp(f, fr, half)))


def rivers(sk: dict, z_hills: np.ndarray, z_all: np.ndarray) -> tuple[list[Channel], dict]:
    """The main river (with its island) and its tributaries. Tributaries are routed first to
    where they join, so the main river widens there. Returns (channels, label anchors)."""
    r = sk["river"]
    bounds = sk["bounds_ne_m"]
    (n0, e0), (n1, e1) = bounds
    rng = np.random.default_rng(r.get("seed", 0))
    irr = r.get("irregularity", 0.7)
    p0, p1 = np.asarray(r["from_ne_m"], float), np.asarray(r["to_ne_m"], float)
    length = float(np.linalg.norm(p1 - p0))
    u = (p1 - p0) / length
    left = np.array([u[1], -u[0]])
    frac = lambda p: float((np.asarray(p) - p0) @ u / length)  # noqa: E731
    mid, half = valley_floor(z_hills, bounds, p0, p1)
    gn, ge = np.gradient(z_all, CELL)

    def slope(p: np.ndarray) -> np.ndarray:
        c = [[(p[0] - n0) / CELL], [(p[1] - e0) / CELL]]
        return np.array(
            [
                map_coordinates(gn, c, order=1, mode="nearest")[0],
                map_coordinates(ge, c, order=1, mode="nearest")[0],
            ]
        )

    isl = r.get("island")
    w0, w1 = r["width_m"]
    tribs = sorted(r.get("tributaries", []), key=lambda t: t["joins_at"])

    def keep(p: np.ndarray) -> float:  # the main river keeps to the valley floor
        f = frac(p)
        ends = min(1.0, max(f, 0.0) / 0.15) * min(1.0, max(1.0 - f, 0.0) / 0.15)
        room = max(min(half(f) - 200.0, 400.0 + 1500.0 * ends), 150.0)
        lateral = float((p - p0) @ left)
        return math.tanh((lateral - ends * mid(f)) / room)  # right of the flow is + bearing

    def main_width(p: np.ndarray) -> float:  # grows a little, and steps up at each tributary
        f = min(max(frac(p), 0.0), 1.0)
        w = w0 * (1.0 + 0.6 * f)
        for t in tribs:
            w += 0.9 * t["width_m"][1] / (1.0 + math.exp(-(f - t["joins_at"]) / 0.01))
        return w

    def calm(p: np.ndarray) -> float:  # straight in from the corner, and past the island
        f = frac(p)
        lead = min(1.0, max(f, 0.0) / 0.08)
        if not isl or not (isl["from"] - 0.03 <= f <= isl["to"] + 0.03):
            return lead
        return 0.15

    locked = SKETCHES / r["course_file"] if r.get("course_file") else None
    if locked is not None and locked.exists():
        # The owner's chosen course, kept exactly (``--lock`` saved it).
        main = np.asarray(json.loads(locked.read_text())["centre_ne_m"], float)
        mw = np.array([main_width(p) for p in main])
    else:
        main, mw = course(
            p0,
            p1,
            main_width,
            rng,
            irr,
            keep=keep,
            calm=calm,
            axis=math.atan2(u[1], u[0]),
            reach_m=r.get("leash_m", 450.0),
            stop=lambda p: frac(p) > 1.0 and not (n0 < p[0] < n1 and e0 < p[1] < e1),
        )
        if locked is not None and LOCK:
            locked.write_text(json.dumps({"centre_ne_m": np.round(main, 2).tolist()}) + "\n")
    rng = np.random.default_rng(r.get("seed", 0) + 1)  # widths and tributaries draw from here on
    channels = [Channel(r.get("name", "river"), main, mw)]
    anchors: dict = {}
    main_tree = KDTree(main)
    flow = np.gradient(main, axis=0)
    flow /= np.linalg.norm(flow, axis=1)[:, None]
    s_main = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(main, axis=0), axis=1))]
    joined: list[tuple[float, float]] = []  # (fraction along the main river, added width)

    for n_t, t in enumerate(tribs):
        trng = np.random.default_rng(r.get("seed", 0) * 100 + n_t)  # each draws on its own
        j = int(np.argmin([abs(frac(p) - t["joins_at"]) for p in main]))
        if t.get("from") == "peak":
            east = z_hills.copy()
            nn_, ee_ = np.meshgrid(
                np.arange(n0, n1 + CELL, CELL), np.arange(e0, e1 + CELL, CELL), indexing="ij"
            )
            east[(nn_ + ee_ < 0) | (ee_ < 1500.0)] = -1.0  # the eastern side of the range
            i_pk = np.unravel_index(np.argmax(east), east.shape)
            summit = np.array([n0 + i_pk[0] * CELL, e0 + i_pk[1] * CELL])
            # Its spring: just below the summit, on the flank facing the river.
            start = summit + 120.0 * (main[j] - summit) / np.linalg.norm(main[j] - summit)
        else:
            start = np.asarray(t["from_ne_m"], float)
        tw0, tw1 = t["width_m"]
        # It makes for a point a little upstream of the junction, on its own bank …
        left_j = np.array([flow[j, 1], -flow[j, 0]])
        side = 1.0 if (start - main[j]) @ left_j > 0 else -1.0
        aim = main[j] + side * left_j * (0.5 * mw[j] + 60.0)
        span = float(np.linalg.norm(aim - start))

        def width_at(p, a=start, b=span, x=tw0, y=tw1):
            return x + (y - x) * min(1.0, float(np.linalg.norm(p - a)) / b)

        f_join = t["joins_at"]

        def meets(
            p: np.ndarray, f_join=f_join
        ) -> bool:  # reaches the river's bank near its junction
            d, i = main_tree.query(p)
            return frac(main[i]) > f_join - 0.03 and d < 0.5 * mw[i] + 25.0

        def apart(
            p: np.ndarray, f_join=f_join, aim=aim
        ) -> float:  # keeps off the river, on its bank
            d, i = main_tree.query(p)
            margin = 0.5 * mw[i] + 110.0
            if frac(main[i]) > f_join - 0.03:
                # Near its junction it bends downstream gradually, to meet the river at about
                # 50° to the flow.
                if d > 300.0:
                    return 0.0
                lft = np.array([flow[i, 1], -flow[i, 0]])
                bank = 1.0 if (p - main[i]) @ lft > 0 else -1.0
                merge = math.atan2(flow[i, 1], flow[i, 0]) + bank * math.radians(50.0)
                want = math.atan2(aim[1] - p[1], aim[0] - p[0])
                return (1.0 - d / 300.0) * _wrap(merge - want)
            if d > 2.0 * margin:
                return 0.0
            to_p = p - main[i]
            away = math.atan2(to_p[1], to_p[0])
            ahead = math.atan2(flow[i, 1], flow[i, 0])  # … and is carried downstream
            want = math.atan2(aim[1] - p[1], aim[0] - p[0])
            push = min(1.0, (2.0 * margin - d) / margin)
            return push * (0.6 * _wrap(away - want) + 0.4 * _wrap(ahead - want))

        line, tw = course(
            start,
            aim,
            width_at,
            trng,
            t.get("irregularity", irr),
            slope=slope,
            arrive=40.0,
            reach_m=180.0,
            stop=meets,
            keep=apart,
        )
        # … then joins it pointing downstream, at about 40° to the flow, as confluences do.
        i = int(main_tree.query(line[-1])[1])
        k = int(np.searchsorted(s_main, s_main[i] + 70.0))
        k = min(k, len(main) - 1)
        left_k = np.array([flow[k, 1], -flow[k, 0]])
        bank = 1.0 if (line[-1] - main[k]) @ left_k > 0 else -1.0
        b_in = math.atan2(flow[k, 1], flow[k, 0]) + bank * math.radians(50.0)
        end = main[k]
        t_out = line[-1] - line[-2]
        t_out /= np.linalg.norm(t_out)
        t_in = np.array([math.cos(b_in), math.sin(b_in)])
        gap = float(np.linalg.norm(end - line[-1]))
        c1, c2 = line[-1] + t_out * gap / 5.0, end - t_in * gap / 2.5
        q = np.linspace(0.0, 1.0, max(int(gap / 4.0), 4))[1:, None]
        merge = (
            (1 - q) ** 3 * line[-1]
            + 3 * (1 - q) ** 2 * q * c1
            + 3 * (1 - q) * q**2 * c2
            + q**3 * end
        )
        line = np.vstack([line, merge])
        tw = np.r_[tw, np.full(len(merge), tw[-1])]
        channels.append(Channel(t["name"], line, tw))
        anchors[t["name"]] = line[len(line) // 3]
        joined.append((frac(main[k]), tw1))

    # Width: a little growth, a step up where each tributary actually joins, a waver; the
    # widest reach is the given width (the entry stays as given).
    f_main = np.clip(np.array([frac(p) for p in main]), 0.0, 1.0)
    mw = w0 * (1.0 + 0.6 * f_main)
    for f_j, add in joined:
        mw = mw + 0.9 * add / (1.0 + np.exp(-(f_main - f_j) / 0.01))
    waver = np.zeros_like(s_main)
    for lam in rng.uniform(250.0, 1200.0, 4):
        waver += 0.25 * np.sin(2 * math.pi * s_main / lam + rng.uniform(0, 2 * math.pi))
    mw = mw * (1.0 + 0.6 * irr * waver)
    mw = w0 + (mw - w0) * (w1 - w0) / (mw.max() - w0)
    channels[0] = Channel(channels[0].name, main, mw)

    if isl:
        f = np.array([frac(p) for p in main])
        k = (f >= isl["from"]) & (f <= isl["to"])
        bmp = np.zeros_like(f)
        bmp[k] = np.sin(math.pi * (f[k] - isl["from"]) / (isl["to"] - isl["from"]))
        x = np.clip((f - isl["from"]) / (isl["to"] - isl["from"]), 0, 1)
        shape = bmp**0.7 * (1 + 0.25 * (1 - 2 * x))  # fuller at its upstream end
        smooth = np.stack([gaussian_filter1d(main[:, i], 25, mode="nearest") for i in (0, 1)], 1)
        d = np.gradient(smooth, axis=0)
        d /= np.linalg.norm(d, axis=1)[:, None]
        normal = np.stack([-d[:, 1], d[:, 0]], 1)
        side = -1.0 if isl.get("side") == "left" else 1.0
        sep = 0.5 * (isl["width_m"] * shape + mw * bmp)
        centre = np.where(k[:, None], smooth, main)
        channels[0] = Channel(
            channels[0].name,
            centre - (side * sep)[:, None] * normal,
            np.where(k, mw * (1.0 - 0.4 * bmp), mw),
        )
        channels.append(Channel("arm", (centre + (side * sep)[:, None] * normal)[k], 0.6 * mw[k]))
        anchors["island"] = centre[k][len(centre[k]) // 2]
    return channels, anchors


def noise(
    nn: np.ndarray, ee: np.ndarray, rng, wavelength: float, octaves: int, ridged: bool = False
) -> np.ndarray:
    """Fractal noise in about [-1, 1] (ridged: [0, 1], sharp crests): random lattices,
    smoothly interpolated, halving in wavelength and amplitude each octave."""
    out, amp, total = np.zeros_like(nn), 1.0, 0.0
    n0, e0 = nn.min(), ee.min()
    for o in range(octaves):
        cell = wavelength / 2**o
        shape = (int((nn.max() - n0) / cell) + 4, int((ee.max() - e0) / cell) + 4)
        lat = rng.standard_normal(shape) / 1.6
        v = map_coordinates(
            lat, [(nn - n0) / cell + 1, (ee - e0) / cell + 1], order=3, mode="nearest"
        )
        v = np.clip(v, -1.0, 1.0)
        out += amp * ((1.0 - np.abs(v)) ** 2 if ridged else v)
        total += amp
        amp *= 0.5
    return out / total


def hill_range(nn: np.ndarray, ee: np.ndarray, rg: dict) -> np.ndarray:
    """A range of hills along a crest line: a broad swell whose width wanders, carrying
    summits, saddles and spurs; scaled so its highest point is ``height_m``."""
    rng = np.random.default_rng(rg.get("seed", 0))
    w = rg["width_m"]
    # Warp the plane a little, so the range's edge and crest are not smooth curves.
    wn = nn + 0.35 * w * noise(nn, ee, rng, 1600.0, 3)
    we = ee + 0.35 * w * noise(nn, ee, rng, 1600.0, 3)
    crest = np.asarray(rg["crest_ne_m"], float)
    t = np.linspace(0.0, 1.0, 400)
    seg = np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(crest, axis=0), axis=1)])
    line = np.stack([np.interp(t * seg[-1], seg, crest[:, i]) for i in (0, 1)], 1)
    dist, _ = KDTree(line).query(np.stack([wn.ravel(), we.ravel()], 1))
    dist = dist.reshape(nn.shape)
    width = w * (1.0 + 0.3 * noise(nn, ee, rng, 1400.0, 2))
    swell = np.exp(-((dist / width) ** 2))
    r = rg.get("roughness", 0.5)
    # Summits and saddles along the crest; spurs and gullies down the flanks.
    lumps = 0.5 + 0.5 * noise(nn, ee, rng, 900.0, 3)
    ridges = noise(nn, ee, rng, 700.0, 4, ridged=True)
    z = swell**1.3 * ((1.0 - r) * (0.4 + 0.6 * lumps) + r * ridges * (0.3 + 0.7 * lumps))
    return rg["height_m"] * z / z.max()


def flatten_valley(hills: np.ndarray, below_m: float, keep: float) -> np.ndarray:
    """A flat valley floor: the low ground (under ``below_m`` of hill) smoothed over about
    60 m and lowered to ``keep`` of its height, easing into the hills above it unchanged."""
    w = np.clip(1.0 - hills / below_m, 0.0, 1.0) ** 2
    return hills - w * (hills - keep * gaussian_filter(hills, 6.0))


def relief(sk: dict, nn: np.ndarray, ee: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(the plain's tilt, the hills) before the river shapes them."""
    tr = sk["terrain"]
    b = math.radians(tr.get("tilt_towards_deg", 0.0))
    h = tr.get("tilt_m_per_km", 0.0) / 1000.0 * (nn * math.cos(b) + ee * math.sin(b))
    hills = np.zeros_like(nn)
    for hl in tr.get("hills", []):
        d2 = (nn - hl["centre_ne_m"][0]) ** 2 + (ee - hl["centre_ne_m"][1]) ** 2
        hills += hl["height_m"] * np.exp(-d2 / hl["radius_m"] ** 2)
    for rg in tr.get("ranges", []):
        hills = np.maximum(hills, hill_range(nn, ee, rg))
    if tr.get("valley_floor"):
        hills = flatten_valley(hills, tr["valley_floor"]["below_m"], tr["valley_floor"]["keep"])
    return h, hills


def terrain(sk: dict, nn: np.ndarray, ee: np.ndarray, base, channels) -> np.ndarray:
    tr = sk["terrain"]
    h, hills = base[0].copy(), base[1].copy()
    rng = np.random.default_rng(1)
    for _ in range(4):
        lam, th, ph = rng.uniform(400, 1000), rng.uniform(0, 2 * math.pi), rng.uniform(0, 6.3)
        h += (
            0.5
            * tr.get("undulation_m", 0.0)
            * np.sin(2 * math.pi * (nn * math.cos(th) + ee * math.sin(th)) / lam + ph)
        )
    if channels:
        grid = np.stack([nn.ravel(), ee.ravel()], 1)
        main = np.vstack([c.line for c in channels if c.name in (channels[0].name, "arm")])
        d_main = KDTree(main).query(grid)[0].reshape(nn.shape)
        hills *= 1.0 - np.exp(-((d_main / tr.get("valley_m", 250.0)) ** 2))
        others = [c.line for c in channels if c.name not in (channels[0].name, "arm")]
        d_all = d_main
        if others:
            d_trib = KDTree(np.vstack(others)).query(grid)[0].reshape(nn.shape)
            hills *= 1.0 - 0.5 * np.exp(-((d_trib / 90.0) ** 2))  # streams cut small valleys
            d_all = np.minimum(d_main, d_trib)
        h -= sk["river"].get("depth_m", 2.0) * np.exp(-((d_all / 25.0) ** 2))
    return h + hills


def band(line: np.ndarray, width: np.ndarray) -> np.ndarray:
    """Outline of a line drawn with a varying width (for filling)."""
    d = np.gradient(line, axis=0)
    d /= np.linalg.norm(d, axis=1)[:, None]
    nrm = np.stack([-d[:, 1], d[:, 0]], 1) * (0.5 * width)[:, None]
    return np.vstack([line + nrm, (line - nrm)[::-1]])


def prepare(name: str):
    """(sketch, north and east grid axes, ground heights, channels, label anchors, city)."""
    sk = yaml.safe_load((SKETCHES / f"{name}.yaml").read_text())
    (n0, e0), (n1, e1) = sk["bounds_ne_m"]
    ns, es = np.arange(n0, n1 + CELL, CELL), np.arange(e0, e1 + CELL, CELL)
    nn, ee = np.meshgrid(ns, es, indexing="ij")
    locked = sk.get("locked", {})
    t_file = SKETCHES / locked["terrain"] if locked.get("terrain") else None
    r_file = SKETCHES / locked["rivers"] if locked.get("rivers") else None
    if t_file is not None and t_file.exists():
        z = np.load(t_file)
        base = (z["tilt"], z["hills"])
    else:
        base = relief(sk, nn, ee)
    if r_file is not None and r_file.exists():
        saved = json.loads(r_file.read_text())
        channels = [
            Channel(c["name"], np.asarray(c["line"]), np.asarray(c["width"]))
            for c in saved["channels"]
        ]
        anchors = {k: np.asarray(v) for k, v in saved["anchors"].items()}
    else:
        channels, anchors = rivers(sk, base[1], base[0] + base[1]) if sk.get("river") else ([], {})
    h = terrain(sk, nn, ee, base, channels)
    return sk, ns, es, h, channels, anchors, settle(sk, channels, h, ns, es)


def settle(sk: dict, channels, h, ns, es) -> City | None:
    """The city, and the villages and farms round it."""
    if not (sk.get("city") and channels):
        return None
    locked = sk.get("locked", {}).get("city")
    path = SKETCHES / locked if locked else None
    if path is not None and path.exists():
        city = load_city(path)  # the city, locked by the owner: exactly as it was built
    else:
        city = build_city(sk["city"], channels, h, ns, es)
        if path is not None and LOCK:
            save_city(city, path)
    if sk.get("countryside"):
        locked = sk.get("locked", {}).get("countryside")
        path = SKETCHES / locked if locked else None
        # Locked villages, roadside houses and farms are used as they are; any not locked are grown.
        kept = load_countryside(path) if path is not None and path.exists() else {}
        groups = add_countryside(sk["countryside"], city, h, ns, es, kept)
        if path is not None and LOCK:
            save_countryside(groups, path)
        if sk.get("land"):
            locked = sk.get("locked", {}).get("land")
            lpath = SKETCHES / locked if locked else None
            kept_land = load_land(lpath) if lpath is not None and lpath.exists() else None
            state = add_land(sk["land"], sk, city, groups, channels, h, ns, es, kept_land)
            if lpath is not None and LOCK:  # (again, if locked: it takes in turbines added since)
                save_land(city, state, lpath)
        if sk.get("power"):
            locked = sk.get("locked", {}).get("power")
            ppath = SKETCHES / locked if locked else None
            if ppath is not None and ppath.exists():
                state = load_power(ppath)
            else:
                state = build_power(sk["power"], city, groups, channels, h, ns, es)
                if state is not None and ppath is not None and LOCK:
                    save_power(state, ppath)
            if state is not None:
                apply_power(city, state)
        if sk.get("infra"):
            locked = sk.get("locked", {}).get("infra")
            ipath = SKETCHES / locked if locked else None
            if ipath is not None and ipath.exists():
                state = load_infra(ipath)
            else:
                state = build_infra(sk["infra"], sk, city, h, ns, es)
                if ipath is not None and LOCK:
                    save_infra(state, ipath)
            apply_infra(city, state)
        if sk.get("ops"):
            locked = sk.get("locked", {}).get("ops")
            opath = SKETCHES / locked if locked else None
            if opath is not None and opath.exists():
                state = load_ops(opath)
            else:
                state = build_ops(sk["ops"], sk, city, h, ns, es, channels)
                if opath is not None and LOCK:
                    save_ops(state, opath)
            apply_ops(city, state)
    return city


def draw(name: str, out: Path) -> Path:
    sk = yaml.safe_load((SKETCHES / f"{name}.yaml").read_text())
    (n0, e0), (n1, e1) = sk["bounds_ne_m"]
    ns, es = np.arange(n0, n1 + CELL, CELL), np.arange(e0, e1 + CELL, CELL)
    nn, ee = np.meshgrid(ns, es, indexing="ij")
    locked = sk.get("locked", {})
    t_file = SKETCHES / locked["terrain"] if locked.get("terrain") else None
    r_file = SKETCHES / locked["rivers"] if locked.get("rivers") else None
    if t_file is not None and t_file.exists():
        z = np.load(t_file)
        base = (z["tilt"], z["hills"])
    else:
        base = relief(sk, nn, ee)
        if t_file is not None and LOCK:
            np.savez_compressed(
                t_file, tilt=base[0].astype(np.float32), hills=base[1].astype(np.float32)
            )
    if r_file is not None and r_file.exists():
        saved = json.loads(r_file.read_text())
        channels = [
            Channel(c["name"], np.asarray(c["line"]), np.asarray(c["width"]))
            for c in saved["channels"]
        ]
        anchors = {k: np.asarray(v) for k, v in saved["anchors"].items()}
    else:
        channels, anchors = rivers(sk, base[1], base[0] + base[1]) if sk.get("river") else ([], {})
        if r_file is not None and LOCK:
            r_file.write_text(
                json.dumps(
                    {
                        "channels": [
                            {
                                "name": c.name,
                                "line": np.round(c.line, 2).tolist(),
                                "width": np.round(c.width, 2).tolist(),
                            }
                            for c in channels
                        ],
                        "anchors": {k: np.round(v, 2).tolist() for k, v in anchors.items()},
                    }
                )
                + "\n"
            )
    h = terrain(sk, nn, ee, base, channels)

    fig, ax = plt.subplots(figsize=(11, 11))
    ls = LightSource(azdeg=315, altdeg=40)
    rgb = ls.shade(
        h,
        cmap=TINT,
        vert_exag=1.5,
        blend_mode="soft",
        dx=CELL,
        dy=CELL,
        vmin=float(h.min()),
        vmax=float(h.min()) + max(float(np.ptp(h)), 20.0),
    )
    ax.imshow(rgb, origin="lower", extent=(e0, e1, n0, n1), zorder=0)
    span = float(np.ptp(h))
    step = 25 if span > 150 else 10 if span > 40 else 5
    levels = np.arange(np.ceil(h.min() / step) * step, h.max() + step, step)
    levels = levels[levels > np.median(h) + step / 2]  # the hills, not the plain's ripples
    major = levels[np.isclose(levels % (4 * step), 0)]
    ax.contour(es, ns, h, levels=levels, colors="#5a4a32", linewidths=0.4, alpha=0.5, zorder=2)
    cs = ax.contour(es, ns, h, levels=major, colors="#4a3a24", linewidths=0.9, alpha=0.7, zorder=2)
    ax.clabel(cs, fontsize=7, fmt="%d m")
    for c in channels:
        o = band(c.line, c.width)
        ax.fill(o[:, 1], o[:, 0], color="#3f86cf", lw=0, zorder=3)
    city = None
    if sk.get("city") and channels:
        city = settle(sk, channels, h, ns, es)
        assert city is not None
        draw_city(ax, city, detail=False)
        ax.imshow(shade_overlay(h, CELL), origin="lower", extent=(e0, e1, n0, n1), zorder=3.25)
    if channels:
        main = channels[0].line
        if "island" in anchors and not sk.get("city"):
            ax.annotate(
                "island",
                xy=(anchors["island"][1], anchors["island"][0]),
                xytext=(60, -60),
                textcoords="offset points",
                fontsize=9,
                arrowprops={"arrowstyle": "-", "color": "#333"},
            )
        for t in sk["river"].get("tributaries", []):
            if t["name"] in anchors:
                q = anchors[t["name"]]
                ax.text(
                    q[1] + 40,
                    q[0] + 40,
                    t["name"],
                    fontsize=8,
                    color="#1d5c99",
                    zorder=6,
                    bbox={"fc": "white", "ec": "none", "alpha": 0.6, "pad": 1},
                )
        inside = (main[:, 0] > n0) & (main[:, 0] < n1) & (main[:, 1] > e0) & (main[:, 1] < e1)
        a, b = main[inside][0], main[inside][-1]
        w0, w1 = sk["river"]["width_m"]
        ax.annotate(
            f"river in ({w0:g} m wide)",
            xy=(a[1], a[0]),
            xytext=(-60, 75),
            textcoords="offset points",
            fontsize=9,
            ha="center",
            arrowprops={"arrowstyle": "<-", "color": "#1d5c99"},
        )
        ax.annotate(
            f"river out (up to {w1:g} m wide)",
            xy=(b[1], b[0]),
            xytext=(110, -40),
            textcoords="offset points",
            fontsize=9,
            ha="center",
            arrowprops={"arrowstyle": "->", "color": "#1d5c99"},
        )
    if not sk.get("city"):
        ax.plot(0, 0, "+", color="#9b1c20", ms=12, mew=2, zorder=5)
        ax.text(40, -90, "centre", color="#9b1c20", fontsize=8, zorder=5)
    ticks = np.arange(np.ceil(min(n0, e0) / 500) * 500, max(n1, e1) + 1, 500)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticks(np.arange(e0, e1 + 1, 100), minor=True)
    ax.set_yticks(np.arange(n0, n1 + 1, 100), minor=True)
    ax.grid(which="minor", color="white", lw=0.3, alpha=0.3)
    ax.grid(which="major", color="white", lw=0.8, alpha=0.6)
    ax.set_xlim(e0, e1)
    ax.set_ylim(n0, n1)
    ax.set_aspect("equal")
    ax.set_xlabel("east of centre (m)")
    ax.set_ylabel("north of centre (m)")
    ne_side, sw_side = h[nn + ee > 0], h[nn + ee < 0]
    ax.set_title(
        f"{sk['name']}: {(e1 - e0) / 1000:g} km square; contours every {step} m. "
        f"North-east of the river to {ne_side.max():.0f} m, south-west to "
        f"{sw_side.max():.0f} m",
        fontsize=11,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    if city is not None:  # and a close-up of the city
        fig, ax = plt.subplots(figsize=(12, 12))
        x0, x1, y0, y1 = sk["city"].get("close_up_ne_m", [-800, 800, -800, 800])
        ax.imshow(rgb, origin="lower", extent=(e0, e1, n0, n1), zorder=0)
        for c in channels:
            o = band(c.line, c.width)
            ax.fill(o[:, 1], o[:, 0], color="#3f86cf", lw=0, zorder=3)
        draw_city(ax, city, detail=True)
        ax.imshow(shade_overlay(h, CELL), origin="lower", extent=(e0, e1, n0, n1), zorder=3.25)
        ax.set_xlim(y0, y1)
        ax.set_ylim(x0, x1)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(y0, y1 + 1, 100), minor=True)
        ax.set_yticks(np.arange(x0, x1 + 1, 100), minor=True)
        ax.grid(which="minor", color="white", lw=0.3, alpha=0.4)
        ax.grid(which="major", color="white", lw=0.8, alpha=0.6)
        ax.set_xlabel("east of centre (m)")
        ax.set_ylabel("north of centre (m)")
        ax.set_title(f"{city.name}: close-up (100 m grid)", fontsize=12)
        fig.savefig(out.with_name(out.stem + "_city.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
    if city is not None and city.ops:  # and the operations map
        fig, ax = plt.subplots(figsize=(13, 13))
        ax.imshow(rgb, origin="lower", extent=(e0, e1, n0, n1), zorder=0)
        for c in channels:
            o = band(c.line, c.width)
            ax.fill(o[:, 1], o[:, 0], color="#3f86cf", lw=0, zorder=3)
        draw_city(ax, city, detail=False, labels=False)
        ax.imshow(shade_overlay(h, CELL), origin="lower", extent=(e0, e1, n0, n1), zorder=3.25)
        draw_ops(ax, city)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.grid(which="major", color="white", lw=0.8, alpha=0.6)
        ax.set_xlim(e0, e1)
        ax.set_ylim(n0, n1)
        ax.set_aspect("equal")
        ax.set_xlabel("east of centre (m)")
        ax.set_ylabel("north of centre (m)")
        ax.set_title(
            f"{sk['name']}: operations (airstrip, landing fields, no-fly zones)", fontsize=12
        )
        fig.savefig(out.with_name(out.stem + "_ops.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw a map sketch as an abstract map sheet.")
    ap.add_argument("name")
    ap.add_argument("--out")
    ap.add_argument(
        "--lock",
        action="store_true",
        help="save what is not yet locked (rivers, city, countryside)",
    )
    ap.add_argument(
        "--unlock",
        action="append",
        default=[],
        help="a village, string of roadside houses or farm to grow again (its name; repeatable)",
    )
    a = ap.parse_args()
    if a.unlock:
        sk = yaml.safe_load((SKETCHES / f"{a.name}.yaml").read_text())
        path = SKETCHES / sk["locked"]["countryside"]
        kept = load_countryside(path)
        for name in a.unlock:
            print(f"unlocked {name}" if kept.pop(name, None) else f"{name}: not locked")
        save_countryside(list(kept.values()), path)
    global LOCK
    LOCK = a.lock
    print(draw(a.name, Path(a.out) if a.out else Path("runs/maps") / f"{a.name}_sketch.png"))


if __name__ == "__main__":
    main()
