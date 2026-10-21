"""Named OpenStreetMap features for a site's Txt2Fly gazetteer (spec 06 §5, T5).

Queries the Overpass API once for named features around the site origin, converts them to the
site's local north/east metres and writes ``configs/sites/<site>.osm.json``:

- places of worship, named buildings, woods, water, fields and farms, parks, villages;
- each with an id (``osm-<type>-<number>``), its name, a kind, a reference point and, for
  areas, a polygon simplified to at most 16 vertices (and the OSM ``height`` when tagged).

The extract is OpenStreetMap data and stays under the ODbL; the file carries the attribution.

It also writes candidate no-fly ``areas`` (motorways, built-up areas grown by a margin,
university sites); the site file chooses which apply, by id.

Usage (shapely merges each no-fly area's pieces; it is needed by this tool only)::

    uv run --with shapely python tools/fetch_gazetteer.py cambridge_west \
        [--radius 1500] [--margin 30] [--from-file overpass.json]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np

from aeroadapt.scenarios import load_site
from aeroadapt.scenarios.site import SITE_DIR
from aeroadapt.txt2fly.geometry import centroid, convex_hull, offset_convex, signed_area, simplify

ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
USER_AGENT = "aeroadapt-gazetteer/0.1 (+https://github.com/gimocimo/aero-adapt)"
ATTRIBUTION = "© OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright)"


def _kind(tags: dict[str, str]) -> str | None:
    """The gazetteer kind of a named feature, or None to leave it out."""
    if tags.get("amenity") == "place_of_worship":
        return "church" if tags.get("building") in ("church", "cathedral") else "chapel"
    if tags.get("natural") == "wood" or tags.get("landuse") == "forest":
        return "wood"
    if tags.get("natural") == "water":
        return "water"
    if tags.get("landuse") in ("farmland", "meadow", "grass", "orchard", "recreation_ground"):
        return "field"
    if tags.get("landuse") == "farmyard" or tags.get("place") == "farm":
        return "farm"
    if tags.get("place") in ("village", "hamlet", "suburb", "locality"):
        return "village"
    if tags.get("leisure") in ("nature_reserve", "park", "pitch", "garden"):
        return "park"
    if tags.get("amenity") == "university":
        return "campus"
    if "building" in tags or tags.get("tourism") == "hotel" or tags.get("amenity") in (
        "pub", "cafe", "community_centre"
    ):  # fmt: skip
        return "building"
    return None


def fetch(site, radius_m: float) -> dict:
    o = site.origin
    dlat = radius_m / 111_000.0
    dlon = radius_m / (111_000.0 * np.cos(np.radians(o.lat_deg)))
    s, w, n, e = o.lat_deg - dlat, o.lon_deg - dlon, o.lat_deg + dlat, o.lon_deg + dlon
    bbox = f"{s:.5f},{w:.5f},{n:.5f},{e:.5f}"
    return _query(
        f'[out:json][timeout:90];(nwr["name"]({bbox});way["highway"="motorway"]({bbox});'
        f'way["landuse"="residential"]({bbox}););out geom tags;'
    )


def fill_relations(raw: dict) -> dict:
    """Overpass leaves out the members of some multipolygons: fetch the outer ways of those
    whose shape matters (woods, fields, water, parks); others keep their bounding box."""
    for el in raw["elements"]:
        tags = el.get("tags", {})
        shaped = _kind(tags) in ("wood", "field", "water", "farm", "park", "campus")
        wanted = tags.get("type") == "multipolygon" and tags.get("name") and shaped
        if el["type"] == "relation" and wanted and not el.get("members"):
            ways = _query(f'[out:json][timeout:30];rel({el["id"]});way(r:"outer");out geom;')
            el["members"] = [{"type": "way", "role": "outer", "geometry": w["geometry"]}
                             for w in ways["elements"] if "geometry" in w]  # fmt: skip
    return raw


def _query(query: str) -> dict:
    data = urllib.parse.urlencode({"data": query}).encode()
    for attempt in range(4):
        for url in ENDPOINTS:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.loads(r.read())
            except (OSError, ValueError) as e:
                print(f"{url}: {e}", flush=True)
        time.sleep(10 * 2**attempt)
    raise SystemExit("Overpass unavailable; retry later")


def _points(el: dict, origin) -> np.ndarray:
    if "geometry" in el:
        g = el["geometry"]
    elif "lat" in el:
        g = [el]
    else:
        g = [p for m in el.get("members", []) if m.get("role") != "inner"
             for p in m.get("geometry", [])]  # fmt: skip
        if not g and "bounds" in el:  # a relation without member geometry: its box
            b = el["bounds"]
            g = [{"lat": b[la], "lon": b[lo]} for la, lo in (("minlat", "minlon"),
                 ("minlat", "maxlon"), ("maxlat", "maxlon"), ("maxlat", "minlon"))]  # fmt: skip
    if not g:
        return np.zeros((0, 2))
    lat = np.array([p["lat"] for p in g])
    lon = np.array([p["lon"] for p in g])
    ned = origin.geodetic_to_ned(lat, lon, np.full_like(lat, origin.h_ellipsoid_m))
    return np.round(ned[:, :2], 1)


def _area(el: dict, pts: np.ndarray) -> np.ndarray | None:
    """The feature's outline: a closed way as drawn, a relation as the hull of its outers."""
    if el["type"] == "way" and len(pts) >= 4 and np.allclose(pts[0], pts[-1]):
        return pts[:-1]
    if el["type"] == "relation" and len(pts) >= 3:
        return convex_hull(pts)
    return None


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def extract(raw: dict, site, radius_m: float) -> list[dict]:
    origin = site.origin.geodetic()
    best: dict[tuple[str, str], dict] = {}
    for el in raw["elements"]:
        tags = el.get("tags", {})
        name, kind = tags.get("name"), _kind(tags)
        if not name or kind is None:
            continue
        pts = _points(el, origin)
        if not len(pts):
            continue
        area = _area(el, pts)
        point = centroid(area) if area is not None else pts.mean(axis=0)
        if np.hypot(*point) > radius_m:
            continue
        f = {"id": f"osm-{el['type']}-{el['id']}", "name": name, "kind": kind,
             "point_ne_m": [round(float(x), 1) for x in point]}  # fmt: skip
        size = 0.0
        if area is not None:
            poly, _ = simplify(area, 16)
            f["polygon_ne_m"] = [[round(float(a), 1), round(float(b), 1)] for a, b in poly]
            size = abs(signed_area(area))
        if tags.get("height"):
            with contextlib.suppress(ValueError):  # a malformed tag: no height
                f["height_m"] = float(tags["height"].split()[0])
        key = (_slug(name), kind)
        if key not in best or size > best[key]["_size"]:  # the same name split in pieces
            best[key] = {**f, "_size": size}
    out = [{k: v for k, v in f.items() if k != "_size"} for f in best.values()]
    return sorted(out, key=lambda f: (f["kind"], f["name"]))


def _rect(p: np.ndarray, q: np.ndarray, half_width: float) -> np.ndarray:
    """The rectangle within ``half_width`` of the segment p-q (extended at both ends)."""
    t = (q - p) / max(np.linalg.norm(q - p), 1e-9)
    nrm = np.array([-t[1], t[0]]) * half_width
    p, q = p - t * half_width, q + t * half_width
    return np.array([p + nrm, q + nrm, q - nrm, p - nrm])


def _rounded(polys: list[np.ndarray]) -> list[list[list[float]]]:
    return [[[round(float(a), 1), round(float(b), 1)] for a, b in poly] for poly in polys]


def areas(raw: dict, site, radius_m: float, margin_m: float) -> list[dict]:
    """Candidate no-fly areas, each a list of polygons:

    - motorways: a rectangle 50 m either side of every segment;
    - settlements: each residential area's outline grown by ``margin_m`` (convex hull, so
      slightly generous), grouped into settlements of areas within 150 m of each other and
      named after the nearest village or suburb;
    - university sites, as mapped.
    """
    origin = site.origin.geodetic()
    reach = 1.5 * radius_m
    roads: dict[str, list[np.ndarray]] = {}
    residential, campus, villages = [], [], []
    for el in raw["elements"]:
        tags = el.get("tags", {})
        pts = _points(el, origin)
        if not len(pts) or np.hypot(*pts.mean(axis=0)) > reach:
            continue
        if tags.get("highway") == "motorway":
            ref = tags.get("ref", "motorway")
            keep = np.hypot(pts[:, 0], pts[:, 1]) < reach
            segments = [i for i in range(len(pts) - 1) if keep[i] or keep[i + 1]]
            roads.setdefault(ref, []).extend(_rect(pts[i], pts[i + 1], 50.0) for i in segments)
        elif tags.get("landuse") == "residential" and _area(el, pts) is not None:
            residential.append(_area(el, pts))
        elif tags.get("amenity") == "university" and _area(el, pts) is not None:
            campus.append((tags.get("name", "campus"), _area(el, pts)))
        elif tags.get("place") in ("village", "suburb", "hamlet") and "lat" in el:
            villages.append((tags["name"], pts[0]))
    out = [
        {"id": f"motorway-{_slug(ref)}", "name": f"{ref} motorway", "kind": "motorway",
         "reason": "motorway, 50 m either side", "polygons_ne_m": _rounded(rects)}
        for ref, rects in roads.items()
    ]  # fmt: skip
    # Union-find over residential areas whose outlines come within 150 m.
    parent = list(range(len(residential)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    lo = np.array([p.min(axis=0) for p in residential])
    hi = np.array([p.max(axis=0) for p in residential])
    for i, p in enumerate(residential):
        gap = np.maximum(np.maximum(lo - hi[i], lo[i] - hi), 0.0)  # bounding-box separation
        for j in np.nonzero(np.hypot(gap[:, 0], gap[:, 1]) < 150.0)[0]:
            if j > i and np.linalg.norm(p[:, None] - residential[j][None], axis=-1).min() < 150.0:
                parent[root(int(j))] = root(i)
    clusters: dict[int, list[np.ndarray]] = {}
    for i, p in enumerate(residential):
        clusters.setdefault(root(i), []).append(p)
    groups = list(clusters.values())
    for g in groups:
        pts = np.vstack(g)
        if abs(signed_area(convex_hull(pts))) < 20_000:  # a few isolated houses
            continue
        c = centroid(convex_hull(pts))
        name = min(villages, key=lambda v: np.linalg.norm(v[1] - c))[0] if villages else "town"
        grown = [offset_convex(convex_hull(q), margin_m) for q in g if len(q) >= 3]
        out.append({"id": f"settlement-{_slug(name)}", "name": name, "kind": "settlement",
                    "reason": f"built-up area, {margin_m:g} m around it",
                    "polygons_ne_m": _rounded(grown)})  # fmt: skip
    for name, poly in campus:
        out.append({"id": f"campus-{_slug(name)}", "name": name, "kind": "campus",
                    "reason": "university site", "polygons_ne_m": _rounded([poly])})  # fmt: skip
    ids: dict[str, int] = {}
    for a in out:  # unique ids (two settlements nearest the same village)
        n = ids[a["id"]] = ids.get(a["id"], 0) + 1
        if n > 1:
            a["id"] = f"{a['id']}-{n}"
        a["polygons_ne_m"] = _union(a["polygons_ne_m"])
    return out


def _union(polys: list[list[list[float]]]) -> list[list[list[float]]]:
    """Merge overlapping polygons into their union's outer rings (holes dropped, which only
    makes an area larger), simplified to 2 m. Overlapping pieces would double-count in a map
    renderer (Cesium's ground polygons cancel where they overlap)."""
    from shapely import Polygon, union_all

    merged = union_all([Polygon(p).buffer(0) for p in polys if len(p) >= 3]).simplify(2.0)
    parts = list(getattr(merged, "geoms", [merged]))
    rings = [np.asarray(g.exterior.coords)[:-1] for g in parts if not g.is_empty]
    return _rounded(rings)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("site")
    ap.add_argument("--radius", type=float, default=1500.0)
    ap.add_argument("--margin", type=float, default=30.0, help="no-fly margin round settlements")
    ap.add_argument("--from-file", type=Path, help="a saved Overpass reply instead of a query")
    args = ap.parse_args()
    site = load_site(args.site)
    raw = (
        json.loads(args.from_file.read_text())
        if args.from_file
        else fill_relations(fetch(site, args.radius))
    )
    features = extract(raw, site, args.radius)
    out = SITE_DIR / f"{site.id}.osm.json"
    doc = {
        "attribution": ATTRIBUTION,
        "source": "OpenStreetMap via the Overpass API (tools/fetch_gazetteer.py)",
        "date": raw.get("osm3s", {}).get("timestamp_osm_base", str(date.today()))[:10],
        "site": site.id,
        "radius_m": args.radius,
        "features": features,
        "areas": areas(raw, site, args.radius, args.margin),
    }
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"{out}: {len(features)} features, {len(doc['areas'])} areas")
    for a in doc["areas"]:
        print(f"  {a['id']}: {a['name']} ({len(a['polygons_ne_m'])} polygons)")


if __name__ == "__main__":
    main()
