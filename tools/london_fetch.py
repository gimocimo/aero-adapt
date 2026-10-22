"""Source data for the London world (docs/08 §2): three files into ``data/london/``.

- ``dsm_1m.tif``: Environment Agency LIDAR Composite DSM (last return), 1 m, cut to the box
  by the agency's WCS; float32 GeoTIFF on the National Grid (EPSG:27700).
- ``dtm_1m.tif``: the matching LIDAR Composite DTM, 1 m.
- ``osm.json``: one Overpass query for the box (with a margin): buildings and building parts,
  roads and paths, railways, water, parks and land use, trees, bridges.

It fetches these and nothing else, and writes ``manifest.json`` (sources, sizes, SHA-256,
licences). The data stays out of git (``data/`` is ignored); the baked world carries the
attribution (docs/08 §9).

Usage::

    uv run python tools/london_fetch.py [--only dsm,dtm,osm] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "london"
USER_AGENT = "aeroadapt-london/0.1 (+https://github.com/gimocimo/aero-adapt)"

# The box on the National Grid (docs/08 §1).
E0, E1, N0, N1 = 531_300, 534_100, 179_700, 181_900
# The same box in WGS84 with a margin of about 60 m (its corners, from the grid by Helmert).
SOUTH, WEST, NORTH, EAST = 51.4998, -0.1107, 51.5213, -0.0677

WCS = {
    "dsm": (
        "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-surface-model-last-return-dsm-1m/wcs",
        "9ba4d5ac-d596-445a-9056-dae3ddec0178__Lidar_Composite_Elevation_LZ_DSM_1m",
    ),
    "dtm": (
        "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs",
        "13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m",
    ),
}
EA_LICENCE = "Open Government Licence v3.0; © Environment Agency copyright and/or database right."
OSM_LICENCE = "ODbL 1.0; © OpenStreetMap contributors (https://www.openstreetmap.org/copyright)."
OVERPASS = "https://overpass-api.de/api/interpreter"


def _overpass_query() -> str:
    b = f"({SOUTH},{WEST},{NORTH},{EAST})"
    parts = [
        f'way["building"]{b}',
        f'relation["building"]{b}',
        f'way["building:part"]{b}',
        f'relation["building:part"]{b}',
        f'way["highway"]{b}',
        f'way["railway"]{b}',
        f'way["bridge"]{b}',
        f'way["man_made"]{b}',
        f'way["natural"]{b}',
        f'relation["natural"]{b}',
        f'way["waterway"]{b}',
        f'way["leisure"]{b}',
        f'relation["leisure"]{b}',
        f'way["landuse"]{b}',
        f'relation["landuse"]{b}',
        f'way["amenity"]{b}',
        f'node["natural"="tree"]{b}',
    ]
    return "[out:json][timeout:180];(" + ";".join(parts) + ";);out body geom;"


def _get(url: str, data: bytes | None = None) -> bytes:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  retrying after {e}")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_lidar(key: str) -> tuple[bytes, str]:
    base, coverage = WCS[key]
    query = urllib.parse.urlencode(
        [
            ("service", "WCS"),
            ("version", "2.0.1"),
            ("request", "GetCoverage"),
            ("CoverageId", coverage),
            ("format", "image/tiff"),
            ("subset", f"E({E0},{E1})"),
            ("subset", f"N({N0},{N1})"),
        ]
    )
    url = f"{base}?{query}"
    body = _get(url)
    if not body.startswith((b"II*\x00", b"MM\x00*")):
        raise RuntimeError(f"{key}: not a TIFF: {body[:300]!r}")
    return body, url


def fetch_osm() -> tuple[bytes, str]:
    body = _get(OVERPASS, urllib.parse.urlencode({"data": _overpass_query()}).encode())
    doc = json.loads(body)
    if not doc.get("elements"):
        raise RuntimeError(f"osm: no elements: {body[:300]!r}")
    return body, f"{OVERPASS} (POST data={_overpass_query()})"


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--only", default="dsm,dtm,osm")
    ap.add_argument("--force", action="store_true", help="fetch again even if the file is there")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    files = {"dsm": "dsm_1m.tif", "dtm": "dtm_1m.tif", "osm": "osm.json"}
    for key in args.only.split(","):
        path = OUT / files[key]
        if path.exists() and not args.force:
            print(f"{path.name}: already here")
            continue
        print(f"{path.name}: fetching")
        body, source = fetch_osm() if key == "osm" else fetch_lidar(key)
        path.write_bytes(body)
        manifest[path.name] = {
            "source": source,
            "licence": OSM_LICENCE if key == "osm" else EA_LICENCE,
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "fetched": datetime.now(UTC).isoformat(timespec="seconds"),
            "box_epsg27700": [E0, N0, E1, N1],
        }
        print(f"  {len(body) / 1e6:.1f} MB")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
