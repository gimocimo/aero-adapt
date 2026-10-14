"""Compute a site's Earth magnetic field with WMM2025 for its YAML config.

Usage: ``uv run python tools/site_magnetic.py LAT_DEG LON_DEG H_ELLIPSOID_M DECIMAL_YEAR``
Example: ``uv run python tools/site_magnetic.py 52.2075 0.0747 63.9 2026.784``

Prints the ``magnetic:`` block to paste into ``configs/sites/<id>.yaml``. The simulator uses
the NED field vector for its magnetometer model. Declination and inclination are kept for
humans and cross-checked against the vector in the tests.
"""

from __future__ import annotations

import sys

from pygeomag import GeoMag

NT_PER_GAUSS = 1e5


def main() -> None:
    lat, lon, h_m, year = (float(a) for a in sys.argv[1:5])
    r = GeoMag(coefficients_file="wmm/WMM_2025.COF").calculate(
        glat=lat, glon=lon, alt=h_m / 1000.0, time=year
    )
    print("magnetic:")
    print("  model: WMM2025")
    print(f"  epoch_year: {year}")
    print(f"  declination_deg: {r.d:.3f}")
    print(f"  inclination_deg: {r.i:.3f}")
    print(f"  total_field_gauss: {r.f / NT_PER_GAUSS:.5f}")
    ned = [r.x / NT_PER_GAUSS, r.y / NT_PER_GAUSS, r.z / NT_PER_GAUSS]
    print(f"  field_ned_gauss: [{ned[0]:.5f}, {ned[1]:.5f}, {ned[2]:.5f}]")


if __name__ == "__main__":
    main()
