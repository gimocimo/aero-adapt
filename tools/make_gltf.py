"""Generate the viewer's calibration UAV model (glTF binary).

The model is authored in body FRD coordinates and converted to glTF axes (+Y up, +Z forward,
+X left), so that it can be used to calibrate the viewer's frame conversions visually:

* yellow nose → forward;
* red wingtip → left (port);
* green wingtip → right (starboard), using aviation navigation-light colours;
* blue fin → up.

Usage: ``uv run python tools/make_gltf.py [output.glb]``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial

OUT = Path(__file__).resolve().parents[1] / "web/skyview/public/models/calibration_uav.glb"

# Rotation taking body-FRD vectors into glTF axes: X_gltf = -y (left), Y_gltf = -z (up), Z_gltf = x.
FRD_TO_GLTF = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])

# name: (extents along FRD x, y, z [m], centre in FRD [m], RGBA)
PARTS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float], tuple[int, ...]]] = {
    "fuselage": ((1.60, 0.16, 0.16), (0.00, 0.00, 0.00), (200, 200, 205, 255)),
    "nose": ((0.15, 0.16, 0.16), (0.875, 0.00, 0.00), (250, 210, 30, 255)),
    "wing": ((0.19, 2.70, 0.03), (0.10, 0.00, -0.06), (240, 240, 240, 255)),
    "wingtip_port": ((0.19, 0.10, 0.035), (0.10, -1.40, -0.06), (220, 30, 30, 255)),
    "wingtip_starboard": ((0.19, 0.10, 0.035), (0.10, 1.40, -0.06), (30, 190, 60, 255)),
    "tailplane": ((0.12, 0.80, 0.02), (-0.75, 0.00, -0.02), (240, 240, 240, 255)),
    "fin": ((0.14, 0.02, 0.30), (-0.75, 0.00, -0.17), (40, 90, 220, 255)),
}


def build_scene() -> trimesh.Scene:
    scene = trimesh.Scene()
    for name, (extents, centre, rgba) in PARTS.items():
        box = trimesh.creation.box(extents=extents)
        box.apply_translation(centre)
        box.vertices = box.vertices @ FRD_TO_GLTF.T
        material = PBRMaterial(
            name=name, baseColorFactor=rgba, metallicFactor=0.0, roughnessFactor=0.8
        )
        box.visual = TextureVisuals(material=material)
        scene.add_geometry(box, node_name=name, geom_name=name)
    return scene


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    data = build_scene().export(file_type="glb")
    assert isinstance(data, bytes)
    out.write_bytes(data)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
