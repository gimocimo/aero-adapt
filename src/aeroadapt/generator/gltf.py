"""glTF model of a generated airframe, from the same geometry the aerodynamics used (M2 spec
§4.1 step 9). Authored in body FRD about the centre of gravity and converted to glTF axes like
the calibration model (``tools/make_gltf.py``), so Skyview places every airframe correctly."""

from __future__ import annotations

import numpy as np
import trimesh
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial

from aeroadapt.generator.design import Design
from aeroadapt.generator.geometry import airplane, layout

FRD_TO_GLTF = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
GEOMETRY_TO_FRD = np.diag([-1.0, 1.0, -1.0])  # x aft → forward, z up → down
COLOURS = {
    "wing": (238, 238, 240, 255),
    "tail": (225, 228, 235, 255),
    "body": (180, 184, 192, 255),
}


def airframe_glb(design: Design, cg: np.ndarray) -> bytes:
    plane = airplane(design, layout(design), (float(cg[0]), float(cg[1]), float(cg[2])))
    to_gltf = FRD_TO_GLTF @ GEOMETRY_TO_FRD
    scene = trimesh.Scene()
    parts: list[tuple[str, np.ndarray, np.ndarray, str]] = []
    for w in plane.wings:
        pts, faces = w.mesh_body(method="tri", chordwise_resolution=10)
        parts.append((w.name, pts, faces, "wing" if w.name == "wing" else "tail"))
    for i, f in enumerate(plane.fuselages):
        pts, faces = f.mesh_body(method="tri", tangential_resolution=16)
        parts.append((f"{f.name}{i}", pts, faces, "body"))
    for name, pts, faces, kind in parts:
        verts = (np.asarray(pts) - cg) @ to_gltf.T
        mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=True)
        mesh.visual = TextureVisuals(
            material=PBRMaterial(
                name=kind, baseColorFactor=COLOURS[kind], metallicFactor=0.0, roughnessFactor=0.7
            )
        )
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    data = scene.export(file_type="glb")
    assert isinstance(data, bytes)
    return data
