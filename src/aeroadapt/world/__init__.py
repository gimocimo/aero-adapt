"""Synthetic worlds (docs/07): authored, generated and known exactly, so the simulator, the
planner, the contingency's survey and the viewer all share one ground truth."""

from __future__ import annotations

from functools import lru_cache

from aeroadapt.world.generate import Building, Dtm, Plot, Structure, Trees, Wire, World, generate
from aeroadapt.world.schema import WORLD_DIR, WorldSpec, load_world_spec

__all__ = [
    "WORLD_DIR",
    "Building",
    "Dtm",
    "Plot",
    "Structure",
    "Trees",
    "Wire",
    "World",
    "WorldSpec",
    "generate",
    "load_world",
    "load_world_spec",
]


@lru_cache(maxsize=8)
def load_world(world_id: str) -> World:
    """The world ``configs/worlds/<id>.yaml``, generated (or, if baked, read) once per process."""
    spec = load_world_spec(world_id)
    if spec.baked is not None:
        from aeroadapt.world.baked import load_baked

        return load_baked(spec)
    return generate(spec)
