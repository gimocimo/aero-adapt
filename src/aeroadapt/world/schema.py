"""The authored description of a synthetic world (``configs/worlds/<id>.yaml``, docs/07 §2).

A world is authored at the level a map-maker would sketch it: the lie of the land, the river,
the roads, the woods, the village and its streets, the farms and the fields that make good
landing grounds. :mod:`aeroadapt.world.generate` expands it, deterministically from its seed,
into every field, hedge, tree and building, with their exact positions and heights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

WORLD_DIR = Path(__file__).resolve().parents[3] / "configs" / "worlds"

NE = tuple[float, float]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Ridge(_Strict):
    """Ground rising from ``from_n_m`` to ``to_n_m`` (north coordinates) by ``rise_m``."""

    from_n_m: float
    to_n_m: float
    rise_m: float


class Hill(_Strict):
    centre_ne_m: NE
    height_m: float
    radius_m: float = Field(gt=0)


class TerrainSpec(_Strict):
    cell_m: float = Field(default=10.0, gt=0)
    east_slope_per_km: float = 0.0  # metres of rise per km eastwards
    ridges: list[Ridge] = Field(default_factory=list)
    hills: list[Hill] = Field(default_factory=list)
    noise_m: float = Field(default=0.0, ge=0)  # amplitude of gentle undulation
    noise_wavelength_m: tuple[float, float] = (350.0, 900.0)


class RiverSpec(_Strict):
    id: str
    name: str
    width_m: float = Field(gt=0)
    depth_m: float = Field(gt=0)
    points_ne_m: list[NE] = Field(min_length=2)


class RoadSpec(_Strict):
    id: str
    name: str | None = None
    kind: Literal["road", "lane", "street", "track"]
    width_m: float = Field(gt=0)
    points_ne_m: list[NE] = Field(min_length=2)
    hedged: bool = True  # hedges along both verges (outside the village)


class WoodSpec(_Strict):
    id: str
    name: str
    kind: Literal["broadleaf", "conifer"] = "broadleaf"
    canopy_m: float = Field(gt=0)
    polygon_ne_m: list[NE] = Field(min_length=3)


class BuildingSpec(_Strict):
    kind: Literal["house", "farmhouse", "barn", "shed", "hangar", "hall", "pub", "nave", "tower"]
    centre_ne_m: NE
    length_m: float = Field(gt=0)
    width_m: float = Field(gt=0)
    heading_deg: float = 0.0  # of the long axis (ridge line)
    eaves_m: float = Field(gt=0)
    ridge_m: float = Field(gt=0)  # top of the roof (spire top for a tower)
    roof: Literal["gable", "flat", "spire"] = "gable"
    name: str | None = None


class StructureSpec(_Strict):
    id: str
    name: str
    kind: Literal["mast", "silo"]
    point_ne_m: NE
    height_m: float = Field(gt=0)
    radius_m: float = Field(default=1.0, gt=0)


class VillageSpec(_Strict):
    id: str
    name: str
    outline_ne_m: list[NE] = Field(min_length=3)  # the built-up area
    streets: list[str]  # road ids houses line up along
    house_spacing_m: tuple[float, float] = (17.0, 26.0)
    green_ne_m: list[NE] | None = None  # kept free of houses
    buildings: list[BuildingSpec] = Field(default_factory=list)  # church, pub, hall, …


class FarmSpec(_Strict):
    id: str
    name: str
    yard_ne_m: list[NE] = Field(min_length=3)
    buildings: list[BuildingSpec]


class AirstripSpec(_Strict):
    id: str
    name: str
    centre_ne_m: NE
    heading_deg: float
    length_m: float
    width_m: float  # the mown area (the runway is painted down its middle)
    buildings: list[BuildingSpec] = Field(default_factory=list)


class FieldSpec(_Strict):
    size_m: tuple[float, float] = (260.0, 340.0)  # grid spacing range
    rotation_deg: float = 0.0
    jitter_m: float = 25.0
    crops: dict[str, float]  # crop → weight
    hedge_height_m: tuple[float, float] = (2.0, 3.0)
    hedge_width_m: float = 2.0
    hedge_tree_spacing_m: float = 90.0  # mean distance between hedgerow trees
    hedge_tree_height_m: tuple[float, float] = (9.0, 15.0)


class ZoneSpec(_Strict):
    """A landing ground, safe zone ``id``: a field drawn here (``polygon_ne_m``), or the field
    or airstrip containing ``at_ne_m``. Its ground is levelled, its hedges are trimmed low and
    no tree stands in its approach funnels."""

    id: str
    at_ne_m: NE | None = None
    polygon_ne_m: list[NE] | None = None
    crop: str | None = None  # the field's crop (its surface)
    name: str | None = None

    @model_validator(mode="after")
    def _one_source(self) -> ZoneSpec:
        if (self.at_ne_m is None) == (self.polygon_ne_m is None):
            raise ValueError(f"zone {self.id}: give either at_ne_m or polygon_ne_m")
        return self


class PlaceSpec(_Strict):
    id: str
    name: str
    kind: str
    point_ne_m: NE
    aliases: list[str] = Field(default_factory=list)


class WorldSpec(_Strict):
    schema_version: int = Field(ge=1, le=1)
    id: str
    name: str
    description: str
    seed: int
    bounds_ne_m: tuple[NE, NE]  # (south-west corner, north-east corner)
    terrain: TerrainSpec
    river: RiverSpec | None = None
    roads: list[RoadSpec] = Field(default_factory=list)
    woods: list[WoodSpec] = Field(default_factory=list)
    village: VillageSpec | None = None
    farms: list[FarmSpec] = Field(default_factory=list)
    airstrip: AirstripSpec | None = None
    structures: list[StructureSpec] = Field(default_factory=list)
    fields: FieldSpec | None = None  # None: no field patchwork (open grassland)
    zones: list[ZoneSpec] = Field(default_factory=list)
    places: list[PlaceSpec] = Field(default_factory=list)


def load_world_spec(name_or_path: str | Path) -> WorldSpec:
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        path = WORLD_DIR / f"{name_or_path}.yaml"
    with path.open() as f:
        return WorldSpec.model_validate(yaml.safe_load(f))
