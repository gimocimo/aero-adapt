"""Txt2Fly plans (spec 06 §4): the typed building blocks an operator approves.

A plan says *what* to do in terms of places and numbers: 1-8 steps, each a discriminated union
on ``do``. The builder turns it into mission items for one airframe in one wind; the checker
decides whether it may fly. Plans come from a file, a form or (week 10) the language model,
and every source is built and checked the same way.

Altitudes are metres above the (flat) site ground; ``None`` keeps the altitude of the step
before (or the aircraft's). The schema stays free of recursion and numeric bounds so that it
can be given to a model as a structured-output format; limits are checked by the checker,
which explains them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- targets
class PlaceRef(_Strict):
    """A place in the site's gazetteer, by id."""

    kind: Literal["place"] = "place"
    id: str


class Here(_Strict):
    """Where the aircraft is when the plan is confirmed."""

    kind: Literal["here"] = "here"


class Home(_Strict):
    kind: Literal["home"] = "home"


class Offset(_Strict):
    """A point at a bearing and distance from a place (or ``here`` / ``home``)."""

    kind: Literal["offset"] = "offset"
    origin: str = Field(description="a place id, 'here' or 'home'")
    bearing_deg: float = Field(description="true bearing from the origin, clockwise from north")
    distance_m: float


class Point(_Strict):
    """A point in the site's local frame, e.g. clicked on the map."""

    kind: Literal["point"] = "point"
    ne_m: tuple[float, float]


class LatLon(_Strict):
    kind: Literal["latlon"] = "latlon"
    lat_deg: float
    lon_deg: float


Target = Annotated[PlaceRef | Here | Home | Offset | Point | LatLon, Field(discriminator="kind")]


# --------------------------------------------------------------------------- steps
class Goto(_Strict):
    """Fly to a place and carry on (a fly-by waypoint)."""

    do: Literal["goto"] = "goto"
    target: Target
    alt_m: float | None = None


class PassOver(_Strict):
    """A straight pass, wings level over the place ("fly by field A")."""

    do: Literal["pass_over"] = "pass_over"
    target: Target
    alt_m: float | None = None
    heading: Literal["along", "into_wind"] | float = Field(
        default="along",
        description="'along' the place's long axis (a point: the course flown to it), "
        "'into_wind', or a true heading in degrees",
    )


class Orbit(_Strict):
    """Circle a place: a number of turns, for a time, or until told otherwise."""

    do: Literal["orbit"] = "orbit"
    target: Target
    radius_m: float | None = Field(default=None, description="None: the aircraft's usual")
    alt_m: float | None = None
    direction: Literal["cw", "ccw"] = "cw"
    turns: float | None = None
    duration_s: float | None = None


class Patrol(_Strict):
    """Visit places in the order given, lap after lap: over the middle of each (``pass_over``)
    or by it (``goto``), on the route from one to the next."""

    do: Literal["patrol"] = "patrol"
    targets: list[Target]
    visit: Literal["pass_over", "goto"] = "pass_over"
    alt_m: float | None = None
    laps: int | None = Field(default=None, description="None: until the battery reserve")


class SetSpeed(_Strict):
    """Airspeed for the steps that follow."""

    do: Literal["set_speed"] = "set_speed"
    speed: Literal["slow", "cruise", "fast"] | float


class Hold(_Strict):
    """Circle where the aircraft is, until told otherwise."""

    do: Literal["hold"] = "hold"
    radius_m: float | None = None


class ReturnHome(_Strict):
    do: Literal["return_home"] = "return_home"


Step = Annotated[
    Goto | PassOver | Orbit | Patrol | SetSpeed | Hold | ReturnHome, Field(discriminator="do")
]
STEP_KINDS = ("goto", "pass_over", "orbit", "patrol", "set_speed", "hold", "return_home")


class Plan(_Strict):
    summary: str = Field(default="", description="one sentence restating the plan")
    assumptions: list[str] = Field(
        default_factory=list, description="every reading chosen where the request was open"
    )
    steps: list[Step]


def load_plan(path: str | Path) -> Plan:
    return Plan.model_validate(yaml.safe_load(Path(path).read_text()))
