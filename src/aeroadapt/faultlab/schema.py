"""Fault specifications as authored in scenario YAML (spec §8). M1 supports time triggers."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TimeTrigger(_Strict):
    at_time_s: float = Field(ge=0)


class PitotBlockage(_Strict):
    """Total-pressure port blocked.

    * ``drain_open``: the trapped total pressure leaks to static with time constant
      ``tau_s``, so indicated airspeed decays towards zero.
    * ``drain_blocked``: total pressure is trapped at its onset value, so the instrument
      behaves like an altimeter (indicated airspeed rises in a climb, falls in a descent).

    ``ramp_s`` blends from a healthy to a fully blocked reading (partial/gradual blockage).
    """

    type: Literal["pitot_blockage"]
    variant: Literal["drain_open", "drain_blocked"]
    trigger: TimeTrigger
    ramp_s: float = Field(default=0.0, ge=0)
    tau_s: float = Field(default=3.0, gt=0)


class ServoJam(_Strict):
    """A control surface stops following commands. The jam position defaults to the
    deflection at onset; the transition into the jam still obeys rate limit and lag.

    ``jam_position_deg`` is a *physical* deflection in Beard & McLain signs (ADR-001), not a
    normalised command.
    """

    type: Literal["servo_jam"]
    channel: Literal["aileron", "elevator", "rudder"]
    trigger: TimeTrigger
    jam_position_deg: float | None = None


class MotorCut(_Strict):
    """ESC desync or cut: no thrust, the propeller windmills, rpm decays, the battery
    supplies avionics current only."""

    type: Literal["motor_cut"]
    trigger: TimeTrigger


FaultSpec = Annotated[PitotBlockage | ServoJam | MotorCut, Field(discriminator="type")]
