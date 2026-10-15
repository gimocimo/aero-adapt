"""Airframe definition schema (validated on load; authored in YAML under configs/airframes)."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Provenance(_Strict):
    source: str
    values_from: str | None = None
    commit: str | None = None
    notes: str | None = None


class MassInertia(_Strict):
    mass_kg: float = Field(gt=0)
    Jx_kgm2: float = Field(gt=0)
    Jy_kgm2: float = Field(gt=0)
    Jz_kgm2: float = Field(gt=0)
    Jxz_kgm2: float

    @model_validator(mode="after")
    def _physically_realisable(self) -> MassInertia:
        # Positive definite and satisfying the triangle inequalities of a real mass distribution.
        jx, jy, jz, jxz = self.Jx_kgm2, self.Jy_kgm2, self.Jz_kgm2, self.Jxz_kgm2
        if jx * jz - jxz**2 <= 0:
            raise ValueError("inertia tensor is not positive definite (Jx*Jz - Jxz^2 <= 0)")
        tr = jx + jy + jz
        disc = math.sqrt(((jx - jz) / 2) ** 2 + jxz**2)
        principal = sorted([jy, (jx + jz) / 2 + disc, (jx + jz) / 2 - disc])
        if principal[0] + principal[1] < principal[2] - 1e-12 * tr:
            raise ValueError(f"principal moments {principal} violate the triangle inequality")
        return self


class Geometry(_Strict):
    wing_area_m2: float = Field(gt=0)
    span_m: float = Field(gt=0)
    chord_m: float = Field(gt=0)
    oswald_e: float = Field(gt=0, le=1)

    @property
    def aspect_ratio(self) -> float:
        return self.span_m**2 / self.wing_area_m2


class Longitudinal(_Strict):
    C_L_0: float
    C_L_alpha: float
    C_L_q: float
    C_L_delta_e: float
    C_D_p: float = Field(ge=0)
    C_D_q: float
    C_D_delta_e: float
    C_m_0: float
    C_m_alpha: float
    C_m_q: float
    C_m_delta_e: float


class Lateral(_Strict):
    C_Y_0: float
    C_Y_beta: float
    C_Y_p: float
    C_Y_r: float
    C_Y_delta_a: float
    C_Y_delta_r: float
    C_ell_0: float
    C_ell_beta: float
    C_ell_p: float
    C_ell_r: float
    C_ell_delta_a: float
    C_ell_delta_r: float
    C_n_0: float
    C_n_beta: float
    C_n_p: float
    C_n_r: float
    C_n_delta_a: float
    C_n_delta_r: float


class Stall(_Strict):
    M: float = Field(gt=0, description="Sigmoid blending rate")
    alpha0_rad: float = Field(gt=0, description="Stall blending angle of attack")


ALPHA_GRID_N = 721  # aero v2: uniform 0.5° grid over [-180°, 180°] (M2 spec §3.1)


class Tables(_Strict):
    """Aero v2 tables on the uniform α grid (``ALPHA_GRID_N`` points from -180° to 180°)."""

    C_L: list[float]
    C_D: list[float]
    C_m: list[float]
    eta_ctrl: list[float]
    eta_damp: list[float]

    @model_validator(mode="after")
    def _on_the_grid(self) -> Tables:
        for name in ("C_L", "C_D", "C_m", "eta_ctrl", "eta_damp"):
            values = getattr(self, name)
            if len(values) != ALPHA_GRID_N or not all(math.isfinite(v) for v in values):
                raise ValueError(f"table {name} needs {ALPHA_GRID_N} finite values")
        return self


class Aero(_Strict):
    """``model`` selects the static-coefficient source (M2 spec §3).

    ``bm_analytic`` evaluates the Beard & McLain formulas (aero v1). ``bm_tabulated`` samples the
    same formulas onto the aero v2 grid, so a B&M airframe can fly in a batch of tabulated ones.
    ``tabulated`` takes ``tables`` (e.g. from the airframe generator); the linear coefficients
    are then the attached-flow fits, kept for documentation and the stack's glide polar.
    """

    model: Literal["bm_analytic", "bm_tabulated", "tabulated"] = "bm_analytic"
    tables: Tables | None = None
    ground_effect: bool = Field(
        default=False, description="Induced-drag reduction near the ground (McCormick)"
    )
    longitudinal: Longitudinal
    lateral: Lateral
    stall: Stall

    @model_validator(mode="after")
    def _tables_match_model(self) -> Aero:
        if (self.model == "tabulated") != (self.tables is not None):
            raise ValueError("tables are required for, and only for, model 'tabulated'")
        return self


class Motor(_Strict):
    KV_rpm_per_volt: float = Field(gt=0)
    R_ohm: float = Field(gt=0)
    i0_a: float = Field(ge=0)

    @property
    def KV_v_s_per_rad(self) -> float:
        """Back-EMF constant in V·s/rad (also the torque constant K_Q in N·m/A)."""
        return 60.0 / (2.0 * math.pi * self.KV_rpm_per_volt)


class Battery(_Strict):
    n_cells: int = Field(gt=0)
    v_cell_v: float = Field(gt=0, description="Ideal-battery cell voltage (B&M V_max)")
    capacity_ah: float = Field(gt=0)
    r_internal_ohm: float = Field(ge=0, description="Pack internal resistance")
    avionics_current_a: float = Field(ge=0)
    ocv_curve: list[tuple[float, float]] = Field(
        min_length=2, description="[state of charge, open-circuit volts per cell]"
    )

    @property
    def v_max(self) -> float:
        return self.n_cells * self.v_cell_v

    @model_validator(mode="after")
    def _ocv_curve_is_valid(self) -> Battery:
        soc = [x for x, _ in self.ocv_curve]
        volts = [v for _, v in self.ocv_curve]
        if soc[0] != 0.0 or soc[-1] != 1.0 or any(b <= a for a, b in pairwise(soc)):
            raise ValueError("ocv_curve SOC must increase strictly from 0.0 to 1.0")
        if any(b < a for a, b in pairwise(volts)):
            raise ValueError("ocv_curve voltage must be non-decreasing with SOC")
        return self


class Propulsion(_Strict):
    prop_diameter_m: float = Field(gt=0)
    C_T: tuple[float, float, float]
    C_Q: tuple[float, float, float]
    windmill_cd: float = Field(ge=0, description="Stopped-prop drag coefficient on disc area")
    motor: Motor
    battery: Battery


class SurfaceActuator(_Strict):
    max_deflection_deg: float = Field(gt=0, le=60)
    rate_limit_dps: float = Field(gt=0)
    tau_s: float = Field(gt=0)
    delay_s: float = Field(ge=0, le=0.15, description="Transport delay (<= 150 ms buffer)")
    deadband_deg: float = Field(ge=0)
    trim_deg: float = 0.0
    reversed: bool = False


class Surfaces(_Strict):
    aileron: SurfaceActuator
    elevator: SurfaceActuator
    rudder: SurfaceActuator


class ThrottleActuator(_Strict):
    tau_s: float = Field(gt=0)
    delay_s: float = Field(ge=0, le=0.15)


class Actuators(_Strict):
    surfaces: Surfaces
    throttle: ThrottleActuator


class Imu(_Strict):
    rate_hz: float = Field(gt=0)
    gyro_noise_rps: float = Field(ge=0)
    gyro_bias_rw: float = Field(ge=0)
    gyro_bias_init_rps: float = Field(ge=0)
    accel_noise_mps2: float = Field(ge=0)
    accel_bias_rw: float = Field(ge=0)
    accel_bias_init_mps2: float = Field(ge=0)


class Baro(_Strict):
    rate_hz: float = Field(gt=0)
    noise_pa: float = Field(ge=0)
    drift_rw_pa_per_sqrt_s: float = Field(ge=0)


class AirspeedSensor(_Strict):
    rate_hz: float = Field(gt=0)
    noise_pa: float = Field(ge=0)


class Gnss(_Strict):
    rate_hz: float = Field(gt=0)
    latency_s: float = Field(ge=0, le=0.3, description="<= 300 ms history buffer")
    pos_sigma_h_m: float = Field(ge=0)
    pos_sigma_v_m: float = Field(ge=0)
    pos_tau_s: float = Field(gt=0)
    vel_sigma_mps: float = Field(ge=0)


class Mag(_Strict):
    rate_hz: float = Field(gt=0)
    noise_gauss: float = Field(ge=0)


class BatterySensor(_Strict):
    rate_hz: float = Field(gt=0)
    voltage_noise_v: float = Field(ge=0)
    current_noise_a: float = Field(ge=0)


class Esc(_Strict):
    rate_hz: float = Field(gt=0)
    rpm_noise_frac: float = Field(ge=0)


class Sensors(_Strict):
    imu: Imu
    baro: Baro
    airspeed: AirspeedSensor
    gnss: Gnss
    mag: Mag
    battery: BatterySensor
    esc: Esc


class Airframe(_Strict):
    schema_version: int = Field(ge=1, le=2)
    id: str
    name: str
    provenance: Provenance
    mass_inertia: MassInertia
    geometry: Geometry
    aero: Aero
    propulsion: Propulsion
    actuators: Actuators
    sensors: Sensors
