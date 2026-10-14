"""Airframe definition schema (validated on load; authored in YAML under configs/airframes)."""

from __future__ import annotations

import math

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


class Aero(_Strict):
    longitudinal: Longitudinal
    lateral: Lateral
    stall: Stall


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
    v_cell_v: float = Field(gt=0)

    @property
    def v_max(self) -> float:
        return self.n_cells * self.v_cell_v


class Propulsion(_Strict):
    prop_diameter_m: float = Field(gt=0)
    C_T: tuple[float, float, float]
    C_Q: tuple[float, float, float]
    motor: Motor
    battery: Battery


class Airframe(_Strict):
    schema_version: int = Field(ge=1, le=1)
    id: str
    name: str
    provenance: Provenance
    mass_inertia: MassInertia
    geometry: Geometry
    aero: Aero
    propulsion: Propulsion
