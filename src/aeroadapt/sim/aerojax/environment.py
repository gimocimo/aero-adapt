"""Environment parameters.

* :class:`Environment` is the minimal (density, gravity) pair used by the force model and by
  the Beard & McLain parity configuration.
* :class:`EnvParams` is the full simulator environment: ISA, wind, turbulence, Earth field.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from aeroadapt.core.frames import wind_from_to_ned
from aeroadapt.sim.aerojax.atmosphere import G0, IsaParams, TurbulenceParams, WindParams


class Environment(NamedTuple):
    rho: float | Array  # air density, kg/m^3
    gravity: float | Array  # m/s^2


# Constants used by Beard & McLain's Aerosonde reference values. Used for N-version parity.
BM_PARITY = Environment(rho=1.2682, gravity=9.81)


class EnvParams(NamedTuple):
    gravity: float
    isa: IsaParams
    wind: WindParams
    turbulence: TurbulenceParams
    magnetic_ned_gauss: Array  # (3,)


def make_env(
    *,
    alt_origin_msl_m: float,
    magnetic_ned_gauss: tuple[float, float, float],
    wind_from_deg: float = 0.0,
    wind_speed_mps: float = 0.0,
    w20_mps: float = 0.0,
    bl_alpha: float = 0.0,
    bl_ref_m: float = 50.0,
    temp_offset_k: float = 0.0,
    qnh_hpa: float = 1013.25,
    gravity: float = G0,
) -> EnvParams:
    """Host-side convenience builder (aviation 'wind from' convention at the boundary)."""
    return EnvParams(
        gravity=gravity,
        isa=IsaParams(temp_offset_k, qnh_hpa * 100.0, alt_origin_msl_m),
        wind=WindParams(wind_from_to_ned(wind_from_deg, wind_speed_mps), bl_alpha, bl_ref_m),
        turbulence=TurbulenceParams(w20_mps),
        magnetic_ned_gauss=jnp.asarray(magnetic_ned_gauss),
    )
