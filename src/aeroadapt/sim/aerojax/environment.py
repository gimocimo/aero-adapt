"""Environment parameters.

Week 1 uses a constant-density atmosphere. The ISA model, wind profiles and Dryden turbulence
arrive in week 2.
"""

from __future__ import annotations

from typing import NamedTuple


class Environment(NamedTuple):
    rho: float  # air density, kg/m^3
    gravity: float  # m/s^2


# Constants used by Beard & McLain's Aerosonde reference values. Used for N-version parity.
BM_PARITY = Environment(rho=1.2682, gravity=9.81)
