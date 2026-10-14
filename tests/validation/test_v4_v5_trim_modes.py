"""V4 trim and V5 linear modes vs Beard & McLain analytic approximations (spec §12)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import least_squares

from aeroadapt.airframes import AirframeParams
from aeroadapt.sim.aerojax import BM_PARITY
from aeroadapt.sim.aerojax.trim import TrimResult, linearise, trim
from reference.bm_numpy import model as ref

VA = 25.0
CASES = {
    "level": {},
    "climb_5deg": {"gamma": math.radians(5.0)},
    "turn_250m": {"radius": 250.0},
}


def _euler_quat(phi: float, theta: float, psi: float) -> np.ndarray:
    """Test-local ZYX Euler → quaternion (independent of aeroadapt.core)."""
    cp, sp = math.cos(phi / 2), math.sin(phi / 2)
    ct, st = math.cos(theta / 2), math.sin(theta / 2)
    cy, sy = math.cos(psi / 2), math.sin(psi / 2)
    return np.array(
        [
            cy * ct * cp + sy * st * sp,
            cy * ct * sp - sy * st * cp,
            cy * st * cp + sy * ct * sp,
            sy * ct * cp - cy * st * sp,
        ]
    )


def _reference_trim(ref_params: ref.Params, gamma=0.0, radius=math.inf) -> np.ndarray:
    psi_dot = 0.0 if math.isinf(radius) else VA * math.cos(gamma) / radius

    def residual(z: np.ndarray) -> np.ndarray:
        a, th, ph = z[:3]
        omega = psi_dot * np.array(
            [-math.sin(th), math.sin(ph) * math.cos(th), math.cos(ph) * math.cos(th)]
        )
        x = np.concatenate(
            [
                [0, 0, -100.0],
                VA * np.array([math.cos(a), 0, math.sin(a)]),
                _euler_quat(ph, th, 0.0),
                omega,
            ]
        )
        d = ref.derivatives(ref_params, x, z[3:], np.zeros(3))
        return np.concatenate([d[3:6], d[10:13], [(-d[2] - VA * math.sin(gamma)) / VA]])

    sol = least_squares(
        residual,
        [0.05, 0.05, 0.0, 0.0, -0.1, 0.0, 0.6],
        method="lm",
        xtol=1e-15,
        ftol=1e-15,
        gtol=1e-15,
    )
    return sol.x


@pytest.mark.parametrize("case", CASES)
def test_v4_trim_converges_and_matches_reference(
    case: str, params: AirframeParams, ref_params: ref.Params
) -> None:
    tr = trim(params, BM_PARITY, VA, **CASES[case])
    assert tr.success and tr.residual <= 1e-8
    z = np.array([tr.alpha, tr.theta, tr.phi, *(float(c) for c in tr.controls)])
    np.testing.assert_allclose(z, _reference_trim(ref_params, **CASES[case]), atol=1e-6)
    assert np.all(np.abs(z[3:6]) < math.radians(25)) and 0 <= z[6] <= 1


def test_v4_turn_bank_angle_is_physical(params: AirframeParams) -> None:
    tr = trim(params, BM_PARITY, VA, radius=250.0)
    assert math.degrees(tr.phi) == pytest.approx(
        math.degrees(math.atan(VA**2 / (BM_PARITY.gravity * 250.0))), abs=1.0
    )


@pytest.fixture(scope="module")
def level_modes(params: AirframeParams) -> tuple[np.ndarray, np.ndarray]:
    tr: TrimResult = trim(params, BM_PARITY, VA)
    lm = linearise(params, BM_PARITY, tr)
    return np.linalg.eigvals(lm.a_lon), np.linalg.eigvals(lm.a_lat)


def test_v5_roll_subsidence(params: AirframeParams, level_modes) -> None:
    _, lat = level_modes
    rho, b = BM_PARITY.rho, params.b
    gam = params.Jx * params.Jz - params.Jxz**2
    c_p_p = params.Jz / gam * params.C_ell_p + params.Jxz / gam * params.C_n_p
    a_phi1 = -0.5 * rho * VA**2 * params.S * b * c_p_p * b / (2 * VA)  # B&M eq. 5.26
    roll = lat[np.argmin(lat.real)].real
    assert roll == pytest.approx(-a_phi1, rel=0.10)


def test_v5_short_period_and_phugoid(params: AirframeParams, level_modes) -> None:
    lon, _ = level_modes
    rho, c = BM_PARITY.rho, params.c
    a_th2 = -rho * VA**2 * c * params.S / (2 * params.Jy) * params.C_m_alpha  # B&M eq. 5.29
    wn = np.sort(np.abs(lon[np.abs(lon.imag) > 1e-6]))
    assert wn[-1] == pytest.approx(math.sqrt(a_th2), rel=0.15)  # short period
    assert wn[0] == pytest.approx(math.sqrt(2) * BM_PARITY.gravity / VA, rel=0.20)  # phugoid
