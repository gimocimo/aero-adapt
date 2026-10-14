"""Airframe schema, provenance and derived quantities."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from aeroadapt.airframes import Airframe, AirframeParams
from aeroadapt.airframes.schema import MassInertia


def test_aerosonde_loads_with_provenance(aerosonde: Airframe) -> None:
    assert aerosonde.id == "aerosonde_bm"
    assert aerosonde.provenance.commit == "10363e0eaeb16a61a7463505f8e122a9eb7e2b4e"


def test_derived_quantities(params: AirframeParams) -> None:
    assert params.AR == pytest.approx(2.8956**2 / 0.55)
    assert params.KV == pytest.approx(60.0 / (2 * math.pi * 145.0))
    assert params.KQ == params.KV
    assert params.V_max == pytest.approx(44.4)


def test_rejects_non_positive_definite_inertia() -> None:
    with pytest.raises(ValidationError, match="positive definite"):
        MassInertia(mass_kg=1, Jx_kgm2=1, Jy_kgm2=1, Jz_kgm2=1, Jxz_kgm2=1.5)


def test_rejects_physically_impossible_inertia() -> None:
    with pytest.raises(ValidationError, match="triangle inequality"):
        MassInertia(mass_kg=1, Jx_kgm2=0.1, Jy_kgm2=0.1, Jz_kgm2=5.0, Jxz_kgm2=0.0)


def test_rejects_unknown_fields(aerosonde: Airframe) -> None:
    data = aerosonde.model_dump()
    data["geometry"]["wingspan_typo"] = 3.0
    with pytest.raises(ValidationError, match="Extra inputs"):
        Airframe.model_validate(data)
