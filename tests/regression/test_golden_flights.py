"""Golden flights (spec §13): the canonical scenarios keep their metrics within tolerance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aeroadapt.scenarios.goldens import GOLDEN_SCENARIOS, compare, metrics

GOLDENS = json.loads((Path(__file__).parent / "goldens.json").read_text())


@pytest.mark.parametrize("scenario_id", GOLDEN_SCENARIOS)
def test_golden_flight(scenario_id: str) -> None:
    problems = compare(GOLDENS[scenario_id], metrics(scenario_id))
    assert not problems, "\n".join(problems)
