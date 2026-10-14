"""Fault specs → fixed-shape slots → labels (ADR-003)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from aeroadapt.faultlab import FaultSpec, build_fault_config, fault_labels, no_faults

SPECS = TypeAdapter(list[FaultSpec])


def test_specs_build_slots_and_labels() -> None:
    specs = SPECS.validate_python(
        [
            {"type": "motor_cut", "trigger": {"at_time_s": 90}},
            {
                "type": "pitot_blockage",
                "variant": "drain_blocked",
                "trigger": {"at_time_s": 200},
                "ramp_s": 5,
            },
            {"type": "servo_jam", "channel": "elevator", "trigger": {"at_time_s": 30}},
            {
                "type": "servo_jam",
                "channel": "aileron",
                "trigger": {"at_time_s": 40},
                "jam_position_deg": 6.0,
            },
        ]
    )
    cfg = build_fault_config(specs)
    assert bool(cfg.motor_cut.active) and float(cfg.motor_cut.onset_s) == 90
    assert bool(cfg.pitot.drain_blocked) and float(cfg.pitot.ramp_s) == 5
    np.testing.assert_array_equal(cfg.servo_jam.active, [True, True, False])
    np.testing.assert_array_equal(cfg.servo_jam.at_onset, [False, True, False])
    assert cfg.servo_jam.jam_rad[0] == pytest.approx(np.deg2rad(6.0))

    lab = fault_labels(cfg, jnp.asarray(35.0))
    assert not bool(lab.motor_cut) and not bool(lab.pitot)
    np.testing.assert_array_equal(lab.servo_jam, [False, True, False])
    lab = fault_labels(cfg, jnp.asarray(200.0))
    assert bool(lab.motor_cut) and bool(lab.pitot)


def test_no_faults_has_same_structure() -> None:
    import jax

    cfg = build_fault_config(
        SPECS.validate_python([{"type": "motor_cut", "trigger": {"at_time_s": 1}}])
    )
    shapes = jax.tree_util.tree_map(np.shape, cfg)
    assert shapes == jax.tree_util.tree_map(np.shape, no_faults())


def test_rejects_duplicates_and_unknown_types() -> None:
    dup = SPECS.validate_python(
        [{"type": "servo_jam", "channel": "rudder", "trigger": {"at_time_s": 1}}] * 2
    )
    with pytest.raises(ValueError, match="duplicate"):
        build_fault_config(dup)
    with pytest.raises(ValidationError):
        SPECS.validate_python([{"type": "wing_falls_off", "trigger": {"at_time_s": 1}}])
