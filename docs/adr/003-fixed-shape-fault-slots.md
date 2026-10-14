# ADR-003: Fixed-shape fault slots

**Status:** Accepted, 14 Oct 2026 (implementation in M1 week 2)

## Context

Fault injection has to work inside `jit` and `vmap`, so that a single compiled program can simulate thousands of scenarios, *each with different faults*, in parallel. Python-level branching on "which fault is active" would force a recompile for every fault combination, or rule out batching entirely.

## Decision

**Fault state.** Each fault type owns a fixed-shape slot in a `FaultState` pytree:
- `active` (bool);
- `variant` (int);
- `onset_t` and `ramp_s`;
- fault-specific parameters.

**Inactive faults.** An inactive slot is an identity transform implemented with masks (`jnp.where`). There is no Python branching on fault state.

**Triggers.** Triggers are resolved to onset times when the scenario resets. M1 supports time-based triggers only; condition-based triggers arrive in M2.

**Injection points:**

| Fault kind | Applied where |
|---|---|
| Sensor | Sensor model output |
| Actuator | Actuator pipeline, after the delay and before the lag |
| Propulsion | Propulsion model |
| Aerodynamic (M2) | Aerodynamic model |
| Environment | Atmosphere |

**Labels.** Every physics step emits ground-truth `FaultLabels`, so every recording doubles as diagnosis training data.

## Consequences

- Adding a fault type changes the pytree structure, and therefore recompiles the program once. That is expected and acceptable.
- Compound faults are free: several slots can be active at once.
- Fault labels are emitted by the simulator, never inferred afterwards, so they are exact.
