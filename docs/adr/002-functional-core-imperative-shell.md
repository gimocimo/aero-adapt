# ADR-002: Functional core (pure JAX) and imperative shell

**Status:** Accepted, 14 Oct 2026

## Context

The same flight-dynamics and control code must serve three uses:

1. single-scenario, host-stepped runs with a live viewer (M1);
2. batched evaluation of thousands of scenarios on a GPU (M2);
3. RL training, including residual RL on top of classical controllers (M3).

Rewriting the code for each use would invite divergence between the versions.

## Decision

- **The fast path is pure JAX.** Everything that runs at 10 Hz or faster is written as pure functions of the form `f(params, state, inputs[, key]) -> (state, outputs)`. This covers the simulator step, navigator, autopilot and contingency v0. These functions:
  - have no side effects;
  - contain no Python control flow on traced values (mode logic uses `lax.switch` / `jnp.where`);
  - are dtype-agnostic;
  - use pytrees (NamedTuples) for parameters and state, so `vmap` can batch over airframes, faults and scenarios.
- **The slow path may be ordinary Python.** Logic at 1 Hz or slower (future Captain/Pathfinder, geodata, I/O, recording) can use any Python.
- **Thin imperative shells** (`AerojaxVehicle`, `FlightStack`, the live server) call the pure functions one step at a time in M1.
- **Precision:** validation runs in float64. Training precision (float32) is decided by V7, and numerics are designed for float32 from the start. Example: the stall-blending sigmoid uses a log-sum-exp form because the textbook form overflows in float32.
- **Type checking:** pyright runs in standard mode across the codebase. Strict mode is enabled per file (`# pyright: strict`) where dependencies are fully typed, e.g. `core/geodesy.py`. JAX's own stubs contain unknown types, so JAX modules cannot pass strict mode without blanket suppressions.

## Consequences

- The M2 batched evaluation and the M3 residual RL reuse the M1 code unchanged.
- Some controller logic (mode machines, waypoint sequencing) is a little more awkward to write in a branch-free style. This is accepted.
- A test will guard against accidental recompilation (planned for week 2).
