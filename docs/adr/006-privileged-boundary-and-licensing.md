# ADR-006: Privileged-information boundary and licensing

**Status:** Accepted, 14 Oct 2026

## Context

There are two separate risks to manage.

1. **Cheating, even by accident.** If flight-stack code can read the simulator's true state, evaluations become meaningless, and faults will not propagate the way they would on real hardware.
2. **Licence contamination.** Some useful sources are GPL-3.0: `mavsim_public` and ArduPilot.

## Decision

**Privileged boundary.** `aeroadapt.stack` must not import `aeroadapt.sim` (and, from week 2, `aeroadapt.faultlab`). This is enforced by import-linter in CI (`.importlinter`). The stack receives a `Vehicle` view without `truth()`. Only the recorder, evaluator and oracle receive `PrivilegedVehicle`.

**N-version independence.** `reference/` (the validation-only NumPy implementation) and `aeroadapt` may not import each other, in either direction. This is also enforced by import-linter.

**Licensing.** The project is Apache-2.0.
- `mavsim_public` is not vendored. Only parameter values are used, with citation (see `NOTICE`).
- ArduPilot-side Lua scripts (M6 onwards) will live in `ardupilot/` under GPL-3.0, with their own licence file.
- ArduPilot SITL runs as an external process over MAVLink.
- *This is not legal advice; re-check before any commercial use.*

**Secrets.** Tokens such as the Cesium ion token live in git-ignored `.env.local` files. The `detect-private-key` pre-commit hook adds a guard.

## Consequences

- Any accidental truth leak into the stack fails CI immediately.
- The reference implementation's independence is structural, not just a matter of good intentions.
