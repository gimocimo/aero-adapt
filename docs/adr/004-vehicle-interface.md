# ADR-004: `Vehicle` interface and ArduPilot-aligned semantics

**Status:** Accepted, 14 Oct 2026 (implementation in M1 week 3)

## Context

The flight stack must fly several vehicles without modification:

- `aerojax` (M1);
- JSBSim (M2);
- ArduPilot SITL (M6);
- a real aircraft (month 7 onwards).

The hardware path uses ArduPilot.

## Decision

**One narrow interface.** Sensor frames go in and actuator commands come out. `step()` advances exactly one control period.

**Message semantics mirror MAVLink and ArduPilot:**
- `time_us` timestamps;
- multi-rate sensors, each with a `fresh` flag;
- GNSS samples timestamped at their *time of validity*, with latency already applied;
- raw differential pressure for airspeed, with indicated airspeed derived by the navigator;
- ESC telemetry rpm.

**Commands use ArduPilot servo functions.** The channels are `aileron`, `elevator`, `rudder` and `throttle`, with aircraft-response signs (ADR-001). Elevon and V-tail mixing (M2) happen in the per-airframe ServoMap, as in ArduPilot. `reversed` flags mirror `SERVOn_REVERSED`.

**Privileged data is a separate view.** Truth and fault labels are available only through `PrivilegedVehicle` (ADR-006).

## Consequences

- Moving to ArduPilot SITL, and later to hardware, means writing an adapter rather than changing the stack.
- Controller parameter names mirror ArduPlane where an equivalent exists, so tunings can move between the two.
