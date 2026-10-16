# Ten thousand aircraft that never existed

*Draft for the M2 post (doc 03: every milestone ends with a public write-up). Numbers are from the
M2 runs (docs/05 §14, week 8).*

Most autopilot work is done on one aircraft. You tune the gains, fly the acceptance tests, and ship.
AeroAdapt's goal is a pilot that copes with aircraft it has never flown and with failures nobody
scripted. You can't measure that on one airframe, so milestone 2 built a fleet.

## A fleet from first principles

The generator draws a design from one of five classes (conventional, T-tail, V-tail, twin-boom
pusher and motor glider), then builds it the way an engineer would:

- **Geometry and aerodynamics:** wing and tail geometry from tail-volume coefficients, then the
  aerodynamics from AeroSandbox's component buildup with NeuralFoil section polars.
- **Mass and propulsion:** a component mass budget that places the battery to hit a static margin,
  and a sized propulsion system.
- **A model for the viewer:** a glTF of the same geometry.

Every candidate is then trimmed and linearised in our JAX simulator, all at once in one batched
call. It is rejected if it can't trim, if its dynamic modes are unstable or poorly damped, or if it
lacks control authority or directional stability. 10,000 aircraft survive, out of 14,784 designs.

The rejections are informative. More than half of all glider designs (57 %) fail on directional
stability: real sailplanes have small fins, and so do ours. About a quarter of the conventional and
T-tail designs fail the same check.

## Flying an aircraft you have never seen

A generated airframe has no hand-tuned parameter file, so the stock autopilot must configure
itself.

1. **Physics first:** speeds come from the polar; trim schedules from batched trims; loop gains from
   the stability derivatives.
2. **Then tuning, as ArduPlane's AUTOTUNE does in flight, but offline:** every airframe flies a
   doublet per axis with a grid of candidate gains, all batched together. The tuner keeps the
   candidate with the largest worst-case margin against the acceptance requirements, measured on
   two independent noise samples.

Then each aircraft sits a reduced version of the M1 acceptance test (V20): steps on every axis,
limit cycles, and line tracking in calm air and in turbulence.

855 of 1,000 pass, short of our 90 % target. The rest miss by 6–18 % on one requirement, most
often limit cycles from sloppy servos, roll steady-state error, sideslip in turns where adverse
yaw is strong, or speed sag when under-powered aircraft climb. They are excluded from the
benchmark with their reasons; the target stands.

The most useful thing V20 found wasn't a tuning problem. Each aircraft's turbulence was seeded by
its position in the batch, so the same aircraft met different gusts depending on its neighbours.
The unchanged tuner then "lost" 14 % of its passing aircraft. Once the noise was tied to the
aircraft's identity, results became reproducible, and tuning could be compared fairly.

## Breaking things on purpose

faultlab v1 has 30 kinds of failure:

- **Sensors:** biased, stuck or drifting airspeed; gyro and accelerometer bias; vibration; compass
  interference; barometer drift; GNSS dropout, drift and spoofing.
- **Control surfaces:** jammed, hard-over, floating, slow, reduced-throw, backlash or reversed.
- **Propulsion:** motor cut, thrust loss, propeller damage, thermal derate.
- **Battery:** cell failure, sag, capacity loss.
- **Airframe:** CG shift, payload drop, wing damage.

Weather adds gusts, wind shear and thermals.

Five pairs are deliberately confounded, because they look the same to a simple monitor:
- a blocked pitot and a sharp tailwind gust;
- GNSS spoofing and a change in wind;
- battery sag and a damaged propeller;
- a CG shift and an elevator rigging error;
- a floating rudder and turbulence.

We checked that the stock stack's monitors really can't tell them apart. Making the gust case
match was instructive: a slow gust is no confound at all, because the aircraft simply accelerates
with it.

## Live

`aeroadapt serve` flies the same compiled simulation the benchmarks use, paced to wall-clock time,
and streams it to Skyview over WebSocket. From the browser you can:

- inject any of the 30 faults, and clear a sensor fault;
- change the wind;
- switch to any aircraft from the fleet.

Every session is recorded, and its command log re-flies it offline byte for byte.

## The benchmark

Bench v0 samples 1,000 scenarios on the test split:

- a random mission over Cambridge;
- wind up to 80 % of cruise speed, with turbulence that follows it;
- gusts, shear and thermals;
- none, one or two failures of any kind.

The stock stack (B1) and the stock stack with the classical motor-failure contingency (B2) fly
every scenario with identical noise.

The classical contingency does its one job: it survives **94 % of motor cuts against 12 %** for
the stock stack alone. Overall survival moves from 83 % to 86 %, because motor cuts are only a
fraction of what goes wrong. Neither baseline survives more than two in three actuator failures.
The stock airspeed check cries wolf on 14 % of fault-free flights, mostly in strong wind. And a partial
power loss fools the motor-failure detector into a glide that sometimes ends badly. Those
failures are the next milestones' targets.

## What this is and isn't

- Everything here is simulation. The aerodynamics are a component buildup, good to about the 10–25 %
  that published comparisons suggest, not wind-tunnel data.
- The ground is flat; the photorealistic map is only a backdrop.
- The autopilot is our reimplementation of ArduPlane's cruise-flight laws, not ArduPlane.

The point of M2 is the instrument: a fleet, faults and a benchmark against which a learned pilot
(M3) can be measured honestly.

On a laptop (Apple M2 Pro), the simulator runs 0.85 million control steps a second for a
heterogeneous fleet with faults; the whole 2,000-flight benchmark takes under 13 minutes.
