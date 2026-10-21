"""``aeroadapt`` command line (spec §10.2).

aeroadapt run <scenario> [--out DIR]           one flight → replay bundle
aeroadapt campaign <campaign> [--out DIR]      seeded variations → outcomes + summary
aeroadapt validate [--tier fast|full]          the V1–V15 test tiers (pytest)
aeroadapt report [--out DIR]                   docs/validation/m2_report.md + figures
aeroadapt bench                                throughput benchmark (V15)
aeroadapt export-viz <bundle> [--to DIR]       copy a bundle's browser files for Skyview
aeroadapt fleet <id> --n N [--start K]         generate a fleet bundle (runs/fleets/<id>)
aeroadapt fleet-accept <fleet> [--n N]         V20: configure, tune and accept a fleet
aeroadapt fleet-sample <fleet> <v20>           the committed sample: V20-accepted, tuned
aeroadapt bench-run <spec> [--n N]             bench v0: B1/B2 over sampled scenarios (runs/bench)
aeroadapt export-fleet <bench> [--to DIR]      a bench's trajectories for Skyview's fleet view
aeroadapt serve [--scenario ID] [--port P]     live server for Skyview (?live=ws://localhost:P)
aeroadapt plan check <plan.yaml> [state …]     Txt2Fly: build and check a typed plan
aeroadapt plan fly <plan.yaml> [state …]       … and fly it in the simulator (replay bundle)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
SKYVIEW_REPLAYS = REPO / "web" / "skyview" / "public" / "replays"


def _run(args: argparse.Namespace) -> int:
    from aeroadapt.scenarios.runner import record, run_scenario
    from aeroadapt.scenarios.schema import load_scenario

    t0 = time.perf_counter()
    sc = load_scenario(args.scenario)
    result = run_scenario(sc)
    out = record(result, Path(args.out) if args.out else RUNS / sc.id)
    print(
        f"{sc.id}: {result.manifest['duration_s']:.1f} s flown in "
        f"{time.perf_counter() - t0:.1f} s → {out}"
    )
    for e in result.events:
        if e["type"] in (
            "fault_onset",
            "failure_detected",
            "zone_selected",
            "touchdown",
            "terminated",
            "airspeed_health",
        ):
            print(f"  {e['time_us'] * 1e-6:8.2f} s  {e['type']:<17} {json.dumps(e['payload'])}")
    return 0


def _campaign(args: argparse.Namespace) -> int:
    from aeroadapt.scenarios.campaign import load_campaign, run_campaign, write_outputs

    spec = load_campaign(args.campaign)
    t0 = time.perf_counter()
    results, _ = run_campaign(spec.scenarios())
    out = write_outputs(Path(args.out) if args.out else RUNS / spec.id, spec, results)
    summary = json.loads((out / "summary.json").read_text())
    print(f"{spec.id}: {spec.n} flights in {time.perf_counter() - t0:.1f} s → {out}")
    print(json.dumps(summary["summary"], indent=2))
    for f in summary["failures"]:
        print(f"  failure {f['scenario']}: {f['cause']}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    marker = ["-m", "not full"] if args.tier == "fast" else []
    return subprocess.call([sys.executable, "-m", "pytest", "-q", *marker], cwd=REPO)


def _report(args: argparse.Namespace) -> int:
    # validation/report.py lives outside the package: it imports the reference implementation.
    cmd = [sys.executable, "-m", "validation.report", *(["--out", args.out] if args.out else [])]
    return subprocess.call([*cmd, *(["--quick"] if args.quick else [])], cwd=REPO)


def _bench(_: argparse.Namespace) -> int:
    return subprocess.call([sys.executable, str(REPO / "tools" / "bench.py")], cwd=REPO)


def _export_viz(args: argparse.Namespace) -> int:
    bundle = Path(args.bundle)
    dest = Path(args.to) if args.to else SKYVIEW_REPLAYS / bundle.name
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "events.jsonl", "viz.arrow"):
        shutil.copy2(bundle / name, dest / name)
    # A fleet airframe's own model travels with its replay (Skyview loads ``viz_model``).
    manifest = json.loads((bundle / "manifest.json").read_text())
    fleet = manifest["scenario"].get("fleet")
    if fleet:
        from aeroadapt.airframes.fleet import fleet_path

        glb = fleet_path(fleet) / "gltf" / f"{manifest['scenario']['airframe']}.glb"
        if glb.exists():
            shutil.copy2(glb, dest / "model.glb")
            manifest["viz_model"] = "model.glb"
            (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    index = dest.parent / "index.json"
    entries = sorted(p.name for p in dest.parent.iterdir() if (p / "viz.arrow").exists())
    index.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"{bundle.name} → {dest} ({len(entries)} replays listed in {index.name})")
    return 0


def _fleet(args: argparse.Namespace) -> int:
    from aeroadapt.generator.fleet import build_fleet

    t0 = time.perf_counter()
    out = build_fleet(
        args.id, args.n, start=args.start, workers=args.workers, gltf=not args.no_gltf
    )
    manifest = json.loads((out / "manifest.json").read_text())
    print(
        f"{manifest['accepted']} accepted of {manifest['designs_tried']} designs in "
        f"{time.perf_counter() - t0:.0f} s → {out}"
    )
    return 0


def _fleet_accept(args: argparse.Namespace) -> int:
    from aeroadapt.airframes.fleet import load_fleet
    from aeroadapt.harness.fleet_v20 import run_fleet_v20, write_fleet_v20
    from aeroadapt.scenarios import load_site

    fleet_dir = Path(args.fleet)
    if not fleet_dir.exists():
        fleet_dir = Path("runs/fleets") / args.fleet
    airframes = load_fleet(fleet_dir)[: args.n]
    t0 = time.perf_counter()
    run = run_fleet_v20(airframes, load_site(args.site), batch=args.batch, seed=args.seed)
    meta = {"fleet": fleet_dir.name, "site": args.site, "seed": args.seed,
            "seconds": round(time.perf_counter() - t0)}  # fmt: skip
    out = write_fleet_v20(Path(args.out or f"runs/v20/{fleet_dir.name}"), run, meta)
    print(f"V20: {run.pass_rate:.1%} of {len(run.results)} airframes passed → {out}")
    return 0


def _fleet_sample(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    from aeroadapt.generator.fleet import subset_fleet
    from aeroadapt.harness.fleet_v20 import airframe_class

    v20 = Path(args.v20)

    def margin(r: dict) -> float:
        """The tightest check, as the fraction of its (upper) limit still unused."""
        return min(1 - v / lim for v, lim, _ in json.loads(r["checks"]).values() if lim > 0)

    by_class: dict[str, list[tuple[float, str]]] = {}
    for r in pq.read_table(v20 / "results.parquet").to_pylist():
        if r["passed"]:
            by_class.setdefault(airframe_class(r["airframe"]), []).append(
                (margin(r), r["airframe"])
            )
    # The airframes with the most margin: the sample re-passes V20 from scratch (full tier).
    ids = [
        i for c in sorted(by_class) for _, i in sorted(by_class[c], reverse=True)[: args.per_class]
    ]
    dst = Path(args.out)
    if dst.exists():
        shutil.rmtree(dst)
    subset_fleet(Path(args.fleet), ids, dst, stack_cfgs=v20 / "stack_cfgs.jsonl")
    print(f"{len(ids)} V20-accepted airframes with their tuned stacks → {dst}")
    return 0


def _bench_run(args: argparse.Namespace) -> int:
    from aeroadapt.bench.run import run_bench, summary, write_bench
    from aeroadapt.bench.sampler import load_bench_spec

    spec = load_bench_spec(args.spec)
    if args.n is not None:
        spec = spec.model_copy(update={"n": args.n})
    out = Path(args.out or f"runs/bench/{spec.id}")
    result = run_bench(spec, out_dir=out)
    write_bench(out, result)
    for b, s in summary(result)["baselines"].items():
        sv, mc = s["survival"], s["motor_cut"]["survival"]
        print(f"{b}: survival {sv['rate']:.1%} [{sv['lo']:.1%}, {sv['hi']:.1%}] of {sv['n']}; "
              f"motor cut {mc['rate']:.1%} of {mc['n']}")  # fmt: skip
    print(f"→ {out}")
    return 0


def _export_fleet(args: argparse.Namespace) -> int:
    import numpy as np
    import pyarrow.parquet as pq

    from aeroadapt.bench.run import TRAJ_FIELDS
    from aeroadapt.scenarios import load_site

    bench = Path(args.bench)
    spec = json.loads((bench / "summary.json").read_text())["spec"]
    rows = pq.read_table(bench / "outcomes.parquet").to_pylist()
    traj = np.load(bench / "trajectories.npz")["traj"].astype("<f4")
    dest = Path(args.to) if args.to else SKYVIEW_REPLAYS.parent / "fleet" / spec["id"]
    dest.mkdir(parents=True, exist_ok=True)
    keep = ("scenario", "baseline", "airframe_class", "faults", "survived", "touchdown",
            "in_zone", "planned_landing")  # fmt: skip
    meta = {
        "id": spec["id"],
        "description": spec.get("description", ""),
        "site": {"origin": load_site(spec["site"]).origin.model_dump(mode="json")},
        "dt_s": 1.0,
        "samples": int(traj.shape[1]),
        "fields": list(TRAJ_FIELDS),
        "flights": [{k: r[k] for k in keep} for r in rows],
    }
    (dest / "fleet.json").write_text(json.dumps(meta) + "\n")
    (dest / "fleet.bin").write_bytes(traj.tobytes())
    print(f"{len(rows)} flights x {traj.shape[1]} s → {dest} ({traj.nbytes / 1e6:.1f} MB)")
    return 0


def _plan_session(args: argparse.Namespace):
    """A live session at the state the options give (txt2fly-demo scenario)."""
    from aeroadapt.airframes.fleet import load_fleet_airframe
    from aeroadapt.live.server import SAMPLE_FLEET
    from aeroadapt.live.session import LiveSession
    from aeroadapt.scenarios.schema import Scenario, load_scenario
    from aeroadapt.stack.config import load_stack_cfg

    d = load_scenario(args.scenario).model_dump(mode="json")
    n, e = (float(x) for x in args.at.split(","))
    speed, frm = (float(x) for x in args.wind.split("@"))
    d["initial"].update(pos_ne_m=[n, e], alt_agl_m=args.alt, heading_deg=args.heading)
    d["environment"]["wind"].update(speed_mps=speed, from_deg=frm)
    d["mission"]["home_ne_m"] = [n, e]
    sc = Scenario.model_validate(d)
    if args.airframe in (None, sc.airframe):
        return LiveSession(sc)
    af = load_fleet_airframe(SAMPLE_FLEET, args.airframe)
    return LiveSession(sc, af, load_stack_cfg(SAMPLE_FLEET, args.airframe),
                       fleet=str(SAMPLE_FLEET.relative_to(REPO)))  # fmt: skip


def _plan(args: argparse.Namespace) -> int:
    from aeroadapt.live.session import CHUNK_S
    from aeroadapt.txt2fly.builder import build
    from aeroadapt.txt2fly.checker import check, describe
    from aeroadapt.txt2fly.gazetteer import load_gazetteer
    from aeroadapt.txt2fly.plan import load_plan

    plan = load_plan(args.plan)
    session = _plan_session(args)
    for _ in range(round(args.settle / CHUNK_S)):  # let the navigator settle first
        session.step()
    gaz = load_gazetteer(session.resolved.site)
    report = check(build(plan, session.aircraft(), gaz), gaz)
    print(describe(report))
    if args.action == "check":
        return 0 if report.ok else 1
    if not report.ok:
        return 1
    err, prop = session.propose(plan)
    if err is None and prop is not None:  # the CLI acknowledges the warnings it printed
        err = session.confirm(prop["id"], prop["warnings"])
    if err:
        print(f"not flown: {err}")
        return 1
    t_end = session.time_s + args.seconds
    while session.time_s < t_end - 1e-6 and not session.done:
        session.step()
    stem = Path(args.plan).stem
    out = session.record(Path(args.out) if args.out else RUNS / "txt2fly" / stem)
    print(f"flown for {args.seconds:g} s → {out} (aeroadapt export-viz {out} to view it)")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import asyncio

    from aeroadapt.live.server import run_server

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_server(args.scenario, port=args.port))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="aeroadapt", description=__doc__.splitlines()[0] if __doc__ else ""
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="fly one scenario and record a replay bundle")
    r.add_argument("scenario", help="scenario id (configs/scenarios) or path")
    r.add_argument("--out", help="bundle directory (default: runs/<id>)")
    r.set_defaults(func=_run)
    c = sub.add_parser("campaign", help="fly a campaign of seeded variations")
    c.add_argument("campaign", help="campaign id (configs/campaigns) or path")
    c.add_argument("--out")
    c.set_defaults(func=_campaign)
    v = sub.add_parser("validate", help="run the validation test tiers")
    v.add_argument("--tier", choices=("fast", "full"), default="fast")
    v.set_defaults(func=_validate)
    rep = sub.add_parser("report", help="build the validation report")
    rep.add_argument("--out")
    rep.add_argument("--quick", action="store_true")
    rep.set_defaults(func=_report)
    b = sub.add_parser("bench", help="throughput benchmark (V15)")
    b.set_defaults(func=_bench)
    e = sub.add_parser("export-viz", help="copy a bundle's browser files for Skyview")
    e.add_argument("bundle")
    e.add_argument("--to")
    e.set_defaults(func=_export_viz)
    f = sub.add_parser("fleet", help="generate a procedural airframe fleet (M2 spec §4)")
    f.add_argument("id")
    f.add_argument("--n", type=int, required=True, help="accepted airframes wanted")
    f.add_argument("--start", type=int, default=0, help="first per-class seed")
    f.add_argument("--workers", type=int, default=None)
    f.add_argument("--no-gltf", action="store_true")
    f.set_defaults(func=_fleet)
    fa = sub.add_parser("fleet-accept", help="V20: auto-configure and accept a fleet (§6)")
    fa.add_argument("fleet", help="fleet bundle directory or id (runs/fleets/<id>)")
    fa.add_argument("--n", type=int, default=None, help="first N airframes (default: all)")
    fa.add_argument("--batch", type=int, default=200)
    fa.add_argument("--site", default="cambridge_west")
    fa.add_argument("--seed", type=int, default=0)
    fa.add_argument("--out", help="output directory (default: runs/v20/<fleet>)")
    fa.set_defaults(func=_fleet_accept)
    fs = sub.add_parser("fleet-sample", help="the sample fleet: V20-accepted airframes, tuned")
    fs.add_argument("fleet", help="fleet bundle directory")
    fs.add_argument("v20", help="the fleet's V20 run directory (fleet-accept output)")
    fs.add_argument("--per-class", type=int, default=2)
    fs.add_argument("--out", default="configs/fleets/sample")
    fs.set_defaults(func=_fleet_sample)
    br = sub.add_parser("bench-run", help="bench v0: baselines over sampled scenarios (§8)")
    br.add_argument("spec", help="bench spec id (configs/bench) or path")
    br.add_argument("--n", type=int, default=None, help="override the number of scenarios")
    br.add_argument("--out", help="output directory (default: runs/bench/<id>)")
    br.set_defaults(func=_bench_run)
    ef = sub.add_parser("export-fleet", help="a bench run's trajectories for the fleet view")
    ef.add_argument("bench", help="bench run directory (runs/bench/<id>)")
    ef.add_argument("--to", help="destination (default: web/skyview/public/fleet/<id>)")
    ef.set_defaults(func=_export_fleet)
    sv = sub.add_parser("serve", help="live server: fly a scenario and inject faults from Skyview")
    sv.add_argument("--scenario", default="m1-nominal")
    sv.add_argument("--port", type=int, default=8765)
    sv.set_defaults(func=_serve)
    pl = sub.add_parser("plan", help="Txt2Fly: check a typed plan, or fly it (docs/06)")
    pl.add_argument("action", choices=("check", "fly"))
    pl.add_argument("plan", help="plan file (YAML)")
    pl.add_argument("--scenario", default="txt2fly-demo", help="site, airframe and weather")
    pl.add_argument("--airframe", help="a sample-fleet airframe id (default: the scenario's)")
    pl.add_argument("--at", default="-150,-200", help="start position north,east (m)")
    pl.add_argument("--alt", type=float, default=100.0, help="start height above ground (m)")
    pl.add_argument("--heading", type=float, default=0.0)
    pl.add_argument("--wind", default="6@250", help="speed@from: m/s and degrees")
    pl.add_argument("--settle", type=float, default=5.0, help="seconds flown before planning")
    pl.add_argument("--seconds", type=float, default=300.0, help="fly: seconds after confirming")
    pl.add_argument("--out", help="fly: bundle directory (default: runs/txt2fly/<plan>)")
    pl.set_defaults(func=_plan)
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
