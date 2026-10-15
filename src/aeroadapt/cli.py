"""``aeroadapt`` command line (spec §10.2).

aeroadapt run <scenario> [--out DIR]           one flight → replay bundle
aeroadapt campaign <campaign> [--out DIR]      seeded variations → outcomes + summary
aeroadapt validate [--tier fast|full]          the V1–V15 test tiers (pytest)
aeroadapt report [--out DIR]                   docs/validation/m2_report.md + figures
aeroadapt bench                                throughput benchmark (V15)
aeroadapt export-viz <bundle> [--to DIR]       copy a bundle's browser files for Skyview
aeroadapt fleet <id> --n N [--start K]         generate a fleet bundle (runs/fleets/<id>)
aeroadapt fleet-accept <fleet> [--n N]         V20: configure, tune and accept a fleet
aeroadapt serve [--scenario ID] [--port P]     live server for Skyview (?live=ws://localhost:P)
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
    from aeroadapt.generator.fleet import load_fleet
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
    sv = sub.add_parser("serve", help="live server: fly a scenario and inject faults from Skyview")
    sv.add_argument("--scenario", default="m1-nominal")
    sv.add_argument("--port", type=int, default=8765)
    sv.set_defaults(func=_serve)
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
