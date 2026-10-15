"""``aeroadapt`` command line (spec §10.2).

aeroadapt run <scenario> [--out DIR]           one flight → replay bundle
aeroadapt campaign <campaign> [--out DIR]      seeded variations → outcomes + summary
aeroadapt validate [--tier fast|full]          the V1–V15 test tiers (pytest)
aeroadapt report [--out DIR]                   docs/validation/m1_report.md + figures
aeroadapt bench                                throughput benchmark (V15)
aeroadapt export-viz <bundle> [--to DIR]       copy a bundle's browser files for Skyview
"""

from __future__ import annotations

import argparse
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
    rep = sub.add_parser("report", help="build the M1 validation report")
    rep.add_argument("--out")
    rep.add_argument("--quick", action="store_true")
    rep.set_defaults(func=_report)
    b = sub.add_parser("bench", help="throughput benchmark (V15)")
    b.set_defaults(func=_bench)
    e = sub.add_parser("export-viz", help="copy a bundle's browser files for Skyview")
    e.add_argument("bundle")
    e.add_argument("--to")
    e.set_defaults(func=_export_viz)
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
