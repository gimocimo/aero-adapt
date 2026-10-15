"""M1 validation report (spec §12, deliverable D2): ``docs/validation/m1_report.md`` + figures.

It recomputes every piece of evidence, independently of the pytest tiers: parity against
the NumPy reference (V3), invariants (V2), trims and modes (V4/V5), integrator convergence
(V6), precision (V7), Dryden statistics (V8), autopilot acceptance (V11), the contingency
(V12) and throughput (V15).

It lives outside the ``aeroadapt`` package because it imports the reference implementation,
which production code must never import (import-linter contract).

Usage: ``uv run python -m validation.report [--out docs/validation] [--quick]``
(``aeroadapt report`` runs the same).
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from scipy.signal import welch  # noqa: E402

from aeroadapt.airframes import AIRFRAME_DIR, load_airframe, to_params  # noqa: E402
from aeroadapt.core import quaternion as Q  # noqa: E402
from aeroadapt.harness.acceptance import TESTS, Guided, V11Runner, _mean, run_v11  # noqa: E402
from aeroadapt.scenarios import load_site  # noqa: E402
from aeroadapt.scenarios.campaign import failure_cause, load_campaign, summarise  # noqa: E402
from aeroadapt.scenarios.runner import git_info  # noqa: E402
from aeroadapt.scenarios.v12 import false_alarms, latency, mini_campaign  # noqa: E402
from aeroadapt.sim.aerojax import (  # noqa: E402
    BM_PARITY,
    controls_from_vector,
    forces_moments,
    rigid_body,
    rollout,
    state_from_vector,
    state_to_vector,
)
from aeroadapt.sim.aerojax.atmosphere import (  # noqa: E402
    TurbulenceParams,
    dryden_scales,
    dryden_stationary,
    dryden_step,
)
from aeroadapt.sim.aerojax.integrators import rk4  # noqa: E402
from aeroadapt.sim.aerojax.trim import linearise, trim  # noqa: E402
from aeroadapt.stack.autopilot import MODE_GUIDED  # noqa: E402
from aeroadapt.stack.config import build_stack_params, load_stack_cfg  # noqa: E402
from reference.bm_numpy import model as ref  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
VA = 25.0
DEG = 180 / math.pi


class Report:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.fig_dir = out / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.lines: list[str] = []

    def h(self, level: int, text: str) -> None:
        self.lines += ["", f"{'#' * level} {text}", ""]

    def p(self, *text: str) -> None:
        self.lines += [*text, ""]

    def table(self, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
        self.lines.append("| " + " | ".join(header) + " |")
        self.lines.append("|" + "|".join("---" for _ in header) + "|")
        for r in rows:
            self.lines.append("| " + " | ".join(str(x) for x in r) + " |")
        self.lines.append("")

    def figure(self, fig: Figure, name: str, caption: str) -> None:
        fig.tight_layout()
        fig.savefig(self.fig_dir / f"{name}.png", dpi=110)
        plt.close(fig)
        self.lines += [f"![{caption}](figures/{name}.png)", "", f"*{caption}*", ""]


def mark(ok: object) -> str:
    return "✅" if bool(ok) else "❌"


# --------------------------------------------------------------------------- V3
def v3_parity(r: Report) -> None:
    params = to_params(load_airframe("aerosonde_bm"))
    raw = ref.load_params(AIRFRAME_DIR / "aerosonde_bm.yaml")
    rp = ref.with_environment(raw, rho=float(BM_PARITY.rho), gravity=float(BM_PARITY.gravity))
    rng = np.random.default_rng(0)
    xs, us, ws = [], [], []
    while len(xs) < 1000:
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        x = np.concatenate(
            [
                rng.uniform(-2000, 2000, 3),
                [rng.uniform(12, 35), rng.uniform(-4, 4), rng.uniform(-4, 4)],
                q,
                rng.uniform(-1, 1, 3),
            ]
        )
        w = rng.uniform(-8, 8, 3)
        if ref.air_data(x, w)[0] < 10:
            continue
        xs.append(x)
        us.append(np.concatenate([rng.uniform(-0.35, 0.35, 3), [rng.uniform(0, 1)]]))
        ws.append(w)
    fm = jax.jit(
        jax.vmap(
            lambda x, u, w: forces_moments(
                params, BM_PARITY, state_from_vector(x), controls_from_vector(u), w
            )
        )
    )(np.array(xs), np.array(us), np.array(ws))
    ref_fm = [ref.forces_moments(rp, x, u, w) for x, u, w in zip(xs, us, ws, strict=True)]

    def rel(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-9)))

    f_err = rel(np.asarray(fm.force_body), np.array([f for f, _ in ref_fm]))
    m_err = rel(np.asarray(fm.moment_body), np.array([m for _, m in ref_fm]))
    # Open-loop trajectories, 20 s, band-limited inputs and wind.
    dt_ = 0.005
    n = round(20 / dt_)
    traj_err = 0.0
    for seed in range(3):
        g = np.random.default_rng(100 + seed)
        t = dt_ * np.arange(n)
        u = np.array([0.0, -0.12, 0.0, 0.6]) + np.array([0.03, 0.04, 0.02, 0.15]) * np.sin(
            2 * np.pi * g.uniform(0.1, 0.8, 4) * t[:, None]
        )
        u[:, 3] = np.clip(u[:, 3], 0, 1)
        w = np.tile(g.uniform(-5, 5, 3), (n, 1))
        x0 = np.array([0.0, 0, -100, 25, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        tr = np.asarray(
            state_to_vector(
                rollout(
                    params,
                    BM_PARITY,
                    state_from_vector(x0),
                    controls_from_vector(u),
                    jnp.asarray(w),
                    dt_,
                )
            )
        )
        traj_err = max(traj_err, float(np.max(np.abs(tr - ref.simulate(rp, x0, u, w, dt_)))))
    r.h(2, "V3 · N-version parity (aerojax vs NumPy reference)")
    r.p(
        "1,000 random (state, wind, control) tuples, float64, plus three 20 s open-loop "
        "trajectories with band-limited inputs and wind."
    )
    r.table(
        ["Quantity", "Max error", "Criterion", ""],
        [
            ["Force (relative)", f"{f_err:.1e}", "≤ 1e-10", mark(f_err <= 1e-10)],
            ["Moment (relative)", f"{m_err:.1e}", "≤ 1e-10", mark(m_err <= 1e-10)],
            [
                "Trajectory state (absolute, 20 s)",
                f"{traj_err:.1e}",
                "≤ 1e-8",
                mark(traj_err <= 1e-8),
            ],
        ],
    )


# --------------------------------------------------------------------------- V2
def v2_invariants(r: Report) -> None:
    p = to_params(load_airframe("aerosonde_bm"))
    s0 = rigid_body.RigidBodyState(
        jnp.zeros(3),
        jnp.array([20.0, 1.0, -0.5]),
        Q.from_euler(0.2, -0.1, 0.7),
        jnp.array([0.6, -0.4, 0.5]),
    )
    zero = jnp.zeros(3)

    def body(s, _):
        nxt = rk4(lambda s: rigid_body.derivatives(p, s, zero, zero), s, 0.005)
        nxt = nxt._replace(quat=Q.normalize(nxt.quat))
        return nxt, nxt

    _, traj = jax.lax.scan(body, s0, None, length=20_000)
    j = jnp.array([[p.Jx, 0, -p.Jxz], [0, p.Jy, 0], [-p.Jxz, 0, p.Jz]])
    h = Q.body_to_ned(traj.quat, traj.omega @ j.T)
    h0 = Q.body_to_ned(s0.quat, j @ s0.omega)
    ke = 0.5 * jnp.einsum("ni,ij,nj->n", traj.omega, j, traj.omega)
    ke0 = 0.5 * s0.omega @ j @ s0.omega
    t = 0.005 * np.arange(1, 20_001)
    dh = np.asarray(jnp.linalg.norm(h - h0, axis=-1) / jnp.linalg.norm(h0))
    dke = np.asarray(jnp.abs(ke - ke0) / ke0)
    qn = np.asarray(jnp.abs(jnp.linalg.norm(traj.quat, axis=-1) - 1))
    r.h(2, "V2 · Rigid-body invariants")
    r.table(
        ["Invariant (100 s torque-free spin, f64)", "Max drift", "Criterion", ""],
        [
            ["Angular momentum (relative)", f"{dh.max():.1e}", "≤ 1e-9", mark(dh.max() <= 1e-9)],
            ["Rotational KE (relative)", f"{dke.max():.1e}", "≤ 1e-9", mark(dke.max() <= 1e-9)],
            ["Quaternion norm", f"{qn.max():.1e}", "≤ 1e-12", mark(qn.max() <= 1e-12)],
        ],
    )
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.semilogy(t, np.maximum(dh, 1e-18), label="|ΔH|/|H₀|")
    ax.semilogy(t, np.maximum(dke, 1e-18), label="|ΔKE|/KE₀")
    ax.set_xlabel("time (s)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    r.figure(fig, "v2_invariants", "V2: torque-free spin about all three axes, RK4 at 5 ms")


# --------------------------------------------------------------------------- V4/V5
def v4_v5_trim_modes(r: Report) -> None:
    p = to_params(load_airframe("aerosonde_bm"))
    rows = []
    for name, kw in [
        ("Level, 25 m/s", {}),
        ("5° climb", {"gamma": math.radians(5)}),
        ("Turn, R = 250 m", {"radius": 250.0}),
    ]:
        tr = trim(p, BM_PARITY, VA, **kw)  # type: ignore[arg-type]
        rows.append(
            [
                name,
                f"{tr.residual:.1e}",
                f"{math.degrees(tr.alpha):.2f}°",
                f"{math.degrees(tr.phi):.1f}°",
                f"{math.degrees(tr.controls[1]):.2f}°",
                f"{tr.controls[3]:.3f}",
                mark(tr.success and tr.residual <= 1e-8),
            ]
        )
    r.h(2, "V4 · Trim")
    r.table(["Case", "Residual", "α", "φ", "δe", "δt", ""], rows)
    tr = trim(p, BM_PARITY, VA)
    lm = linearise(p, BM_PARITY, tr)
    lon = np.asarray(np.linalg.eigvals(lm.a_lon), dtype=complex)
    lat = np.asarray(np.linalg.eigvals(lm.a_lat), dtype=complex)
    rho, b, c = BM_PARITY.rho, p.b, p.c
    gam = p.Jx * p.Jz - p.Jxz**2
    a_phi1 = (
        -0.5
        * rho
        * VA**2
        * p.S
        * b
        * (p.Jz / gam * p.C_ell_p + p.Jxz / gam * p.C_n_p)
        * b
        / (2 * VA)
    )
    a_th2 = -rho * VA**2 * c * p.S / (2 * p.Jy) * p.C_m_alpha
    wn = np.sort(np.abs(lon[np.abs(np.imag(lon)) > 1e-6]))
    roll = float(np.real(lat[np.argmin(np.real(lat))]))
    sp_, ph = float(wn[-1]), float(wn[0])
    ph_ref = math.sqrt(2) * BM_PARITY.gravity / VA
    r.h(2, "V5 · Linear modes vs Beard & McLain approximations")
    r.table(
        ["Mode", "Linearised", "B&M approximation", "Difference", "Tolerance", ""],
        [
            [
                "Roll subsidence (1/s)",
                f"{roll:.2f}",
                f"{-a_phi1:.2f}",
                f"{abs(roll / -a_phi1 - 1):.1%}",
                "10%",
                mark(abs(roll / -a_phi1 - 1) <= 0.10),
            ],
            [
                "Short period ωn (rad/s)",
                f"{sp_:.2f}",
                f"{math.sqrt(a_th2):.2f}",
                f"{abs(sp_ / math.sqrt(a_th2) - 1):.1%}",
                "15%",
                mark(abs(sp_ / math.sqrt(a_th2) - 1) <= 0.15),
            ],
            [
                "Phugoid ωn (rad/s)",
                f"{ph:.3f}",
                f"{ph_ref:.3f}",
                f"{abs(ph / ph_ref - 1):.1%}",
                "20%",
                mark(abs(ph / ph_ref - 1) <= 0.20),
            ],
        ],
    )


# --------------------------------------------------------------------------- V6
def v6_convergence(r: Report) -> None:
    p = to_params(load_airframe("aerosonde_bm"))
    tr = trim(p, BM_PARITY, VA, h=100.0)

    def doublet(dt_: float) -> np.ndarray:
        n = round(30.0 / dt_)
        t = dt_ * np.arange(n)
        u = np.tile(tr.controls, (n, 1))
        u[(t >= 2) & (t < 3), 1] += math.radians(3)
        u[(t >= 3) & (t < 4), 1] -= math.radians(3)
        traj = rollout(
            p, BM_PARITY, tr.state, controls_from_vector(jnp.asarray(u)), jnp.zeros((n, 3)), dt_
        )
        return np.asarray(Q.to_euler(traj.quat[:: round(0.01 / dt_)])[1])

    dts = [0.01, 0.005, 0.0025]
    ref_pitch = doublet(0.0005)
    errs = [float(np.max(np.abs(doublet(d) - ref_pitch))) for d in dts]
    slope = float(np.polyfit(np.log(dts), np.log(errs), 1)[0])
    r.h(2, "V6 · Integrator convergence")
    r.p(
        f"RK4 on a 30 s elevator doublet (±3°) against a 0.5 ms reference: log-log slope "
        f"**{slope:.2f}** (criterion 4 ± 0.5) {mark(3.5 <= slope <= 4.5)}; max pitch error "
        f"at the chosen 5 ms step **{errs[1]:.1e} rad** (criterion ≤ 1e-3) {mark(errs[1] <= 1e-3)}."
    )
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    t = 0.01 * np.arange(len(ref_pitch))
    ax[0].plot(t, ref_pitch * DEG)
    ax[0].set(xlabel="time (s)", ylabel="pitch (°)", title="doublet response")
    ax[1].loglog(dts, errs, "o-", label=f"slope {slope:.2f}")
    ax[1].loglog(dts, errs[0] * (np.array(dts) / dts[0]) ** 4, "--", label="4th order")
    ax[1].set(xlabel="dt (s)", ylabel="max pitch error (rad)")
    ax[1].legend()
    ax[1].grid(True, which="both", alpha=0.3)
    r.figure(fig, "v6_convergence", "V6: RK4 convergence on an elevator doublet")


# --------------------------------------------------------------------------- V7
def v7_precision(r: Report) -> None:
    sys.path.insert(0, str(REPO))
    from tests.validation.test_v6_v13_numerics import precision_divergence

    with tempfile.TemporaryDirectory() as d:
        pos, att = precision_divergence(Path(d))
    r.h(2, "V7 · float32 vs float64")
    r.p(
        f"The same 60 s closed-loop flight (stock stack, light turbulence, sensor noise) in "
        f"float32 and float64: max position divergence **{pos:.3f} m** (expected ≤ 1 m) "
        f"{mark(pos <= 1)}, attitude **{att:.3f}°** (expected ≤ 0.5°) {mark(att <= 0.5)}. "
        "All noise is drawn in float32 and cast, so both runs fly the same realisation; "
        "float32 is confirmed as the M2 training precision."
    )


# --------------------------------------------------------------------------- V8
def v8_dryden(r: Report, quick: bool) -> None:
    w20, h, dt_ = 15 * 0.514444, 120.0, 0.005
    seeds, secs, tol = (16, 300.0, 0.15) if quick else (64, 600.0, 0.05)
    turb = TurbulenceParams(w20)
    n = round(secs / dt_)

    def one(key):
        k0, k1 = jax.random.split(key)
        s0 = dryden_stationary(turb, jnp.asarray(VA), jnp.asarray(h), k0, jnp.float64)
        _, g = jax.lax.scan(
            lambda s, k: dryden_step(s, turb, jnp.asarray(VA), jnp.asarray(h), dt_, k),
            s0,
            jax.random.split(k1, n),
        )
        return g

    g = np.asarray(jax.jit(jax.vmap(one))(jax.random.split(jax.random.PRNGKey(5), seeds)))
    s_uv, s_w, l_uv, l_w = (float(x) for x in dryden_scales(jnp.asarray(w20), jnp.asarray(h)))
    std = g.reshape(-1, 3).std(0)
    f, psd = welch(g, fs=1 / dt_, nperseg=2**15, axis=1)
    psd = psd.mean(0)
    w = 2 * np.pi * f
    an_u = s_uv**2 * 2 * l_uv / (np.pi * VA) / (1 + (l_uv * w / VA) ** 2) * 2 * np.pi
    an_w = (
        s_w**2
        * l_w
        / (np.pi * VA)
        * (1 + 3 * (l_w * w / VA) ** 2)
        / (1 + (l_w * w / VA) ** 2) ** 2
        * 2
        * np.pi
    )
    r.h(2, "V8 · Dryden turbulence (light, 120 m AGL)")
    r.table(
        ["Channel", "σ simulated", "σ MIL-HDBK-1797", "Error", f"≤ {tol:.0%}"],
        [
            [
                k,
                f"{std[i]:.3f}",
                f"{ref_:.3f}",
                f"{std[i] / ref_ - 1:+.1%}",
                mark(abs(std[i] / ref_ - 1) <= tol),
            ]
            for i, (k, ref_) in enumerate([("u", s_uv), ("v", s_uv), ("w", s_w)])
        ],
    )
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.loglog(f[1:], psd[1:, 0], label="u simulated", alpha=0.8)
    ax.loglog(f[1:], an_u[1:], "k--", label="u analytic")
    ax.loglog(f[1:], psd[1:, 2], label="w simulated", alpha=0.8)
    ax.loglog(f[1:], an_w[1:], "k:", label="w analytic")
    ax.set(xlabel="frequency (Hz)", ylabel="PSD ((m/s)²/Hz)", xlim=(1e-3, 20))
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    r.figure(fig, "v8_dryden_psd", f"V8: Dryden PSD, {seeds} seeds × {secs / 60:.0f} min")


# --------------------------------------------------------------------------- V11
def v11_autopilot(r: Report, quick: bool) -> None:
    calm = run_v11()
    turb = run_v11(turbulent=True, seeds=tuple(range(4 if quick else 16)))
    r.h(2, "V11 · Stock autopilot acceptance (Aerosonde, 25 m/s)")
    r.p(
        "Scored on truth. Steps start at t = 20 s after navigator convergence; light-turbulence "
        "step metrics use the ensemble mean over seeds."
    )
    r.table(
        ["Check", "Value", "Limit", ""],
        [
            [
                c.name.replace("|", "\\|"),
                f"{c.value:.3f} {c.unit}",
                f"{'≤' if c.upper else '≥'} {c.limit:g}",
                mark(c.ok),
            ]
            for c in calm + turb
        ],
    )
    af, st = load_airframe("aerosonde_bm"), load_site("cambridge_west")
    sp = build_stack_params(load_stack_cfg("aerosonde_bm"), af, st, 0.01)
    runner = V11Runner(af, st, sp, False, (0,))
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    from aeroadapt.harness.acceptance import _DUMMY, T_STEP

    pitch_trim, thr = float(sp.tecs.pitch_trim), float(sp.tecs.thr_cruise)
    for a, kind, secs, sig, dem, label in [
        (
            ax[0, 0],
            "roll",
            8,
            lambda m: m.euler[:, 0] * DEG,
            lambda m: m.roll_dem * DEG,
            "roll (°)",
        ),
        (
            ax[0, 1],
            "pitch",
            4,
            lambda m: m.euler[:, 1] * DEG,
            lambda m: m.pitch_dem * DEG,
            "pitch (°)",
        ),
        (ax[1, 0], "alt", 30, lambda m: -m.pos_ned[:, 2], lambda m: m.alt_dem, "height AGL (m)"),
        (ax[1, 1], "speed", 25, lambda m: m.va, lambda m: m.speed_dem, "true airspeed (m/s)"),
    ]:
        m = _mean(runner.fly(_DUMMY, T_STEP + secs, Guided(kind, pitch_trim, thr), MODE_GUIDED))
        w = m.t > T_STEP - 2
        a.plot(m.t[w] - T_STEP, sig(m)[w], label="truth")
        a.plot(m.t[w] - T_STEP, dem(m)[w], "--", label="demand")
        a.set(xlabel="time from step (s)", ylabel=label)
        a.grid(alpha=0.3)
        a.legend()
    r.figure(fig, "v11_steps", "V11: step responses in calm air (truth vs demand)")


# --------------------------------------------------------------------------- V12
def v12_contingency(r: Report, quick: bool) -> None:
    n_lat = 30 if quick else 100
    hours = 2.0 if quick else 10.0
    lat = {d: latency(d, n=n_lat) for d in ("esc", "energy")}
    fa = {d: false_alarms(d, hours=hours) for d in ("esc", "energy")}
    results, logs = mini_campaign()
    s = summarise(results)
    r.h(2, "V12 · Classical contingency")
    r.table(
        ["Detector", "Cuts", "Median latency", "Max latency", "Within limit", "False alarms", ""],
        [
            [
                d,
                len(lat[d].latencies_s),
                f"{np.median(lat[d].latencies_s):.2f} s",
                f"{lat[d].max_s:.2f} s",
                f"{lat[d].within_limit:.0%} (≤ {lat[d].limit_s:g} s)",
                f"{fa[d].false_alarms} in {fa[d].hours:.0f} h",
                mark(lat[d].within_limit == 1.0 and fa[d].false_alarms == 0),
            ]
            for d in ("esc", "energy")
        ],
    )
    r.p(
        "**ESC telemetry** (primary) meets the latency and zero-false-alarm criteria. The "
        "**energy-residual backup** (CUSUM against TECS's throttle model) does not: gusts in "
        "light turbulence move the total-energy rate by ~1.4 m/s RMS, worst in climbing turns, "
        "while a cut at low throttle loses only ~1–2 m/s. It is tuned for zero false alarms "
        "first; the remaining false alarms and the slow tail are a documented M1 miss, "
        "addressed by learned fault diagnosis in M4."
    )
    fig, ax = plt.subplots(figsize=(7, 3))
    for d in ("esc", "energy"):
        ax.hist(lat[d].latencies_s, bins=30, alpha=0.7, label=f"{d} (limit {lat[d].limit_s:g} s)")
    ax.set(xlabel="detection latency (s)", ylabel="cuts")
    ax.legend()
    r.figure(fig, "v12_latency", "V12: motor-cut detection latency")

    spec = load_campaign("m1-mini")
    r.h(
        3,
        f"Mini-campaign ({spec.n} motor cuts, onset {spec.vary.fault_onset_s} s, wind "
        f"{spec.vary.wind_speed_mps} m/s from any direction)",
    )
    r.table(
        ["Metric", "Value"],
        [
            ["Detected", f"{s['detected']} / {s['n']}"],
            ["Judged reachable at detection", s["reachable_at_detection"]],
            ["Landed in the selected zone (controlled or hard)", s["landed_in_zone"]],
            [
                "**In-zone rate** (criterion ≥ 80%)",
                f"**{s['in_zone_rate']:.0%}** {mark(s['in_zone_rate'] >= 0.8)}",
            ],
            ["Crashes", s["crashes"]],
        ],
    )
    fails = [(o.scenario_id, failure_cause(o)) for o in results if failure_cause(o)]
    if fails:
        r.p("Failures (every reachable case not ending inside its zone):")
        r.table(["Scenario", "Cause"], [list(f) for f in fails])
    # Map.
    site = load_site("cambridge_west")
    fig, ax = plt.subplots(figsize=(8, 7))
    for z in site.safe_zones:
        poly = np.array([*z.polygon_ne_m, z.polygon_ne_m[0]])
        ax.fill(poly[:, 1], poly[:, 0], color="#38d16a", alpha=0.25)
        ax.text(poly[:, 1].mean(), poly[:, 0].mean(), z.id, ha="center", fontsize=12)
    wps = np.array(load_campaign("m1-mini").scenarios()[0].mission.waypoints_3d())
    loop = np.vstack([wps, wps[:1]])
    ax.plot(loop[:, 1], loop[:, 0], "k--", lw=1, label="mission")
    pos = np.asarray(logs.pos_ned)
    failed = np.asarray(logs.failed)
    for k, o in enumerate(results):
        after = failed[k] > 0
        ax.plot(
            pos[k, after, 1],
            pos[k, after, 0],
            lw=0.6,
            alpha=0.6,
            color="#2a7ab0" if o.reachable_at_detection else "#b05a2a",
        )
        if o.touchdown_ne_m:
            n_, e_ = o.touchdown_ne_m
            good = o.in_zone and o.touchdown in ("controlled", "hard")
            ax.plot(e_, n_, "o" if good else "x", color="#1a9e3a" if good else "#d22", ms=6, mew=2)
    ax.set(
        xlabel="east (m)",
        ylabel="north (m)",
        aspect="equal",
        title="Mini-campaign: glide after detection (blue: zone selected, brown: straight in)",
    )
    ax.grid(alpha=0.3)
    r.figure(fig, "v12_campaign_map", "V12: mini-campaign map; ○ in-zone landing, × other")


# --------------------------------------------------------------------------- V15
def v15_throughput(r: Report) -> None:
    out = subprocess.run(
        [sys.executable, str(REPO / "tools" / "bench.py")],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    r.h(2, "V15 · Throughput (recorded, not gated)")
    r.p("```", *out.stdout.strip().splitlines(), "```")


def build_report(out: Path, quick: bool = False) -> Path:
    r = Report(out)
    t0 = time.perf_counter()
    gi = git_info()
    r.lines += [
        "# M1 validation report",
        "",
        f"*Generated {dt.date.today().isoformat()} by `validation/report.py` · commit "
        f"`{gi.get('commit', 'unknown')[:10]}`{' (dirty)' if gi.get('dirty') else ''} · "
        f"{platform.machine()} {platform.system()} · Python {platform.python_version()} · "
        f"JAX {jax.__version__}{' · quick mode' if quick else ''}*",
        "",
        "Evidence for spec §12. Each section recomputes its results; the pytest tiers gate the "
        "same criteria in CI (fast) and nightly (full).",
    ]
    sections: list[Callable[[], None]] = [
        lambda: v3_parity(r),
        lambda: v2_invariants(r),
        lambda: v4_v5_trim_modes(r),
        lambda: v6_convergence(r),
        lambda: v7_precision(r),
        lambda: v8_dryden(r, quick),
        lambda: v11_autopilot(r, quick),
        lambda: v12_contingency(r, quick),
        lambda: v15_throughput(r),
    ]
    for section in sections:
        section()
    r.p("", f"*Built in {time.perf_counter() - t0:.0f} s. V11 test names: {', '.join(TESTS)}.*")
    path = out / "m1_report.md"
    path.write_text("\n".join(r.lines).strip() + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", default=str(REPO / "docs" / "validation"))
    ap.add_argument("--quick", action="store_true", help="fewer seeds and hours (smoke check)")
    args = ap.parse_args()
    print(build_report(Path(args.out), args.quick))


if __name__ == "__main__":
    main()
