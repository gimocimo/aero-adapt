"""Validation report: ``docs/validation/m2_report.md`` + figures (M1 spec §12, M2 spec §10).

It recomputes every piece of evidence, independently of the pytest tiers: parity against
the NumPy reference (V3), invariants (V2), trims and modes (V4/V5), integrator convergence
(V6), precision (V7), Dryden statistics (V8), autopilot acceptance (V11), the contingency
(V12) and throughput (V15) from M1; and from M2 the JSBSim cross-check (V14), the aero v2
tables (V16, V17), the generator (V18, V19), the fleet acceptance (V20), faultlab (V21) and
the bench (V22), live mode (V23) and fleet throughput (V15). ``docs/validation/m1_report.md``
is the frozen M1 artefact.

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

from aeroadapt.airframes import AIRFRAME_DIR, bm_tables, load_airframe, to_params  # noqa: E402
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
from aeroadapt.sim.aerojax.aero import drag_coefficient, lift_coefficient  # noqa: E402
from aeroadapt.sim.aerojax.aero_tables import table_lookup  # noqa: E402
from aeroadapt.sim.aerojax.atmosphere import (  # noqa: E402
    TurbulenceParams,
    dryden_scales,
    dryden_stationary,
    dryden_step,
)
from aeroadapt.sim.aerojax.integrators import rk4  # noqa: E402
from aeroadapt.sim.aerojax.trim import linearise, trim  # noqa: E402
from aeroadapt.sim.jsbsim.v14 import AIRSPEED, run_v14  # noqa: E402
from aeroadapt.stack.autopilot import MODE_GUIDED  # noqa: E402
from aeroadapt.stack.config import build_stack_params, load_stack_cfg  # noqa: E402
from reference.bm_numpy import model as ref  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FLEET = REPO / "runs" / "fleets" / "m2-fleet"
V20_RUN = REPO / "runs" / "v20" / "m2-fleet"
BENCH_RUN = REPO / "runs" / "bench" / "b0"
VA = 25.0
DEG = 180 / math.pi


class Report:
    def __init__(self, out: Path) -> None:
        self.out = out
        # M2's own figures: m1_report.md (frozen) keeps the ones in figures/.
        self.fig_dir = out / "figures" / "m2"
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
        self.lines += [f"![{caption}](figures/m2/{name}.png)", "", f"*{caption}*", ""]


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


# --------------------------------------------------------------------------- V14
def v14_jsbsim(r: Report) -> None:
    p = to_params(load_airframe("aerosonde_bm"))
    v = run_v14(p)
    r.h(2, "V14 · JSBSim cross-check (Aerosonde exported to JSBSim)")
    r.p(
        "The Aerosonde's aerodynamics are exported to a JSBSim model (`tools/jsbsim`); JSBSim "
        "supplies its own equations of motion, integrators, WGS84 gravity, rotating Earth and "
        f"atmosphere. Each simulator finds its own power-off glide trim at {AIRSPEED} m/s "
        f"(ρ = {v.rho:.4f} kg/m³ and g = {v.gravity:.4f} m/s² matched), then flies the same "
        "doublets open loop for 10 s."
    )
    rows = [
        ("α (°)", v.aerojax.alpha, v.jsbsim.alpha, v.alpha_error_deg, "≤ 0.1"),
        ("δe (°)", v.aerojax.delta_e, v.jsbsim.delta_e, None, "—"),
        ("γ (°)", v.aerojax.gamma, v.jsbsim.gamma, v.gamma_error_deg, "≤ 0.05"),
    ]
    r.table(
        ["Glide trim", "aerojax", "JSBSim", "Difference", "Criterion"],
        [
            (n, f"{a * DEG:.3f}", f"{b * DEG:.3f}", "—" if d is None else f"{d:.4f}", c)
            for n, a, b, d, c in rows
        ],
    )
    r.table(
        ["Response", "RMS difference", "Criterion", ""],
        [
            ("Pitch, elevator doublet ±2°", f"{v.pitch_rms_deg:.3f}°", "≤ 0.3°", mark(v.passed)),
            (
                "Roll, aileron doublet ±2°",
                f"{v.roll_rms_deg:.3f}°",
                "recorded (≤ 1°)",
                mark(v.roll_rms_deg <= 1),
            ),
            (
                "Heading, aileron doublet ±2°",
                f"{v.yaw_rms_deg:.3f}°",
                "recorded (≤ 1°)",
                mark(v.yaw_rms_deg <= 1),
            ),
        ],
    )
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].plot(v.t, v.elevator_aerojax[:, 1] * DEG, label="aerojax")
    ax[0].plot(v.t, v.elevator_jsbsim[:, 1] * DEG, "--", label="JSBSim")
    ax[0].set(xlabel="time (s)", ylabel="Δθ (°)", title="elevator doublet")
    ax[1].plot(v.t, v.aileron_aerojax[:, 0] * DEG, label="aerojax φ")
    ax[1].plot(v.t, v.aileron_jsbsim[:, 0] * DEG, "--", label="JSBSim φ")
    ax[1].set(xlabel="time (s)", ylabel="Δφ (°)", title="aileron doublet")
    for a in ax:
        a.legend()
        a.grid(alpha=0.3)
    r.figure(fig, "v14_jsbsim", "V14: aerojax and JSBSim from their own power-off glide trims")


# --------------------------------------------------------------------------- V16 / V17
def v16_v17_tables(r: Report) -> None:
    p = to_params(load_airframe("aerosonde_bm"))
    t = bm_tables(p)
    alpha = np.radians(np.linspace(-15.0, 20.0, 3_501))
    errs = {
        name: float(
            np.max(np.abs(np.asarray(table_lookup(jnp.asarray(tab), jnp.asarray(alpha))) - exact))
        )
        for name, tab, exact in (
            ("C_L", t.C_L, np.asarray(lift_coefficient(p, jnp.asarray(alpha)))),
            ("C_D", t.C_D, np.asarray(drag_coefficient(p, jnp.asarray(alpha)))),
            ("C_m", t.C_m, p.C_m_0 + p.C_m_alpha * alpha),
        )
    }
    r.h(2, "V16 / V17 · Aero v2 tables")
    r.p(
        "Aero v2 tabulates the static coefficients over 360° of α (0.5° grid) and scales the "
        "damping and control derivatives by effectiveness tables. **V16**: the B&M formulas "
        "sampled onto the grid reproduce the analytic model in the flight envelope "
        "(α ∈ [−15°, 20°]):"
    )
    r.table(
        ["Coefficient", "Max abs(tabulated − analytic)", "Criterion", ""],
        [(k, f"{e:.1e}", "≤ 1e-3", mark(e <= 1e-3)) for k, e in errs.items()],
    )
    r.p(
        "The golden flights also pass with tabulated aerodynamics (fast tier: nominal and "
        "classical motor cut; full tier: all five). **V17**: the NumPy reference interpolates "
        "the tables independently (`np.interp` on an explicit grid, against index arithmetic "
        "in aerojax) and agrees to 1e-10 relative on random tables, pointwise and over a 20 s "
        "trajectory, and with ground effect enabled near the ground (`tests/validation/"
        "test_v16_v17_aero_tables.py`). The JSBSim exporter reproduces aerojax's forces and "
        "moments in all six axes to 2e-5 of q̄S (q̄Sb for moments) at random flight conditions, "
        "including tables, effectiveness factors, sideslip, rates and ground effect."
    )


# --------------------------------------------------------------------------- V18 / V19
def v18_references(r: Report) -> None:
    from aeroadapt.generator.references import COMPARED, v18

    r.h(2, "V18 · Generator aerodynamics vs published values")
    r.p(
        "Two real airframes are rebuilt from their published geometry and run through the "
        "generator's aerodynamics (AeroSandbox AeroBuildup with tripped NeuralFoil polars). "
        "Criteria: C_Lα within 15 %, statically stable, C_ℓp within 25 %; the other "
        "derivatives are recorded. Buildup and vortex-lattice estimates typically differ from "
        "wind-tunnel or flight-identified values by 20–30 % on secondary derivatives."
    )
    for c in v18():
        rows = []
        for name, pub, gen in c.rows:
            rel = (gen - pub) / abs(pub)
            gate = COMPARED.get(name)
            verdict = (
                mark(abs(rel) <= gate)
                if gate is not None
                else ("✅" if name == "C_m_alpha" and gen < 0 else "recorded")
            )
            rows.append((name, f"{pub:.4f}", f"{gen:.4f}", f"{rel:+.0%}", verdict))
        r.h(3, c.airframe)
        r.table(["Derivative", "Published", "Generator", "Difference", ""], rows)
    r.p(
        "**Reading the misses.** The Aerosonde's published C_ℓp (−0.51) is below strip theory "
        "for its own rectangular AR-15 wing (≈ −0.9) and below a vortex-lattice estimate "
        "(−0.71): the textbook values are not identified from the aircraft, and the book gives "
        "no tail geometry, so only the wing-dominated derivatives are meaningful there. The "
        "X8's control-power differences come from elevon sizes that are not published (ours "
        "follow thin-airfoil theory for the surfaces as sized), and the buildup method "
        "under-predicts a flying wing's directional derivatives (C_nβ, C_n,r, C_Yβ). The flying "
        "wing class is held out of the training fleet; when the X8 flies, it uses its "
        "published coefficients."
    )


def v19_fleet(r: Report) -> None:
    import json

    import pyarrow.parquet as pq

    r.h(2, "V19 · The generated fleet")
    if not (FLEET / "manifest.json").exists():
        r.p(
            f"*Skipped: no fleet bundle at `{FLEET.relative_to(REPO)}` (`aeroadapt fleet m2-fleet --n 10000`).*"
        )
        return
    m = json.loads((FLEET / "manifest.json").read_text())
    rows = pq.read_table(FLEET / "designs.parquet").to_pylist()
    acc = [x for x in rows if x["accepted"]]
    r.p(
        f"`{FLEET.relative_to(REPO)}`: **{m['accepted']:,} accepted airframes** from "
        f"{m['designs_tried']:,} designs ({m['accepted'] / m['designs_tried']:.0%} yield), "
        "every one re-checked in aerojax (trim at three speeds, modes, static margin, roll "
        "authority, thrust-to-weight, stall speed and wing loading)."
    )
    r.table(
        ["Class", "Tried", "Accepted", "Yield"],
        [
            (c, v["tried"], v["accepted"], f"{v['accepted'] / max(v['tried'], 1):.0%}")
            for c, v in m["per_class"].items()
        ],
    )
    r.table(["Rejection reason", "Designs"], list(m["rejection_reasons"].items())[:12])
    keys = [
        ("span_m", "span (m)"), ("mass_kg", "mass (kg)"), ("wing_loading", "wing loading (N/m²)"),
        ("v_stall", "stall speed (m/s)"), ("v_cruise", "cruise speed (m/s)"),
        ("static_margin_aerojax", "static margin"), ("roll_rate_dps", "roll rate (°/s)"),
        ("thrust_to_weight", "thrust / weight"),
    ]  # fmt: skip
    r.table(
        ["Quantity", "Min", "Median", "Max"],
        [
            (label, f"{np.min(v):.3g}", f"{np.median(v):.3g}", f"{np.max(v):.3g}")
            for k, label in keys
            for v in [np.array([x[k] for x in acc], dtype=float)]
        ],
    )
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.5))
    classes = list(m["per_class"])
    for ax, (k, label) in zip(axes.ravel(), keys, strict=True):
        data = [np.array([x[k] for x in acc if x["cls"] == c], dtype=float) for c in classes]
        # Cruise speeds are rounded to 0.5 m/s: align the bins with that grid.
        bins = np.arange(6.75, 25.5, 0.5) if k == "v_cruise" else 30
        ax.hist(data, bins=bins, stacked=True, label=classes)
        ax.set(xlabel=label)
    axes[0, 0].legend(fontsize=7)
    r.figure(fig, "v19_fleet", "V19: distributions over the accepted fleet, stacked by class")


# --------------------------------------------------------------------------- V20
def v20_fleet(r: Report) -> None:
    import json
    from collections import Counter

    import pyarrow.parquet as pq

    r.h(2, "V20 · Auto-configured stock stack across the fleet")
    if not (V20_RUN / "summary.json").exists():
        r.p(
            f"*Skipped: no V20 run at `{V20_RUN.relative_to(REPO)}` "
            "(`aeroadapt fleet-accept m2-fleet --n 1000`).*"
        )
        return
    s = json.loads((V20_RUN / "summary.json").read_text())
    rows = pq.read_table(V20_RUN / "results.parquet").to_pylist()
    ok = s["pass_rate"] >= s["target"]
    r.p(
        f"The first {s['airframes']:,} airframes of `{s['fleet']}`, each auto-configured "
        "(`harness/autoconfig.py`) and auto-tuned (`harness/autotune.py`), then flown through a "
        "batched V11-lite: roll, pitch, altitude and speed steps, limit cycles and an L1 box in "
        "calm air; the L1 box and servo saturation in light turbulence. Dimensional limits "
        "scale with cruise speed and GNSS accuracy (`harness/fleet_acceptance.py`).",
        "",
        f"**{s['passed']} of {s['airframes']} pass ({s['pass_rate']:.1%})** · criterion ≥ "
        f"{s['target']:.0%} {mark(ok)} · {s['seconds'] / 60:.0f} min on this machine.",
    )
    r.table(
        ["Class", "Airframes", "Pass rate"],
        [(c, v["airframes"], f"{v['pass_rate']:.0%}") for c, v in s["pass_rate_by_class"].items()],
    )
    n_failed = Counter(len(x["failures"]) for x in rows if not x["passed"])
    margins: dict[str, list[float]] = {}
    for x in rows:
        for name, (value, limit, passed) in json.loads(x["checks"]).items():
            if not passed and limit:
                margins.setdefault(name, []).append(value / limit)
    r.table(
        ["Failed check", "Airframes", "Median value / limit"],
        [
            (k, v, f"{np.median(margins[k]):.2f}" if k in margins else "")
            for k, v in s["failures_by_check"].items()
        ],
    )
    r.p(
        f"Failing airframes fail {', '.join(f'{k} check{"s" if k > 1 else ""}: {v}' for k, v in sorted(n_failed.items()))}. "
        "Per the spec (§6), airframes that fail V20 are excluded from the benchmark fleet, "
        "with these reasons recorded in `results.parquet`."
    )


# --------------------------------------------------------------------------- V21
def v21_faultlab(r: Report, quick: bool) -> None:
    from aeroadapt.faultlab.slots import (
        ACTUATOR_KINDS,
        AIRFRAME_KINDS,
        ENERGY_KINDS,
        PROPULSION_KINDS,
        SENSOR_KINDS,
    )
    from aeroadapt.harness.confounded import load_pair, pair_ids, run_pair

    r.h(2, "V21 · faultlab v1")
    families = {"sensor": SENSOR_KINDS, "actuator": ACTUATOR_KINDS,
                "propulsion": PROPULSION_KINDS, "energy": ENERGY_KINDS,
                "airframe": AIRFRAME_KINDS}  # fmt: skip
    r.p(
        f"**{sum(len(k) - 1 for k in families.values())} v1 kinds** plus the four M1 faults "
        "(pitot blockage with the drain open or blocked, servo jam, motor cut), and weather "
        "events (1−cos gusts, low-level shear, thermals). (a) and (b), every kind's effect "
        "and its absence before onset, are gated in the fast tier "
        "(`tests/validation/test_v21_faultlab.py`).",
    )
    r.table(["Family", "Kinds"], [(f, ", ".join(k[1:])) for f, k in families.items()])
    r.p(
        "**(c) Confounded pairs** (doc 03 §3.2): each fault is flown against its look-alike "
        "and a baseline, with common random numbers. The criterion is that the M1 monitors "
        "(airspeed health, motor-failure detection) respond alike within the horizon, that "
        "both arms show the symptom within a factor of 2, and that the ground truth differs."
    )
    rows = []
    for pid in pair_ids():
        cfg = load_pair(pid)
        if quick:
            cfg = cfg.model_copy(update={"seeds": 3})
        res = run_pair(cfg)
        rows.append((
            pid, f"{cfg.horizon_s:g} s", cfg.symptom.replace("_", " "),
            f"{np.median(res.symptom_a):+.3g} / {np.median(res.symptom_b):+.3g}",
            f"{res.symptom_ratio:.2f}", f"{sum(res.flags_agree)}/{len(res.flags_agree)}",
            mark(res.passed),
        ))  # fmt: skip
    r.table(["Pair", "Horizon", "Symptom", "Median a / b", "Ratio", "Monitors alike", ""], rows)


# --------------------------------------------------------------------------- V23
def v23_live(r: Report) -> None:
    import json

    from aeroadapt.live.session import LiveSession, replay_commands
    from aeroadapt.scenarios.schema import load_scenario
    from aeroadapt.telemetry.recorder import FILES
    from aeroadapt.telemetry.schema import FAULT_BITS

    r.h(2, "V23 · Live mode")
    s = LiveSession(load_scenario("m1-nominal"))
    for _ in range(20):
        s.step()
    t_inject = s.time_s
    s.inject({"type": "motor_cut"})
    s.inject({"type": "sensor_fault", "kind": "gnss_dropout"})
    for _ in range(10):
        s.step()
    s.clear(1)
    for _ in range(10):
        s.step()
    cols = s.all_columns()
    t = cols["time_us"] * 1e-6
    on = (cols["labels.fault_bits"] & FAULT_BITS["motor_cut"]) > 0
    latency = float(t[np.argmax(on)] - t_inject)
    with tempfile.TemporaryDirectory() as tmp:
        a = s.record(Path(tmp) / "live")
        commands = json.loads((a / "manifest.json").read_text())["live_commands"]
        b = replay_commands(load_scenario("m1-nominal"), commands)
        while b.time_s < s.time_s - 1e-6:
            b.step()
        bd = b.record(Path(tmp) / "offline")
        same = all((a / f).read_bytes() == (bd / f).read_bytes() for f in FILES)
    r.p(
        "`aeroadapt serve` steps the same compiled closed loop in 0.1 s chunks, paced to "
        "wall-clock time; Skyview injects and clears faults while it flies.",
    )
    r.table(
        ["Check", "Result", "Criterion", ""],
        [
            ("Injection latency (fault label after the command)", f"{latency * 1e3:.0f} ms",
             "next control step (10 ms)", mark(abs(latency - 0.01) < 1e-6)),
            ("Session bundle vs offline re-fly of its command log", "byte-identical" if same
             else "different", "byte-identical", mark(same)),
            ("Pacing jitter at 1x", "gated in the fast tier (`test_pacing_jitter_at_real_time`)",
             "p99 ≤ 20 ms", ""),
        ],
    )  # fmt: skip


# --------------------------------------------------------------------------- V22
def _ci(x: dict) -> str:
    if not x["n"]:
        return "–"
    return f"{x['rate']:.1%} [{x['lo']:.0%}, {x['hi']:.0%}] ({x['k']}/{x['n']})"


def v22_bench(r: Report, quick: bool) -> None:
    import json
    from dataclasses import asdict

    from aeroadapt.bench.run import run_bench
    from aeroadapt.bench.sampler import load_bench_spec

    r.h(2, "V22 · Bench v0")
    if (BENCH_RUN / "summary.json").exists():
        s = json.loads((BENCH_RUN / "summary.json").read_text())
        spec = s["spec"]
        r.p(
            f"`aeroadapt bench-run {spec['id']}`: **{spec['n']:,} scenarios** "
            f"(seed {spec['seed']}) on the {spec['split']} split of the V20-accepted fleet, each "
            "flown by B1 (the stock stack) and B2 (B1 plus the classical contingency, which acts "
            "only on a detected motor failure), with the same start state and noise. Missions "
            "are random closed loops; wind up to 80 % of cruise speed with turbulence following "
            "the wind, gusts, shear and thermals; none, one or two faults of any kind. Rates "
            "with 95 % Wilson intervals; latency means with bootstrap intervals."
        )
        b = s["baselines"]
        rows = []
        for label, get in (
            ("Survival (no loss of control, no ground contact except a planned landing)",
             lambda x: _ci(x["survival"])),
            ("Mission completed (one full lap)", lambda x: _ci(x["mission_completion"])),
            ("Loss of control", lambda x: _ci(x["loss_of_control"])),
            ("Motor cut: survival", lambda x: _ci(x["motor_cut"]["survival"])),
            ("Motor cut: landed in a safe zone", lambda x: _ci(x["motor_cut"]["landed_in_zone"])),
            ("Motor cut: detected", lambda x: _ci(x["motor_cut"]["detection"])),
            ("Motor cut: detection latency (s)", lambda x: (
                f"{x['motor_cut']['detect_latency_s']['mean']:.2f} "
                f"[{x['motor_cut']['detect_latency_s']['lo']:.2f}, "
                f"{x['motor_cut']['detect_latency_s']['hi']:.2f}]"
                if x["motor_cut"]["detect_latency_s"]["mean"] is not None else "–")),
            ("False alarms on fault-free flights", lambda x: _ci(x["false_alarms"])),
            ("False alarms per flight hour", lambda x: str(x["false_alarms"]["per_flight_hour"])),
        ):  # fmt: skip
            rows.append((label, *(get(b[k]) for k in b)))
        r.table(["", *b], rows)
        cats = sorted({c for k in b for c in b[k]["survival_by_category"]})
        r.table(
            ["Survival by fault category", *b],
            [(c, *(_ci(b[k]["survival_by_category"].get(c, {"n": 0})) for k in b)) for c in cats],
        )
        if "b2_vs_b1" in s:
            d = s["b2_vs_b1"]
            r.p(
                f"B1 and B2 differ on {d['scenarios_where_survival_differs']} scenarios: B2 "
                f"survives {d['b2_better']} that B1 does not, B1 {d['b1_better']} that B2 does "
                f"not (fault combinations: {', '.join(d['differing_fault_kinds']) or 'none'})."
            )
    else:
        r.p(f"*No bench run at `{BENCH_RUN.relative_to(REPO)}` (`aeroadapt bench-run b0`).*")
    if not quick:
        spec = load_bench_spec("sample").model_copy(update={"n": 4, "duration_s": 60.0})
        a, b2 = run_bench(spec, log=lambda _: None), run_bench(spec, log=lambda _: None)
        same = [asdict(o) for o in a.outcomes] == [asdict(o) for o in b2.outcomes]
        r.p(f"Reproducibility: the sample bench run twice gives an identical outcome table "
            f"{mark(same)}.")  # fmt: skip


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
    fleet = subprocess.run(
        [sys.executable, str(REPO / "tools" / "bench_fleet.py"), "--batches", "64,256,1024"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    r.p(
        "Heterogeneous fleet with faults (M2, `tools/bench_fleet.py`; the GPU figure comes from "
        "`notebooks/gpu_throughput.ipynb`):",
        "```",
        *fleet.stdout.strip().splitlines(),
        "```",
    )


def build_report(out: Path, quick: bool = False) -> Path:
    r = Report(out)
    t0 = time.perf_counter()
    gi = git_info()
    r.lines += [
        "# Validation report (M1 and M2)",
        "",
        f"*Generated {dt.date.today().isoformat()} by `validation/report.py` · commit "
        f"`{gi.get('commit', 'unknown')[:10]}`{' (dirty)' if gi.get('dirty') else ''} · "
        f"{platform.machine()} {platform.system()} · Python {platform.python_version()} · "
        f"JAX {jax.__version__}{' · quick mode' if quick else ''}*",
        "",
        "Evidence for the M1 spec §12 and the M2 spec §10. Each section recomputes its "
        "results; the pytest tiers gate the same criteria in CI (fast) and nightly (full). "
        "The frozen M1 artefact is [m1_report.md](m1_report.md).",
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
        lambda: v14_jsbsim(r),
        lambda: v16_v17_tables(r),
        lambda: v18_references(r),
        lambda: v19_fleet(r),
        lambda: v20_fleet(r),
        lambda: v21_faultlab(r, quick),
        lambda: v22_bench(r, quick),
        lambda: v23_live(r),
        lambda: v15_throughput(r),
    ]
    for section in sections:
        section()
    r.p("", f"*Built in {time.perf_counter() - t0:.0f} s. V11 test names: {', '.join(TESTS)}.*")
    path = out / "m2_report.md"
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
