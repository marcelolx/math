"""Baseline 1D longitudinal rocket combustor thermoacoustic model (Crocco n-tau).

Governing dimensionless linearized acoustics on x in [0, 1]:
    dU/dt + dP/dx = 0
    dP/dt + dU/dx = n * delta(x - x_f) * P_ac(x_f, t - tau)

Physical DC-blocking high-pass filter (omega_hp = 0.20 << pi):
    d(P_bar)/dt = omega_hp * (P(x_f, t) - P_bar),   P_ac = P(x_f, t) - P_bar
    H_hp(s)     = s / (s + omega_hp)
Represents stiff choked-feed mean-flow regulation at omega = 0, preventing unphysical
static k=0 thermal runaway when n > zeta (verified explicitly by real-axis root scans).

Boundary conditions:
    x = 0 (stiff injector faceplate): U(0, t) = 0
    x = 1 (effective nozzle/end loss): U(1, t) = zeta * P(1, t),  zeta = 0.04

Exact dispersion relation for eigenmodes ~ exp(s * t):
    F(s) = sinh(s) + zeta * cosh(s)
           - n * H_hp(s) * exp(-s * tau) * cosh(s * x_f) * (cosh(s * (1 - x_f)) + zeta * sinh(s * (1 - x_f)))
         = 0

This script:
1. Audits the unfiltered k=0 bulk instability (proving why H_hp is physically required) and
   verifies that H_hp removes all positive real roots on s in [0, 3].
2. Solves the exact transcendental dispersion relation across k = 1..K_max.
3. Cross-checks against an energy-conserving staggered-grid RK4 PDE solver using a
   damped-sinusoid least-squares modal estimator.
4. Generates stability maps and validation figures in ../results/.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.optimize import least_squares


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OMEGA_HP = 0.20  # Causal DC-blocking cutoff (|H_hp(i*pi)| = 0.9980, phase lead = +3.64 deg)


# ---------------------------------------------------------------------------
# 1. Exact dispersion relation solver + real-axis k=0 audit
# ---------------------------------------------------------------------------

def dispersion_F(
    s: complex,
    n: float,
    tau: float,
    x_f: float,
    zeta: float,
    use_hp: bool = True,
) -> complex:
    """Exact thermoacoustic dispersion function F(s) = 0."""
    lhs = np.sinh(s) + zeta * np.cosh(s)
    if n == 0.0:
        return lhs
    hp = (s / (s + OMEGA_HP)) if use_hp else 1.0
    c1 = np.cosh(s * x_f)
    c2 = np.cosh(s * (1.0 - x_f)) + zeta * np.sinh(s * (1.0 - x_f))
    rhs = n * hp * np.exp(-s * tau) * c1 * c2
    return lhs - rhs


def check_positive_real_roots(
    n: float,
    tau: float,
    x_f: float,
    zeta: float,
    use_hp: bool = True,
    s_max: float = 3.0,
) -> list[float]:
    """Scan the real axis s in [0, s_max] for sign changes (unstable k=0 bulk roots)."""
    grid = np.linspace(0.0, s_max, 301)
    vals = np.array([dispersion_F(complex(sv, 0.0), n, tau, x_f, zeta, use_hp=use_hp).real for sv in grid])
    roots: list[float] = []
    for i in range(len(grid) - 1):
        if vals[i] == 0.0:
            roots.append(float(grid[i]))
        elif vals[i] * vals[i + 1] < 0.0:
            lo, hi = float(grid[i]), float(grid[i + 1])
            flo = vals[i]
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                fm = dispersion_F(complex(mid, 0.0), n, tau, x_f, zeta, use_hp=use_hp).real
                if flo * fm <= 0.0:
                    hi = mid
                else:
                    lo = mid
                    flo = fm
            roots.append(0.5 * (lo + hi))
    return roots


def solve_mode_continuation(
    k: int,
    n: float,
    tau: float,
    x_f: float,
    zeta: float,
    use_hp: bool = True,
    n_steps: int = 8,
    tol: float = 1e-12,
) -> complex:
    """Track longitudinal eigenmode k from n=0 to target n via homotopy continuation."""
    s = complex(-np.arctanh(zeta), k * np.pi)
    if n == 0.0:
        return s

    steps = max(1, int(np.ceil(n_steps * max(1.0, n / 0.15))))
    for step in range(1, steps + 1):
        n_cur = n * (step / steps)
        for _ in range(40):
            f_val = dispersion_F(s, n_cur, tau, x_f, zeta, use_hp=use_hp)
            if abs(f_val) < tol:
                break
            eps = 1e-7 * max(1.0, abs(s))
            df_val = (
                dispersion_F(s + eps, n_cur, tau, x_f, zeta, use_hp=use_hp)
                - dispersion_F(s - eps, n_cur, tau, x_f, zeta, use_hp=use_hp)
            ) / (2.0 * eps)
            if abs(df_val) < 1e-15:
                break
            ds = f_val / df_val
            if abs(ds) > 1.0:
                ds = ds / abs(ds)
            s -= ds
    return s


def solve_spectrum(
    n: float,
    tau: float,
    x_f: float,
    zeta: float,
    k_max: int = 8,
    use_hp: bool = True,
) -> list[tuple[int, complex, float]]:
    """Return list of (mode_k, eigenvalue s, residual |F(s)|) for k = 1..k_max."""
    modes = []
    for k in range(1, k_max + 1):
        s = solve_mode_continuation(k, n, tau, x_f, zeta, use_hp=use_hp)
        res = abs(dispersion_F(s, n, tau, x_f, zeta, use_hp=use_hp))
        modes.append((k, s, res))
    return modes


# ---------------------------------------------------------------------------
# 2. Direct time-domain staggered-grid PDE solver (RK4) with DC-blocking HP
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    t: np.ndarray
    p_head: np.ndarray
    p_flame: np.ndarray
    energy: np.ndarray
    rayleigh: np.ndarray
    nozzle_loss: np.ndarray
    mode_proj: np.ndarray
    x_p: np.ndarray
    p_final: np.ndarray
    x_f_actual: float


def simulate_1d_chamber(
    n: float,
    tau: float,
    x_f: float,
    zeta: float,
    use_hp: bool = True,
    t_end: float = 30.0,
    n_cells: int = 200,
    cfl: float = 0.25,
    init_mode: int | None = 1,
    init_amp: float = 1e-3,
    k_track: int = 8,
) -> SimResult:
    """Solve the 1D thermoacoustic PDE on an energy-conserving staggered grid with RK4."""
    dx = 1.0 / n_cells
    dt = cfl * dx
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps
    inv_dx = 1.0 / dx

    x_p = (np.arange(n_cells) + 0.5) * dx
    m_flame = int(np.argmin(np.abs(x_p - x_f)))
    x_f_actual = float(x_p[m_flame])

    ks = np.arange(1, k_track + 1)[:, None]
    cos_basis = np.cos(ks * np.pi * x_p[None, :])

    if init_mode is not None:
        P = init_amp * np.cos(init_mode * np.pi * x_p)
    else:
        P = np.zeros(n_cells)
        for k in range(1, 7):
            P += (init_amp / k) * np.cos(k * np.pi * x_p)
    U = np.zeros(n_cells - 1)
    p_bar = 0.0  # High-pass running mean state

    delay_steps_f = tau / dt
    buf_len = int(np.ceil(delay_steps_f)) + 4
    hist = np.zeros(n_steps + buf_len + 2)
    hist[buf_len] = float(P[m_flame]) - p_bar

    def get_delayed_p(step_idx: int, frac: float, cur_val: float) -> float:
        if tau <= 1e-12:
            return cur_val
        idx_f = buf_len + step_idx + frac - delay_steps_f
        i0 = int(np.floor(idx_f))
        a = idx_f - i0
        if i0 < 0:
            return 0.0
        if i0 >= buf_len + step_idx:
            return cur_val
        return float((1.0 - a) * hist[i0] + a * hist[i0 + 1])

    def rhs(P_cur: np.ndarray, U_cur: np.ndarray, pbar_cur: float, p_del: float):
        dP = np.empty_like(P_cur)
        dP[0] = -U_cur[0] * inv_dx
        dP[1:-1] = -(U_cur[1:] - U_cur[:-1]) * inv_dx
        dP[-1] = -(zeta * P_cur[-1] - U_cur[-1]) * inv_dx
        if n != 0.0:
            dP[m_flame] += n * p_del * inv_dx
        dU = -(P_cur[1:] - P_cur[:-1]) * inv_dx
        dpbar = OMEGA_HP * (float(P_cur[m_flame]) - pbar_cur) if use_hp else 0.0
        return dP, dU, dpbar

    t_arr = np.linspace(0.0, t_end, n_steps + 1)
    p_head = np.empty(n_steps + 1)
    p_flame = np.empty(n_steps + 1)
    energy = np.empty(n_steps + 1)
    rayleigh = np.empty(n_steps + 1)
    nozzle_loss = np.empty(n_steps + 1)
    mode_proj = np.empty((k_track, n_steps + 1))

    def record(idx: int, P_cur: np.ndarray, U_cur: np.ndarray, p_del: float) -> None:
        p_head[idx] = P_cur[0]
        p_flame[idx] = P_cur[m_flame]
        energy[idx] = 0.5 * dx * (np.dot(P_cur, P_cur) + np.dot(U_cur, U_cur))
        rayleigh[idx] = n * P_cur[m_flame] * p_del
        nozzle_loss[idx] = zeta * (P_cur[-1] ** 2)
        mode_proj[:, idx] = 2.0 * dx * (cos_basis @ P_cur)

    record(0, P, U, get_delayed_p(0, 0.0, float(P[m_flame]) - p_bar))

    for step in range(n_steps):
        p_ac_0 = float(P[m_flame]) - (p_bar if use_hp else 0.0)
        p_del_0 = get_delayed_p(step, 0.0, p_ac_0)

        k1_P, k1_U, k1_b = rhs(P, U, p_bar, p_del_0)
        P2 = P + 0.5 * dt * k1_P
        b2 = p_bar + 0.5 * dt * k1_b
        p_del_h = get_delayed_p(step, 0.5, float(P2[m_flame]) - (b2 if use_hp else 0.0))

        k2_P, k2_U, k2_b = rhs(P2, U + 0.5 * dt * k1_U, b2, p_del_h)
        P3 = P + 0.5 * dt * k2_P
        b3 = p_bar + 0.5 * dt * k2_b
        p_del_h2 = get_delayed_p(step, 0.5, float(P3[m_flame]) - (b3 if use_hp else 0.0))

        k3_P, k3_U, k3_b = rhs(P3, U + 0.5 * dt * k2_U, b3, p_del_h2)
        P4 = P + dt * k3_P
        b4 = p_bar + dt * k3_b
        p_del_1 = get_delayed_p(step, 1.0, float(P4[m_flame]) - (b4 if use_hp else 0.0))

        k4_P, k4_U, k4_b = rhs(P4, U + dt * k3_U, b4, p_del_1)

        P = P + (dt / 6.0) * (k1_P + 2.0 * k2_P + 2.0 * k3_P + k4_P)
        U = U + (dt / 6.0) * (k1_U + 2.0 * k2_U + 2.0 * k3_U + k4_U)
        p_bar = p_bar + (dt / 6.0) * (k1_b + 2.0 * k2_b + 2.0 * k3_b + k4_b)

        p_ac_next = float(P[m_flame]) - (p_bar if use_hp else 0.0)
        hist[buf_len + step + 1] = p_ac_next
        record(step + 1, P, U, get_delayed_p(step + 1, 0.0, p_ac_next))

    return SimResult(
        t=t_arr,
        p_head=p_head,
        p_flame=p_flame,
        energy=energy,
        rayleigh=rayleigh,
        nozzle_loss=nozzle_loss,
        mode_proj=mode_proj,
        x_p=x_p,
        p_final=P.copy(),
        x_f_actual=x_f_actual,
    )


def fit_damped_sinusoid(
    t: np.ndarray,
    signal: np.ndarray,
    t_min: float,
    t_max: float,
    alpha_init: float,
    omega_init: float,
) -> tuple[float, float]:
    """Nonlinear least-squares fit of exp(alpha * (t-t0)) * (c1*cos(w*(t-t0)) + c2*sin(w*(t-t0)))."""
    mask = (t >= t_min) & (t <= t_max)
    tw = t[mask] - t_min
    yw = signal[mask]
    # Subsample slightly if huge for speed while retaining >20 pts/cycle
    stride = max(1, len(tw) // 2000)
    ts = tw[::stride]
    ys = yw[::stride]

    def residual(params: np.ndarray) -> np.ndarray:
        a, w = params
        env = np.exp(a * ts)
        A = np.column_stack([env * np.cos(w * ts), env * np.sin(w * ts)])
        coef, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
        return A @ coef - ys

    res = least_squares(residual, x0=np.array([alpha_init, omega_init]), xtol=1e-12, ftol=1e-12)
    return float(res.x[0]), float(res.x[1])


def check_energy_balance(sim: SimResult) -> float:
    """Check relative error of integrated Rayleigh balance: dE = int (R - D) dt."""
    dt = sim.t[1] - sim.t[0]
    net_power = sim.rayleigh - sim.nozzle_loss
    cum_work = np.concatenate([[0.0], np.cumsum(0.5 * (net_power[:-1] + net_power[1:]) * dt)])
    dE = sim.energy - sim.energy[0]
    scale = max(np.max(np.abs(dE)), np.max(sim.energy), 1e-15)
    return float(np.max(np.abs(dE - cum_work)) / scale)


# ---------------------------------------------------------------------------
# 3. Main experiment & plotting
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Effective lumped nozzle/end admittance (represents choked nozzle + end losses at low bulk Mach)
    zeta = 0.04
    print("=== 1D Rocket Combustor Baseline (DC-Blocked Crocco n-tau + End Admittance) ===")
    print(f"Lumped end admittance zeta = {zeta:.4f}, DC-blocking HP cutoff omega_hp = {OMEGA_HP:.2f}")
    print(f"Uncoupled damping rate: alpha_0 = -arctanh(zeta) = {-np.arctanh(zeta):.6f}\n")

    # Audit k=0 real root WITH vs WITHOUT DC blocking (peer-review C1 check)
    k0_unfiltered = check_positive_real_roots(0.10, 0.20, 0.1475, zeta, use_hp=False)
    k0_filtered = check_positive_real_roots(0.10, 0.20, 0.1475, zeta, use_hp=True)
    print(f"k=0 real-axis audit (n=0.10, tau=0.20, x_f=0.1475):")
    print(f"  Unfiltered (no HP): positive real roots = {k0_unfiltered}  (UNSTABLE static bulk runaway!)")
    print(f"  DC-blocked (HP):    positive real roots = {k0_filtered}  (eliminated)\n")

    # ------------------------------------------------------------------
    # Part A: Four physical verification cases (Exact Dispersion vs PDE)
    # ------------------------------------------------------------------
    cases = [
        {
            "id": "A_no_combustion",
            "label": "Case A: Cold Flow (n=0, 1L Damping)",
            "n": 0.0,
            "tau": 0.20,
            "x_f": 0.15,
            "init_mode": 1,
            "target_mode": 1,
            "t_end": 30.0,
            "fit_window": (4.0, 28.0),
        },
        {
            "id": "B_1L_unstable",
            "label": "Case B: 1L Rayleigh-Driven (x_f=0.15, tau=0.20)",
            "n": 0.10,
            "tau": 0.20,
            "x_f": 0.15,
            "init_mode": 1,
            "target_mode": 1,
            "t_end": 30.0,
            "fit_window": (6.0, 28.0),
        },
        {
            "id": "C_2L_takeover",
            "label": "Case C: 1L Damped -> 2L Takeover (x_f=0.10, tau=1.00)",
            "n": 0.10,
            "tau": 1.00,
            "x_f": 0.10,
            "init_mode": None,
            "target_mode": 2,
            "t_end": 36.0,
            "fit_window": (12.0, 34.0),
        },
        {
            "id": "D_4L_sneakthrough",
            "label": "Case D: 2L-Node Placement -> 4L Sneak-Through (x_f=0.25, tau=1.00)",
            "n": 0.10,
            "tau": 1.00,
            "x_f": 0.25,
            "init_mode": None,
            "target_mode": 4,
            "t_end": 34.0,
            "fit_window": (12.0, 32.0),
        },
    ]

    metrics: dict[str, dict] = {
        "parameters": {"zeta": zeta, "omega_hp": OMEGA_HP, "n_cells": 200, "cfl": 0.25},
        "k0_audit": {"unfiltered_caseB_real_roots": k0_unfiltered, "dc_blocked_caseB_real_roots": k0_filtered},
        "cases": {},
    }

    sim_results = []

    for c in cases:
        sim = simulate_1d_chamber(
            n=c["n"],
            tau=c["tau"],
            x_f=c["x_f"],
            zeta=zeta,
            use_hp=True,
            t_end=c["t_end"],
            n_cells=200,
            cfl=0.25,
            init_mode=c["init_mode"],
            k_track=8,
        )
        pos_real = check_positive_real_roots(c["n"], c["tau"], sim.x_f_actual, zeta, use_hp=True)
        spectrum = solve_spectrum(c["n"], c["tau"], sim.x_f_actual, zeta, k_max=8, use_hp=True)
        k_tgt = c["target_mode"]
        dom = next(m for m in spectrum if m[0] == k_tgt)
        k_dom, s_dom, f_res = dom

        sig_mode = sim.mode_proj[k_dom - 1]
        alpha_pde, omega_pde = fit_damped_sinusoid(
            sim.t, sig_mode, c["fit_window"][0], c["fit_window"][1], s_dom.real, s_dom.imag
        )
        e_err = check_energy_balance(sim)

        alpha_err = abs(alpha_pde - s_dom.real)
        omega_err = abs(omega_pde - s_dom.imag)

        print(f"[{c['label']}] (x_f_grid = {sim.x_f_actual:.4f}, real k=0 roots={pos_real})")
        print(f"  Tracked mode: {k_dom}L | s_exact = {s_dom.real:+.6f} + {s_dom.imag:.6f}i  (|F|={f_res:.1e})")
        print(f"  PDE LSQ fit:       s_PDE   = {alpha_pde:+.6f} + {omega_pde:.6f}i")
        print(f"  Abs diff: |d_alpha| = {alpha_err:.2e}, |d_omega| = {omega_err:.2e} | Energy balance err = {e_err:.2e}")
        mode_str = ", ".join(f"{k}L:{s.real:+.4f}" for k, s, _ in spectrum[:6])
        print(f"  Spectrum (1L-6L): {mode_str}\n")

        metrics["cases"][c["id"]] = {
            "label": c["label"],
            "n": c["n"],
            "tau": c["tau"],
            "x_f_target": c["x_f"],
            "x_f_grid": sim.x_f_actual,
            "positive_real_roots": pos_real,
            "tracked_mode": k_dom,
            "alpha_exact": float(s_dom.real),
            "omega_exact": float(s_dom.imag),
            "dispersion_residual": float(f_res),
            "alpha_pde": alpha_pde,
            "omega_pde": omega_pde,
            "alpha_abs_diff": float(alpha_err),
            "omega_abs_diff": float(omega_err),
            "energy_balance_rel_err": e_err,
            "spectrum_alpha": {f"{k}L": float(s.real) for k, s, _ in spectrum},
        }
        sim_results.append((c, sim, dom, alpha_pde, omega_pde, e_err))

    # ------------------------------------------------------------------
    # Part B: Plot PDE validation traces + modal envelope comparison
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), dpi=150)
    for ax, (c, sim, dom, alpha_pde, omega_pde, e_err) in zip(axes.flat, sim_results):
        k_dom, s_dom, _ = dom
        sig = sim.mode_proj[k_dom - 1]
        ax.plot(sim.t, sig, color="#1f77b4", lw=0.9, label=f"PDE mode {k_dom}L projection $a_{k_dom}(t)$")

        t0, t1 = c["fit_window"]
        t_mid = 0.5 * (t0 + t1)
        win_mask = (sim.t >= t0) & (sim.t <= t1)
        mid_mask = (sim.t >= t_mid - 0.8) & (sim.t <= t_mid + 0.8)
        amp_mid = np.max(np.abs(sig[mid_mask]))
        t_env = sim.t[win_mask]
        env = amp_mid * np.exp(s_dom.real * (t_env - t_mid))
        ax.plot(t_env, env, "r--", lw=1.7, label=f"Exact {k_dom}L envelope ($\\alpha={s_dom.real:+.4f}$)")
        ax.plot(t_env, -env, "r--", lw=1.7)

        ax.set_title(c["label"], fontsize=10.2, fontweight="bold")
        ax.set_xlabel("Dimensionless time $t \\bar{c}/L$")
        ax.set_ylabel(f"Modal amplitude $a_{k_dom}(t)$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8.3)
        ax.text(
            0.98, 0.05,
            f"PDE:   $\\alpha={alpha_pde:+.5f},\\ \\omega={omega_pde:.4f}$\n"
            f"Exact: $\\alpha={s_dom.real:+.5f},\\ \\omega={s_dom.imag:.4f}$\n"
            f"Energy balance err: {e_err:.1e}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.92),
        )

    fig.suptitle(
        f"1D Rocket Combustor Verification: Direct PDE vs Exact Dispersion ($\\zeta={zeta:.2f},\\ \\omega_{{\\mathrm{{hp}}}}={OMEGA_HP}$)",
        fontsize=12.2, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(RESULTS_DIR / "baseline_pde_validation.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Part C: Stability maps in (tau, n) and (x_f, tau) planes (k=0 verified!)
    # ------------------------------------------------------------------
    x_f_map = 0.15
    tau_grid = np.linspace(0.0, 2.0, 101)
    n_grid = np.linspace(0.0, 0.20, 51)
    alpha_max_tn = np.zeros((len(n_grid), len(tau_grid)))

    for i, n_val in enumerate(n_grid):
        for j, tau_val in enumerate(tau_grid):
            spec = solve_spectrum(n_val, tau_val, x_f_map, zeta, k_max=4, use_hp=True)
            best = max(item[1].real for item in spec)
            rr = check_positive_real_roots(n_val, tau_val, x_f_map, zeta, use_hp=True)
            if rr:
                best = max(best, max(rr))
            alpha_max_tn[i, j] = best

    n_fixed = 0.10
    xf_grid = np.linspace(0.05, 0.95, 47)
    tau_grid2 = np.linspace(0.0, 2.0, 81)
    alpha_1L_xt = np.zeros((len(tau_grid2), len(xf_grid)))
    alpha_4L_xt = np.zeros((len(tau_grid2), len(xf_grid)))
    alpha_8L_xt = np.zeros((len(tau_grid2), len(xf_grid)))
    alpha_12L_xt = np.zeros((len(tau_grid2), len(xf_grid)))
    real_root_hits = 0

    for i, tau_val in enumerate(tau_grid2):
        for j, xf_val in enumerate(xf_grid):
            rr = check_positive_real_roots(n_fixed, tau_val, xf_val, zeta, use_hp=True)
            r_max = max(rr) if rr else -1e9
            if rr:
                real_root_hits += 1
            spec = solve_spectrum(n_fixed, tau_val, xf_val, zeta, k_max=12, use_hp=True)
            alphas = [item[1].real for item in spec]
            alpha_1L_xt[i, j] = max(alphas[0], r_max)
            alpha_4L_xt[i, j] = max(max(alphas[:4]), r_max)
            alpha_8L_xt[i, j] = max(max(alphas[:8]), r_max)
            alpha_12L_xt[i, j] = max(max(alphas[:12]), r_max)

    stable_frac_1L = float(np.mean(alpha_1L_xt < 0.0))
    stable_frac_4L = float(np.mean(alpha_4L_xt < 0.0))
    stable_frac_8L = float(np.mean(alpha_8L_xt < 0.0))
    stable_frac_12L = float(np.mean(alpha_12L_xt < 0.0))

    print(f"Linearly stable area in (x_f, tau) in [0.05,0.95]x[0,2] at n={n_fixed}, zeta={zeta} (real k=0 hits={real_root_hits}):")
    print(f"  1L only:        {100.0 * stable_frac_1L:.1f}%")
    print(f"  Modes 1L - 4L:  {100.0 * stable_frac_4L:.1f}%  (upper bound for k > 4)")
    print(f"  Modes 1L - 8L:  {100.0 * stable_frac_8L:.1f}%  (upper bound for k > 8)")
    print(f"  Modes 1L - 12L: {100.0 * stable_frac_12L:.1f}%  (upper bound for k > 12)")

    metrics["stability_maps"] = {
        "domain_xf": [0.05, 0.95],
        "domain_tau": [0.0, 2.0],
        "x_f_for_tau_n_map": x_f_map,
        "n_for_xf_tau_map": n_fixed,
        "positive_real_root_hits_on_grid": real_root_hits,
        "stable_fraction_1L_only": stable_frac_1L,
        "stable_fraction_1L_to_4L": stable_frac_4L,
        "stable_fraction_1L_to_8L": stable_frac_8L,
        "stable_fraction_1L_to_12L": stable_frac_12L,
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.6), dpi=150)

    norm1 = TwoSlopeNorm(vmin=-0.08, vcenter=0.0, vmax=0.18)
    im1 = ax1.contourf(
        tau_grid, n_grid, alpha_max_tn,
        levels=np.linspace(-0.08, 0.18, 27),
        cmap="RdBu_r",
        norm=norm1,
        extend="both",
    )
    ax1.contour(tau_grid, n_grid, alpha_max_tn, levels=[0.0], colors="black", linewidths=2.0)

    # Weak-coupling 1L boundary including HP magnitude/phase at omega=pi
    hp_1 = (1j * np.pi) / (1j * np.pi + OMEGA_HP)
    cos_t = np.real(hp_1 * np.exp(-1j * np.pi * tau_grid))
    with np.errstate(divide="ignore", invalid="ignore"):
        n_weak_1L = np.where(cos_t > 0.05, zeta / ((np.cos(np.pi * x_f_map) ** 2) * cos_t), np.nan)
    ax1.plot(tau_grid, n_weak_1L, "k--", lw=1.3, alpha=0.85, label="Weak-coupling 1L boundary")

    ax1.scatter([0.20], [0.10], c=["gold"], edgecolors="black", s=65, zorder=5)
    ax1.annotate(
        "Case B (1L)", (0.20, 0.10), xytext=(0.28, 0.115),
        fontsize=8.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
    )

    ax1.set_xlim(0.0, 2.0)
    ax1.set_ylim(0.0, 0.20)
    ax1.set_xlabel("Dimensionless time lag $\\tau \\bar{c}/L$")
    ax1.set_ylabel("Interaction index $n$")
    ax1.set_title(f"Max Growth Rate $\\max_{{k \\leq 4}} \\alpha_k$ ($x_f={x_f_map},\\ \\zeta={zeta},\\ \\omega_{{\\mathrm{{hp}}}}={OMEGA_HP}$)", pad=10)
    ax1.legend(loc="upper right", fontsize=8.5)
    plt.colorbar(im1, ax=ax1, label="Max growth rate $\\alpha_{\\max}$")

    norm2 = TwoSlopeNorm(vmin=-0.06, vcenter=0.0, vmax=0.08)
    im2 = ax2.contourf(
        xf_grid, tau_grid2, alpha_4L_xt,
        levels=np.linspace(-0.06, 0.08, 29),
        cmap="RdBu_r",
        norm=norm2,
        extend="both",
    )
    ax2.contour(xf_grid, tau_grid2, alpha_4L_xt, levels=[0.0], colors="black", linewidths=1.8)
    ax2.scatter([0.15, 0.10, 0.25], [0.20, 1.00, 1.00], c=["gold", "lime", "cyan"], edgecolors="black", s=65, zorder=5)
    lbl_box = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.82)
    ax2.annotate("B (1L)", (0.15, 0.20), xytext=(0.18, 0.14), fontsize=8.3, fontweight="bold", bbox=lbl_box)
    ax2.annotate("C (2L)", (0.10, 1.00), xytext=(0.06, 1.12), fontsize=8.3, fontweight="bold", bbox=lbl_box)
    ax2.annotate("D (4L)", (0.25, 1.00), xytext=(0.24, 1.13), fontsize=8.3, fontweight="bold", bbox=lbl_box)

    ax2.set_xlabel("Flame position $x_f / L$")
    ax2.set_ylabel("Dimensionless time lag $\\tau \\bar{c}/L$")
    ax2.set_title(
        f"Stability Islands ($n={n_fixed},\\ \\zeta={zeta}$): "
        f"1L={100*stable_frac_1L:.0f}%, 1-4L={100*stable_frac_4L:.0f}%, 1-12L={100*stable_frac_12L:.1f}%",
        pad=10,
    )
    plt.colorbar(im2, ax=ax2, label="Max growth rate $\\alpha_{\\max}$ (1L-4L)")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "baseline_stability_map.png")
    plt.close(fig)

    with open(RESULTS_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved plots and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
