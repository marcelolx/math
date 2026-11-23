"""Baseline 1D longitudinal rocket combustor thermoacoustic model (Crocco n-tau).

Governing dimensionless linearized acoustics on x in [0, 1]:
    dU/dt + dP/dx = 0
    dP/dt + dU/dx = n * delta(x - x_f) * P(x_f, t - tau)

Boundary conditions:
    x = 0 (stiff injector faceplate): U(0, t) = 0
    x = 1 (compact choked nozzle):    U(1, t) = zeta * P(1, t),  zeta = ((gamma - 1) / 2) * M_bar

Exact dispersion relation for eigenmodes ~ exp(s * t):
    F(s) = sinh(s) + zeta * cosh(s)
           - n * exp(-s * tau) * cosh(s * x_f) * (cosh(s * (1 - x_f)) + zeta * sinh(s * (1 - x_f)))
         = 0

This script:
1. Solves the exact transcendental dispersion relation across longitudinal modes k = 1..K_max.
2. Runs a direct time-domain staggered-grid PDE solver (RK4) with exact discrete energy conservation.
3. Validates PDE modal growth rates, frequencies, and the discrete Rayleigh energy balance against exact theory.
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


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# 1. Exact dispersion relation solver
# ---------------------------------------------------------------------------

def dispersion_F(s: complex, n: float, tau: float, x_f: float, zeta: float) -> complex:
    """Exact thermoacoustic dispersion function F(s) = 0."""
    lhs = np.sinh(s) + zeta * np.cosh(s)
    c1 = np.cosh(s * x_f)
    c2 = np.cosh(s * (1.0 - x_f)) + zeta * np.sinh(s * (1.0 - x_f))
    rhs = n * np.exp(-s * tau) * c1 * c2
    return lhs - rhs


def dispersion_dF(s: complex, n: float, tau: float, x_f: float, zeta: float) -> complex:
    """Analytical derivative dF/ds for Newton-Raphson root finding."""
    dlhs = np.cosh(s) + zeta * np.sinh(s)
    c1 = np.cosh(s * x_f)
    dc1 = x_f * np.sinh(s * x_f)
    a = 1.0 - x_f
    c2 = np.cosh(s * a) + zeta * np.sinh(s * a)
    dc2 = a * (np.sinh(s * a) + zeta * np.cosh(s * a))
    e = np.exp(-s * tau)
    drhs = n * e * (-tau * c1 * c2 + dc1 * c2 + c1 * dc2)
    return dlhs - drhs


def solve_mode_continuation(
    k: int,
    n: float,
    tau: float,
    x_f: float,
    zeta: float,
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
            f_val = dispersion_F(s, n_cur, tau, x_f, zeta)
            if abs(f_val) < tol:
                break
            df_val = dispersion_dF(s, n_cur, tau, x_f, zeta)
            if abs(df_val) < 1e-15:
                break
            ds = f_val / df_val
            if abs(ds) > 1.0:
                ds = ds / abs(ds)
            s -= ds
    for _ in range(30):
        f_val = dispersion_F(s, n, tau, x_f, zeta)
        if abs(f_val) < tol:
            break
        df_val = dispersion_dF(s, n, tau, x_f, zeta)
        if abs(df_val) < 1e-15:
            break
        s -= f_val / df_val
    return s


def solve_spectrum(
    n: float,
    tau: float,
    x_f: float,
    zeta: float,
    k_max: int = 8,
) -> list[tuple[int, complex, float]]:
    """Return list of (mode_k, eigenvalue s, residual |F(s)|) for k = 1..k_max."""
    modes = []
    for k in range(1, k_max + 1):
        s = solve_mode_continuation(k, n, tau, x_f, zeta)
        res = abs(dispersion_F(s, n, tau, x_f, zeta))
        modes.append((k, s, res))
    return modes


# ---------------------------------------------------------------------------
# 2. Direct time-domain staggered-grid PDE solver (RK4)
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    t: np.ndarray
    p_head: np.ndarray        # Pressure at injector faceplate (cell 0)
    p_flame: np.ndarray       # Pressure at flame cell m
    energy: np.ndarray        # Total discrete acoustic energy E_h(t)
    rayleigh: np.ndarray      # Instantaneous Rayleigh driving R(t) = n * P_m(t) * P_m(t - tau)
    nozzle_loss: np.ndarray   # Instantaneous nozzle loss D(t) = zeta * P_{N-1}(t)^2
    mode_proj: np.ndarray     # Modal projections a_k(t) = 2 * int P(x,t) cos(k*pi*x) dx, shape (K, nt)
    x_p: np.ndarray
    p_final: np.ndarray
    x_f_actual: float


def simulate_1d_chamber(
    n: float,
    tau: float,
    x_f: float,
    zeta: float,
    t_end: float = 30.0,
    n_cells: int = 200,
    cfl: float = 0.25,
    init_mode: int | None = 1,
    init_amp: float = 1e-3,
    k_track: int = 8,
) -> SimResult:
    """Solve the 1D thermoacoustic PDE on an energy-conserving staggered grid with RK4.

    Exact discrete Rayleigh energy balance:
        dE_h/dt = n * P[m](t) * P[m](t - tau) - zeta * P[N - 1](t)^2
    """
    dx = 1.0 / n_cells
    dt = cfl * dx
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps

    x_p = (np.arange(n_cells) + 0.5) * dx
    m_flame = int(np.argmin(np.abs(x_p - x_f)))
    x_f_actual = float(x_p[m_flame])

    # Cosine basis matrix for modal projections (k = 1..k_track)
    ks = np.arange(1, k_track + 1)[:, None]
    cos_basis = np.cos(ks * np.pi * x_p[None, :])  # shape (k_track, n_cells)

    if init_mode is not None:
        P = init_amp * np.cos(init_mode * np.pi * x_p)
        U = np.zeros(n_cells - 1)
    else:
        # Broadband seed across modes 1..6 so the most unstable mode emerges naturally
        P = np.zeros(n_cells)
        for k in range(1, 7):
            P += (init_amp / k) * np.cos(k * np.pi * x_p)
        U = np.zeros(n_cells - 1)

    delay_steps_f = tau / dt
    buf_len = int(np.ceil(delay_steps_f)) + 4
    hist = np.zeros(n_steps + buf_len + 2)
    hist[buf_len] = float(P[m_flame])

    def get_delayed_p(step_idx: int, frac: float) -> float:
        if tau <= 0.0:
            return hist[buf_len + step_idx]
        idx_f = buf_len + step_idx + frac - delay_steps_f
        i0 = int(np.floor(idx_f))
        a = idx_f - i0
        if i0 < 0:
            return 0.0
        if i0 + 1 >= len(hist):
            return hist[-1]
        return (1.0 - a) * hist[i0] + a * hist[i0 + 1]

    inv_dx = 1.0 / dx

    def rhs(P_cur: np.ndarray, U_cur: np.ndarray, p_del: float) -> tuple[np.ndarray, np.ndarray]:
        dP = np.empty_like(P_cur)
        dP[0] = -U_cur[0] * inv_dx
        dP[1:-1] = -(U_cur[1:] - U_cur[:-1]) * inv_dx
        dP[-1] = -(zeta * P_cur[-1] - U_cur[-1]) * inv_dx
        if n != 0.0:
            dP[m_flame] += n * p_del * inv_dx
        dU = -(P_cur[1:] - P_cur[:-1]) * inv_dx
        return dP, dU

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

    record(0, P, U, get_delayed_p(0, 0.0))

    for step in range(n_steps):
        p_del_0 = get_delayed_p(step, 0.0)
        p_del_h = get_delayed_p(step, 0.5)
        p_del_1 = get_delayed_p(step, 1.0)

        k1_P, k1_U = rhs(P, U, p_del_0)
        k2_P, k2_U = rhs(P + 0.5 * dt * k1_P, U + 0.5 * dt * k1_U, p_del_h)
        k3_P, k3_U = rhs(P + 0.5 * dt * k2_P, U + 0.5 * dt * k2_U, p_del_h)
        k4_P, k4_U = rhs(P + dt * k3_P, U + dt * k3_U, p_del_1)

        P = P + (dt / 6.0) * (k1_P + 2.0 * k2_P + 2.0 * k3_P + k4_P)
        U = U + (dt / 6.0) * (k1_U + 2.0 * k2_U + 2.0 * k3_U + k4_U)

        hist[buf_len + step + 1] = float(P[m_flame])
        record(step + 1, P, U, get_delayed_p(step + 1, 0.0))

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


def estimate_mode_growth_and_freq(
    t: np.ndarray,
    signal: np.ndarray,
    t_min: float,
    t_max: float,
) -> tuple[float, float]:
    """Estimate exponential growth rate alpha (via Hilbert-free RMS/peak envelope) and omega (via FFT)."""
    mask = (t >= t_min) & (t <= t_max)
    tw = t[mask]
    yw = signal[mask]

    # Analytic signal envelope via FFT Hilbert transform (clean, no discrete peak-aliasing!)
    n_pts = len(yw)
    Y = np.fft.fft(yw)
    h = np.zeros(n_pts)
    h[0] = 1.0
    if n_pts % 2 == 0:
        h[n_pts // 2] = 1.0
        h[1 : n_pts // 2] = 2.0
    else:
        h[1 : (n_pts + 1) // 2] = 2.0
    env = np.abs(np.fft.ifft(Y * h))

    # Trim 15% from each edge to eliminate Hilbert boundary ringing
    trim = int(0.15 * n_pts)
    t_in = tw[trim:-trim]
    env_in = env[trim:-trim]
    alpha_est = float(np.polyfit(t_in, np.log(np.maximum(env_in, 1e-30)), 1)[0])

    # Detrended FFT for precise frequency
    y_flat = yw * np.exp(-alpha_est * (tw - tw[0]))
    dt_s = tw[1] - tw[0]
    n_pad = 16 * len(y_flat)
    spec = np.abs(np.fft.rfft(y_flat * np.hanning(len(y_flat)), n=n_pad))
    freqs = np.fft.rfftfreq(n_pad, d=dt_s)
    idx = int(np.argmax(spec[1:]) + 1)
    omega_est = float(2.0 * np.pi * freqs[idx])

    return alpha_est, omega_est


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

    gamma = 1.25
    m_bar = 0.32
    zeta = 0.5 * (gamma - 1.0) * m_bar  # = 0.04

    print("=== 1D Rocket Combustor Baseline (Crocco n-tau + Choked Nozzle) ===")
    print(f"gamma = {gamma}, M_bar = {m_bar} -> nozzle admittance zeta = {zeta:.4f}")
    print(f"Uncoupled nozzle damping rate: alpha_0 = -arctanh(zeta) = {-np.arctanh(zeta):.6f}\n")

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
            "init_mode": None,   # Broadband seed (1L-6L)
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
            "init_mode": None,   # Broadband seed (1L-6L)
            "target_mode": 4,
            "t_end": 34.0,
            "fit_window": (12.0, 32.0),
        },
    ]

    metrics: dict[str, dict] = {
        "parameters": {"gamma": gamma, "M_bar": m_bar, "zeta": zeta, "n_cells": 200, "cfl": 0.25},
        "cases": {},
    }

    sim_results = []

    for c in cases:
        sim = simulate_1d_chamber(
            n=c["n"],
            tau=c["tau"],
            x_f=c["x_f"],
            zeta=zeta,
            t_end=c["t_end"],
            n_cells=200,
            cfl=0.25,
            init_mode=c["init_mode"],
            k_track=8,
        )
        spectrum = solve_spectrum(c["n"], c["tau"], sim.x_f_actual, zeta, k_max=8)
        k_tgt = c["target_mode"]
        dom = next(m for m in spectrum if m[0] == k_tgt)
        k_dom, s_dom, f_res = dom

        # Measure PDE growth rate & frequency on the projected mode a_k(t)
        sig_mode = sim.mode_proj[k_dom - 1]
        alpha_pde, omega_pde = estimate_mode_growth_and_freq(sim.t, sig_mode, *c["fit_window"])
        e_err = check_energy_balance(sim)

        alpha_err = abs(alpha_pde - s_dom.real)
        omega_err = abs(omega_pde - s_dom.imag)

        print(f"[{c['label']}] (x_f_grid = {sim.x_f_actual:.4f})")
        print(f"  Tracked mode: {k_dom}L | s_exact = {s_dom.real:+.6f} + {s_dom.imag:.6f}i  (|F|={f_res:.1e})")
        print(f"  PDE modal fit:     s_PDE   = {alpha_pde:+.6f} + {omega_pde:.6f}i")
        print(f"  Abs diff: |d_alpha| = {alpha_err:.2e}, |d_omega| = {omega_err:.2e} | Energy balance err = {e_err:.2e}")
        mode_str = ", ".join(f"{k}L:{s.real:+.4f}" for k, s, _ in spectrum[:6])
        print(f"  Spectrum (1L-6L): {mode_str}\n")

        metrics["cases"][c["id"]] = {
            "label": c["label"],
            "n": c["n"],
            "tau": c["tau"],
            "x_f_target": c["x_f"],
            "x_f_grid": sim.x_f_actual,
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

        # Exact exponential envelope anchored at midpoint of fit window
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
        f"1D Rocket Combustor Verification: Direct PDE vs Exact Dispersion ($\\zeta={zeta:.2f}$)",
        fontsize=12.5, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(RESULTS_DIR / "baseline_pde_validation.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Part C: Stability maps in (tau, n) and (x_f, tau) planes
    # ------------------------------------------------------------------
    x_f_map = 0.15
    tau_grid = np.linspace(0.0, 2.0, 121)
    n_grid = np.linspace(0.0, 0.20, 61)
    alpha_max_tn = np.zeros((len(n_grid), len(tau_grid)))

    for i, n_val in enumerate(n_grid):
        for j, tau_val in enumerate(tau_grid):
            spec = solve_spectrum(n_val, tau_val, x_f_map, zeta, k_max=4)
            alpha_max_tn[i, j] = max(item[1].real for item in spec)

    n_fixed = 0.10
    xf_grid = np.linspace(0.02, 0.98, 81)
    tau_grid2 = np.linspace(0.0, 2.0, 101)
    alpha_1L_xt = np.zeros((len(tau_grid2), len(xf_grid)))
    alpha_4L_xt = np.zeros((len(tau_grid2), len(xf_grid)))
    alpha_8L_xt = np.zeros((len(tau_grid2), len(xf_grid)))

    for i, tau_val in enumerate(tau_grid2):
        for j, xf_val in enumerate(xf_grid):
            spec = solve_spectrum(n_fixed, tau_val, xf_val, zeta, k_max=8)
            alphas = [item[1].real for item in spec]
            alpha_1L_xt[i, j] = alphas[0]
            alpha_4L_xt[i, j] = max(alphas[:4])
            alpha_8L_xt[i, j] = max(alphas[:8])

    stable_frac_1L = float(np.mean(alpha_1L_xt < 0.0))
    stable_frac_4L = float(np.mean(alpha_4L_xt < 0.0))
    stable_frac_8L = float(np.mean(alpha_8L_xt < 0.0))

    print(f"Linearly stable area in (x_f, tau) at n={n_fixed}, zeta={zeta}:")
    print(f"  1L only:       {100.0 * stable_frac_1L:.1f}%")
    print(f"  Modes 1L - 4L: {100.0 * stable_frac_4L:.1f}%")
    print(f"  Modes 1L - 8L: {100.0 * stable_frac_8L:.1f}%")

    metrics["stability_maps"] = {
        "x_f_for_tau_n_map": x_f_map,
        "n_for_xf_tau_map": n_fixed,
        "stable_fraction_1L_only": stable_frac_1L,
        "stable_fraction_1L_to_4L": stable_frac_4L,
        "stable_fraction_1L_to_8L": stable_frac_8L,
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.6), dpi=150)

    # Panel 1: (tau, n) plane with zero-centered diverging colormap
    norm1 = TwoSlopeNorm(vmin=-0.08, vcenter=0.0, vmax=0.18)
    im1 = ax1.contourf(
        tau_grid, n_grid, alpha_max_tn,
        levels=np.linspace(-0.08, 0.18, 27),
        cmap="RdBu_r",
        norm=norm1,
        extend="both",
    )
    ax1.contour(tau_grid, n_grid, alpha_max_tn, levels=[0.0], colors="black", linewidths=2.0)

    cos_t = np.cos(np.pi * tau_grid)
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
    ax1.set_title(
        f"Max Growth Rate $\\max_{{k \\leq 4}} \\alpha_k$ ($x_f={x_f_map},\\ \\zeta={zeta}$)",
        pad=10,
    )
    ax1.legend(loc="upper right", fontsize=8.5)
    plt.colorbar(im1, ax=ax1, label="Max growth rate $\\alpha_{\\max}$")

    # Panel 2: (x_f, tau) plane with zero-centered diverging colormap
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
    ax2.annotate("C (2L)", (0.10, 1.00), xytext=(0.04, 1.12), fontsize=8.3, fontweight="bold", bbox=lbl_box)
    ax2.annotate("D (4L)", (0.25, 1.00), xytext=(0.24, 1.13), fontsize=8.3, fontweight="bold", bbox=lbl_box)

    ax2.set_xlabel("Flame position $x_f / L$")
    ax2.set_ylabel("Dimensionless time lag $\\tau \\bar{c}/L$")
    ax2.set_title(
        f"Stability Islands ($n={n_fixed},\\ \\zeta={zeta}$): "
        f"1L={100*stable_frac_1L:.0f}%, 1-4L={100*stable_frac_4L:.0f}%, 1-8L={100*stable_frac_8L:.0f}%",
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
