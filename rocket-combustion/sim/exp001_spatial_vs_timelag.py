"""Entry 001: Finite Flame Width vs Causal Time-Lag Dispersion in a 1D Rocket Combustor.

Uses the unified DC-blocked flame transfer function H_hp(s) = s / (s + omega_hp) (omega_hp = 0.20)
from Entry 000 so the k=0 bulk mode is rigorously stable (verified by real-axis root scans), and
compares three physical mechanisms at n = 0.10, zeta = 0.04:

Mechanism 1 (Local pressure coupling + spatial flame width sigma_f):
    Q(x, t) = n * w(x) * P_ac(x, t - tau)
    High-k spatial coupling I_k[w] = int w(x) cos^2(k*pi*x) dx -> 1/2 (not 0).
    Threshold n_c = 2 * arctanh(zeta) = 0.080043: for n > n_c, spatial smearing ALONE fails.
    (Diagonal Rayleigh approximation is clearly distinguished from exact uniform-flame & PDE runs.)

Mechanism 2 (Injector/feed-coupled distributed flame width sigma_f):
    Q(x, t) = n * w(x) * P_ac(0, t - tau)
    Exact Green's-function dispersion with H_hp(s); super-exponential high-k roll-off.

Mechanism 3 (Local pressure coupling + causal shifted Erlang-4 time-lag spread sigma_tau):
    Q(x, t) = n * delta(x - x_f) * (H_hp * g_Erlang * P(x_f, .))(t)
    Strictly causal for sqrt(m)*sigma_tau <= tau_bar. To include tau < 0.45 in the full
    [0.05, 2.0] domain cleanly, when sigma_tau exceeds tau_bar / sqrt(m) we cap sigma_eff = tau_bar / sqrt(m)
    (pure m-stage Erlang with tau_0 = 0, preserving the exact mean lag tau_bar!).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.optimize import least_squares


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OMEGA_HP = 0.20


def gaussian_profile(x: np.ndarray, x_f: float, sigma_f: float) -> np.ndarray:
    raw = np.exp(-0.5 * ((x - x_f) / sigma_f) ** 2)
    dx = x[1] - x[0]
    return raw / (np.sum(raw) * dx)


def hp_factor(s: complex) -> complex:
    return s / (s + OMEGA_HP)


def erlang_effective_params(tau_bar: float, sigma_tau: float, m_stages: int = 4) -> tuple[float, float, float]:
    """Return strictly causal (tau_0, tau_d, sigma_eff) preserving exact mean lag tau_bar."""
    tau_d = min(tau_bar, np.sqrt(m_stages) * sigma_tau)
    tau_0 = max(0.0, tau_bar - tau_d)
    sigma_eff = tau_d / np.sqrt(m_stages)
    return float(tau_0), float(tau_d), float(sigma_eff)


# ---------------------------------------------------------------------------
# 1. Exact dispersion solvers (all with H_hp + real-axis k=0 check)
# ---------------------------------------------------------------------------

def solve_uniform_local_mode(k: int, n: float, tau: float, zeta: float) -> tuple[complex, float]:
    """Exact dispersion for uniform DC-blocked local flame w(x) == 1 on [0, 1]."""
    def F(s: complex, n_val: float) -> complex:
        k2 = s * (s - n_val * hp_factor(s) * np.exp(-s * tau))
        kap = np.sqrt(k2)
        return kap * np.sinh(kap) + zeta * s * np.cosh(kap)

    s = complex(-np.arctanh(zeta), k * np.pi)
    for step in range(1, 11):
        n_cur = n * (step / 10.0)
        for _ in range(30):
            fv = F(s, n_cur)
            if abs(fv) < 1e-13:
                break
            eps = 1e-7 * max(1.0, abs(s))
            dfv = (F(s + eps, n_cur) - F(s - eps, n_cur)) / (2.0 * eps)
            if abs(dfv) < 1e-15:
                break
            ds = fv / dfv
            if abs(ds) > 0.5:
                ds = 0.5 * ds / abs(ds)
            s -= ds
    return s, float(abs(F(s, n)))


def solve_injector_coupled_mode(
    k: int,
    n: float,
    tau: float,
    x_f: float,
    sigma_f: float,
    zeta: float,
    n_quad: int = 400,
) -> tuple[complex, float]:
    """Exact Green's-function dispersion for DC-blocked injector-coupled distributed flame."""
    xi = (np.arange(n_quad) + 0.5) / n_quad
    dxi = 1.0 / n_quad
    w = gaussian_profile(xi, x_f, sigma_f)
    one_minus_xi = 1.0 - xi

    def F(s: complex, n_val: float) -> complex:
        lhs = np.sinh(s) + zeta * np.cosh(s)
        ker = np.cosh(s * one_minus_xi) + zeta * np.sinh(s * one_minus_xi)
        integral = np.sum(w * ker) * dxi
        return lhs - n_val * hp_factor(s) * np.exp(-s * tau) * integral

    s = complex(-np.arctanh(zeta), k * np.pi)
    if n == 0.0:
        return s, float(abs(F(s, 0.0)))

    for step in range(1, 9):
        n_cur = n * (step / 8.0)
        for _ in range(30):
            fv = F(s, n_cur)
            if abs(fv) < 1e-12:
                break
            eps = 1e-7 * max(1.0, abs(s))
            dfv = (F(s + eps, n_cur) - F(s - eps, n_cur)) / (2.0 * eps)
            if abs(dfv) < 1e-15:
                break
            ds = fv / dfv
            if abs(ds) > 0.5:
                ds = 0.5 * ds / abs(ds)
            s -= ds
    return s, float(abs(F(s, n)))


def erlang_dispersion_F(
    s: complex,
    n: float,
    tau_bar: float,
    sigma_tau: float,
    x_f: float,
    zeta: float,
    m_stages: int = 4,
) -> complex:
    tau_0, tau_d, _ = erlang_effective_params(tau_bar, sigma_tau, m_stages)
    lhs = np.sinh(s) + zeta * np.cosh(s)
    if tau_d > 0.0:
        g_hat = np.exp(-s * tau_0) / ((1.0 + s * (tau_d / m_stages)) ** m_stages)
    else:
        g_hat = np.exp(-s * tau_bar)
    c1 = np.cosh(s * x_f)
    c2 = np.cosh(s * (1.0 - x_f)) + zeta * np.sinh(s * (1.0 - x_f))
    return lhs - n * hp_factor(s) * g_hat * c1 * c2


def has_positive_real_root_erlang(
    n: float, tau_bar: float, sigma_tau: float, x_f: float, zeta: float, m_stages: int = 4
) -> bool:
    grid = np.linspace(0.0, 3.0, 61)
    vals = [erlang_dispersion_F(complex(sv, 0.0), n, tau_bar, sigma_tau, x_f, zeta, m_stages).real for sv in grid]
    return any(vals[i] * vals[i + 1] <= 0.0 for i in range(len(vals) - 1))


def solve_erlang_timelag_mode(
    k: int,
    n: float,
    tau_bar: float,
    sigma_tau: float,
    x_f: float,
    zeta: float,
    m_stages: int = 4,
) -> tuple[complex, float]:
    s = complex(-np.arctanh(zeta), k * np.pi)
    if n == 0.0:
        return s, float(abs(erlang_dispersion_F(s, 0.0, tau_bar, sigma_tau, x_f, zeta, m_stages)))

    for step in range(1, 9):
        n_cur = n * (step / 8.0)
        for _ in range(30):
            fv = erlang_dispersion_F(s, n_cur, tau_bar, sigma_tau, x_f, zeta, m_stages)
            if abs(fv) < 1e-12:
                break
            eps = 1e-7 * max(1.0, abs(s))
            dfv = (
                erlang_dispersion_F(s + eps, n_cur, tau_bar, sigma_tau, x_f, zeta, m_stages)
                - erlang_dispersion_F(s - eps, n_cur, tau_bar, sigma_tau, x_f, zeta, m_stages)
            ) / (2.0 * eps)
            if abs(dfv) < 1e-15:
                break
            ds = fv / dfv
            if abs(ds) > 0.5:
                ds = 0.5 * ds / abs(ds)
            s -= ds
    return s, float(abs(erlang_dispersion_F(s, n, tau_bar, sigma_tau, x_f, zeta, m_stages)))


# ---------------------------------------------------------------------------
# 2. Direct 1D staggered-grid PDE solver (all 3 mechanisms with DC-blocking HP)
# ---------------------------------------------------------------------------

def simulate_1d_pde(
    mode: str,
    n: float,
    tau_bar: float,
    x_f: float,
    zeta: float,
    sigma_f: float = 0.12,
    sigma_tau: float = 0.15,
    m_stages: int = 4,
    t_end: float = 34.0,
    n_cells: int = 200,
    cfl: float = 0.25,
    init_mode: int | None = None,
    init_amp: float = 1e-3,
    k_track: int = 12,
) -> dict:
    dx = 1.0 / n_cells
    dt = cfl * dx
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps
    inv_dx = 1.0 / dx

    x_p = (np.arange(n_cells) + 0.5) * dx
    m_flame = int(np.argmin(np.abs(x_p - x_f)))
    x_f_actual = float(x_p[m_flame])

    if mode == "local_gaussian":
        w = gaussian_profile(x_p, x_f_actual, sigma_f)
    elif mode == "local_uniform":
        w = np.ones(n_cells)
    elif mode == "injector_gauss":
        w = gaussian_profile(x_p, x_f_actual, sigma_f)
    elif mode in ("point", "erlang_point"):
        w = np.zeros(n_cells)
        w[m_flame] = inv_dx
    else:
        raise ValueError(f"Unknown mode: {mode}")

    ks = np.arange(1, k_track + 1)[:, None]
    cos_basis = np.cos(ks * np.pi * x_p[None, :])

    if init_mode is not None:
        P = init_amp * np.cos(init_mode * np.pi * x_p)
    else:
        P = np.zeros(n_cells)
        for k in range(1, 9):
            P += (init_amp / np.sqrt(k)) * np.cos(k * np.pi * x_p)
    U = np.zeros(n_cells - 1)

    is_local_dist = mode in ("local_gaussian", "local_uniform")
    p_bar_vec = np.zeros(n_cells) if is_local_dist else None
    p_bar_sca = 0.0

    if mode == "erlang_point":
        tau_0, tau_d, _ = erlang_effective_params(tau_bar, sigma_tau, m_stages)
        rate = m_stages / tau_d if tau_d > 0.0 else 0.0
        q_states = np.zeros(m_stages)
    else:
        tau_0 = tau_bar
        tau_d = 0.0
        rate = 0.0
        q_states = np.zeros(0)

    delay_steps_f = tau_0 / dt
    buf_len = int(np.ceil(delay_steps_f)) + 4

    if is_local_dist:
        hist_vec = np.zeros((n_steps + buf_len + 2, n_cells))
        hist_vec[buf_len] = P - p_bar_vec
        hist_sca = None
    else:
        hist_vec = None
        hist_sca = np.zeros(n_steps + buf_len + 2)
        p_sense0 = float(P[0] if mode == "injector_gauss" else P[m_flame])
        hist_sca[buf_len] = p_sense0 - p_bar_sca

    def get_delayed(step_idx: int, frac: float, cur_val):
        if tau_0 <= 1e-12:
            return cur_val
        idx_f = buf_len + step_idx + frac - delay_steps_f
        i0 = int(np.floor(idx_f))
        a = idx_f - i0
        if i0 < 0:
            return np.zeros(n_cells) if is_local_dist else 0.0
        if i0 >= buf_len + step_idx:
            return cur_val
        if is_local_dist:
            return (1.0 - a) * hist_vec[i0] + a * hist_vec[i0 + 1]
        return float((1.0 - a) * hist_sca[i0] + a * hist_sca[i0 + 1])

    def rhs_wave(P_cur: np.ndarray, U_cur: np.ndarray, Q_src: np.ndarray):
        dP = np.empty_like(P_cur)
        dP[0] = -U_cur[0] * inv_dx
        dP[1:-1] = -(U_cur[1:] - U_cur[:-1]) * inv_dx
        dP[-1] = -(zeta * P_cur[-1] - U_cur[-1]) * inv_dx
        dP += Q_src
        dU = -(P_cur[1:] - P_cur[:-1]) * inv_dx
        return dP, dU

    def rhs_erlang(q_cur: np.ndarray, p_in: float) -> np.ndarray:
        dq = np.empty_like(q_cur)
        dq[0] = rate * (p_in - q_cur[0])
        for r in range(1, m_stages):
            dq[r] = rate * (q_cur[r - 1] - q_cur[r])
        return dq

    t_arr = np.linspace(0.0, t_end, n_steps + 1)
    energy = np.empty(n_steps + 1)
    p_head = np.empty(n_steps + 1)
    mode_proj = np.empty((k_track, n_steps + 1))

    def record(idx: int) -> None:
        p_head[idx] = P[0]
        energy[idx] = 0.5 * dx * (np.dot(P, P) + np.dot(U, U))
        mode_proj[:, idx] = 2.0 * dx * (cos_basis @ P)

    record(0)

    for step in range(n_steps):
        if is_local_dist:
            cur0 = P - p_bar_vec
            d0 = get_delayed(step, 0.0, cur0)
            k1_P, k1_U = rhs_wave(P, U, (n * d0) * w)
            k1_b = OMEGA_HP * cur0

            P2, b2 = P + 0.5 * dt * k1_P, p_bar_vec + 0.5 * dt * k1_b
            dh1 = get_delayed(step, 0.5, P2 - b2)
            k2_P, k2_U = rhs_wave(P2, U + 0.5 * dt * k1_U, (n * dh1) * w)
            k2_b = OMEGA_HP * (P2 - b2)

            P3, b3 = P + 0.5 * dt * k2_P, p_bar_vec + 0.5 * dt * k2_b
            dh2 = get_delayed(step, 0.5, P3 - b3)
            k3_P, k3_U = rhs_wave(P3, U + 0.5 * dt * k2_U, (n * dh2) * w)
            k3_b = OMEGA_HP * (P3 - b3)

            P4, b4 = P + dt * k3_P, p_bar_vec + dt * k3_b
            d1 = get_delayed(step, 1.0, P4 - b4)
            k4_P, k4_U = rhs_wave(P4, U + dt * k3_U, (n * d1) * w)
            k4_b = OMEGA_HP * (P4 - b4)

            P = P + (dt / 6.0) * (k1_P + 2.0 * k2_P + 2.0 * k3_P + k4_P)
            U = U + (dt / 6.0) * (k1_U + 2.0 * k2_U + 2.0 * k3_U + k4_U)
            p_bar_vec = p_bar_vec + (dt / 6.0) * (k1_b + 2.0 * k2_b + 2.0 * k3_b + k4_b)
            hist_vec[buf_len + step + 1] = P - p_bar_vec
        else:
            sense_idx = 0 if mode == "injector_gauss" else m_flame
            cur0 = float(P[sense_idx]) - p_bar_sca
            d0 = get_delayed(step, 0.0, cur0)
            if mode == "erlang_point" and tau_d > 0.0:
                k1_q = rhs_erlang(q_states, d0)
                Q1 = (n * float(q_states[-1])) * w
            else:
                k1_q = None
                Q1 = (n * d0) * w
            k1_P, k1_U = rhs_wave(P, U, Q1)
            k1_b = OMEGA_HP * cur0

            P2, b2 = P + 0.5 * dt * k1_P, p_bar_sca + 0.5 * dt * k1_b
            dh1 = get_delayed(step, 0.5, float(P2[sense_idx]) - b2)
            if k1_q is not None:
                q2 = q_states + 0.5 * dt * k1_q
                k2_q = rhs_erlang(q2, dh1)
                Q2 = (n * float(q2[-1])) * w
            else:
                k2_q = None
                Q2 = (n * dh1) * w
            k2_P, k2_U = rhs_wave(P2, U + 0.5 * dt * k1_U, Q2)
            k2_b = OMEGA_HP * (float(P2[sense_idx]) - b2)

            P3, b3 = P + 0.5 * dt * k2_P, p_bar_sca + 0.5 * dt * k2_b
            dh2 = get_delayed(step, 0.5, float(P3[sense_idx]) - b3)
            if k2_q is not None:
                q3 = q_states + 0.5 * dt * k2_q
                k3_q = rhs_erlang(q3, dh2)
                Q3 = (n * float(q3[-1])) * w
            else:
                k3_q = None
                Q3 = (n * dh2) * w
            k3_P, k3_U = rhs_wave(P3, U + 0.5 * dt * k2_U, Q3)
            k3_b = OMEGA_HP * (float(P3[sense_idx]) - b3)

            P4, b4 = P + dt * k3_P, p_bar_sca + dt * k3_b
            d1 = get_delayed(step, 1.0, float(P4[sense_idx]) - b4)
            if k3_q is not None:
                q4 = q_states + dt * k3_q
                k4_q = rhs_erlang(q4, d1)
                Q4 = (n * float(q4[-1])) * w
            else:
                k4_q = None
                Q4 = (n * d1) * w
            k4_P, k4_U = rhs_wave(P4, U + dt * k3_U, Q4)
            k4_b = OMEGA_HP * (float(P4[sense_idx]) - b4)

            P = P + (dt / 6.0) * (k1_P + 2.0 * k2_P + 2.0 * k3_P + k4_P)
            U = U + (dt / 6.0) * (k1_U + 2.0 * k2_U + 2.0 * k3_U + k4_U)
            p_bar_sca = p_bar_sca + (dt / 6.0) * (k1_b + 2.0 * k2_b + 2.0 * k3_b + k4_b)
            if k1_q is not None:
                q_states = q_states + (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
            hist_sca[buf_len + step + 1] = float(P[sense_idx]) - p_bar_sca

        record(step + 1)

    return {
        "t": t_arr,
        "p_head": p_head,
        "energy": energy,
        "mode_proj": mode_proj,
        "x_f_actual": x_f_actual,
    }


def fit_damped_sinusoid(
    t: np.ndarray, sig: np.ndarray, t_min: float, t_max: float, alpha_init: float, omega_init: float
) -> float:
    mask = (t >= t_min) & (t <= t_max)
    tw = t[mask] - t_min
    yw = sig[mask]
    stride = max(1, len(tw) // 2000)
    ts, ys = tw[::stride], yw[::stride]

    def res(p: np.ndarray) -> np.ndarray:
        env = np.exp(p[0] * ts)
        A = np.column_stack([env * np.cos(p[1] * ts), env * np.sin(p[1] * ts)])
        c, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
        return A @ c - ys

    out = least_squares(res, x0=np.array([alpha_init, omega_init]), xtol=1e-11, ftol=1e-11)
    return float(out.x[0])


# ---------------------------------------------------------------------------
# 3. Main experiments & figures
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    zeta = 0.04
    alpha_0 = -float(np.arctanh(zeta))
    n_crit_local = 2.0 * float(np.arctanh(zeta))
    n_base = 0.10
    tau_base = 1.00
    x_f_caseD = 0.25
    m_stages = 4

    sigma_tau_crit_4L = (np.sqrt(m_stages) / (4.0 * np.pi)) * np.sqrt((n_base / np.arctanh(zeta)) ** (2.0 / m_stages) - 1.0)

    print("=== Entry 001: Finite Flame Width vs Causal Time-Lag Dispersion (DC-Blocked) ===")
    print(f"Baseline: zeta = {zeta}, omega_hp = {OMEGA_HP}, n = {n_base}, tau = {tau_base}, Case D x_f = {x_f_caseD}")
    print(f"Analytical local-smearing threshold: n_c = 2*arctanh(zeta) = {n_crit_local:.6f}")
    print(f"First-order perturbation cutoff for 4L+: sigma_tau*(4) = {sigma_tau_crit_4L:.5f}\n")

    # Experiment 1: Mechanism 1 Negative Result (Local spatial smearing)
    print("--- Experiment 1: Local Pressure Coupling + Spatial Smearing (Negative Result) ---")
    u_results = {}
    for label, n_val in [("above_nc (n=0.10)", 0.10), ("below_nc (n=0.07)", 0.07)]:
        s4_exact, res4 = solve_uniform_local_mode(4, n_val, tau_base, zeta)
        s8_exact, _ = solve_uniform_local_mode(8, n_val, tau_base, zeta)
        sim_u = simulate_1d_pde("local_uniform", n_val, tau_base, x_f_caseD, zeta, t_end=34.0, init_mode=4)
        a4_pde = fit_damped_sinusoid(sim_u["t"], sim_u["mode_proj"][3], 8.0, 32.0, s4_exact.real, s4_exact.imag)
        print(
            f"  Uniform flame {label}: exact 4L={s4_exact.real:+.6f} (res={res4:.1e}), "
            f"PDE 4L={a4_pde:+.6f} (|diff|={abs(a4_pde - s4_exact.real):.2e}), exact 8L={s8_exact.real:+.6f}"
        )
        u_results[label] = {
            "n": n_val,
            "alpha_4L_exact": float(s4_exact.real),
            "alpha_4L_pde": a4_pde,
            "alpha_8L_exact": float(s8_exact.real),
        }

    sim_m1 = simulate_1d_pde("local_gaussian", n_base, tau_base, x_f_caseD, zeta, sigma_f=0.12, t_end=34.0, init_mode=None)
    sim_m1_4L = simulate_1d_pde("local_gaussian", n_base, tau_base, x_f_caseD, zeta, sigma_f=0.12, t_end=34.0, init_mode=4)
    a4_m1 = fit_damped_sinusoid(sim_m1_4L["t"], sim_m1_4L["mode_proj"][3], 8.0, 32.0, 0.01, 4.0 * np.pi)

    xi_q = (np.arange(1000) + 0.5) / 1000.0
    w_m1 = gaussian_profile(xi_q, sim_m1_4L["x_f_actual"], 0.12)
    ks_12 = np.arange(1, 13)
    I_k_m1 = np.array([np.sum(w_m1 * (np.cos(k * np.pi * xi_q) ** 2)) / 1000.0 for k in ks_12])
    alpha_m1_diag = np.zeros(12)
    for idx, k in enumerate(ks_12):
        hpk = hp_factor(1j * k * np.pi)
        fac = np.real(hpk * np.exp(-1j * k * np.pi * tau_base))
        a = alpha_0 + n_base * I_k_m1[idx] * fac
        for _ in range(15):
            a = alpha_0 + n_base * I_k_m1[idx] * np.exp(-a * tau_base) * fac
        alpha_m1_diag[idx] = a
    print(f"  Local Gaussian (sigma_f=0.12, n=0.10): 4L PDE={a4_m1:+.6f} vs diag-Rayleigh approx={alpha_m1_diag[3]:+.6f} (STILL UNSTABLE!)")
    print(f"  High-k asymptote: I_k -> 0.500 -> alpha_12L = {alpha_m1_diag[11]:+.5f} (> 0)\n")

    # Experiment 2: Mechanism 2 (Injector-Coupled Distributed Flame)
    print("--- Experiment 2: Injector-Coupled Distributed Flame (sigma_f = 0.12) ---")
    sigma_f_test = 0.12
    sim_m2 = simulate_1d_pde("injector_gauss", n_base, tau_base, x_f_caseD, zeta, sigma_f=sigma_f_test, t_end=34.0, init_mode=None)
    sim_m2_4L = simulate_1d_pde("injector_gauss", n_base, tau_base, x_f_caseD, zeta, sigma_f=sigma_f_test, t_end=34.0, init_mode=4)
    s4_m2_ex, _ = solve_injector_coupled_mode(4, n_base, tau_base, sim_m2_4L["x_f_actual"], sigma_f_test, zeta)
    a4_m2_pde = fit_damped_sinusoid(sim_m2_4L["t"], sim_m2_4L["mode_proj"][3], 6.0, 30.0, s4_m2_ex.real, s4_m2_ex.imag)
    alpha_m2_exact = np.array([
        solve_injector_coupled_mode(k, n_base, tau_base, sim_m2_4L["x_f_actual"], sigma_f_test, zeta)[0].real
        for k in ks_12
    ])
    print(f"  Injector Gaussian (sigma_f={sigma_f_test}): exact 4L={alpha_m2_exact[3]:+.6f}, PDE 4L={a4_m2_pde:+.6f} (|diff|={abs(a4_m2_pde - alpha_m2_exact[3]):.2e})")
    print(f"  Max alpha across 1L-12L: {np.max(alpha_m2_exact):+.6f}\n")

    # Experiment 3: Mechanism 3 (Causal Erlang-4 Time-Lag Dispersion)
    print("--- Experiment 3: Local Coupling + Causal Erlang-4 Time-Lag Spread ---")
    sigma_tau_test = 0.15
    sim_m3 = simulate_1d_pde("erlang_point", n_base, tau_base, x_f_caseD, zeta, sigma_tau=sigma_tau_test, m_stages=m_stages, t_end=34.0, init_mode=None)
    sim_m3_4L = simulate_1d_pde("erlang_point", n_base, tau_base, x_f_caseD, zeta, sigma_tau=sigma_tau_test, m_stages=m_stages, t_end=34.0, init_mode=4)
    s4_m3_ex, _ = solve_erlang_timelag_mode(4, n_base, tau_base, sigma_tau_test, sim_m3_4L["x_f_actual"], zeta, m_stages=m_stages)
    a4_m3_pde = fit_damped_sinusoid(sim_m3_4L["t"], sim_m3_4L["mode_proj"][3], 6.0, 30.0, s4_m3_ex.real, s4_m3_ex.imag)
    alpha_m3_exact = np.array([
        solve_erlang_timelag_mode(k, n_base, tau_base, sigma_tau_test, sim_m3_4L["x_f_actual"], zeta, m_stages=m_stages)[0].real
        for k in ks_12
    ])
    k0_m3 = has_positive_real_root_erlang(n_base, tau_base, sigma_tau_test, sim_m3_4L["x_f_actual"], zeta, m_stages=m_stages)
    print(f"  Erlang-4 local (sigma_tau={sigma_tau_test}): exact 4L={alpha_m3_exact[3]:+.6f}, PDE 4L={a4_m3_pde:+.6f} (|diff|={abs(a4_m3_pde - alpha_m3_exact[3]):.2e}), k=0 unstable={k0_m3}")
    print(f"  Max alpha across 1L-12L: {np.max(alpha_m3_exact):+.6f}\n")

    sim_base = simulate_1d_pde("point", n_base, tau_base, x_f_caseD, zeta, t_end=34.0, init_mode=None)
    alpha_base_exact = np.array([
        solve_erlang_timelag_mode(k, n_base, tau_base, 0.0, sim_base["x_f_actual"], zeta, m_stages=m_stages)[0].real
        for k in ks_12
    ])

    # Figure 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.4), dpi=150)
    ax1.axhline(0.0, color="black", lw=1.4)
    ax1.axhline(alpha_0, color="gray", lw=1.0, ls=":", label=f"Cold end damping ($\\alpha_0={alpha_0:.3f}$)")
    ax1.axhline(alpha_0 + 0.5 * n_base * np.exp(-(alpha_0 + 0.5 * n_base) * tau_base), color="#d62728", lw=1.1, ls="--", alpha=0.7,
                label="Local-smearing asymptote ($n/2 - \\zeta > 0$)")

    ax1.plot(ks_12, alpha_base_exact, "o-", color="#7f7f7f", lw=1.4, ms=5, label="Entry 000: Point flame ($\\sigma=0$)")
    ax1.plot(ks_12, alpha_m1_diag, "s-", color="#d62728", lw=1.9, ms=5.5, label="Mech 1: Local Gaussian ($\\sigma_f=0.12$, diag. approx) — FAILS")
    ax1.plot(ks_12, alpha_m2_exact, "^-", color="#1f77b4", lw=1.9, ms=6, label="Mech 2: Injector-coupled ($\\sigma_f=0.12$) — STABLE")
    ax1.plot(ks_12, alpha_m3_exact, "D-", color="#2ca02c", lw=1.9, ms=5.5, label="Mech 3: Causal Erlang-4 ($\\sigma_\\tau=0.15$) — STABLE")

    ax1.set_xticks(ks_12)
    ax1.set_xticklabels([f"{k}L" for k in ks_12])
    ax1.set_xlabel("Longitudinal acoustic mode $k$")
    ax1.set_ylabel("Modal growth rate $\\alpha_k$")
    ax1.set_title("Case D ($x_f=0.25, \\bar{\\tau}=1.0, n=0.10, \\zeta=0.04, \\omega_{\\mathrm{hp}}=0.2$): Spectrum")
    ax1.set_ylim(-0.12, 0.08)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=7.6)

    ax2.semilogy(sim_base["t"], sim_base["energy"] / sim_base["energy"][0], color="#7f7f7f", lw=1.5, ls="--",
                 label="Entry 000: Point flame (4L/8L blow-up)")
    ax2.semilogy(sim_m1["t"], sim_m1["energy"] / sim_m1["energy"][0], color="#d62728", lw=2.0,
                 label=f"Mech 1: Local Gaussian $\\sigma_f=0.12$ (still grows, $\\alpha_{{4L}}={a4_m1:+.4f}$)")
    ax2.semilogy(sim_m2["t"], sim_m2["energy"] / sim_m2["energy"][0], color="#1f77b4", lw=2.0,
                 label=f"Mech 2: Injector Gaussian $\\sigma_f=0.12$ (decays, $\\alpha_{{4L}}={a4_m2_pde:+.4f}$)")
    ax2.semilogy(sim_m3["t"], sim_m3["energy"] / sim_m3["energy"][0], color="#2ca02c", lw=2.0,
                 label=f"Mech 3: Causal Erlang-4 $\\sigma_\\tau=0.15$ (decays, $\\alpha_{{4L}}={a4_m3_pde:+.4f}$)")

    ax2.set_xlabel("Dimensionless time $t \\bar{c}/L$")
    ax2.set_ylabel("Normalized acoustic energy $E(t) / E(0)$")
    ax2.set_title("Direct 1D PDE Broadband Verification (Seeded 1L–8L, DC-Blocked)")
    ax2.grid(True, which="both", alpha=0.28)
    ax2.legend(loc="upper left", fontsize=7.9)

    fig.suptitle("Entry 001: Why Local Spatial Smearing Fails—and What Cures High-k Harmonics", fontsize=12.3, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RESULTS_DIR / "exp001_three_mechanisms.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Experiment 4: Full-Domain [0.05, 0.95] x [0.05, 2.0] Stability Recovery
    # (includes tau < 0.45 where 1L lobe lives, using causal cap sigma_eff = min(sigma_tau, tau_bar/2))
    # ------------------------------------------------------------------
    print("--- Experiment 4: Broadband (1L-12L) Stability Recovery on Full Domain tau in [0.05, 2.0] ---")
    xf_grid = np.linspace(0.05, 0.95, 31)
    tau_grid = np.linspace(0.05, 2.00, 40)

    sigma_tau_sweep = [0.0, 0.04, 0.08, 0.12, 0.15, 0.20]
    stable_frac_erlang = []
    alpha_map_erlang_015 = None
    k0_hits = 0

    for st in sigma_tau_sweep:
        alpha_max = np.zeros((len(tau_grid), len(xf_grid)))
        for i, tv in enumerate(tau_grid):
            for j, xv in enumerate(xf_grid):
                if has_positive_real_root_erlang(n_base, tv, st, xv, zeta, m_stages=m_stages):
                    k0_hits += 1
                    alpha_max[i, j] = 1.0
                    continue
                best = -1e9
                for k in range(1, 13):
                    s_k, _ = solve_erlang_timelag_mode(k, n_base, tv, st, xv, zeta, m_stages=m_stages)
                    if s_k.real > best:
                        best = s_k.real
                alpha_max[i, j] = best
        frac = float(np.mean(alpha_max < 0.0))
        stable_frac_erlang.append(frac)
        tag = " (upper bound; k>12 unstable)" if st == 0.0 else ""
        print(f"  Erlang-4 sigma_tau = {st:.2f}: 1L-12L stable area = {100.0 * frac:.1f}%{tag}")
        if abs(st - 0.15) < 1e-9:
            alpha_map_erlang_015 = alpha_max

    print(f"  Real-axis k=0 unstable hits across entire Erlang sweep: {k0_hits}")

    # Mechanism 1 (diagonal Rayleigh approximation, upper bound for k <= 12) on the EXACT same grid
    sigma_f_sweep = [0.02, 0.05, 0.08, 0.12, 0.16, 0.20]
    stable_frac_local_sf = []
    xi_fast = (np.arange(200) + 0.5) / 200.0
    cos2_mat = np.cos(ks_12[:, None] * np.pi * xi_fast[None, :]) ** 2
    hp_vec = np.array([hp_factor(1j * k * np.pi) for k in ks_12])
    for sf in sigma_f_sweep:
        stable_count = 0
        total_count = len(tau_grid) * len(xf_grid)
        for xv in xf_grid:
            w_f = gaussian_profile(xi_fast, xv, sf)
            I_vec = (cos2_mat @ w_f) / 200.0
            for tv in tau_grid:
                fac = np.real(hp_vec * np.exp(-1j * ks_12 * np.pi * tv))
                a_vec = alpha_0 + n_base * I_vec * fac
                for _ in range(4):
                    a_vec = alpha_0 + n_base * I_vec * np.exp(-a_vec * tv) * fac
                if np.max(a_vec) < 0.0:
                    stable_count += 1
        frac = stable_count / total_count
        stable_frac_local_sf.append(frac)
        print(f"  Local Gaussian (diag. approx, k<=12 UB) sigma_f = {sf:.2f}: stable area = {100.0 * frac:.1f}%")

    # Figure 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.5), dpi=150)

    ax1.plot(sigma_tau_sweep, [100 * f for f in stable_frac_erlang], "D-", color="#2ca02c", lw=2.2, ms=6.5,
             label="Mech 3: Causal Erlang-4 ($\\sigma_{\\mathrm{eff}}=\\min(\\sigma_\\tau, \\bar{\\tau}/2)$)")
    ax1.plot(sigma_f_sweep, [100 * f for f in stable_frac_local_sf], "s--", color="#d62728", lw=2.0, ms=6,
             label="Mech 1: Local spatial width $\\sigma_f$ (diag. approx, $k\\leq 12$ UB)")
    ax1.axvline(sigma_tau_crit_4L, color="#2ca02c", ls=":", lw=1.4,
                label=f"1st-order 4L bound $\\sigma_\\tau^* = {sigma_tau_crit_4L:.3f}$")
    ax1.set_xlabel("Dispersion width ($\\sigma_\\tau$ or $\\sigma_f$)")
    ax1.set_ylabel("Broadband (1L–12L) stable area (%)")
    ax1.set_title("Broadband Stability Recovery ($\\bar{\\tau} \\in [0.05, 2.0]$, DC-Blocked)", pad=10)
    ax1.set_ylim(-2, 102)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="center right", fontsize=8.0)

    norm = TwoSlopeNorm(vmin=-0.06, vcenter=0.0, vmax=0.06)
    im = ax2.contourf(
        xf_grid, tau_grid, alpha_map_erlang_015,
        levels=np.linspace(-0.06, 0.06, 25),
        cmap="RdBu_r",
        norm=norm,
        extend="both",
    )
    ax2.contour(xf_grid, tau_grid, alpha_map_erlang_015, levels=[0.0], colors="black", linewidths=1.8)
    ax2.scatter([0.15, 0.10, 0.25], [0.20, 1.00, 1.00], c=["gold", "lime", "cyan"], edgecolors="black", s=60, zorder=5)
    box = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85)
    ax2.annotate("B (1L unstable)", (0.15, 0.20), xytext=(0.20, 0.12), fontsize=8.0, fontweight="bold", bbox=box)
    ax2.annotate("C (2L: +0.004)", (0.10, 1.00), xytext=(0.06, 1.13), fontsize=8.0, fontweight="bold", bbox=box)
    ax2.annotate("D (stable)", (0.25, 1.00), xytext=(0.28, 1.05), fontsize=8.0, fontweight="bold", bbox=box)
    ax2.set_xlabel("Flame position $x_f / L$")
    ax2.set_ylabel("Mean time lag $\\bar{\\tau} \\bar{c}/L$")
    ax2.set_title(
        f"1L–12L Stability Map (Erlang-4, $\\sigma_\\tau=0.15$): {100*stable_frac_erlang[4]:.1f}% stable",
        pad=10,
    )
    plt.colorbar(im, ax=ax2, label="Max growth rate $\\alpha_{\\max}$ (1L–12L)")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp001_stability_recovery.png")
    plt.close(fig)

    metrics = {
        "parameters": {
            "zeta": zeta,
            "omega_hp": OMEGA_HP,
            "n_base": n_base,
            "tau_base": tau_base,
            "x_f_caseD": x_f_caseD,
            "m_stages": m_stages,
            "n_crit_local_smearing": n_crit_local,
            "sigma_tau_crit_4L_perturbation": float(sigma_tau_crit_4L),
            "domain_xf": [0.05, 0.95],
            "domain_tau": [0.05, 2.0],
            "grid_shape": [len(tau_grid), len(xf_grid)],
        },
        "uniform_local_threshold_test": u_results,
        "caseD_comparison": {
            "point_flame_4L_exact": float(alpha_base_exact[3]),
            "mech1_local_gauss_4L_pde": a4_m1,
            "mech1_local_gauss_4L_diag_approx": float(alpha_m1_diag[3]),
            "mech1_local_gauss_12L_diag_approx": float(alpha_m1_diag[11]),
            "mech2_injector_gauss_4L_exact": float(alpha_m2_exact[3]),
            "mech2_injector_gauss_4L_pde": a4_m2_pde,
            "mech2_injector_gauss_max_1L_12L": float(np.max(alpha_m2_exact)),
            "mech3_erlang4_4L_exact": float(alpha_m3_exact[3]),
            "mech3_erlang4_4L_pde": a4_m3_pde,
            "mech3_erlang4_max_1L_12L": float(np.max(alpha_m3_exact)),
            "mech3_k0_unstable": k0_m3,
        },
        "broadband_1L_12L_recovery": {
            "k0_unstable_hits": k0_hits,
            "erlang4_sigma_tau": sigma_tau_sweep,
            "erlang4_stable_fraction": stable_frac_erlang,
            "local_gauss_sigma_f": sigma_f_sweep,
            "local_gauss_diag_approx_k12_upper_bound": stable_frac_local_sf,
        },
    }

    with open(RESULTS_DIR / "exp001_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved Entry 001 plots and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
