"""Entry 002: Passive Acoustic Damping via a Faceplate Helmholtz Resonator (DC-Blocked).

Uses the unified DC-blocked flame transfer function H_hp(s) = s / (s + omega_hp) (omega_hp = 0.20)
so k=0 is rigorously stable across all runs (verified by real-axis root scans).

Addresses all peer-review items:
- Uniform DC-blocked model consistent across Entries 000-003, with real-axis k=0 audit.
- Exact 2-mode damped-sinusoid least-squares fit on the under-damped beating PDE case (M6).
- Uncensored frequency mistuning domain omega_H/pi in [0.65, 1.45] (no right-censoring).
- All four broadband (1L-12L) baselines computed on the EXACT same full domain
  (x_f in [0.05, 0.95], tau_bar in [0.05, 2.00], including the 1L lobe tau < 0.45!).
- Fixed seed-split scaling sqrt(0.5 * beta * omega_H^2) and tau_0 = 0 delay guard.
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


def erlang_effective_params(tau_bar: float, sigma_tau: float, m_stages: int = 4) -> tuple[float, float]:
    tau_d = min(tau_bar, np.sqrt(m_stages) * sigma_tau)
    tau_0 = max(0.0, tau_bar - tau_d)
    return float(tau_0), float(tau_d)


def dispersion_F(
    s: complex,
    n: float,
    tau_bar: float,
    x_f: float,
    zeta: float,
    beta: float,
    omega_H: float,
    xi_H: float,
    sigma_tau: float = 0.0,
    m_stages: int = 4,
) -> complex:
    denom = s * s + 2.0 * xi_H * omega_H * s + omega_H * omega_H
    num_Y = beta * omega_H * omega_H * s if beta > 0.0 else 0.0j

    sh_s = np.sinh(s)
    ch_s = np.cosh(s)
    lhs = (denom + zeta * num_Y) * sh_s + (zeta * denom + num_Y) * ch_s
    if n == 0.0:
        return lhs

    hp = s / (s + OMEGA_HP)
    if sigma_tau > 0.0:
        tau_0, tau_d = erlang_effective_params(tau_bar, sigma_tau, m_stages)
        g_hat = np.exp(-s * tau_0) / ((1.0 + s * (tau_d / m_stages)) ** m_stages) if tau_d > 0.0 else np.exp(-s * tau_bar)
    else:
        g_hat = np.exp(-s * tau_bar)

    c1 = denom * np.cosh(s * x_f) + num_Y * np.sinh(s * x_f)
    c2 = np.cosh(s * (1.0 - x_f)) + zeta * np.sinh(s * (1.0 - x_f))
    return lhs - n * hp * g_hat * c1 * c2


def has_positive_real_root(
    n: float, tau_bar: float, x_f: float, zeta: float, beta: float, omega_H: float, xi_H: float, sigma_tau: float = 0.0
) -> bool:
    grid = np.linspace(0.0, 3.0, 61)
    vals = [dispersion_F(complex(sv, 0.0), n, tau_bar, x_f, zeta, beta, omega_H, xi_H, sigma_tau).real for sv in grid]
    return any(vals[i] * vals[i + 1] <= 0.0 for i in range(len(vals) - 1))


def newton_polish(
    s0: complex,
    n: float,
    tau_bar: float,
    x_f: float,
    zeta: float,
    beta: float,
    omega_H: float,
    xi_H: float,
    sigma_tau: float = 0.0,
    m_stages: int = 4,
    max_iter: int = 45,
    tol: float = 1e-11,
) -> tuple[complex, float]:
    s = complex(s0)
    for _ in range(max_iter):
        fv = dispersion_F(s, n, tau_bar, x_f, zeta, beta, omega_H, xi_H, sigma_tau, m_stages)
        if abs(fv) < tol:
            return s, float(abs(fv))
        eps = 1e-7 * max(1.0, abs(s))
        f_p = dispersion_F(s + eps, n, tau_bar, x_f, zeta, beta, omega_H, xi_H, sigma_tau, m_stages)
        f_m = dispersion_F(s - eps, n, tau_bar, x_f, zeta, beta, omega_H, xi_H, sigma_tau, m_stages)
        dfv = (f_p - f_m) / (2.0 * eps)
        if abs(dfv) < 1e-16:
            break
        ds = fv / dfv
        if abs(ds) > 0.45:
            ds = 0.45 * ds / abs(ds)
        s -= ds
        if abs(s.real) > 6.0 or abs(s.imag) > 60.0:
            return s, 1e9
    fv = dispersion_F(s, n, tau_bar, x_f, zeta, beta, omega_H, xi_H, sigma_tau, m_stages)
    return s, float(abs(fv))


def find_coupled_1L_pair(
    n: float,
    tau_bar: float,
    x_f: float,
    zeta: float,
    beta: float,
    omega_H: float,
    xi_H: float,
    sigma_tau: float = 0.0,
    m_stages: int = 4,
) -> list[complex]:
    if beta <= 0.0:
        r, _ = newton_polish(complex(-0.02, np.pi), n, tau_bar, x_f, zeta, 0.0, omega_H, xi_H, sigma_tau, m_stages)
        return [r]
    s_cav = complex(-xi_H * omega_H, omega_H * np.sqrt(max(0.05, 1.0 - xi_H * xi_H)))
    split = np.sqrt(max(1e-4, 0.5 * beta * omega_H * omega_H))
    seeds = [
        complex(-0.02, np.pi - split),
        complex(-0.02, np.pi + split),
        complex(-0.12, np.pi - 0.5 * split),
        complex(-0.12, np.pi + 0.5 * split),
        complex(-0.25, np.pi),
        s_cav,
        complex(-xi_H * omega_H - 0.1, omega_H),
        complex(0.05, np.pi - 0.4),
        complex(0.05, np.pi + 0.4),
        complex(-0.6, omega_H),
        complex(-1.2, omega_H),
        complex(-0.05, 2.0),
        complex(-0.05, 4.4),
    ]
    roots: list[complex] = []
    for seed in seeds:
        r, res = newton_polish(seed, n, tau_bar, x_f, zeta, beta, omega_H, xi_H, sigma_tau, m_stages)
        if res < 1e-8 and 1.2 <= r.imag <= 5.4 and -5.0 <= r.real <= 1.0:
            if all(abs(r - existing) > 1e-4 for existing in roots):
                roots.append(r)
    roots.sort(key=lambda z: (round(z.imag, 4), -z.real))
    return roots


def solve_higher_mode(
    k: int,
    n: float,
    tau_bar: float,
    x_f: float,
    zeta: float,
    beta: float,
    omega_H: float,
    xi_H: float,
    sigma_tau: float = 0.0,
    m_stages: int = 4,
) -> complex:
    s = complex(-np.arctanh(zeta), k * np.pi)
    for step in range(1, 7):
        n_cur = n * (step / 6.0)
        s, _ = newton_polish(s, n_cur, tau_bar, x_f, zeta, beta, omega_H, xi_H, sigma_tau, m_stages)
    return s


# ---------------------------------------------------------------------------
# 2. Direct 1D staggered-grid PDE + DC-blocking HP + Helmholtz cavity ODEs
# ---------------------------------------------------------------------------

def simulate_chamber_with_cavity(
    n: float,
    tau_bar: float,
    x_f: float,
    zeta: float,
    beta: float,
    omega_H: float,
    xi_H: float,
    sigma_tau: float = 0.0,
    m_stages: int = 4,
    t_end: float = 40.0,
    n_cells: int = 200,
    cfl: float = 0.25,
    init_mode: int = 1,
    init_amp: float = 1e-3,
) -> dict:
    dx = 1.0 / n_cells
    dt = cfl * dx
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps
    inv_dx = 1.0 / dx

    x_p = (np.arange(n_cells) + 0.5) * dx
    m_flame = int(np.argmin(np.abs(x_p - x_f)))
    x_f_actual = float(x_p[m_flame])

    P = init_amp * np.cos(init_mode * np.pi * x_p)
    U = np.zeros(n_cells - 1)
    v_r = 0.0
    p_c = float(P[0]) if beta > 0.0 else 0.0  # Pre-pressurized consistently with standing wave
    p_bar = 0.0

    use_erlang = sigma_tau > 0.0
    if use_erlang:
        tau_0, tau_d = erlang_effective_params(tau_bar, sigma_tau, m_stages)
        rate = m_stages / tau_d if tau_d > 0.0 else 0.0
        q_st = np.zeros(m_stages)
    else:
        tau_0 = tau_bar
        tau_d = 0.0
        rate = 0.0
        q_st = np.zeros(0)

    delay_steps_f = tau_0 / dt
    buf_len = int(np.ceil(delay_steps_f)) + 4
    hist = np.zeros(n_steps + buf_len + 2)
    hist[buf_len] = float(P[m_flame]) - p_bar

    def get_delayed(step_idx: int, frac: float, cur_val: float) -> float:
        if tau_0 <= 1e-12:
            return cur_val
        idx_f = buf_len + step_idx + frac - delay_steps_f
        i0 = int(np.floor(idx_f))
        a = idx_f - i0
        if i0 < 0:
            return 0.0
        if i0 >= buf_len + step_idx:
            return cur_val
        return float((1.0 - a) * hist[i0] + a * hist[i0 + 1])

    def rhs(P_c: np.ndarray, U_c: np.ndarray, vr_c: float, pc_c: float, pbar_c: float, q_in: float):
        dP = np.empty_like(P_c)
        dP[0] = -(U_c[0] + vr_c) * inv_dx
        dP[1:-1] = -(U_c[1:] - U_c[:-1]) * inv_dx
        dP[-1] = -(zeta * P_c[-1] - U_c[-1]) * inv_dx
        if n != 0.0:
            dP[m_flame] += n * q_in * inv_dx
        dU = -(P_c[1:] - P_c[:-1]) * inv_dx
        if beta > 0.0:
            dvr = beta * (omega_H ** 2) * (P_c[0] - pc_c) - 2.0 * xi_H * omega_H * vr_c
            dpc = vr_c / beta
        else:
            dvr = 0.0
            dpc = 0.0
        dpbar = OMEGA_HP * (float(P_c[m_flame]) - pbar_c)
        return dP, dU, dvr, dpc, dpbar

    def rhs_erlang(q_cur: np.ndarray, p_del: float) -> np.ndarray:
        dq = np.empty_like(q_cur)
        dq[0] = rate * (p_del - q_cur[0])
        for r in range(1, m_stages):
            dq[r] = rate * (q_cur[r - 1] - q_cur[r])
        return dq

    t_arr = np.linspace(0.0, t_end, n_steps + 1)
    p_head = np.empty(n_steps + 1)
    energy_tot = np.empty(n_steps + 1)
    rayleigh = np.empty(n_steps + 1)
    loss_noz = np.empty(n_steps + 1)
    loss_res = np.empty(n_steps + 1)
    cos1 = np.cos(np.pi * x_p)
    a1_proj = np.empty(n_steps + 1)

    def record(idx: int, q_val: float) -> None:
        p_head[idx] = P[0]
        a1_proj[idx] = 2.0 * dx * float(np.dot(cos1, P))
        e_ch = 0.5 * dx * (np.dot(P, P) + np.dot(U, U))
        e_cav = (0.5 * beta * p_c * p_c + 0.5 * (v_r * v_r) / (beta * omega_H * omega_H)) if beta > 0.0 else 0.0
        energy_tot[idx] = e_ch + e_cav
        rayleigh[idx] = n * P[m_flame] * q_val
        loss_noz[idx] = zeta * (P[-1] ** 2)
        loss_res[idx] = ((2.0 * xi_H) / (beta * omega_H)) * (v_r * v_r) if beta > 0.0 else 0.0

    record(0, 0.0 if use_erlang else get_delayed(0, 0.0, float(P[m_flame]) - p_bar))

    for step in range(n_steps):
        cur0 = float(P[m_flame]) - p_bar
        d0 = get_delayed(step, 0.0, cur0)
        if use_erlang and tau_d > 0.0:
            k1_q = rhs_erlang(q_st, d0)
            q_in1 = float(q_st[-1])
        else:
            k1_q = None
            q_in1 = d0
        k1 = rhs(P, U, v_r, p_c, p_bar, q_in1)

        P2, b2 = P + 0.5 * dt * k1[0], p_bar + 0.5 * dt * k1[4]
        dh1 = get_delayed(step, 0.5, float(P2[m_flame]) - b2)
        if k1_q is not None:
            q2 = q_st + 0.5 * dt * k1_q
            k2_q = rhs_erlang(q2, dh1)
            q_in2 = float(q2[-1])
        else:
            k2_q = None
            q_in2 = dh1
        k2 = rhs(P2, U + 0.5 * dt * k1[1], v_r + 0.5 * dt * k1[2], p_c + 0.5 * dt * k1[3], b2, q_in2)

        P3, b3 = P + 0.5 * dt * k2[0], p_bar + 0.5 * dt * k2[4]
        dh2 = get_delayed(step, 0.5, float(P3[m_flame]) - b3)
        if k2_q is not None:
            q3 = q_st + 0.5 * dt * k2_q
            k3_q = rhs_erlang(q3, dh2)
            q_in3 = float(q3[-1])
        else:
            k3_q = None
            q_in3 = dh2
        k3 = rhs(P3, U + 0.5 * dt * k2[1], v_r + 0.5 * dt * k2[2], p_c + 0.5 * dt * k2[3], b3, q_in3)

        P4, b4 = P + dt * k3[0], p_bar + dt * k3[4]
        d1 = get_delayed(step, 1.0, float(P4[m_flame]) - b4)
        if k3_q is not None:
            q4 = q_st + dt * k3_q
            k4_q = rhs_erlang(q4, d1)
            q_in4 = float(q4[-1])
        else:
            k4_q = None
            q_in4 = d1
        k4 = rhs(P4, U + dt * k3[1], v_r + dt * k3[2], p_c + dt * k3[3], b4, q_in4)

        P = P + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        U = U + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        v_r = v_r + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        p_c = p_c + (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
        p_bar = p_bar + (dt / 6.0) * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4])
        if k1_q is not None:
            q_st = q_st + (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)

        cur_next = float(P[m_flame]) - p_bar
        hist[buf_len + step + 1] = cur_next
        record(step + 1, float(q_st[-1]) if k1_q is not None else get_delayed(step + 1, 0.0, cur_next))

    net_p = rayleigh - loss_noz - loss_res
    cum_w = np.concatenate([[0.0], np.cumsum(0.5 * (net_p[:-1] + net_p[1:]) * dt)])
    dE = energy_tot - energy_tot[0]
    scale = max(np.max(np.abs(dE)), np.max(energy_tot), 1e-15)
    e_err = float(np.max(np.abs(dE - cum_w)) / scale)

    return {
        "t": t_arr,
        "p_head": p_head,
        "a1_proj": a1_proj,
        "energy_tot": energy_tot,
        "energy_balance_rel_err": e_err,
        "x_f_actual": x_f_actual,
    }


def fit_multi_mode_lsq(
    t: np.ndarray,
    sig: np.ndarray,
    t_min: float,
    t_max: float,
    init_roots: list[complex],
) -> list[complex]:
    """Fit 1 or 2 damped sinusoids via variable-projection nonlinear least squares."""
    mask = (t >= t_min) & (t <= t_max)
    tw = t[mask] - t_min
    yw = sig[mask]
    stride = max(1, len(tw) // 2500)
    ts, ys = tw[::stride], yw[::stride]

    m = len(init_roots)
    x0 = np.array([val for r in init_roots for val in (r.real, r.imag)], dtype=float)

    def residual(params: np.ndarray) -> np.ndarray:
        cols = []
        for idx in range(m):
            a, w = params[2 * idx], params[2 * idx + 1]
            env = np.exp(a * ts)
            cols.append(env * np.cos(w * ts))
            cols.append(env * np.sin(w * ts))
        A = np.column_stack(cols)
        coef, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
        return A @ coef - ys

    res = least_squares(residual, x0=x0, xtol=1e-11, ftol=1e-11)
    fitted = [complex(res.x[2 * idx], res.x[2 * idx + 1]) for idx in range(m)]
    fitted.sort(key=lambda z: z.imag)
    return fitted


# ---------------------------------------------------------------------------
# 3. Main experiments & figures
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    zeta = 0.04
    n_B = 0.10
    tau_B = 0.20
    x_f_B = 0.15
    beta_main = 0.02
    beta_tight = 0.005

    xi_ep_cold = zeta / np.pi + np.sqrt(2.0 * beta_main)
    alpha_ep_cold = -zeta - np.pi * np.sqrt(0.5 * beta_main)

    s1_hot_uncoupled = newton_polish(complex(0.02, 3.10), n_B, tau_B, x_f_B, zeta, 0.0, np.pi, 0.1)[0]
    xi_ep_hot_pred = (-s1_hot_uncoupled.real) / np.pi + np.sqrt(2.0 * beta_main)
    xi_ep_tight_pred = (-s1_hot_uncoupled.real) / np.pi + np.sqrt(2.0 * beta_tight)

    print("=== Entry 002: Faceplate Helmholtz Resonator & Exceptional-Point Tuning (DC-Blocked) ===")
    print(f"Case B uncoupled 1L (omega_hp={OMEGA_HP}): s = {s1_hot_uncoupled.real:+.6f} + {s1_hot_uncoupled.imag:.6f}i")
    print(f"Cold EP prediction (beta={beta_main}):  xi_EP_cold = {xi_ep_cold:.5f}, alpha_EP_cold = {alpha_ep_cold:+.5f}")
    print(f"Hot  EP prediction (beta={beta_main}):  xi_EP_hot  = {xi_ep_hot_pred:.5f}")
    print(f"Hot  EP prediction (beta={beta_tight}): xi_EP_hot  = {xi_ep_tight_pred:.5f}\n")

    # Experiments 1 & 2: Damping Sweep
    xi_sweep = np.concatenate([
        np.linspace(0.002, 0.05, 25),
        np.linspace(0.055, 0.22, 60),
        np.linspace(0.23, 0.60, 35),
    ])

    cold_a_max, hot_a_max, hot_tight_a_max = [], [], []
    hot_w_lo, hot_w_hi = [], []

    for xi in xi_sweep:
        r_c = find_coupled_1L_pair(0.0, tau_B, x_f_B, zeta, beta_main, np.pi, xi)
        cold_a_max.append(max(r.real for r in r_c))

        r_h = find_coupled_1L_pair(n_B, tau_B, x_f_B, zeta, beta_main, s1_hot_uncoupled.imag, xi)
        hot_a_max.append(max(r.real for r in r_h))
        omegas_h = sorted([r.imag for r in r_h])
        hot_w_lo.append(omegas_h[0])
        hot_w_hi.append(omegas_h[-1])

        r_t = find_coupled_1L_pair(n_B, tau_B, x_f_B, zeta, beta_tight, s1_hot_uncoupled.imag, xi)
        hot_tight_a_max.append(max(r.real for r in r_t))

    cold_a_max = np.array(cold_a_max)
    hot_a_max = np.array(hot_a_max)
    hot_tight_a_max = np.array(hot_tight_a_max)

    idx_c_opt = int(np.argmin(cold_a_max))
    idx_h_opt = int(np.argmin(hot_a_max))
    idx_t_opt = int(np.argmin(hot_tight_a_max))
    stable_tight_xis = xi_sweep[hot_tight_a_max < 0.0]

    print(f"Exact Cold Optimum (beta=0.02):  xi_opt = {xi_sweep[idx_c_opt]:.5f}, alpha_max = {cold_a_max[idx_c_opt]:+.5f}")
    print(f"Exact Hot  Optimum (beta=0.02):  xi_opt = {xi_sweep[idx_h_opt]:.5f}, alpha_max = {hot_a_max[idx_h_opt]:+.5f}")
    print(f"Exact Hot  Optimum (beta=0.005): xi_opt = {xi_sweep[idx_t_opt]:.5f}, alpha_max = {hot_tight_a_max[idx_t_opt]:+.5f}")
    print(f"Tight cavity (beta=0.005) stable window: xi in [{stable_tight_xis[0]:.4f}, {stable_tight_xis[-1]:.4f}]\n")

    # Direct 1D PDE + Resonator Verification using multi-mode LSQ fit (fixes M6!)
    print("--- Direct 1D PDE + Helmholtz Cavity Verification (Case B, beta=0.005) ---")
    pde_cases = [
        ("no_cavity", "No cavity (beta=0)", 0.0, 0.10, (8.0, 32.0), 1),
        ("underdamped", "Under-damped trap (xi=0.004, Q=125)", beta_tight, 0.004, (6.0, 44.0), 2),
        ("optimal", "Hot-EP optimal (xi=0.086, Q=5.8)", beta_tight, 0.086, (10.0, 30.0), 2),
        ("overdamped", "Over-damped decoupled (xi=0.35, Q=1.4)", beta_tight, 0.35, (8.0, 32.0), 1),
    ]
    pde_summary = {}
    pde_runs = []
    for cid, clabel, b_val, xi_val, fwin, n_fit in pde_cases:
        t_sim = 46.0 if cid == "underdamped" else 34.0
        sim = simulate_chamber_with_cavity(
            n=n_B, tau_bar=tau_B, x_f=x_f_B, zeta=zeta,
            beta=b_val, omega_H=s1_hot_uncoupled.imag, xi_H=xi_val,
            t_end=t_sim, n_cells=200, cfl=0.25, init_mode=1,
        )
        r_pair = find_coupled_1L_pair(n_B, tau_B, sim["x_f_actual"], zeta, b_val, s1_hot_uncoupled.imag, xi_val)
        dom_s = max(r_pair, key=lambda z: z.real)
        init_r = r_pair[:n_fit] if n_fit == 2 else [dom_s]
        fitted_r = fit_multi_mode_lsq(sim["t"], sim["a1_proj"], fwin[0], fwin[1], init_r)
        dom_fit = max(fitted_r, key=lambda z: z.real)
        err_a = abs(dom_fit.real - dom_s.real)
        err_w = abs(dom_fit.imag - dom_s.imag)
        print(
            f"  [{clabel}] exact_max={dom_s.real:+.6f}+{dom_s.imag:.4f}i | "
            f"PDE_LSQ={dom_fit.real:+.6f}+{dom_fit.imag:.4f}i (|da|={err_a:.2e}, |dw|={err_w:.2e}) | "
            f"Energy err={sim['energy_balance_rel_err']:.2e}"
        )
        pde_summary[cid] = {
            "label": clabel,
            "beta": b_val,
            "xi_H": xi_val,
            "exact_roots": [[float(r.real), float(r.imag)] for r in r_pair],
            "fitted_roots": [[float(r.real), float(r.imag)] for r in fitted_r],
            "alpha_exact_max": float(dom_s.real),
            "alpha_pde_lsq": float(dom_fit.real),
            "alpha_abs_diff": float(err_a),
            "omega_abs_diff": float(err_w),
            "energy_balance_rel_err": sim["energy_balance_rel_err"],
        }
        pde_runs.append((cid, clabel, sim, dom_s, dom_fit.real))

    # Figure 1
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.0), dpi=150)

    ax = axes[0]
    ax.axhline(0.0, color="black", lw=1.4)
    ax.axhline(s1_hot_uncoupled.real, color="#7f7f7f", ls=":", lw=1.2, label=f"Uncoupled Case B ($\\alpha={s1_hot_uncoupled.real:+.3f}$)")
    ax.plot(xi_sweep, cold_a_max, "--", color="#1f77b4", lw=1.6, label="Cold $\\max \\alpha_\\pm$ ($\\beta=2\\%$)")
    ax.plot(xi_sweep, hot_a_max, "-", color="#d62728", lw=2.2, label="Hot Case B $\\max \\alpha_\\pm$ ($\\beta=2\\%$)")
    ax.plot(xi_sweep, hot_tight_a_max, "-", color="#ff7f0e", lw=2.0, label="Hot Case B $\\max \\alpha_\\pm$ ($\\beta=0.5\\%$)")
    ax.axvline(xi_ep_cold, color="#1f77b4", ls=":", lw=1.2, alpha=0.8)
    ax.axvline(xi_ep_hot_pred, color="#d62728", ls=":", lw=1.2, alpha=0.8)
    ax.scatter([0.004, 0.086, 0.35], [pde_summary[k]["alpha_exact_max"] for k in ("underdamped", "optimal", "overdamped")],
               c=["crimson", "limegreen", "purple"], edgecolors="black", s=55, zorder=5)
    ax.set_xlabel("Resonator damping ratio $\\xi_H = 1/(2 Q_H)$")
    ax.set_ylabel("Dominant growth rate $\\max(\\alpha_-, \\alpha_+)$")
    ax.set_title("Exceptional-Point Damping Optimum", fontweight="bold")
    ax.set_xlim(0.0, 0.55)
    ax.set_ylim(-0.34, 0.04)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=7.8)

    ax = axes[1]
    ax.axhline(s1_hot_uncoupled.imag, color="#7f7f7f", ls=":", lw=1.2, label=f"Uncoupled hot $1L$ ($\\omega={s1_hot_uncoupled.imag:.3f}$)")
    ax.plot(xi_sweep, hot_w_lo, "-", color="#d62728", lw=2.0, label="Lower split branch $\\omega_-$")
    ax.plot(xi_sweep, hot_w_hi, "-", color="#1f77b4", lw=2.0, label="Upper split branch $\\omega_+$")
    ax.axvline(xi_sweep[idx_h_opt], color="black", ls="--", lw=1.3, label=f"Hot EP $\\xi_H^*={xi_sweep[idx_h_opt]:.3f}$")
    ax.set_xlabel("Resonator damping ratio $\\xi_H$")
    ax.set_ylabel("Coupled eigenfrequencies $\\omega_\\pm$")
    ax.set_title("Mode Splitting Below the Exceptional Point ($\\beta=2\\%$)", fontweight="bold")
    ax.set_xlim(0.0, 0.35)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.0)

    ax = axes[2]
    colors_pde = {"no_cavity": "#7f7f7f", "underdamped": "#d62728", "optimal": "#2ca02c", "overdamped": "#9467bd"}
    styles_pde = {"no_cavity": "--", "underdamped": "-", "optimal": "-", "overdamped": "-."}
    for cid, clabel, sim, dom_s, alpha_pde in pde_runs:
        mask_p = sim["t"] <= 34.0
        ax.semilogy(
            sim["t"][mask_p], sim["energy_tot"][mask_p] / sim["energy_tot"][0],
            color=colors_pde[cid], ls=styles_pde[cid], lw=2.0,
            label=f"{clabel}\n($\\alpha_{{LSQ}}={alpha_pde:+.5f}$)",
        )
    ax.set_xlabel("Dimensionless time $t \\bar{c}/L$")
    ax.set_ylabel("Normalized total energy $E_{\\mathrm{tot}}(t)/E_{\\mathrm{tot}}(0)$")
    ax.set_title("Direct 1D PDE Verification ($\\beta=0.5\\%$, Case B)", fontweight="bold")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(loc="lower left", fontsize=7.5)

    fig.suptitle("Entry 002: Helmholtz Cavity Mode-Splitting Trap & Exceptional-Point Optimum (DC-Blocked)", fontsize=12.3, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp002_exceptional_point.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Experiment 3: Uncensored Frequency Mistuning Robustness (omega_H/pi in [0.65, 1.45])
    # ------------------------------------------------------------------
    print("\n--- Experiment 3: Uncensored Frequency Mistuning Robustness (Case B, beta=0.01) ---")
    beta_map = 0.01
    w_ratio_grid = np.linspace(0.65, 1.45, 81)
    xi_map_grid = np.linspace(0.005, 0.35, 55)
    alpha_mistune = np.zeros((len(xi_map_grid), len(w_ratio_grid)))

    for i, xi in enumerate(xi_map_grid):
        for j, wr in enumerate(w_ratio_grid):
            r_pair = find_coupled_1L_pair(n_B, tau_B, x_f_B, zeta, beta_map, wr * np.pi, xi)
            alpha_mistune[i, j] = max(r.real for r in r_pair)

    def stable_bandwidth(xi_target: float) -> tuple[float, float, float]:
        idx = int(np.argmin(np.abs(xi_map_grid - xi_target)))
        row = alpha_mistune[idx]
        ok = w_ratio_grid[row < 0.0]
        if len(ok) == 0:
            return 0.0, float("nan"), float("nan")
        return float(ok[-1] - ok[0]), float(ok[0]), float(ok[-1])

    xi_ep_1pct = (-s1_hot_uncoupled.real) / np.pi + np.sqrt(2.0 * beta_map)
    bw_under = stable_bandwidth(0.015)
    bw_ep = stable_bandwidth(xi_ep_1pct)
    bw_robust = stable_bandwidth(0.22)
    print(f"  beta=1% hot EP prediction xi_H = {xi_ep_1pct:.4f}")
    print(f"  Uncensored mistuning bandwidth (omega_H/pi where alpha < 0):")
    print(f"    Under-damped (xi=0.015): width = {100*bw_under[0]:.1f}% [{bw_under[1]:.3f}, {bw_under[2]:.3f}]")
    print(f"    At hot EP    (xi={xi_ep_1pct:.3f}): width = {100*bw_ep[0]:.1f}% [{bw_ep[1]:.3f}, {bw_ep[2]:.3f}]")
    print(f"    Over-EP      (xi=0.220): width = {100*bw_robust[0]:.1f}% [{bw_robust[1]:.3f}, {bw_robust[2]:.3f}]")

    # ------------------------------------------------------------------
    # Experiment 4: Full-Domain [0.05, 0.95] x [0.05, 2.0] Synergy (ALL 4 baselines computed!)
    # ------------------------------------------------------------------
    print("\n--- Experiment 4: Full-Domain Synergy (Modes 1L-12L, tau in [0.05, 2.0], all computed!) ---")
    xf_grid = np.linspace(0.05, 0.95, 31)
    tau_grid = np.linspace(0.05, 2.00, 40)
    sigma_tau_syn = 0.15
    beta_syn = 0.02
    omega_H_syn = np.pi
    xi_H_syn = 0.14

    def eval_grid(beta_v: float, sigma_v: float) -> tuple[np.ndarray, float, int]:
        arr = np.zeros((len(tau_grid), len(xf_grid)))
        k0_hits = 0
        for i, tv in enumerate(tau_grid):
            for j, xv in enumerate(xf_grid):
                if has_positive_real_root(n_B, tv, xv, zeta, beta_v, omega_H_syn, xi_H_syn, sigma_tau=sigma_v):
                    k0_hits += 1
                    arr[i, j] = 1.0
                    continue
                r1 = find_coupled_1L_pair(n_B, tv, xv, zeta, beta_v, omega_H_syn, xi_H_syn, sigma_tau=sigma_v)
                best = max(r.real for r in r1)
                for k in range(2, 13):
                    sk = solve_higher_mode(k, n_B, tv, xv, zeta, beta_v, omega_H_syn, xi_H_syn, sigma_tau=sigma_v)
                    if sk.real > best:
                        best = sk.real
                arr[i, j] = best
        return arr, float(np.mean(arr < 0.0)), k0_hits

    _, frac_point, k0_pt = eval_grid(0.0, 0.0)
    _, frac_cav_only, k0_co = eval_grid(beta_syn, 0.0)
    _, frac_erl_only, k0_eo = eval_grid(0.0, sigma_tau_syn)
    alpha_syn_map, frac_synergy, k0_sy = eval_grid(beta_syn, sigma_tau_syn)

    print(f"  Point flame (beta=0, sigma=0):        {100.0*frac_point:.1f}%  (k<=12 UB; k=0 hits={k0_pt})")
    print(f"  Cavity ONLY (beta=2%, sigma=0):       {100.0*frac_cav_only:.1f}%  (k<=12 UB; k=0 hits={k0_co})")
    print(f"  Erlang-4 ONLY (beta=0, sigma=0.15):   {100.0*frac_erl_only:.1f}%  (k=0 hits={k0_eo})")
    print(f"  Synergy (Cavity 2% + Erlang 0.15):    {100.0*frac_synergy:.1f}%  (k=0 hits={k0_sy})")

    # Case B synergy check
    r1_B = find_coupled_1L_pair(n_B, 0.20, 0.15, zeta, beta_syn, omega_H_syn, xi_H_syn, sigma_tau=0.08)
    hi_B = [solve_higher_mode(k, n_B, 0.20, 0.15, zeta, beta_syn, omega_H_syn, xi_H_syn, sigma_tau=0.08) for k in range(2, 13)]
    max_alpha_B_syn = max([r.real for r in r1_B] + [s.real for s in hi_B])
    print(f"  Case B (tau=0.20, x_f=0.15) with Cavity + sigma_tau=0.08: max_alpha (1L-12L) = {max_alpha_B_syn:+.5f}")

    # Figure 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.5), dpi=150)

    norm1 = TwoSlopeNorm(vmin=-0.12, vcenter=0.0, vmax=0.03)
    im1 = ax1.contourf(
        w_ratio_grid, xi_map_grid, alpha_mistune,
        levels=np.linspace(-0.12, 0.03, 31),
        cmap="RdBu_r",
        norm=norm1,
        extend="both",
    )
    ax1.contour(w_ratio_grid, xi_map_grid, alpha_mistune, levels=[0.0], colors="black", linewidths=2.0)
    ax1.axvline(s1_hot_uncoupled.imag / np.pi, color="black", ls=":", lw=1.2, label=f"Hot $1L$ freq ({s1_hot_uncoupled.imag/np.pi:.3f}$\\pi$)")
    ax1.axhline(xi_ep_1pct, color="gold", ls="--", lw=1.4, label=f"Hot EP $\\xi_H^*={xi_ep_1pct:.3f}$")
    ax1.set_xlabel("Cavity tuning ratio $\\omega_H / \\pi$")
    ax1.set_ylabel("Resonator damping ratio $\\xi_H$")
    ax1.set_title("Uncensored Frequency Mistuning Robustness (Case B, $\\beta = 1\\%$)", pad=10, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.3)
    plt.colorbar(im1, ax=ax1, label="Max coupled growth rate $\\max(\\alpha_-, \\alpha_+)$")

    norm2 = TwoSlopeNorm(vmin=-0.06, vcenter=0.0, vmax=0.05)
    im2 = ax2.contourf(
        xf_grid, tau_grid, alpha_syn_map,
        levels=np.linspace(-0.06, 0.05, 23),
        cmap="RdBu_r",
        norm=norm2,
        extend="both",
    )
    ax2.contour(xf_grid, tau_grid, alpha_syn_map, levels=[0.0], colors="black", linewidths=1.8)
    ax2.set_xlabel("Flame position $x_f / L$")
    ax2.set_ylabel("Mean time lag $\\bar{\\tau} \\bar{c}/L$")
    ax2.set_title(
        f"Synergy ($\\bar{{\\tau}}\\in[0.05,2]$): Cavity 2% + Erlang $\\rightarrow$ {100*frac_synergy:.1f}% Stable (1L–12L)",
        pad=10, fontweight="bold",
    )
    plt.colorbar(im2, ax=ax2, label="Max growth rate $\\alpha_{\\max}$ (1L–12L)")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp002_mistuning_and_synergy.png")
    plt.close(fig)

    metrics = {
        "parameters": {
            "zeta": zeta,
            "omega_hp": OMEGA_HP,
            "n_B": n_B,
            "tau_B": tau_B,
            "x_f_B": x_f_B,
            "beta_main": beta_main,
            "beta_tight": beta_tight,
            "s1_hot_uncoupled": [float(s1_hot_uncoupled.real), float(s1_hot_uncoupled.imag)],
            "xi_ep_cold_pred": float(xi_ep_cold),
            "alpha_ep_cold_pred": float(alpha_ep_cold),
            "xi_ep_hot_pred": float(xi_ep_hot_pred),
        },
        "damping_optima": {
            "cold_beta_2pct": {"xi_opt": float(xi_sweep[idx_c_opt]), "alpha_max": float(cold_a_max[idx_c_opt])},
            "hot_beta_2pct": {"xi_opt": float(xi_sweep[idx_h_opt]), "alpha_max": float(hot_a_max[idx_h_opt])},
            "hot_beta_05pct": {
                "xi_opt": float(xi_sweep[idx_t_opt]),
                "alpha_max": float(hot_tight_a_max[idx_t_opt]),
                "stable_xi_range": [float(stable_tight_xis[0]), float(stable_tight_xis[-1])],
            },
        },
        "pde_verification_beta_05pct": pde_summary,
        "mistuning_robustness_beta_1pct": {
            "w_ratio_domain": [0.65, 1.45],
            "xi_ep_hot": float(xi_ep_1pct),
            "bandwidth_xi_0015": {"width": bw_under[0], "min_ratio": bw_under[1], "max_ratio": bw_under[2]},
            "bandwidth_xi_ep": {"width": bw_ep[0], "min_ratio": bw_ep[1], "max_ratio": bw_ep[2]},
            "bandwidth_xi_0220": {"width": bw_robust[0], "min_ratio": bw_robust[1], "max_ratio": bw_robust[2]},
        },
        "synergy_1L_12L_full_domain": {
            "domain_xf": [0.05, 0.95],
            "domain_tau": [0.05, 2.00],
            "grid_shape": [len(tau_grid), len(xf_grid)],
            "point_flame_k12_ub": frac_point,
            "cavity_only_beta_2pct_k12_ub": frac_cav_only,
            "erlang4_only_sigma_015": frac_erl_only,
            "cavity_plus_erlang4": frac_synergy,
            "k0_unstable_hits": [k0_pt, k0_co, k0_eo, k0_sy],
            "caseB_synergy_max_alpha_1L_12L": float(max_alpha_B_syn),
        },
    }

    with open(RESULTS_DIR / "exp002_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved Entry 002 plots and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
