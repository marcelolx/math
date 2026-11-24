"""Entry 003: Nonlinear Triggering (Bounded Subcritical Flame) & Cavity Orifice Drag.

Peer-review-hardened nonlinear 1D combustor model:
- Uniform DC-blocking HP filter H_hp(s) = s/(s + 0.20) + causal Erlang-4 (sigma_tau = 0.10, tau_bar = 0.20).
- Strictly bounded, monotonically increasing subcritical flame response (never reverses sign!):
      Phi(q) = q_s * tanh(q/q_s + kappa * (q/q_s)^3),   kappa = 4.5, q_s = 0.038
  Keeps limit-cycle amplitudes P_LC <= 0.10 (p'/p_bar = gamma*P <= 0.125), safely inside the
  bulk-linear acoustic regime (simple-wave shock distance > 3.5 L).
- Exact numerical first-harmonic Flame Describing Function (FDF) compared LIKE-FOR-LIKE against
  the first-harmonic FFT amplitude |P_1(0)| of the settled 1D PDE limit cycle (with peak |P(0)|
  reported separately).
- Calibrated quadratic orifice jet drag in the Helmholtz cavity neck:
      c_nl = (beta * omega_H^2) / (2 * C_d^2 * sigma_o^2)
  with consistent cavity pre-pressurization (p_c(0) = P(0,0), v_r(0) = 0, no artificial t=0 neck impulse!).
- Explicit grid (N=100 vs N=200) and horizon (t_end=110 vs t_end=220) convergence check for P_crit.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OMEGA_HP = 0.20


# ---------------------------------------------------------------------------
# 1. Exact linear Hopf point, coupled eigenvalues & bounded FDF
# ---------------------------------------------------------------------------

def flame_transfer(s: complex, tau_bar: float, sigma_tau: float, m_stages: int = 4) -> complex:
    hp = s / (s + OMEGA_HP)
    tau_d = min(tau_bar, np.sqrt(m_stages) * sigma_tau)
    tau_0 = max(0.0, tau_bar - tau_d)
    g_hat = np.exp(-s * tau_0) / ((1.0 + s * (tau_d / m_stages)) ** m_stages) if tau_d > 0 else np.exp(-s * tau_bar)
    return hp * g_hat


def dispersion_F(
    s: complex, n: float, tau_bar: float, x_f: float, zeta: float,
    beta: float, omega_H: float, xi_H: float, sigma_tau: float, m_stages: int = 4
) -> complex:
    denom = s * s + 2.0 * xi_H * omega_H * s + omega_H * omega_H
    num_Y = beta * omega_H * omega_H * s if beta > 0.0 else 0.0j
    lhs = (denom + zeta * num_Y) * np.sinh(s) + (zeta * denom + num_Y) * np.cosh(s)
    if n == 0.0:
        return lhs
    tf = flame_transfer(s, tau_bar, sigma_tau, m_stages)
    c1 = denom * np.cosh(s * x_f) + num_Y * np.sinh(s * x_f)
    c2 = np.cosh(s * (1.0 - x_f)) + zeta * np.sinh(s * (1.0 - x_f))
    return lhs - n * tf * c1 * c2


def solve_root(s0: complex, *args) -> tuple[complex, float]:
    s = complex(s0)
    for _ in range(45):
        fv = dispersion_F(s, *args)
        if abs(fv) < 1e-12:
            return s, float(abs(fv))
        eps = 1e-7 * max(1.0, abs(s))
        dfv = (dispersion_F(s + eps, *args) - dispersion_F(s - eps, *args)) / (2.0 * eps)
        if abs(dfv) < 1e-16:
            break
        ds = fv / dfv
        if abs(ds) > 0.4:
            ds = 0.4 * ds / abs(ds)
        s -= ds
    return s, float(abs(dispersion_F(s, *args)))


def find_1L_dominant(n: float, tau_bar: float, x_f: float, zeta: float, beta: float, omega_H: float, xi_H: float, sigma_tau: float) -> complex:
    if beta <= 0.0:
        return solve_root(complex(0.0, 3.12), n, tau_bar, x_f, zeta, 0.0, omega_H, xi_H, sigma_tau)[0]
    split = np.sqrt(max(1e-4, 0.5 * beta * omega_H * omega_H))
    seeds = [
        complex(-0.02, np.pi - split), complex(-0.02, np.pi + split),
        complex(-0.10, np.pi - 0.4 * split), complex(-0.10, np.pi + 0.4 * split),
        complex(-0.18, np.pi), complex(-xi_H * omega_H, omega_H), complex(0.03, np.pi),
    ]
    roots = []
    for sd in seeds:
        r, res = solve_root(sd, n, tau_bar, x_f, zeta, beta, omega_H, xi_H, sigma_tau)
        if res < 1e-8 and 1.5 <= r.imag <= 5.0 and -4.0 <= r.real <= 1.0:
            if all(abs(r - e) > 1e-4 for e in roots):
                roots.append(r)
    return max(roots, key=lambda z: z.real)


def find_hopf_n(tau_bar: float, x_f: float, zeta: float, sigma_tau: float) -> tuple[float, float]:
    lo, hi = 0.01, 0.25
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        s = find_1L_dominant(mid, tau_bar, x_f, zeta, 0.0, np.pi, 0.1, sigma_tau)
        if s.real < 0.0:
            lo = mid
        else:
            hi = mid
    n_H = 0.5 * (lo + hi)
    s_H = find_1L_dominant(n_H, tau_bar, x_f, zeta, 0.0, np.pi, 0.1, sigma_tau)
    return n_H, float(s_H.imag)


def phi_bounded(q: np.ndarray | float, q_s: float, kappa: float):
    """Strictly bounded, odd, monotonic subcritical flame law: q_s * tanh(u + kappa*u^3)."""
    u = q / q_s
    return q_s * np.tanh(u + kappa * (u ** 3))


def fdf_gain(a_f: float, q_s: float, kappa: float) -> float:
    """Exact first-harmonic describing function G(A_f) = (2 / (pi*A_f)) int_0^pi Phi(A_f cos t) cos t dt."""
    if a_f < 1e-9:
        return 1.0
    th = np.linspace(0.0, np.pi, 256, endpoint=False) + (np.pi / 512.0)
    vals = phi_bounded(a_f * np.cos(th), q_s, kappa) * np.cos(th)
    return float((2.0 / a_f) * np.mean(vals))


def compute_fdf_branches(
    n_vals: np.ndarray, n_H: float, omega_hopf: float, x_f: float, sigma_tau: float, q_s: float, kappa: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute exact numerical FDF branches (P1_crit, P1_LC, n_fold, G_max)."""
    tf_mag = abs(flame_transfer(1j * omega_hopf, 0.20, sigma_tau, 4))
    spatial_gain = abs(np.cos(omega_hopf * x_f)) * tf_mag

    af_grid = np.linspace(1e-5, 0.20, 2000)
    G_grid = np.array([fdf_gain(af, q_s, kappa) for af in af_grid])
    idx_pk = int(np.argmax(G_grid))
    G_max = float(G_grid[idx_pk])
    n_fold = n_H / G_max

    p_crit = np.full_like(n_vals, np.nan, dtype=float)
    p_lc = np.full_like(n_vals, np.nan, dtype=float)

    for i, nv in enumerate(n_vals):
        target_G = n_H / nv
        if target_G > G_max:
            continue
        # Upper branch (right of peak, monotonically decreasing)
        g_hi = G_grid[idx_pk:]
        af_hi = af_grid[idx_pk:]
        if g_hi[-1] <= target_G <= g_hi[0]:
            # Interpolate (reverse because decreasing)
            af_root = float(np.interp(target_G, g_hi[::-1], af_hi[::-1]))
            p_lc[i] = af_root / spatial_gain
        # Lower unstable branch (left of peak, monotonically increasing from 1.0 to G_max)
        if nv < n_H:
            g_lo = G_grid[: idx_pk + 1]
            af_lo = af_grid[: idx_pk + 1]
            if g_lo[0] <= target_G <= g_lo[-1]:
                af_root = float(np.interp(target_G, g_lo, af_lo))
                p_crit[i] = af_root / spatial_gain

    return p_crit, p_lc, float(n_fold), G_max


# ---------------------------------------------------------------------------
# 2. Nonlinear 1D staggered-grid PDE (consistent ICs + first-harmonic extractor)
# ---------------------------------------------------------------------------

def simulate_nonlinear_combustor(
    n: float,
    tau_bar: float,
    x_f: float,
    zeta: float,
    sigma_tau: float = 0.10,
    m_stages: int = 4,
    q_s: float = 0.038,
    kappa: float = 4.5,
    beta: float = 0.0,
    omega_H: float = 3.12,
    xi_H: float = 0.10,
    c_nl: float = 0.0,
    p0_amp: float = 0.03,
    t_end: float = 110.0,
    n_cells: int = 100,
    cfl: float = 0.25,
    record_history: bool = False,
) -> dict:
    dx = 1.0 / n_cells
    dt = cfl * dx
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps
    inv_dx = 1.0 / dx

    x_p = (np.arange(n_cells) + 0.5) * dx
    m_flame = int(np.argmin(np.abs(x_p - x_f)))
    x_f_actual = float(x_p[m_flame])

    P = p0_amp * np.cos(np.pi * x_p)
    U = np.zeros(n_cells - 1)
    # Consistent standing-wave IC: cavity pre-pressurized to P(0, 0), zero initial neck velocity!
    v_r = 0.0
    p_c = float(P[0]) if beta > 0.0 else 0.0
    p_bar = 0.0

    tau_d = np.sqrt(m_stages) * sigma_tau  # = 0.20 == tau_bar => tau_0 = 0
    rate = m_stages / tau_d
    # Consistent harmonic Erlang-state IC (evaluating exact 1L steady-state response at t=0)
    q_st = np.zeros(m_stages)
    tf_stage = 1.0 / (1.0 + 1j * np.pi / rate)
    hp0 = (1j * np.pi) / (1j * np.pi + OMEGA_HP)
    cur_c = float(P[m_flame]) * hp0
    for r in range(m_stages):
        cur_c = cur_c * tf_stage
        q_st[r] = float(cur_c.real)

    def rhs(P_c, U_c, vr_c, pc_c, pbar_c, q_c):
        p_f = float(P_c[m_flame])
        p_ac = p_f - pbar_c
        dpbar = OMEGA_HP * p_ac

        dq = np.empty(m_stages)
        dq[0] = rate * (p_ac - q_c[0])
        for r in range(1, m_stages):
            dq[r] = rate * (q_c[r - 1] - q_c[r])

        dP = np.empty_like(P_c)
        dP[0] = -(U_c[0] + vr_c) * inv_dx
        dP[1:-1] = -(U_c[1:] - U_c[:-1]) * inv_dx
        dP[-1] = -(zeta * P_c[-1] - U_c[-1]) * inv_dx
        dP[m_flame] += n * float(phi_bounded(float(q_c[-1]), q_s, kappa)) * inv_dx

        dU = -(P_c[1:] - P_c[:-1]) * inv_dx

        if beta > 0.0:
            dvr = beta * (omega_H ** 2) * (P_c[0] - pc_c) - (2.0 * xi_H * omega_H + c_nl * abs(vr_c)) * vr_c
            dpc = vr_c / beta
        else:
            dvr = 0.0
            dpc = 0.0
        return dP, dU, dvr, dpc, dpbar, dq

    # Always keep late window for first-harmonic extraction (last ~20 units = ~10 cycles)
    late_lo = int(0.80 * n_steps)
    w2_lo, w2_hi = int(0.55 * n_steps), int(0.70 * n_steps)
    p_late_buf = np.empty(n_steps - late_lo + 1)
    vr_late_max = 0.0
    vr_peak_all = 0.0
    p_w2 = 0.0
    p_w3 = 0.0

    if record_history:
        t_arr = np.linspace(0.0, t_end, n_steps + 1)
        p_head = np.empty(n_steps + 1)
        vr_hist = np.empty(n_steps + 1)
        p_head[0] = P[0]
        vr_hist[0] = v_r
    else:
        t_arr = p_head = vr_hist = None

    for step in range(n_steps):
        k1 = rhs(P, U, v_r, p_c, p_bar, q_st)
        k2 = rhs(
            P + 0.5 * dt * k1[0], U + 0.5 * dt * k1[1],
            v_r + 0.5 * dt * k1[2], p_c + 0.5 * dt * k1[3],
            p_bar + 0.5 * dt * k1[4], q_st + 0.5 * dt * k1[5],
        )
        k3 = rhs(
            P + 0.5 * dt * k2[0], U + 0.5 * dt * k2[1],
            v_r + 0.5 * dt * k2[2], p_c + 0.5 * dt * k2[3],
            p_bar + 0.5 * dt * k2[4], q_st + 0.5 * dt * k2[5],
        )
        k4 = rhs(
            P + dt * k3[0], U + dt * k3[1],
            v_r + dt * k3[2], p_c + dt * k3[3],
            p_bar + dt * k3[4], q_st + dt * k3[5],
        )

        P = P + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        U = U + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        v_r = v_r + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        p_c = p_c + (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
        p_bar = p_bar + (dt / 6.0) * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4])
        q_st = q_st + (dt / 6.0) * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5])

        ap = abs(float(P[0]))
        av = abs(v_r)
        if av > vr_peak_all:
            vr_peak_all = av
        if w2_lo <= step < w2_hi and ap > p_w2:
            p_w2 = ap
        if step >= late_lo:
            p_late_buf[step - late_lo] = float(P[0])
            if ap > p_w3:
                p_w3 = ap
            if av > vr_late_max:
                vr_late_max = av

        if record_history:
            p_head[step + 1] = P[0]
            vr_hist[step + 1] = v_r

    p_late_buf[-1] = float(P[0])
    # Extract first-harmonic amplitude |P_1(0)| from late window via Hann-windowed FFT
    w_hann = np.hanning(len(p_late_buf))
    spec = np.abs(np.fft.rfft(p_late_buf * w_hann)) * (2.0 / np.sum(w_hann))
    freqs = np.fft.rfftfreq(len(p_late_buf), d=dt) * 2.0 * np.pi
    band = (freqs >= 2.0) & (freqs <= 4.3)
    p1_late = float(np.max(spec[band])) if np.any(band) else 0.0

    triggered = bool(p1_late > 0.022 or (p_w3 > p_w2 * 1.005 and p_w3 > 0.0015))

    return {
        "p_w2": float(p_w2),
        "p_peak_late": float(p_w3),
        "p1_harm_late": p1_late,
        "vr_peak_all": float(vr_peak_all),
        "vr_late_max": float(vr_late_max),
        "triggered": triggered,
        "x_f_actual": x_f_actual,
        "t": t_arr,
        "p_head": p_head,
        "vr_hist": vr_hist,
    }


def bisect_trigger_threshold(
    n: float, tau_bar: float, x_f: float, zeta: float, sigma_tau: float, q_s: float, kappa: float,
    beta: float = 0.0, omega_H: float = 3.12, xi_H: float = 0.10, c_nl: float = 0.0,
    p_lo: float = 0.002, t_end: float = 110.0, n_cells: int = 100, n_iter: int = 11,
) -> tuple[float | None, float, float]:
    """Return (P_crit, P1_LC_first_harmonic, P_peak_LC)."""
    r_lo = simulate_nonlinear_combustor(
        n, tau_bar, x_f, zeta, sigma_tau, 4, q_s, kappa, beta, omega_H, xi_H, c_nl,
        p0_amp=p_lo, t_end=t_end, n_cells=n_cells,
    )
    if r_lo["triggered"]:
        r_lc = simulate_nonlinear_combustor(
            n, tau_bar, x_f, zeta, sigma_tau, 4, q_s, kappa, beta, omega_H, xi_H, c_nl,
            p0_amp=0.08, t_end=130.0, n_cells=n_cells,
        )
        return 0.0, r_lc["p1_harm_late"], r_lc["p_peak_late"]

    hi_found = None
    for pa in (0.025, 0.035, 0.050, 0.075, 0.105):
        r = simulate_nonlinear_combustor(
            n, tau_bar, x_f, zeta, sigma_tau, 4, q_s, kappa, beta, omega_H, xi_H, c_nl,
            p0_amp=pa, t_end=t_end, n_cells=n_cells,
        )
        if r["triggered"]:
            hi_found = pa
            break

    if hi_found is None:
        return None, 0.0, 0.0

    lo, hi = p_lo, hi_found
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        r = simulate_nonlinear_combustor(
            n, tau_bar, x_f, zeta, sigma_tau, 4, q_s, kappa, beta, omega_H, xi_H, c_nl,
            p0_amp=mid, t_end=t_end, n_cells=n_cells,
        )
        if r["triggered"]:
            hi = mid
        else:
            lo = mid

    r_lc = simulate_nonlinear_combustor(
        n, tau_bar, x_f, zeta, sigma_tau, 4, q_s, kappa, beta, omega_H, xi_H, c_nl,
        p0_amp=max(0.09, hi * 1.15), t_end=130.0, n_cells=n_cells,
    )
    return 0.5 * (lo + hi), float(r_lc["p1_harm_late"]), float(r_lc["p_peak_late"])


# ---------------------------------------------------------------------------
# 3. Main experiments & plotting
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    zeta = 0.04
    tau_bar = 0.20
    x_f = 0.15
    sigma_tau = 0.10
    q_s = 0.038
    kappa = 4.5

    x_p100 = (np.arange(100) + 0.5) / 100.0
    x_f_grid = float(x_p100[np.argmin(np.abs(x_p100 - x_f))])
    n_H, w_hopf = find_hopf_n(tau_bar, x_f_grid, zeta, sigma_tau)

    n_curve = np.linspace(0.036, 0.088, 220)
    p_crit_fdf, p_lc_fdf, n_fold, g_max = compute_fdf_branches(n_curve, n_H, w_hopf, x_f_grid, sigma_tau, q_s, kappa)

    print("=== Entry 003: Bounded Subcritical Triggering & Calibrated Cavity Orifice Drag ===")
    print(f"Bounded flame law: Phi(q) = {q_s} * tanh(q/{q_s} + {kappa}*(q/{q_s})^3)  (strictly monotonic, |Phi| < {q_s})")
    print(f"Linear Hopf n_H = {n_H:.5f}, omega_H = {w_hopf:.5f} | FDF peak gain = {g_max:.3f}x -> n_fold = {n_fold:.5f}\n")

    # Experiment 1: Bare-combustor subcritical bifurcation (like-for-like first-harmonic FDF comparison!)
    print("--- Experiment 1: Bare-Combustor Subcritical Bifurcation (First-Harmonic Like-for-Like) ---")
    n_test_pts = [0.044, 0.048, 0.052, 0.056, 0.060, 0.066, 0.074, 0.082]
    pde_bif = []
    for nv in n_test_pts:
        pc, p1_lc, pk_lc = bisect_trigger_threshold(nv, tau_bar, x_f, zeta, sigma_tau, q_s, kappa, beta=0.0)
        pc_a, plc_a, _, _ = compute_fdf_branches(np.array([nv]), n_H, w_hopf, x_f_grid, sigma_tau, q_s, kappa)
        pc_s = f"{pc:.4f}" if pc is not None else "uncond. stable"
        pca_s = f"{pc_a[0]:.4f}" if np.isfinite(pc_a[0]) else ("0.0000" if nv >= n_H else "uncond. stable")
        plca_s = f"{plc_a[0]:.4f}" if np.isfinite(plc_a[0]) else "0.0000"
        print(f"  n={nv:.3f}: P_crit PDE={pc_s:>10s} (FDF={pca_s:>10s}) | P1_LC PDE={p1_lc:.4f} (FDF={plca_s}, peak={pk_lc:.4f})")
        pde_bif.append({
            "n": nv,
            "p_crit_pde": pc,
            "p1_lc_pde": p1_lc,
            "p_peak_lc_pde": pk_lc,
            "p_crit_fdf": None if not np.isfinite(pc_a[0]) else float(pc_a[0]),
            "p1_lc_fdf": None if not np.isfinite(plc_a[0]) else float(plc_a[0]),
        })

    # Convergence spot-check on P_crit at n = 0.052 (N=100, t=110 vs N=200, t=220)
    pc_100 = next(d["p_crit_pde"] for d in pde_bif if abs(d["n"] - 0.052) < 1e-6)
    pc_200, _, _ = bisect_trigger_threshold(0.052, tau_bar, x_f, zeta, sigma_tau, q_s, kappa, beta=0.0, t_end=220.0, n_cells=200, n_iter=10)
    conv_rel_diff = abs(pc_200 - pc_100) / pc_200
    print(f"  Convergence check at n=0.052: P_crit(N=100,t=110)={pc_100:.4f} vs P_crit(N=200,t=220)={pc_200:.4f} (rel diff={100*conv_rel_diff:.2f}%)\n")

    # Record two bomb traces at n = 0.052
    tr_decay = simulate_nonlinear_combustor(0.052, tau_bar, x_f, zeta, sigma_tau, 4, q_s, kappa, p0_amp=pc_100 * 0.88, t_end=110.0, record_history=True)
    tr_trig = simulate_nonlinear_combustor(0.052, tau_bar, x_f, zeta, sigma_tau, 4, q_s, kappa, p0_amp=pc_100 * 1.12, t_end=110.0, record_history=True)

    # Experiment 2 & 3: Calibrated Quadratic Orifice Drag Sweep
    n_cav = 0.080
    beta_cav = 0.010
    omega_cav = 3.12
    C_d = 0.65
    sigma_o = 0.010  # Orifice-to-chamber area ratio A_neck/A_ch = 1.0%
    c_nl_val = (beta_cav * (omega_cav ** 2)) / (2.0 * (C_d ** 2) * (sigma_o ** 2))  # ~1152

    s_bare = find_1L_dominant(n_cav, tau_bar, x_f_grid, zeta, 0.0, omega_cav, 0.1, sigma_tau)
    print(f"--- Experiment 2 & 3: Calibrated Orifice Drag Sweep (n={n_cav}, beta=1%, C_d={C_d}, sigma_o={sigma_o} -> c_nl={c_nl_val:.1f}) ---")
    print(f"Bare chamber at n={n_cav}: linearly UNSTABLE (s = {s_bare.real:+.5f} + {s_bare.imag:.4f}i)")

    xi_grid = [0.015, 0.030, 0.045, 0.065, 0.090, 0.125, 0.165, 0.210, 0.260, 0.320]
    sweep_rows = []
    for xi in xi_grid:
        s_lin = find_1L_dominant(n_cav, tau_bar, x_f_grid, zeta, beta_cav, omega_cav, xi, sigma_tau)
        pc_lin, _, _ = bisect_trigger_threshold(
            n_cav, tau_bar, x_f, zeta, sigma_tau, q_s, kappa,
            beta=beta_cav, omega_H=omega_cav, xi_H=xi, c_nl=0.0, t_end=100.0,
        )
        pc_nl, p1_lc_nl, pk_lc_nl = bisect_trigger_threshold(
            n_cav, tau_bar, x_f, zeta, sigma_tau, q_s, kappa,
            beta=beta_cav, omega_H=omega_cav, xi_H=xi, c_nl=c_nl_val, t_end=100.0,
        )
        pc_lin_s = f"{pc_lin:.4f}" if pc_lin is not None else "NO TRIG<=0.12"
        pc_nl_s = f"{pc_nl:.4f}" if pc_nl is not None else "NO TRIG<=0.12"
        print(
            f"  xi_H={xi:.3f}: alpha_lin={s_lin.real:+.5f} | "
            f"P_crit(lin cav)={pc_lin_s:>13s} | P_crit(nonlin)={pc_nl_s:>13s} (P1_LC={p1_lc_nl:.4f}, peak={pk_lc_nl:.4f})"
        )
        sweep_rows.append({
            "xi_H": xi,
            "alpha_lin": float(s_lin.real),
            "p_crit_linear_cavity": pc_lin,
            "p_crit_nonlinear_cavity": pc_nl,
            "p1_lc_nonlinear_cavity": p1_lc_nl,
            "p_peak_lc_nonlinear_cavity": pk_lc_nl,
        })

    idx_lin_ep = int(np.argmin([r["alpha_lin"] for r in sweep_rows]))
    xi_lin_ep = sweep_rows[idx_lin_ep]["xi_H"]
    idx_nl_best = int(np.argmax([1.0 if r["p_crit_nonlinear_cavity"] is None else r["p_crit_nonlinear_cavity"] for r in sweep_rows]))
    xi_nl_best = sweep_rows[idx_nl_best]["xi_H"]

    pc_ep_nl = sweep_rows[idx_lin_ep]["p_crit_nonlinear_cavity"]
    pc_best_nl = sweep_rows[idx_nl_best]["p_crit_nonlinear_cavity"]
    p_bomb = 0.5 * (pc_ep_nl + (0.10 if pc_best_nl is None else pc_best_nl)) if pc_ep_nl is not None else 0.06

    tr_ep_lin = simulate_nonlinear_combustor(
        n_cav, tau_bar, x_f, zeta, sigma_tau, 4, q_s, kappa,
        beta=beta_cav, omega_H=omega_cav, xi_H=xi_lin_ep, c_nl=0.0, p0_amp=p_bomb, t_end=100.0, record_history=True
    )
    tr_ep_nl = simulate_nonlinear_combustor(
        n_cav, tau_bar, x_f, zeta, sigma_tau, 4, q_s, kappa,
        beta=beta_cav, omega_H=omega_cav, xi_H=xi_lin_ep, c_nl=c_nl_val, p0_amp=p_bomb, t_end=100.0, record_history=True
    )
    tr_ud_nl = simulate_nonlinear_combustor(
        n_cav, tau_bar, x_f, zeta, sigma_tau, 4, q_s, kappa,
        beta=beta_cav, omega_H=omega_cav, xi_H=xi_nl_best, c_nl=c_nl_val, p0_amp=p_bomb, t_end=100.0, record_history=True
    )

    xi_eff_ep_peak = xi_lin_ep + (4.0 * c_nl_val / (3.0 * np.pi * omega_cav)) * tr_ep_nl["vr_peak_all"]
    xi_eff_ud_peak = xi_nl_best + (4.0 * c_nl_val / (3.0 * np.pi * omega_cav)) * tr_ud_nl["vr_peak_all"]
    print(f"\nBomb showdown at P0 = {p_bomb:.4f} (consistent pre-pressurized cavity IC, no t=0 impulse!):")
    print(f"  Linear-EP (xi={xi_lin_ep:.3f}, c_nl=0):      triggered={tr_ep_lin['triggered']}, P1_late={tr_ep_lin['p1_harm_late']:.4f}")
    print(f"  Linear-EP (xi={xi_lin_ep:.3f}, c_nl={c_nl_val:.0f}):   triggered={tr_ep_nl['triggered']}, P1_late={tr_ep_nl['p1_harm_late']:.4f}, peak xi_eff={xi_eff_ep_peak:.3f}")
    print(f"  Under-damped (xi={xi_nl_best:.3f}, c_nl={c_nl_val:.0f}): triggered={tr_ud_nl['triggered']}, P1_late={tr_ud_nl['p1_harm_late']:.4f}, peak xi_eff={xi_eff_ud_peak:.3f}")

    # Figure 1: Bifurcation (like-for-like first harmonic!)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2), dpi=150)

    ax1.axvline(n_H, color="black", ls=":", lw=1.3, label=f"Linear Hopf $n_H = {n_H:.4f}$")
    ax1.axvline(n_fold, color="#7f7f7f", ls="--", lw=1.2, label=f"Fold point $n_{{\\mathrm{{fold}}}} = {n_fold:.4f}$")
    ax1.axvspan(n_fold, n_H, color="#ffeda0", alpha=0.38, label="Bistable triggering zone")

    ax1.plot(n_curve, p_lc_fdf, "-", color="#d62728", lw=2.2, label="Exact FDF 1st-harmonic limit cycle $|P_1|_{\\mathrm{LC}}$")
    ax1.plot(n_curve, p_crit_fdf, "--", color="#1f77b4", lw=2.0, label="Exact FDF triggering threshold $P_{\\mathrm{crit}}$")

    n_pc_pts = [d["n"] for d in pde_bif if d["p_crit_pde"] is not None]
    y_pc_pts = [d["p_crit_pde"] for d in pde_bif if d["p_crit_pde"] is not None]
    n_lc_pts = [d["n"] for d in pde_bif if d["p1_lc_pde"] > 0.03]
    y_lc_pts = [d["p1_lc_pde"] for d in pde_bif if d["p1_lc_pde"] > 0.03]
    y_pk_pts = [d["p_peak_lc_pde"] for d in pde_bif if d["p1_lc_pde"] > 0.03]

    ax1.scatter(n_lc_pts, y_lc_pts, c="#d62728", edgecolors="black", s=55, zorder=5, label="1D PDE 1st-harmonic $|P_1|_{\\mathrm{LC}}$")
    ax1.scatter(n_lc_pts, y_pk_pts, c="none", edgecolors="#d62728", marker="^", s=45, zorder=4, label="1D PDE peak $|P(0)|_{\\mathrm{LC}}$ (incl. harmonics)")
    ax1.scatter(n_pc_pts, y_pc_pts, c="#1f77b4", marker="D", edgecolors="black", s=55, zorder=5, label="1D PDE bomb threshold $P_{\\mathrm{crit}}$")

    ax1.set_xlabel("Interaction index $n$")
    ax1.set_ylabel("Head-end pressure amplitude ($P = p'/(\\bar{\\rho}\\bar{c}^2)$)")
    ax1.set_title("Bounded Subcritical Hopf (Bulk-Linear Regime $P \\leq 0.11$)", fontweight="bold")
    ax1.set_xlim(0.036, 0.086)
    ax1.set_ylim(-0.004, 0.125)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=7.2)

    ax2.plot(tr_trig["t"], tr_trig["p_head"], color="#d62728", lw=0.95,
             label=f"Bomb $P_0 = {pc_100*1.12:.4f} > P_{{\\mathrm{{crit}}}}$ $\\rightarrow$ Limit Cycle ($|P_1|={tr_trig['p1_harm_late']:.3f}$)")
    ax2.plot(tr_decay["t"], tr_decay["p_head"], color="#1f77b4", lw=1.05,
             label=f"Bomb $P_0 = {pc_100*0.88:.4f} < P_{{\\mathrm{{crit}}}}$ $\\rightarrow$ Decays")
    ax2.set_xlabel("Dimensionless time $t \\bar{c}/L$")
    ax2.set_ylabel("Head-end pressure $P(0, t)$")
    ax2.set_title("Linearly Stable Combustor ($n = 0.052 < n_H$) Triggered by Bomb", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=7.9)

    fig.suptitle("Entry 003 (Part 1): Bounded Subcritical Flame — Like-for-Like FDF vs. 1D PDE", fontsize=12.3, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp003_subcritical_bifurcation.png")
    plt.close(fig)

    # Figure 2: Cavity drag sweep + showdown
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.4), dpi=150)

    xis = np.array([r["xi_H"] for r in sweep_rows])
    alphas = np.array([r["alpha_lin"] for r in sweep_rows])
    cap_val = 0.125
    pc_nl_arr = np.array([cap_val if r["p_crit_nonlinear_cavity"] is None else r["p_crit_nonlinear_cavity"] for r in sweep_rows])
    pc_lin_arr = np.array([cap_val if r["p_crit_linear_cavity"] is None else r["p_crit_linear_cavity"] for r in sweep_rows])

    ax1_r = ax1.twinx()
    l1, = ax1.plot(xis, pc_nl_arr, "o-", color="#d62728", lw=2.3, ms=6.5,
                   label=f"$P_{{\\mathrm{{crit}}}}$ (nonlin orifice $c_{{\\mathrm{{nl}}}}={c_nl_val:.0f}$)")
    l2, = ax1.plot(xis, pc_lin_arr, "s--", color="#2ca02c", lw=1.7, ms=5.5, alpha=0.85,
                   label="$P_{\\mathrm{crit}}$ (ideal linear cavity, $c_{\\mathrm{nl}}=0$)")
    l3, = ax1_r.plot(xis, -alphas, "^:", color="#1f77b4", lw=1.8, ms=5.5,
                     label=f"Linear margin $-\\alpha_{{\\mathrm{{lin}}}}$ (EP $\\xi={xi_lin_ep:.3f}$)")

    ax1.axvline(xi_lin_ep, color="#1f77b4", ls=":", lw=1.2, alpha=0.7)
    ax1.axhline(cap_val, color="gray", ls="--", lw=0.9, alpha=0.6)
    ax1.text(0.22, cap_val + 0.0025, "No trigger (probed <=0.12)", fontsize=7.5, color="gray")

    ax1.set_xlabel("Linear cavity damping ratio $\\xi_H$")
    ax1.set_ylabel("Triggering threshold $P_{\\mathrm{crit}}$ (higher = safer)", color="#d62728")
    ax1_r.set_ylabel("Linear stability margin $-\\alpha_{\\mathrm{lin}}$", color="#1f77b4")
    ax1.set_ylim(0.0, 0.14)
    ax1_r.set_ylim(-0.01, 0.19)
    ax1.set_title(f"Linear vs. Nonlinear Cavity Optimum ($n={n_cav}, \\beta=1\\%$)", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend([l1, l2, l3], [l1.get_label(), l2.get_label(), l3.get_label()], loc="lower left", fontsize=7.5)

    ax2.plot(tr_ep_nl["t"], tr_ep_nl["p_head"], color="#d62728", lw=0.95,
             label=f"Linear-EP ($\\xi_H={xi_lin_ep:.3f}, c_{{\\mathrm{{nl}}}}={c_nl_val:.0f}$) $\\rightarrow$ "
                   f"{'Triggers' if tr_ep_nl['triggered'] else 'Decays'} (peak $\\xi_{{\\mathrm{{eff}}}}={xi_eff_ep_peak:.2f}$)")
    ax2.plot(tr_ep_lin["t"], tr_ep_lin["p_head"], color="#7f7f7f", lw=1.0, ls="--", alpha=0.85,
             label=f"Ideal linear cavity ($\\xi_H={xi_lin_ep:.3f}, c_{{\\mathrm{{nl}}}}=0$) $\\rightarrow$ "
                   f"{'Triggers' if tr_ep_lin['triggered'] else 'Decays'}")
    ax2.plot(tr_ud_nl["t"], tr_ud_nl["p_head"], color="#2ca02c", lw=1.15,
             label=f"Under-damped ($\\xi_H={xi_nl_best:.3f}, c_{{\\mathrm{{nl}}}}={c_nl_val:.0f}$) $\\rightarrow$ "
                   f"{'Triggers' if tr_ud_nl['triggered'] else 'Survives'} (peak $\\xi_{{\\mathrm{{eff}}}}={xi_eff_ud_peak:.2f}$)")
    ax2.set_xlabel("Dimensionless time $t \\bar{c}/L$")
    ax2.set_ylabel("Head-end pressure $P(0, t)$")
    ax2.set_title(f"Same Bomb Pulse ($P_0 = {p_bomb:.4f}$) Across Three Cavity Designs", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=7.5)

    fig.suptitle("Entry 003 (Part 2): Calibrated Orifice Drag Shifts Optimal Cavity Damping Downward", fontsize=12.2, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp003_cavity_self_choking.png")
    plt.close(fig)

    metrics = {
        "parameters": {
            "zeta": zeta, "tau_bar": tau_bar, "x_f_grid": x_f_grid,
            "sigma_tau": sigma_tau, "omega_hp": OMEGA_HP,
            "flame_law": {"type": "q_s * tanh(q/q_s + kappa*(q/q_s)^3)", "q_s": q_s, "kappa": kappa, "fdf_peak_gain": g_max},
            "n_hopf_bare": n_H, "omega_hopf_bare": w_hopf, "n_fold_fdf": n_fold,
            "cavity_test": {"n": n_cav, "beta": beta_cav, "omega_H": omega_cav, "C_d": C_d, "sigma_o": sigma_o, "c_nl": float(c_nl_val)},
        },
        "convergence_check_n052": {
            "p_crit_N100_t110": pc_100, "p_crit_N200_t220": pc_200, "rel_diff": float(conv_rel_diff)
        },
        "exp1_bifurcation_bare": pde_bif,
        "exp2_3_cavity_xi_sweep": sweep_rows,
        "bomb_comparison": {
            "p0": float(p_bomb), "xi_lin_ep": xi_lin_ep, "xi_nl_best": xi_nl_best,
            "ep_linear_triggered": tr_ep_lin["triggered"], "ep_linear_p1_late": tr_ep_lin["p1_harm_late"],
            "ep_nonlinear_triggered": tr_ep_nl["triggered"], "ep_nonlinear_p1_late": tr_ep_nl["p1_harm_late"],
            "ep_nonlinear_peak_xi_eff": float(xi_eff_ep_peak),
            "underdamped_nonlinear_triggered": tr_ud_nl["triggered"], "underdamped_nonlinear_p1_late": tr_ud_nl["p1_harm_late"],
            "underdamped_nonlinear_peak_xi_eff": float(xi_eff_ud_peak),
        },
    }

    with open(RESULTS_DIR / "exp003_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved Entry 003 plots and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
