"""Entry 003: Nonlinear Triggering (Subcritical Hopf Bifurcation) & Cavity Orifice Self-Choking.

Extends our verified 1D closed-choked rocket combustor + causal Erlang-4 time lag +
faceplate Helmholtz resonator with:

0. Physical DC-blocking high-pass filter (omega_hp = 0.20 << pi) on flame pressure sensing:
       H_hp(s) = s / (s + omega_hp)
   Prevents unphysical static k=0 bulk thermal runaway for n > zeta (real choked-feed
   injectors regulate mean flow at omega = 0) while leaving 1L acoustics (>99.8%) intact.

1. Subcritical cubic-quintic flame response (secondary droplet shattering + burnout ceiling):
       Q(x, t) = delta(x - x_f) * n * (q + c3 * q^3 - c5 * q^5),   c3 > 0, c5 > 0
   First-harmonic Flame Describing Function (FDF):
       n_eff(A_f) = n * (1 + (3/4)*c3*A_f^2 - (5/8)*c5*A_f^4)

2. Quadratic orifice jet dissipation in the Helmholtz cavity neck (Ingard/Cummings drag):
       dv_r/dt + (2*xi_H*omega_H + c_nl * |v_r|) * v_r = beta * omega_H^2 * (P(0, t) - P_c)
   Exact passive energy dissipation:
       D_res = (2*xi_H / (beta*omega_H)) * v_r^2 + (c_nl / (beta*omega_H^2)) * |v_r|^3 >= 0
   First-harmonic effective cavity damping ratio:
       xi_eff(V_r) = xi_H + (4 * c_nl / (3 * pi * omega_H)) * V_r

Experiments:
1. Subcritical Hopf bifurcation diagram (no cavity): analytical describing function vs
   direct 1D PDE binary-search bomb tests (triggering threshold P_crit & settled limit cycle P_LC).
2. Nonlinear orifice self-choking trap: compare ideal linear cavity (c_nl=0) vs realistic
   nonlinear orifice drag (c_nl=350) on a beta=1% cavity-stabilized combustor.
3. Design sweep P_crit(xi_H): show that the linear exceptional point self-chokes under finite
   bomb pulses, while deliberately under-damping xi_H < xi_EP maximizes nonlinear bomb survival!
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

OMEGA_HP = 0.20  # High-pass cutoff (<< pi) to prevent unphysical k=0 DC runaway


# ---------------------------------------------------------------------------
# 1. Exact linear Hopf point & coupled eigenvalues (with HP + Erlang-4 + cavity)
# ---------------------------------------------------------------------------

def flame_transfer(s: complex, tau_bar: float, sigma_tau: float, m_stages: int = 4) -> complex:
    """Causal DC-blocked Erlang-m transfer function H_hp(s) * g_hat(s)."""
    hp = s / (s + OMEGA_HP)
    if sigma_tau > 0.0:
        tau_d = np.sqrt(m_stages) * sigma_tau
        tau_0 = max(0.0, tau_bar - tau_d)
        g_hat = np.exp(-s * tau_0) / ((1.0 + s * (tau_d / m_stages)) ** m_stages)
    else:
        g_hat = np.exp(-s * tau_bar)
    return hp * g_hat


def dispersion_F(
    s: complex,
    n: float,
    tau_bar: float,
    x_f: float,
    zeta: float,
    beta: float,
    omega_H: float,
    xi_H: float,
    sigma_tau: float,
    m_stages: int = 4,
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


def solve_root(
    s0: complex,
    n: float,
    tau_bar: float,
    x_f: float,
    zeta: float,
    beta: float,
    omega_H: float,
    xi_H: float,
    sigma_tau: float,
    m_stages: int = 4,
) -> tuple[complex, float]:
    s = complex(s0)
    for _ in range(45):
        fv = dispersion_F(s, n, tau_bar, x_f, zeta, beta, omega_H, xi_H, sigma_tau, m_stages)
        if abs(fv) < 1e-12:
            return s, float(abs(fv))
        eps = 1e-7 * max(1.0, abs(s))
        dfv = (
            dispersion_F(s + eps, n, tau_bar, x_f, zeta, beta, omega_H, xi_H, sigma_tau, m_stages)
            - dispersion_F(s - eps, n, tau_bar, x_f, zeta, beta, omega_H, xi_H, sigma_tau, m_stages)
        ) / (2.0 * eps)
        if abs(dfv) < 1e-16:
            break
        ds = fv / dfv
        if abs(ds) > 0.4:
            ds = 0.4 * ds / abs(ds)
        s -= ds
    fv = dispersion_F(s, n, tau_bar, x_f, zeta, beta, omega_H, xi_H, sigma_tau, m_stages)
    return s, float(abs(fv))


def find_1L_dominant(
    n: float,
    tau_bar: float,
    x_f: float,
    zeta: float,
    beta: float,
    omega_H: float,
    xi_H: float,
    sigma_tau: float,
) -> complex:
    if beta <= 0.0:
        s, _ = solve_root(complex(0.0, 3.12), n, tau_bar, x_f, zeta, 0.0, omega_H, xi_H, sigma_tau)
        return s
    split = np.sqrt(max(1e-4, 0.5 * beta * omega_H * omega_H))
    seeds = [
        complex(-0.02, np.pi - split),
        complex(-0.02, np.pi + split),
        complex(-0.10, np.pi - 0.4 * split),
        complex(-0.10, np.pi + 0.4 * split),
        complex(-0.18, np.pi),
        complex(-xi_H * omega_H, omega_H),
        complex(0.03, np.pi),
    ]
    roots = []
    for sd in seeds:
        r, res = solve_root(sd, n, tau_bar, x_f, zeta, beta, omega_H, xi_H, sigma_tau)
        if res < 1e-8 and 1.5 <= r.imag <= 5.0 and -4.0 <= r.real <= 1.0:
            if all(abs(r - e) > 1e-4 for e in roots):
                roots.append(r)
    return max(roots, key=lambda z: z.real)


def find_hopf_n(tau_bar: float, x_f: float, zeta: float, sigma_tau: float) -> tuple[float, float]:
    """Bisect exact linear Hopf bifurcation n_H (where alpha_1L = 0) with no cavity."""
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


def analytical_fdf_branches(
    n_vals: np.ndarray,
    n_H: float,
    omega_hopf: float,
    x_f: float,
    sigma_tau: float,
    c3: float,
    c5: float,
    m_stages: int = 4,
) -> tuple[np.ndarray, np.ndarray, float]:
    """First-harmonic Flame Describing Function (FDF) branches (P_crit, P_LC, n_fold)."""
    tf_mag = abs(flame_transfer(1j * omega_hopf, 0.20, sigma_tau, m_stages))
    spatial_gain = abs(np.cos(omega_hopf * x_f)) * tf_mag

    gain_max = 1.0 + (9.0 * c3 * c3) / (40.0 * c5)
    n_fold = n_H / gain_max

    p_crit = np.full_like(n_vals, np.nan, dtype=float)
    p_lc = np.full_like(n_vals, np.nan, dtype=float)

    peak_af2 = (3.0 * c3) / (5.0 * c5)
    for i, nv in enumerate(n_vals):
        disc = peak_af2 ** 2 - (8.0 / (5.0 * c5)) * (n_H / nv - 1.0)
        if disc >= 0.0:
            rt = np.sqrt(disc)
            af2_hi = peak_af2 + rt
            af2_lo = peak_af2 - rt
            if af2_hi > 0.0:
                p_lc[i] = np.sqrt(af2_hi) / spatial_gain
            if af2_lo > 0.0 and nv < n_H:
                p_crit[i] = np.sqrt(af2_lo) / spatial_gain
    return p_crit, p_lc, float(n_fold)


# ---------------------------------------------------------------------------
# 2. Nonlinear 1D staggered-grid PDE + HP + Erlang-4 + nonlinear cavity
# ---------------------------------------------------------------------------

def simulate_nonlinear_combustor(
    n: float,
    tau_bar: float,
    x_f: float,
    zeta: float,
    sigma_tau: float = 0.10,
    m_stages: int = 4,
    c3: float = 180.0,
    c5: float = 8000.0,
    beta: float = 0.0,
    omega_H: float = 3.12,
    xi_H: float = 0.10,
    c_nl: float = 0.0,
    p0_amp: float = 0.05,
    t_end: float = 85.0,
    n_cells: int = 100,
    cfl: float = 0.25,
    record_history: bool = False,
) -> dict:
    """Direct 1D RK4 PDE with DC-blocked Erlang-4 subcritical heat release and nonlinear cavity.

    With tau_bar = 0.20, m_stages = 4, sigma_tau = 0.10:
        tau_d = sqrt(4)*0.10 = 0.20 => dead-time tau_0 = 0 (exact 4-stage ODE cascade!).
    """
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
    v_r = 0.0
    p_c = 0.0

    # High-pass filter state p_bar: d(p_bar)/dt = OMEGA_HP * (P[m_flame] - p_bar), p_ac = P[m_flame] - p_bar
    p_bar = 0.0

    # Causal Erlang-m cascade (tau_0 = 0 when tau_d == tau_bar)
    tau_d = np.sqrt(m_stages) * sigma_tau
    rate = m_stages / tau_d
    # Initialize q_st smoothly near steady periodic phase of 1L standing wave
    q_st = np.full(m_stages, float(P[m_flame]))

    def phi_flame(q_val: float) -> float:
        q2 = q_val * q_val
        return q_val * (1.0 + c3 * q2 - c5 * q2 * q2)

    def rhs(
        P_c: np.ndarray,
        U_c: np.ndarray,
        vr_c: float,
        pc_c: float,
        pbar_c: float,
        q_c: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, np.ndarray]:
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
        dP[m_flame] += n * phi_flame(float(q_c[-1])) * inv_dx

        dU = -(P_c[1:] - P_c[:-1]) * inv_dx

        if beta > 0.0:
            dvr = beta * (omega_H ** 2) * (P_c[0] - pc_c) - (2.0 * xi_H * omega_H + c_nl * abs(vr_c)) * vr_c
            dpc = vr_c / beta
        else:
            dvr = 0.0
            dpc = 0.0
        return dP, dU, dvr, dpc, dpbar, dq

    if record_history:
        t_arr = np.linspace(0.0, t_end, n_steps + 1)
        p_head = np.empty(n_steps + 1)
        vr_hist = np.empty(n_steps + 1)
        p_head[0] = P[0]
        vr_hist[0] = v_r
    else:
        t_arr = p_head = vr_hist = None

    # Track peak amplitudes in three windows to unambiguously classify growth vs decay
    w1_lo, w1_hi = int(0.15 * n_steps), int(0.30 * n_steps)
    w2_lo, w2_hi = int(0.55 * n_steps), int(0.70 * n_steps)
    w3_lo = int(0.85 * n_steps)

    p_w1 = p_w2 = p_w3 = 0.0
    vr_peak_all = 0.0
    blew_up = False

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
        if w1_lo <= step < w1_hi and ap > p_w1:
            p_w1 = ap
        if w2_lo <= step < w2_hi and ap > p_w2:
            p_w2 = ap
        if step >= w3_lo and ap > p_w3:
            p_w3 = ap

        if record_history:
            p_head[step + 1] = P[0]
            vr_hist[step + 1] = v_r

        if not np.isfinite(ap) or ap > 2.0:
            blew_up = True
            p_w3 = float("inf")
            break

    # Triggered iff amplitude is either growing (p_w3 > p_w2) OR settled on a large limit cycle (> 0.11)
    triggered = bool(blew_up or p_w3 > 0.11 or (p_w3 > p_w2 * 1.005 and p_w3 > 0.002))

    return {
        "p_w1": float(p_w1),
        "p_w2": float(p_w2),
        "p_late_max": float(p_w3),
        "vr_peak_all": float(vr_peak_all),
        "triggered": triggered,
        "x_f_actual": x_f_actual,
        "t": t_arr,
        "p_head": p_head,
        "vr_hist": vr_hist,
    }


def bisect_trigger_threshold(
    n: float,
    tau_bar: float,
    x_f: float,
    zeta: float,
    sigma_tau: float,
    c3: float,
    c5: float,
    beta: float = 0.0,
    omega_H: float = 3.12,
    xi_H: float = 0.10,
    c_nl: float = 0.0,
    p_lo: float = 0.004,
    t_end: float = 95.0,
    n_iter: int = 11,
) -> tuple[float | None, float]:
    """Binary search for exact critical bomb-pulse amplitude P_crit separating decay from limit cycle."""
    # Check linear stability at tiny pulse
    r_lo = simulate_nonlinear_combustor(
        n, tau_bar, x_f, zeta, sigma_tau, 4, c3, c5, beta, omega_H, xi_H, c_nl,
        p0_amp=p_lo, t_end=t_end,
    )
    if r_lo["triggered"]:
        # Measure settled limit cycle starting near upper branch
        r_lc = simulate_nonlinear_combustor(
            n, tau_bar, x_f, zeta, sigma_tau, 4, c3, c5, beta, omega_H, xi_H, c_nl,
            p0_amp=0.17, t_end=110.0,
        )
        return 0.0, r_lc["p_late_max"]

    # Probe range of finite bomb amplitudes (include subcritical sweet spot 0.14..0.24)
    hi_found = None
    for pa in (0.08, 0.12, 0.16, 0.20, 0.24):
        r = simulate_nonlinear_combustor(
            n, tau_bar, x_f, zeta, sigma_tau, 4, c3, c5, beta, omega_H, xi_H, c_nl,
            p0_amp=pa, t_end=t_end,
        )
        if r["triggered"]:
            hi_found = pa
            break

    if hi_found is None:
        return None, 0.0

    lo, hi = p_lo, hi_found
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        r = simulate_nonlinear_combustor(
            n, tau_bar, x_f, zeta, sigma_tau, 4, c3, c5, beta, omega_H, xi_H, c_nl,
            p0_amp=mid, t_end=t_end,
        )
        if r["triggered"]:
            hi = mid
        else:
            lo = mid

    # Dedicated settled limit-cycle measurement well above threshold
    r_lc = simulate_nonlinear_combustor(
        n, tau_bar, x_f, zeta, sigma_tau, 4, c3, c5, beta, omega_H, xi_H, c_nl,
        p0_amp=max(0.18, hi * 1.1), t_end=110.0,
    )
    return 0.5 * (lo + hi), float(r_lc["p_late_max"])


# ---------------------------------------------------------------------------
# 3. Main experiments & plotting
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    zeta = 0.04
    tau_bar = 0.20
    x_f = 0.15
    sigma_tau = 0.10  # Pure 4-stage causal Erlang (tau_0 = 0), cleanly kills 3L/5L harmonics!
    c3 = 180.0
    c5 = 8000.0

    x_p100 = (np.arange(100) + 0.5) / 100.0
    x_f_grid = float(x_p100[np.argmin(np.abs(x_p100 - x_f))])

    n_H, w_hopf = find_hopf_n(tau_bar, x_f_grid, zeta, sigma_tau)
    print("=== Entry 003: Nonlinear Subcritical Triggering & Cavity Orifice Self-Choking ===")
    print(f"Bare combustor (x_f={x_f_grid:.4f}, tau={tau_bar}, sigma_tau={sigma_tau}, omega_hp={OMEGA_HP}, zeta={zeta}):")
    print(f"  Exact linear Hopf point: n_H = {n_H:.5f}, omega_H = {w_hopf:.5f}")

    # ------------------------------------------------------------------
    # Experiment 1: Bare-Combustor Subcritical Bifurcation Diagram
    # ------------------------------------------------------------------
    n_curve = np.linspace(0.032, 0.095, 220)
    p_crit_fdf, p_lc_fdf, n_fold = analytical_fdf_branches(n_curve, n_H, w_hopf, x_f_grid, sigma_tau, c3, c5)
    print(f"  Analytical FDF fold point: n_fold = {n_fold:.5f} (bistable window n in [{n_fold:.4f}, {n_H:.4f}])\n")

    print("--- Experiment 1: Direct 1D PDE Bomb-Test Bifurcation Verification (No Cavity) ---")
    n_test_pts = [0.036, 0.042, 0.048, 0.054, 0.060, 0.066, 0.076, 0.086]
    pde_bif = []
    for nv in n_test_pts:
        pc, plc = bisect_trigger_threshold(nv, tau_bar, x_f, zeta, sigma_tau, c3, c5, beta=0.0)
        pc_a, plc_a, _ = analytical_fdf_branches(np.array([nv]), n_H, w_hopf, x_f_grid, sigma_tau, c3, c5)
        pc_str = f"{pc:.4f}" if pc is not None else "uncond. stable"
        pca_str = f"{pc_a[0]:.4f}" if np.isfinite(pc_a[0]) else ("0.0000" if nv >= n_H else "uncond. stable")
        plca_str = f"{plc_a[0]:.4f}" if np.isfinite(plc_a[0]) else "0.0000"
        print(f"  n={nv:.3f}: P_crit PDE={pc_str:>14s} (FDF={pca_str:>14s}) | P_LC PDE={plc:.4f} (FDF={plca_str})")
        pde_bif.append({
            "n": nv,
            "p_crit_pde": pc,
            "p_lc_pde": plc,
            "p_crit_fdf": None if not np.isfinite(pc_a[0]) else float(pc_a[0]),
            "p_lc_fdf": None if not np.isfinite(plc_a[0]) else float(plc_a[0]),
        })

    # Two bomb traces in the bistable linearly stable zone (n = 0.054 < n_H)
    n_bistable = 0.054
    pc_bi = next(d["p_crit_pde"] for d in pde_bif if abs(d["n"] - n_bistable) < 1e-6)
    tr_decay = simulate_nonlinear_combustor(
        n_bistable, tau_bar, x_f, zeta, sigma_tau, 4, c3, c5, p0_amp=pc_bi * 0.88, t_end=85.0, record_history=True
    )
    tr_trig = simulate_nonlinear_combustor(
        n_bistable, tau_bar, x_f, zeta, sigma_tau, 4, c3, c5, p0_amp=pc_bi * 1.12, t_end=85.0, record_history=True
    )

    # ------------------------------------------------------------------
    # Experiment 2 & 3: Nonlinear Orifice Drag Sweep (beta = 1%, n = 0.088)
    # ------------------------------------------------------------------
    n_cav = 0.088       # Linearly UNSTABLE bare chamber (n > n_H), strong subcritical boost!
    beta_cav = 0.010    # 1.0% cavity volume ratio
    omega_cav = 3.12
    c_nl_val = 450.0    # Realistic sharp-edged orifice quadratic drag (A_neck/A_ch ~ 1%)

    s_bare = find_1L_dominant(n_cav, tau_bar, x_f_grid, zeta, 0.0, omega_cav, 0.1, sigma_tau)
    print(f"\n--- Experiment 2 & 3: Nonlinear Orifice Drag Sweep (n={n_cav}, beta={beta_cav}, c_nl={c_nl_val}) ---")
    print(f"Bare chamber at n={n_cav}: linearly UNSTABLE (s = {s_bare.real:+.5f} + {s_bare.imag:.4f}i)")

    xi_grid = [0.015, 0.030, 0.045, 0.065, 0.090, 0.125, 0.165, 0.210, 0.260, 0.320]
    sweep_rows = []
    for xi in xi_grid:
        s_lin = find_1L_dominant(n_cav, tau_bar, x_f_grid, zeta, beta_cav, omega_cav, xi, sigma_tau)
        pc_lin, _ = bisect_trigger_threshold(
            n_cav, tau_bar, x_f, zeta, sigma_tau, c3, c5,
            beta=beta_cav, omega_H=omega_cav, xi_H=xi, c_nl=0.0, t_end=85.0,
        )
        pc_nl, plc_nl = bisect_trigger_threshold(
            n_cav, tau_bar, x_f, zeta, sigma_tau, c3, c5,
            beta=beta_cav, omega_H=omega_cav, xi_H=xi, c_nl=c_nl_val, t_end=85.0,
        )
        pc_lin_s = f"{pc_lin:.4f}" if pc_lin is not None else "NO TRIGGER"
        pc_nl_s = f"{pc_nl:.4f}" if pc_nl is not None else "NO TRIGGER"
        print(
            f"  xi_H={xi:.3f}: alpha_lin={s_lin.real:+.5f} | "
            f"P_crit(linear cav)={pc_lin_s:>10s} | P_crit(nonlin c_nl={int(c_nl_val)})={pc_nl_s:>10s} (P_LC={plc_nl:.4f})"
        )
        sweep_rows.append({
            "xi_H": xi,
            "alpha_lin": float(s_lin.real),
            "p_crit_linear_cavity": pc_lin,
            "p_crit_nonlinear_cavity": pc_nl,
            "p_lc_nonlinear_cavity": plc_nl,
        })

    # Identify linear EP xi and nonlinear best xi from the sweep
    idx_lin_ep = int(np.argmin([r["alpha_lin"] for r in sweep_rows]))
    xi_lin_ep = sweep_rows[idx_lin_ep]["xi_H"]
    # Pick best nonlinear xi (preferring NO TRIGGER or highest P_crit)
    def nl_score(r: dict) -> float:
        return 1.0 if r["p_crit_nonlinear_cavity"] is None else r["p_crit_nonlinear_cavity"]
    idx_nl_best = int(np.argmax([nl_score(r) for r in sweep_rows]))
    xi_nl_best = sweep_rows[idx_nl_best]["xi_H"]

    # Choose a bomb pulse right between the linear-EP P_crit and the under-damped P_crit!
    pc_ep_nl = sweep_rows[idx_lin_ep]["p_crit_nonlinear_cavity"]
    pc_best_nl = sweep_rows[idx_nl_best]["p_crit_nonlinear_cavity"]
    if pc_ep_nl is not None:
        if pc_best_nl is None:
            p_bomb = min(0.18, max(0.12, pc_ep_nl * 1.35))
        else:
            p_bomb = 0.5 * (pc_ep_nl + pc_best_nl)
    else:
        p_bomb = 0.14

    tr_ep_lin = simulate_nonlinear_combustor(
        n_cav, tau_bar, x_f, zeta, sigma_tau, 4, c3, c5,
        beta=beta_cav, omega_H=omega_cav, xi_H=xi_lin_ep, c_nl=0.0, p0_amp=p_bomb, t_end=80.0, record_history=True
    )
    tr_ep_nl = simulate_nonlinear_combustor(
        n_cav, tau_bar, x_f, zeta, sigma_tau, 4, c3, c5,
        beta=beta_cav, omega_H=omega_cav, xi_H=xi_lin_ep, c_nl=c_nl_val, p0_amp=p_bomb, t_end=80.0, record_history=True
    )
    tr_ud_nl = simulate_nonlinear_combustor(
        n_cav, tau_bar, x_f, zeta, sigma_tau, 4, c3, c5,
        beta=beta_cav, omega_H=omega_cav, xi_H=xi_nl_best, c_nl=c_nl_val, p0_amp=p_bomb, t_end=80.0, record_history=True
    )

    xi_eff_ep_peak = xi_lin_ep + (4.0 * c_nl_val / (3.0 * np.pi * omega_cav)) * tr_ep_nl["vr_peak_all"]
    xi_eff_ud_peak = xi_nl_best + (4.0 * c_nl_val / (3.0 * np.pi * omega_cav)) * tr_ud_nl["vr_peak_all"]

    print(f"\nLinear EP xi_H = {xi_lin_ep:.3f} vs Nonlinear-optimal xi_H = {xi_nl_best:.3f}")
    print(f"Bomb test at P0 = {p_bomb:.3f} (n = {n_cav}, beta = 1.0%, c_nl = {c_nl_val}):")
    print(f"  Linear-EP cavity (xi={xi_lin_ep:.3f}, c_nl=0):   triggered={tr_ep_lin['triggered']}, P_late={tr_ep_lin['p_late_max']:.4f}")
    print(f"  Linear-EP cavity (xi={xi_lin_ep:.3f}, c_nl={int(c_nl_val)}): triggered={tr_ep_nl['triggered']}, P_late={tr_ep_nl['p_late_max']:.4f}, peak xi_eff={xi_eff_ep_peak:.3f}")
    print(f"  Under-damped     (xi={xi_nl_best:.3f}, c_nl={int(c_nl_val)}): triggered={tr_ud_nl['triggered']}, P_late={tr_ud_nl['p_late_max']:.4f}, peak xi_eff={xi_eff_ud_peak:.3f}")

    # ------------------------------------------------------------------
    # Figure 1: Subcritical Hopf Bifurcation & Bare-Combustor Bomb Tests
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2), dpi=150)

    ax1.axvline(n_H, color="black", ls=":", lw=1.3, label=f"Linear Hopf $n_H = {n_H:.4f}$")
    ax1.axvline(n_fold, color="#7f7f7f", ls="--", lw=1.2, label=f"Fold point $n_{{\\mathrm{{fold}}}} = {n_fold:.4f}$")
    ax1.axvspan(n_fold, n_H, color="#ffeda0", alpha=0.38, label="Bistable triggering zone")

    ax1.plot(n_curve, p_lc_fdf, "-", color="#d62728", lw=2.2, label="Analytical FDF limit cycle $P_{\\mathrm{LC}}$")
    ax1.plot(n_curve, p_crit_fdf, "--", color="#1f77b4", lw=2.0, label="Analytical FDF bomb threshold $P_{\\mathrm{crit}}$")

    n_pc_pts = [d["n"] for d in pde_bif if d["p_crit_pde"] is not None]
    y_pc_pts = [d["p_crit_pde"] for d in pde_bif if d["p_crit_pde"] is not None]
    n_lc_pts = [d["n"] for d in pde_bif if d["p_lc_pde"] > 0.05]
    y_lc_pts = [d["p_lc_pde"] for d in pde_bif if d["p_lc_pde"] > 0.05]
    ax1.scatter(n_lc_pts, y_lc_pts, c="#d62728", edgecolors="black", s=55, zorder=5, label="1D PDE settled limit cycle")
    ax1.scatter(n_pc_pts, y_pc_pts, c="#1f77b4", marker="D", edgecolors="black", s=55, zorder=5, label="1D PDE bomb threshold")

    ax1.set_xlabel("Interaction index $n$")
    ax1.set_ylabel("Head-end pressure amplitude $|P(0)|$")
    ax1.set_title("Subcritical Hopf Bifurcation (Bare Combustor, $\\sigma_\\tau=0.10$)", fontweight="bold")
    ax1.set_xlim(0.032, 0.090)
    ax1.set_ylim(-0.005, 0.25)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=7.8)

    ax2.plot(tr_trig["t"], tr_trig["p_head"], color="#d62728", lw=0.95,
             label=f"Bomb $P_0 = {pc_bi*1.12:.3f} > P_{{\\mathrm{{crit}}}}$ $\\rightarrow$ Limit Cycle ($P_{{\\mathrm{{LC}}}}={tr_trig['p_late_max']:.3f}$)")
    ax2.plot(tr_decay["t"], tr_decay["p_head"], color="#1f77b4", lw=1.05,
             label=f"Bomb $P_0 = {pc_bi*0.88:.3f} < P_{{\\mathrm{{crit}}}}$ $\\rightarrow$ Decays")
    ax2.set_xlabel("Dimensionless time $t \\bar{c}/L$")
    ax2.set_ylabel("Head-end pressure $P(0, t)$")
    ax2.set_title(f"Linearly Stable Combustor ($n={n_bistable} < n_H$) Triggered by Bomb Pulse", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8.0)

    fig.suptitle("Entry 003 (Part 1): Subcritical Triggering — Analytical Describing Function vs. 1D PDE", fontsize=12.3, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp003_subcritical_bifurcation.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Nonlinear Cavity Self-Choking Trap & Linear-vs-Nonlinear Optimum Shift
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.4), dpi=150)

    xis = np.array([r["xi_H"] for r in sweep_rows])
    alphas = np.array([r["alpha_lin"] for r in sweep_rows])
    cap_val = 0.25
    pc_nl_arr = np.array([cap_val if r["p_crit_nonlinear_cavity"] is None else r["p_crit_nonlinear_cavity"] for r in sweep_rows])
    pc_lin_arr = np.array([cap_val if r["p_crit_linear_cavity"] is None else r["p_crit_linear_cavity"] for r in sweep_rows])

    ax1_r = ax1.twinx()
    l1, = ax1.plot(xis, pc_nl_arr, "o-", color="#d62728", lw=2.3, ms=6.5,
                   label=f"Bomb threshold $P_{{\\mathrm{{crit}}}}$ (nonlin orifice $c_{{\\mathrm{{nl}}}}={int(c_nl_val)}$)")
    l2, = ax1.plot(xis, pc_lin_arr, "s--", color="#2ca02c", lw=1.7, ms=5.5, alpha=0.85,
                   label="Bomb threshold $P_{\\mathrm{crit}}$ (ideal linear cavity, $c_{\\mathrm{nl}}=0$)")
    l3, = ax1_r.plot(xis, -alphas, "^:", color="#1f77b4", lw=1.8, ms=5.5,
                     label=f"Linear damping margin $-\\alpha_{{\\mathrm{{lin}}}}$ (peaks at EP $\\xi={xi_lin_ep:.3f}$)")

    ax1.axvline(xi_lin_ep, color="#1f77b4", ls=":", lw=1.2, alpha=0.7)
    ax1.axhline(cap_val, color="gray", ls="--", lw=0.9, alpha=0.6)
    ax1.text(0.24, cap_val + 0.005, "No trigger (<=0.24)", fontsize=7.5, color="gray")

    ax1.set_xlabel("Linear cavity damping ratio $\\xi_H$")
    ax1.set_ylabel("Triggering threshold $P_{\\mathrm{crit}}$ (higher = safer)", color="#d62728")
    ax1_r.set_ylabel("Linear stability margin $-\\alpha_{\\mathrm{lin}}$", color="#1f77b4")
    ax1.set_ylim(0.0, 0.28)
    ax1_r.set_ylim(-0.01, 0.18)
    ax1.set_title(f"Linear vs. Nonlinear Cavity Optimum ($n={n_cav}, \\beta=1.0\\%$)", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend([l1, l2, l3], [l1.get_label(), l2.get_label(), l3.get_label()], loc="lower left", fontsize=7.5)

    # Panel 2: Same bomb pulse hitting all 3 cavity designs
    ax2.plot(tr_ep_nl["t"], tr_ep_nl["p_head"], color="#d62728", lw=0.95,
             label=f"Linear-EP ($\\xi_H={xi_lin_ep:.3f}, c_{{\\mathrm{{nl}}}}={int(c_nl_val)}$) $\\rightarrow$ "
                   f"{'CHOKES & TRIGGERS' if tr_ep_nl['triggered'] else 'Decays'} (peak $\\xi_{{\\mathrm{{eff}}}}={xi_eff_ep_peak:.2f}$)")
    ax2.plot(tr_ep_lin["t"], tr_ep_lin["p_head"], color="#7f7f7f", lw=1.0, ls="--", alpha=0.85,
             label=f"Ideal Linear-EP ($\\xi_H={xi_lin_ep:.3f}, c_{{\\mathrm{{nl}}}}=0$) $\\rightarrow$ "
                   f"{'Triggers' if tr_ep_lin['triggered'] else 'Decays'}")
    ax2.plot(tr_ud_nl["t"], tr_ud_nl["p_head"], color="#2ca02c", lw=1.15,
             label=f"Under-damped ($\\xi_H={xi_nl_best:.3f}, c_{{\\mathrm{{nl}}}}={int(c_nl_val)}$) $\\rightarrow$ "
                   f"{'Triggers' if tr_ud_nl['triggered'] else 'SURVIVES BOMB'} (peak $\\xi_{{\\mathrm{{eff}}}}={xi_eff_ud_peak:.2f}$)")
    ax2.set_xlabel("Dimensionless time $t \\bar{c}/L$")
    ax2.set_ylabel("Head-end pressure $P(0, t)$")
    ax2.set_title(f"Same Bomb Pulse ($P_0 = {p_bomb:.3f}$) Across Three Cavity Designs", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=7.6)

    fig.suptitle("Entry 003 (Part 2): Nonlinear Orifice Self-Choking & Why Under-Damping Wins Bomb Tests", fontsize=12.3, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp003_cavity_self_choking.png")
    plt.close(fig)

    metrics = {
        "parameters": {
            "zeta": zeta,
            "tau_bar": tau_bar,
            "x_f_grid": x_f_grid,
            "sigma_tau": sigma_tau,
            "omega_hp": OMEGA_HP,
            "c3": c3,
            "c5": c5,
            "n_hopf_bare": n_H,
            "omega_hopf_bare": w_hopf,
            "n_fold_fdf": n_fold,
            "cavity_test": {"n": n_cav, "beta": beta_cav, "omega_H": omega_cav, "c_nl": c_nl_val},
        },
        "exp1_bifurcation_bare": pde_bif,
        "exp2_3_cavity_xi_sweep": sweep_rows,
        "bomb_comparison": {
            "p0": float(p_bomb),
            "xi_lin_ep": xi_lin_ep,
            "xi_nl_best": xi_nl_best,
            "ep_linear_triggered": tr_ep_lin["triggered"],
            "ep_linear_p_late": tr_ep_lin["p_late_max"],
            "ep_nonlinear_triggered": tr_ep_nl["triggered"],
            "ep_nonlinear_p_late": tr_ep_nl["p_late_max"],
            "ep_nonlinear_peak_xi_eff": float(xi_eff_ep_peak),
            "underdamped_nonlinear_triggered": tr_ud_nl["triggered"],
            "underdamped_nonlinear_p_late": tr_ud_nl["p_late_max"],
            "underdamped_nonlinear_peak_xi_eff": float(xi_eff_ud_peak),
        },
    }

    with open(RESULTS_DIR / "exp003_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved Entry 003 plots and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
