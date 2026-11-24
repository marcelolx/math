"""Entry 001 (grid-stability): Optimal Spatial Placement of Grid-Forming Inverters & Droop Saturation Cliff.

Hardened after Peer Review #4:
- Uses an exact saturated rotating-frame static equilibrium existence check (solving
  sum_i [D_base,i * omega_ss + h_i * tanh(D_gfm,i * omega_ss / h_i)] = -Delta_P via brentq
  and verifying Fiedler stability lambda_2(L) > 0) combined with long-horizon (T = 80s)
  pole-slip monitoring (verified against T = 120s). Eliminates both short-horizon ghost-crawl
  false positives (C1) and slow-settling false negatives (C2).
- Tests ALL 5 placement policies (including Area A only and Area B only!) under BOTH
  unsaturated droop (Exp 1) and primary-droop active-power saturation (Exp 2A & 2B).
- Fixes compute_linear_metrics H_2 norm to use the exact [Q^T delta, omega] realization
  (removing only the true zero mode [1; 0]), which is exact for arbitrary non-proportional M, D.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import brentq

from baseline_power_grid import (
    build_K_matrix,
    find_equilibrium,
    linear_swing_spectrum,
    make_12bus_grid,
    swing_energy,
)


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# 1. Saturated Rotating-Frame Equilibrium Check & Nonlinear Swing Integrator
# ---------------------------------------------------------------------------

def droop_vec(om: np.ndarray | float, D_gfm: np.ndarray, h_gfm: np.ndarray) -> np.ndarray:
    has_cap = np.isfinite(h_gfm) & (h_gfm > 0.0) & (D_gfm > 0.0)
    safe_h = np.where(has_cap, h_gfm, 1.0)
    lin = D_gfm * om
    return np.where(has_cap, safe_h * np.tanh(lin / safe_h), lin)


def check_saturated_rotating_equilibrium(
    D_base: np.ndarray,
    D_gfm: np.ndarray,
    h_gfm: np.ndarray,
    P_post: np.ndarray,
    K_post: np.ndarray,
    theta_guess: np.ndarray | None = None,
) -> tuple[bool, float, float]:
    """Exact post-fault synchronous rotating-frame equilibrium check under saturated droop.

    In a synchronous rotating frame with common frequency omega_ss:
        sum_i [D_base,i * omega_ss + droop_i(omega_ss)] = sum_i P_post,i
    Since D_base,i > 0 and droop_i is strictly increasing, omega_ss is unique.
    Then P_rot,i = P_post,i - D_base,i * omega_ss - droop_i(omega_ss) satisfies sum P_rot = 0;
    a stable post-fault synchronous state exists iff P_rot = K_post sin(dtheta) has a
    Fiedler-stable solution (lambda_2(L) > 0).
    """
    net_P = float(np.sum(P_post))
    if abs(net_P) < 1e-14:
        om_ss = 0.0
    else:
        def f_bal(om: float) -> float:
            return float(np.sum(D_base * om + droop_vec(om, D_gfm, h_gfm))) - net_P
        lo, hi = -20.0, 20.0
        while f_bal(lo) > 0.0:
            lo *= 2.0
        while f_bal(hi) < 0.0:
            hi *= 2.0
        om_ss = float(brentq(f_bal, lo, hi, xtol=1e-13, rtol=1e-13))

    P_rot = P_post - D_base * om_ss - droop_vec(om_ss, D_gfm, h_gfm)
    ok, _, lam2 = find_equilibrium(P_rot, K_post, theta_guess=theta_guess)
    return bool(ok), om_ss, float(lam2)


def simulate_swing_saturated(
    M: np.ndarray,
    D_base: np.ndarray,
    D_gfm: np.ndarray,
    h_gfm: np.ndarray,
    P_pre: np.ndarray,
    K_pre: np.ndarray,
    P_post: np.ndarray,
    K_post: np.ndarray,
    theta0: np.ndarray,
    t_fault: float = 1.0,
    dt: float = 0.004,
    t_end: float = 80.0,
    check_energy: bool = False,
) -> dict:
    """Horizon-robust RK4 nonlinear swing simulation with smooth primary-droop saturation."""
    ok_eq, om_ss, lam2_eq = check_saturated_rotating_equilibrium(D_base, D_gfm, h_gfm, P_post, K_post, theta_guess=theta0)
    if (not ok_eq) and (not check_energy):
        return {
            "synchronized": False, "eq_exists": False, "om_ss": om_ss,
            "max_angle_spread_rad": float("inf"), "energy_balance_rel_err": None,
        }

    n = len(M)
    inv_M = 1.0 / M
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps
    t_arr = np.linspace(0.0, t_end, n_steps + 1)

    has_cap = np.isfinite(h_gfm) & (h_gfm > 0.0) & (D_gfm > 0.0)
    safe_h = np.where(has_cap, h_gfm, 1.0)

    def droop_torque(om: np.ndarray) -> np.ndarray:
        lin = D_gfm * om
        return np.where(has_cap, safe_h * np.tanh(lin / safe_h), lin)

    def rhs(th_v: np.ndarray, om_v: np.ndarray, K_m: np.ndarray, P_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        diff = th_v[:, None] - th_v[None, :]
        pe = np.sum(K_m * np.sin(diff), axis=1)
        damp = D_base * om_v + droop_torque(om_v)
        return om_v, inv_M * (P_v - pe - damp)

    th = theta0.copy()
    om = np.zeros(n)
    max_spread = 0.0

    H_post_list = []
    diss_post_list = []
    t_post_start_step = int(round(t_fault / dt))

    for step in range(n_steps):
        if t_arr[step] >= t_fault:
            K_cur, P_cur = K_post, P_post
        else:
            K_cur, P_cur = K_pre, P_pre

        if check_energy and step >= t_post_start_step:
            H_post_list.append(swing_energy(th, om, M, P_post, K_post))
            diss_post_list.append(float(np.sum((D_base * om + droop_torque(om)) * om)))

        k1_t, k1_o = rhs(th, om, K_cur, P_cur)
        k2_t, k2_o = rhs(th + 0.5 * dt * k1_t, om + 0.5 * dt * k1_o, K_cur, P_cur)
        k3_t, k3_o = rhs(th + 0.5 * dt * k2_t, om + 0.5 * dt * k2_o, K_cur, P_cur)
        k4_t, k4_o = rhs(th + dt * k3_t, om + dt * k3_o, K_cur, P_cur)

        th = th + (dt / 6.0) * (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t)
        om = om + (dt / 6.0) * (k1_o + 2.0 * k2_o + 2.0 * k3_o + k4_o)

        th_rel = th - np.mean(th)
        sp = float(np.max(th_rel) - np.min(th_rel))
        if sp > max_spread:
            max_spread = sp

        if (not check_energy) and sp > np.pi:
            return {
                "synchronized": False, "eq_exists": ok_eq, "om_ss": om_ss,
                "max_angle_spread_rad": sp, "energy_balance_rel_err": None,
            }

    e_err = None
    if check_energy and len(H_post_list) > 5:
        H_arr = np.array(H_post_list)
        d_arr = np.array(diss_post_list)
        cum_d = np.concatenate([[0.0], np.cumsum(0.5 * (d_arr[:-1] + d_arr[1:]) * dt)])
        dH = H_arr - H_arr[0]
        e_err = float(np.max(np.abs(dH + cum_d)) / max(float(np.max(np.abs(dH))), 1e-12))

    # Horizon-robust survival: post-fault stable rotating equilibrium exists AND no pole slip over [0, t_end]
    synced = bool(ok_eq and (max_spread < np.pi))
    return {
        "synchronized": synced, "eq_exists": ok_eq, "om_ss": om_ss,
        "max_angle_spread_rad": float(max_spread), "energy_balance_rel_err": e_err,
    }


def compute_linear_metrics(theta_eq: np.ndarray, M: np.ndarray, D_tot_vec: np.ndarray, K: np.ndarray) -> dict:
    """Exact post-fault linearized inter-area damping alpha_dom and exact H_2 synchronization norm.

    Uses the exact (2n - 1)-dimensional state realization x = [Q_orth^T delta; omega] removing
    ONLY the true zero shift mode [1; 0], preserving exact COI-internal coupling for arbitrary
    non-proportional M_i and D_i (fixes Peer Review #4 M1).
    """
    n = len(M)
    ev = linear_swing_spectrum(theta_eq, M, D_tot_vec, K)
    osc = [z for z in ev if abs(z.imag) > 0.05]
    alpha_dom = float(max(z.real for z in osc)) if osc else 0.0

    ones = np.ones((n, 1)) / np.sqrt(n)
    U_svd, _, _ = np.linalg.svd(ones, full_matrices=True)
    Q_orth = U_svd[:, 1:]  # (n, n-1) orthonormal complement of 1

    diff = theta_eq[:, None] - theta_eq[None, :]
    W = K * np.cos(diff)
    L = np.diag(np.sum(W, axis=1)) - W
    Minv = np.diag(1.0 / M)

    # Exact state x = [z_ang (n-1); omega (n)], where z_ang = Q_orth^T delta:
    #   dz_ang/dt = Q_orth^T omega
    #   domega/dt = -M^{-1} L Q_orth z_ang - M^{-1} D omega + M^{-1} Q_orth u
    # Since L*1 = 0, L delta = L Q_orth z_ang is EXACT!
    A_exact = np.block([
        [np.zeros((n - 1, n - 1)), Q_orth.T],
        [-Minv @ L @ Q_orth, -Minv @ np.diag(D_tot_vec)],
    ])
    B_exact = np.vstack([np.zeros((n - 1, n - 1)), Minv @ Q_orth])
    C_exact = np.hstack([np.eye(n - 1), np.zeros((n - 1, n))])

    P_lyap = solve_continuous_lyapunov(A_exact, -(B_exact @ B_exact.T))
    h2_angle = float(np.sqrt(max(np.trace(C_exact @ P_lyap @ C_exact.T), 0.0)))
    return {"alpha_dom": alpha_dom, "h2_angle_norm": h2_angle}


def bisect_dP_crit(
    M: np.ndarray,
    D_base: np.ndarray,
    D_gfm: np.ndarray,
    h_gfm: np.ndarray,
    P_pre: np.ndarray,
    K_pre: np.ndarray,
    K_post: np.ndarray,
    theta0: np.ndarray,
    fault_bus: int,
    dP_hi: float = 2.4,
    n_iter: int = 11,
    t_end: float = 80.0,
) -> float:
    lo, hi = 0.0, dP_hi
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        P_post = P_pre.copy()
        P_post[fault_bus] -= mid
        res = simulate_swing_saturated(
            M, D_base, D_gfm, h_gfm, P_pre, K_pre, P_post, K_post, theta0,
            t_fault=1.0, dt=0.004, t_end=t_end,
        )
        if res["synchronized"]:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


# ---------------------------------------------------------------------------
# 2. Main Experiments & Figures
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Entry 001 (grid-stability): GFM Spatial Placement & Droop Saturation Cliff (Hardened) ===\n")

    n12, edges12, P12, gen_buses = make_12bus_grid()
    K12_pre = build_K_matrix(n12, edges12)
    edges12_post = [e for e in edges12 if not (e[0] == 5 and e[1] == 9)]
    K12_post = build_K_matrix(n12, edges12_post)

    ok_pre, th_pre, _ = find_equilibrium(P12, K12_pre)
    ok_post0, th_post0, _ = find_equilibrium(P12, K12_post)
    assert ok_pre and ok_post0

    diff_p = th_post0[:, None] - th_post0[None, :]
    W_p = K12_post * np.cos(diff_p)
    L_p = np.diag(np.sum(W_p, axis=1)) - W_p
    _, evecs_L = np.linalg.eigh(L_p)
    v2 = evecs_L[:, 1]
    w_fiedler = np.array([v2[b] ** 2 for b in gen_buses])
    w_fiedler /= np.sum(w_fiedler)

    M_GFM_TOT = 5.4
    D_GFM_TOT = 4.8

    M_base = np.full(n12, 0.12)
    D_base = np.full(n12, 0.12)
    for b in gen_buses:
        M_base[b] = 0.08
        D_base[b] = 0.04

    policies = {
        "area_A_only":       np.array([1 / 3, 1 / 3, 1 / 3, 0.0,   0.0,   0.0]),
        "area_B_only":       np.array([0.0,   0.0,   0.0,   1 / 3, 1 / 3, 1 / 3]),
        "tie_gateways_2_6":  np.array([0.0,   0.0,   0.5,   0.5,   0.0,   0.0]),
        "fiedler_weighted":  w_fiedler,
        "uniform_6_buses":   np.full(6, 1.0 / 6.0),
    }
    all_policy_names = list(policies.keys())

    def build_MD(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        M = M_base.copy()
        D_gfm = np.zeros(n12)
        for idx, b in enumerate(gen_buses):
            M[b] += weights[idx] * M_GFM_TOT
            D_gfm[b] = weights[idx] * D_GFM_TOT
        return M, D_gfm

    # Exact energy-balance verification with tanh droop saturation
    M_uni, Dg_uni = build_MD(policies["uniform_6_buses"])
    h_test = np.zeros(n12)
    for b in gen_buses:
        h_test[b] = 0.25
    P_test = P12.copy()
    P_test[6] -= 0.75
    res_econ = simulate_swing_saturated(
        M_uni, D_base, Dg_uni, h_test, P12, K12_pre, P_test, K12_post, th_pre,
        t_fault=1.0, dt=0.002, t_end=20.0, check_energy=True,
    )
    print(f"Verification: Exact nonlinear energy-balance residual with tanh droop saturation: {res_econ['energy_balance_rel_err']:.2e}\n")

    # Experiment 1: Unsaturated Fixed-Budget Placement (horizon-robust T=80s + static eq check)
    print("--- Experiment 1: Unsaturated Fixed-Budget GFM Placement (T=80s + Static Equilibrium Check) ---")
    h_inf = np.full(n12, np.inf)
    exp1_summary = {}
    for name in all_policy_names:
        w = policies[name]
        M, Dg = build_MD(w)
        lin = compute_linear_metrics(th_post0, M, D_base + Dg, K12_post)
        dP_A = bisect_dP_crit(M, D_base, Dg, h_inf, P12, K12_pre, K12_post, th_pre, fault_bus=0, t_end=80.0)
        dP_B = bisect_dP_crit(M, D_base, Dg, h_inf, P12, K12_pre, K12_post, th_pre, fault_bus=6, t_end=80.0)
        dP_worst = min(dP_A, dP_B)
        print(
            f"  [{name:18s}] alpha_dom={lin['alpha_dom']:+.4f} | exact H2_ang={lin['h2_angle_norm']:.4f} | "
            f"dP_crit(A)={dP_A:.3f} | dP_crit(B)={dP_B:.3f} | WORST={dP_worst:.3f}"
        )
        exp1_summary[name] = {
            "weights": [float(x) for x in w],
            "alpha_dom": lin["alpha_dom"],
            "h2_angle_norm": lin["h2_angle_norm"],
            "dP_crit_areaA": dP_A,
            "dP_crit_areaB": dP_B,
            "dP_crit_worst": dP_worst,
        }

    # Experiment 2A: Equal per-bus primary-droop headroom ceiling h_bus across ALL 5 policies!
    print("\n--- Experiment 2A: Equal Per-Bus Primary-Droop Cap h_bus (ALL 5 Policies, T=80s) ---")
    h_bus_grid = [0.10, 0.16, 0.24, 0.35, 0.55, 0.90, 2.00]
    exp2_per_bus = {p: [] for p in all_policy_names}
    for hb in h_bus_grid:
        row_str = f"  h_bus={hb:.2f}: "
        for pname in all_policy_names:
            M, Dg = build_MD(policies[pname])
            h_vec = np.zeros(n12)
            for idx, b in enumerate(gen_buses):
                if policies[pname][idx] > 1e-6:
                    h_vec[b] = hb
            dP_A = bisect_dP_crit(M, D_base, Dg, h_vec, P12, K12_pre, K12_post, th_pre, fault_bus=0, t_end=80.0)
            dP_B = bisect_dP_crit(M, D_base, Dg, h_vec, P12, K12_pre, K12_post, th_pre, fault_bus=6, t_end=80.0)
            dP_w = min(dP_A, dP_B)
            exp2_per_bus[pname].append({"h_bus": hb, "dP_crit_A": dP_A, "dP_crit_B": dP_B, "dP_crit_worst": dP_w})
            row_str += f"{pname[:6]}={dP_w:.3f} | "
        print(row_str)

    # Experiment 2B: Fixed total droop headroom H_tot allocated proportionally across ALL 5 policies!
    print("\n--- Experiment 2B: Fixed Total Droop Headroom H_tot Proportional Allocation (ALL 5 Policies, T=80s) ---")
    H_tot_grid = [0.60, 0.90, 1.20, 1.80, 3.00]
    exp2_prop = {p: [] for p in all_policy_names}
    for Ht in H_tot_grid:
        row_str = f"  H_tot={Ht:.2f}: "
        for pname in all_policy_names:
            M, Dg = build_MD(policies[pname])
            h_vec = np.zeros(n12)
            for idx, b in enumerate(gen_buses):
                h_vec[b] = policies[pname][idx] * Ht
            dP_A = bisect_dP_crit(M, D_base, Dg, h_vec, P12, K12_pre, K12_post, th_pre, fault_bus=0, t_end=80.0)
            dP_B = bisect_dP_crit(M, D_base, Dg, h_vec, P12, K12_pre, K12_post, th_pre, fault_bus=6, t_end=80.0)
            dP_w = min(dP_A, dP_B)
            exp2_prop[pname].append({"H_tot": Ht, "dP_crit_A": dP_A, "dP_crit_B": dP_B, "dP_crit_worst": dP_w})
            row_str += f"{pname[:6]}={dP_w:.3f} | "
        print(row_str)

    # Horizon convergence check at T=80s vs T=120s on Area B only (h_bus=0.24, Bus 6) & Gateways (h_bus=0.24, Bus 6)
    M_b, Dg_b = build_MD(policies["area_B_only"])
    h_b = np.zeros(n12)
    for b in [6, 7, 8]:
        h_b[b] = 0.24
    dP_b_120 = bisect_dP_crit(M_b, D_base, Dg_b, h_b, P12, K12_pre, K12_post, th_pre, fault_bus=6, t_end=120.0)
    dP_b_80 = [x["dP_crit_B"] for x in exp2_per_bus["area_B_only"] if abs(x["h_bus"] - 0.24) < 1e-9][0]

    M_g, Dg_g = build_MD(policies["tie_gateways_2_6"])
    h_g = np.zeros(n12)
    for b in [2, 6]:
        h_g[b] = 0.24
    dP_g_120 = bisect_dP_crit(M_g, D_base, Dg_g, h_g, P12, K12_pre, K12_post, th_pre, fault_bus=6, t_end=120.0)
    dP_g_80 = [x["dP_crit_B"] for x in exp2_per_bus["tie_gateways_2_6"] if abs(x["h_bus"] - 0.24) < 1e-9][0]
    print(f"\n  Horizon checks (T=80s vs T=120s, h_bus=0.24, Bus 6):")
    print(f"    Area B only: T=80 -> {dP_b_80:.4f} vs T=120 -> {dP_b_120:.4f} (|diff|={abs(dP_b_80-dP_b_120):.2e})")
    print(f"    Gateways:    T=80 -> {dP_g_80:.4f} vs T=120 -> {dP_g_120:.4f} (|diff|={abs(dP_g_80-dP_g_120):.2e})")

    # Figure 1: Unsaturated Fixed-Budget Placement
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.2), dpi=150)
    p_labels = ["Area A\nOnly", "Area B\nOnly", "Tie-Line\nGateways\n{2, 6}", "Fiedler\nWeighted", "Uniform\n6 Buses"]
    x = np.arange(len(all_policy_names))
    w_bar = 0.26

    ax1.bar(x - w_bar, [exp1_summary[p]["dP_crit_areaA"] for p in all_policy_names], w_bar, color="#1f77b4", edgecolor="black", label="Fault in Area A (Bus 0)")
    ax1.bar(x,         [exp1_summary[p]["dP_crit_areaB"] for p in all_policy_names], w_bar, color="#ff7f0e", edgecolor="black", label="Fault in Area B (Bus 6)")
    ax1.bar(x + w_bar, [exp1_summary[p]["dP_crit_worst"] for p in all_policy_names], w_bar, color="#2ca02c", edgecolor="black", label="Worst-Case $\\min(\\Delta P_A, \\Delta P_B)$")
    ax1.set_xticks(x)
    ax1.set_xticklabels(p_labels, fontsize=8.8)
    ax1.set_ylabel("Max surviving generator-loss $\\Delta P_{\\mathrm{crit}}$ (pu)")
    ax1.set_title("Unsaturated Fixed-Budget GFM Placement ($T=80$ + Static Check)", fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.28)
    ax1.legend(loc="upper left", fontsize=8.0)

    ax2.bar(x - 0.18, [-exp1_summary[p]["alpha_dom"] for p in all_policy_names], 0.36, color="#9467bd", edgecolor="black", label="Linear inter-area damping $-\\alpha_{\\mathrm{dom}}$")
    ax2_r = ax2.twinx()
    ax2_r.plot(x, [exp1_summary[p]["h2_angle_norm"] for p in all_policy_names], "D-", color="#d62728", lw=2.3, ms=7, label="Exact linear $H_2$ angle norm (lower = better)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(p_labels, fontsize=8.8)
    ax2.set_ylabel("Slowest oscillatory damping $-\\alpha_{\\mathrm{dom}}$", color="#9467bd")
    ax2_r.set_ylabel("Exact linear $H_2$ angle norm", color="#d62728")
    ax2.set_title("Linear Damping & Exact $H_2$ Norm Across Policies", fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.28)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.0)

    fig.suptitle("Grid Stability Entry 001 (Part 1): Horizon-Robust Spatial Allocation of Limited GFM Inverters", fontsize=11.9, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp001_gfm_placement.png")
    plt.close(fig)

    # Figure 2: Primary-Droop Headroom Saturation Cliff (ALL 5 Policies!)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.3), dpi=150)
    style = {
        "area_B_only":      ("^-", "#9467bd", "Area B only (importing side)"),
        "fiedler_weighted": ("s--", "#ff7f0e", "Fiedler-weighted (all 6)"),
        "uniform_6_buses":  ("D-", "#2ca02c", "Uniform distributed (all 6)"),
        "tie_gateways_2_6": ("o-", "#d62728", "Tie-line gateways {2, 6}"),
        "area_A_only":      ("v:", "#7f7f7f", "Area A only (exporting side)"),
    }
    for pname in ["area_B_only", "fiedler_weighted", "uniform_6_buses", "tie_gateways_2_6", "area_A_only"]:
        fmt, col, lbl = style[pname]
        ax1.plot([r["h_bus"] for r in exp2_per_bus[pname]], [r["dP_crit_worst"] for r in exp2_per_bus[pname]],
                 fmt, color=col, lw=2.2, ms=6, label=lbl)
        ax2.plot([r["H_tot"] for r in exp2_prop[pname]], [r["dP_crit_worst"] for r in exp2_prop[pname]],
                 fmt, color=col, lw=2.2, ms=6, label=lbl)

    ax1.set_xlabel("Per-bus primary-droop ceiling $h_{\\mathrm{bus}}$ (pu)")
    ax1.set_ylabel("Worst-case surviving fault $\\Delta P_{\\mathrm{crit}}$ (pu)")
    ax1.set_title("Case A: Equal Per-Bus Droop Cap (Import-Side & Distributed Win)", fontweight="bold")
    ax1.grid(True, alpha=0.28)
    ax1.legend(loc="lower right", fontsize=7.8)

    ax2.set_xlabel("Fixed total droop headroom $H_{\\mathrm{tot}} = \\sum h_i$ (pu)")
    ax2.set_ylabel("Worst-case surviving fault $\\Delta P_{\\mathrm{crit}}$ (pu)")
    ax2.set_title("Case B: Fixed Total Headroom Allocated Proportionally", fontweight="bold")
    ax2.grid(True, alpha=0.28)
    ax2.legend(loc="lower right", fontsize=7.8)

    fig.suptitle("Grid Stability Entry 001 (Part 2): Horizon-Robust Primary-Droop Saturation Cliff ($T=80$)", fontsize=11.9, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp001_headroom_saturation.png")
    plt.close(fig)

    metrics = {
        "energy_balance_saturation_err": res_econ["energy_balance_rel_err"],
        "fiedler_weights": [float(x) for x in w_fiedler],
        "exp1_unsaturated_placement": exp1_summary,
        "exp2_per_bus_headroom_cliff": exp2_per_bus,
        "exp2_proportional_headroom_cliff": exp2_prop,
        "horizon_checks_T80_vs_T120_h024": {
            "area_B_only_bus6": {"T80": dP_b_80, "T120": dP_b_120, "abs_diff": float(abs(dP_b_80 - dP_b_120))},
            "tie_gateways_bus6": {"T80": dP_g_80, "T120": dP_g_120, "abs_diff": float(abs(dP_g_80 - dP_g_120))},
        },
    }
    with open(RESULTS_DIR / "exp001_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved hardened grid-stability Entry 001 plots and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
