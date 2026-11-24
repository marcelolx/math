"""Entry 001 (grid-stability): Optimal Spatial Placement of Grid-Forming Inverters & Current-Limit Saturation Cliff.

Investigates two critical design questions for 100% inverter-based renewable AC power grids
using our verified 12-bus two-area meshed AC swing model from Entry 000:

1. Fixed-Budget Spatial Placement (Linear H_2 / Eigenvalue Damping vs. True Nonlinear N-1 Fault Survival):
   Given a fixed total Grid-Forming (GFM) virtual inertia & primary droop budget
   (M_GFM_tot = 5.4, D_GFM_tot = 4.8 — equivalent to 3 full GFM units spread across 6 generator buses),
   compare 5 concrete placement strategies:
     (a) Area-A concentrated (exporting area only; Area B is 100% GFL)
     (b) Area-B concentrated (importing area only; Area A is 100% GFL)
     (c) Tie-line gateway anchors (100% of GFM budget on buses {2, 6} framing surviving tie line 2-6)
     (d) Fiedler-vector weighted (budget weighted by post-fault Fiedler eigenvector v_2,i^2)
     (e) Uniform distributed GFM (budget shared equally across all 6 generator buses)
   Measures BOTH linearized post-fault inter-area damping / H_2 norm AND exact worst-case
   nonlinear N-1 surviving generator-trip power min(Delta_P_crit_A, Delta_P_crit_B).

2. Realistic Power-Electronics Headroom / Droop Saturation Cliff:
   Solid-state inverters cannot overload like rotating copper turbines: their primary droop
   response saturates once active-power emergency injection hits headroom h_i (proportional
   to local GFM rating, i.e., sum_i h_i = H_tot fixed across all policies OR fixed per-bus
   inverter rating h_bus!).
   Specifically, droop response is bounded smoothly via:
       P_droop,i(omega_i) = -h_i * tanh(D_gfm,i * omega_i / h_i)
   so for small |omega_i|, P_droop,i ~ -D_gfm,i * omega_i (exact linear droop!), while for large
   excursions |P_droop,i| <= h_i (strict semiconductor headroom limit!).
   Note: energy dissipation D_diss(omega) = -omega_i * P_droop,i(omega_i) >= 0 identically,
   preserving exact Lyapunov monotonicity dH/dt = sum_i omega_i * P_droop,i - sum_i D_base,i * omega_i^2 <= 0!

   Tests BOTH:
     - Case I (Proportional headroom: h_i proportional to GFM allocation, sum h_i = H_tot)
     - Case II (Per-bus semiconductor ceiling: each generator bus has equal inverter MVA rating
       so concentrated GFM at 2 gateway buses has only 2*h_bus headroom vs 6*h_bus distributed!)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_continuous_lyapunov

from baseline_power_grid import (
    build_K_matrix,
    find_equilibrium,
    linear_swing_spectrum,
    make_12bus_grid,
    swing_energy,
)


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# 1. Nonlinear Swing Integrator with Smooth Semiconductor Droop Saturation
# ---------------------------------------------------------------------------

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
    dt: float = 0.0025,
    t_end: float = 28.0,
    check_energy: bool = False,
) -> dict:
    """RK4 integration of nonlinear AC swing network with saturated GFM droop:
        M_i ddot(th_i) + D_base,i dot(th_i) + h_gfm,i * tanh(D_gfm,i * dot(th_i) / h_gfm,i)
            = P_i - sum_j K_ij sin(th_i - th_j)
    When h_gfm,i = inf (or <= 0), uses unsaturated linear droop D_gfm,i * dot(th_i).
    """
    n = len(M)
    inv_M = 1.0 / M
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps
    t_arr = np.linspace(0.0, t_end, n_steps + 1)

    has_cap = np.isfinite(h_gfm) & (h_gfm > 0.0) & (D_gfm > 0.0)
    safe_h = np.where(has_cap, h_gfm, 1.0)

    def droop_torque(om: np.ndarray) -> np.ndarray:
        lin = D_gfm * om
        sat = safe_h * np.tanh(lin / safe_h)
        return np.where(has_cap, sat, lin)

    def rhs(th_v: np.ndarray, om_v: np.ndarray, K_m: np.ndarray, P_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        diff = th_v[:, None] - th_v[None, :]
        pe = np.sum(K_m * np.sin(diff), axis=1)
        damp = D_base * om_v + droop_torque(om_v)
        return om_v, inv_M * (P_v - pe - damp)

    th = theta0.copy()
    om = np.zeros(n)

    M_tot = float(np.sum(M))
    max_spread = 0.0
    max_f_coi = 0.0
    max_om_diff_late = 0.0
    late_start = int(0.85 * n_steps)

    H_post_list = []
    diss_post_list = []
    t_post_start_step = int(round(t_fault / dt))

    for step in range(n_steps):
        t_now = t_arr[step]
        if t_now >= t_fault:
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

        om_coi = float(np.dot(M, om)) / M_tot
        f_coi_abs = abs(om_coi) / (2.0 * np.pi)
        if f_coi_abs > max_f_coi:
            max_f_coi = f_coi_abs

        th_rel = th - np.mean(th)
        sp = float(np.max(th_rel) - np.min(th_rel))
        if sp > max_spread:
            max_spread = sp
        if step >= late_start:
            od = float(np.max(np.abs(om - om_coi)))
            if od > max_om_diff_late:
                max_om_diff_late = od

        # Early bailout on catastrophic pole-slip blackout
        if (not check_energy) and sp > 2.0 * np.pi:
            return {
                "synchronized": False,
                "max_angle_spread_rad": sp,
                "f_nadir_coi_hz": max_f_coi,
                "final_om_rel_max": float(np.max(np.abs(om - om_coi))),
                "energy_balance_rel_err": None,
            }

    e_err = None
    if check_energy and len(H_post_list) > 5:
        H_arr = np.array(H_post_list)
        d_arr = np.array(diss_post_list)
        cum_d = np.concatenate([[0.0], np.cumsum(0.5 * (d_arr[:-1] + d_arr[1:]) * dt)])
        dH = H_arr - H_arr[0]
        e_err = float(np.max(np.abs(dH + cum_d)) / max(float(np.max(np.abs(dH))), 1e-12))

    # Synchronized if no pole slip (< pi), inter-bus relative frequency decayed, AND COI frequency stays bounded (< 1.5 Hz)
    synced = (max_spread < np.pi) and (max_om_diff_late < 0.04) and (max_f_coi < 1.5)
    return {
        "synchronized": bool(synced),
        "max_angle_spread_rad": float(max_spread),
        "f_nadir_coi_hz": float(max_f_coi),
        "final_om_rel_max": float(max_om_diff_late),
        "energy_balance_rel_err": e_err,
    }


def compute_linear_metrics(theta_eq: np.ndarray, M: np.ndarray, D_tot_vec: np.ndarray, K: np.ndarray) -> dict:
    """Exact post-fault linearized inter-area damping alpha_dom and reduced H_2 synchronization norm."""
    n = len(M)
    ev = linear_swing_spectrum(theta_eq, M, D_tot_vec, K)
    osc = [z for z in ev if abs(z.imag) > 0.05]
    alpha_dom = float(max(z.real for z in osc)) if osc else 0.0

    # Exact H_2 norm of internal angle synchronization modes (projected orthogonal to uniform shift 1)
    # Build orthonormal basis Q (n x (n-1)) orthogonal to 1/sqrt(n)
    ones = np.ones((n, 1)) / np.sqrt(n)
    U_svd, _, _ = np.linalg.svd(ones, full_matrices=True)
    Q_orth = U_svd[:, 1:]  # shape (n, n-1)

    diff = theta_eq[:, None] - theta_eq[None, :]
    W = K * np.cos(diff)
    L = np.diag(np.sum(W, axis=1)) - W
    Minv = np.diag(1.0 / M)

    # Project state (delta_red = Q^T delta, omega = omega) -> Wait: project onto M-orthogonal complement
    # so uniform frequency drift does not blow up H_2! Use state x = [Q^T delta, Q^T omega] approximately?
    # Better: exact 2(n-1) reduced system in coordinates delta_hat = Q^T delta, v_hat = Q^T M^(1/2) omega?
    # Exact projection removing zero mode [1; 0] and COI mode [0; 1] using Moore-Penrose / Schur is clean;
    # Here compute exact H_2 norm of angle differences C*x = Q^T * delta for zero-sum power impulse B = [0; M^{-1} Q]:
    # Since B has zero mean in M-weighted sense? Wait: sum_i (M * M^{-1} Q)_i = 1^T Q = 0 -> EXACTLY ZERO COI EXCITATION!
    # Therefore for zero-sum impulse disturbances P_dist = Q u, COI frequency stays identically 0!
    # Check exact controllability in reduced coordinates using Sylvester/Lyapunov on projected subspace:
    M_sqrt = np.diag(np.sqrt(M))
    M_isqrt = np.diag(1.0 / np.sqrt(M))
    m_vec = np.sqrt(M) / np.sqrt(np.sum(M))
    U_m, _, _ = np.linalg.svd(m_vec[:, None], full_matrices=True)
    Vm = U_m[:, 1:]  # (n, n-1) orthonormal, Vm^T sqrt(M) 1 = 0!

    # Transformed coordinates y = Vm^T M^{1/2} delta, z = Vm^T M^{1/2} omega:
    # M^{1/2} ddot(delta) + M^{-1/2} D M^{-1/2} (M^{1/2} dot(delta)) + M^{-1/2} L M^{-1/2} (M^{1/2} delta) = M^{-1/2} p_dist
    L_hat = Vm.T @ (M_isqrt @ L @ M_isqrt) @ Vm
    D_hat = Vm.T @ (M_isqrt @ np.diag(D_tot_vec) @ M_isqrt) @ Vm
    A_red = np.block([
        [np.zeros((n - 1, n - 1)), np.eye(n - 1)],
        [-L_hat, -D_hat],
    ])
    # Zero-sum unit power impulse covariance across buses: B_red = [0; Vm^T M^{-1/2} Q_orth]
    B_red = np.vstack([np.zeros((n - 1, n - 1)), Vm.T @ M_isqrt @ Q_orth])
    # Output = inter-bus angle deviations: C_red = [Q_orth^T M^{-1/2} Vm, 0]
    C_red = np.hstack([Q_orth.T @ M_isqrt @ Vm, np.zeros((n - 1, n - 1))])
    P_lyap = solve_continuous_lyapunov(A_red, -(B_red @ B_red.T))
    h2_angle = float(np.sqrt(max(np.trace(C_red @ P_lyap @ C_red.T), 0.0)))
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
) -> float:
    """Bisect maximum surviving generator-loss contingency Delta_P at fault_bus + tie-line trip."""
    lo, hi = 0.0, dP_hi
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        P_post = P_pre.copy()
        P_post[fault_bus] -= mid
        res = simulate_swing_saturated(
            M, D_base, D_gfm, h_gfm, P_pre, K_pre, P_post, K_post, theta0,
            t_fault=1.0, dt=0.004, t_end=26.0,
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
    print("=== Entry 001 (grid-stability): GFM Spatial Placement & Semiconductor Saturation Cliff ===\n")

    n12, edges12, P12, gen_buses = make_12bus_grid()
    K12_pre = build_K_matrix(n12, edges12)
    edges12_post = [e for e in edges12 if not (e[0] == 5 and e[1] == 9)]
    K12_post = build_K_matrix(n12, edges12_post)

    ok_pre, th_pre, _ = find_equilibrium(P12, K12_pre)
    ok_post0, th_post0, _ = find_equilibrium(P12, K12_post)
    assert ok_pre and ok_post0

    # Compute post-fault Fiedler eigenvector v_2 on K12_post for Fiedler-weighted placement!
    diff_p = th_post0[:, None] - th_post0[None, :]
    W_p = K12_post * np.cos(diff_p)
    L_p = np.diag(np.sum(W_p, axis=1)) - W_p
    evals_L, evecs_L = np.linalg.eigh(L_p)
    v2 = evecs_L[:, 1]  # Fiedler eigenvector (inter-area mode!)
    w_fiedler_raw = np.array([v2[b] ** 2 for b in gen_buses])
    w_fiedler = w_fiedler_raw / np.sum(w_fiedler_raw)

    # Fixed total GFM hardware budget across the 6 generator buses [0, 1, 2, 6, 7, 8]:
    M_GFM_TOT = 5.4
    D_GFM_TOT = 4.8

    # Base grid background (100% inverter grid: tiny GFL residual on gen buses, small load damping)
    M_base = np.full(n12, 0.12)
    D_base = np.full(n12, 0.12)
    for b in gen_buses:
        M_base[b] = 0.08
        D_base[b] = 0.04

    # 5 spatial allocation weight vectors over gen_buses = [0, 1, 2, 6, 7, 8] (each sums to 1.0!)
    policies = {
        "area_A_only":       np.array([1 / 3, 1 / 3, 1 / 3, 0.0,   0.0,   0.0]),
        "area_B_only":       np.array([0.0,   0.0,   0.0,   1 / 3, 1 / 3, 1 / 3]),
        "tie_gateways_2_6":  np.array([0.0,   0.0,   0.5,   0.5,   0.0,   0.0]),
        "fiedler_weighted":  w_fiedler,
        "uniform_6_buses":   np.full(6, 1.0 / 6.0),
    }

    def build_MD(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        M = M_base.copy()
        D_gfm = np.zeros(n12)
        for idx, b in enumerate(gen_buses):
            M[b] += weights[idx] * M_GFM_TOT
            D_gfm[b] = weights[idx] * D_GFM_TOT
        return M, D_gfm

    # ------------------------------------------------------------------
    # Verification Check: Exact Nonlinear Lyapunov Energy Balance with Saturation
    # ------------------------------------------------------------------
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
    print(f"Verification: Exact nonlinear Lyapunov balance with tanh droop saturation: rel err = {res_econ['energy_balance_rel_err']:.2e}\n")

    # ------------------------------------------------------------------
    # Experiment 1: Unsaturated Fixed-Budget Spatial Placement Comparison
    # ------------------------------------------------------------------
    print("--- Experiment 1: Unsaturated Fixed-Budget GFM Placement (M_tot=5.4, D_tot=4.8) ---")
    h_inf = np.full(n12, np.inf)
    exp1_summary = {}
    for name, w in policies.items():
        M, Dg = build_MD(w)
        lin = compute_linear_metrics(th_post0, M, D_base + Dg, K12_post)
        dP_A = bisect_dP_crit(M, D_base, Dg, h_inf, P12, K12_pre, K12_post, th_pre, fault_bus=0)
        dP_B = bisect_dP_crit(M, D_base, Dg, h_inf, P12, K12_pre, K12_post, th_pre, fault_bus=6)
        dP_worst = min(dP_A, dP_B)
        print(
            f"  [{name:18s}] alpha_dom={lin['alpha_dom']:+.4f} | H2_ang={lin['h2_angle_norm']:.4f} | "
            f"dP_crit(Bus0/A)={dP_A:.3f} | dP_crit(Bus6/B)={dP_B:.3f} | WORST-CASE={dP_worst:.3f}"
        )
        exp1_summary[name] = {
            "weights": [float(x) for x in w],
            "alpha_dom": lin["alpha_dom"],
            "h2_angle_norm": lin["h2_angle_norm"],
            "dP_crit_areaA": dP_A,
            "dP_crit_areaB": dP_B,
            "dP_crit_worst": dP_worst,
        }

    # ------------------------------------------------------------------
    # Experiment 2: Semiconductor Inverter Current/Headroom Saturation Cliff
    # ------------------------------------------------------------------
    print("\n--- Experiment 2: Inverter Headroom Saturation Cliff (Per-Bus Cap h_bus vs Proportional Cap) ---")
    # Case II (Per-bus semiconductor ceiling): every generator bus has the SAME physical inverter MVA ceiling h_bus!
    # Concentrated gateways (only 2 active GFM buses) therefore have total droop headroom 2*h_bus,
    # whereas uniform distributed GFM (all 6 buses participating) shares 6*h_bus headroom!
    h_bus_grid = [0.10, 0.16, 0.24, 0.35, 0.55, 0.90, 2.00]
    sat_policies = ["tie_gateways_2_6", "fiedler_weighted", "uniform_6_buses"]
    exp2_per_bus = {p: [] for p in sat_policies}

    print("  [2A] Equal per-bus inverter headroom ceiling h_bus (tests active-bus headroom sharing):")
    for hb in h_bus_grid:
        row_str = f"    h_bus={hb:.2f}: "
        for pname in sat_policies:
            M, Dg = build_MD(policies[pname])
            h_vec = np.zeros(n12)
            for idx, b in enumerate(gen_buses):
                if policies[pname][idx] > 1e-6:
                    h_vec[b] = hb
            dP_A = bisect_dP_crit(M, D_base, Dg, h_vec, P12, K12_pre, K12_post, th_pre, fault_bus=0)
            dP_B = bisect_dP_crit(M, D_base, Dg, h_vec, P12, K12_pre, K12_post, th_pre, fault_bus=6)
            dP_w = min(dP_A, dP_B)
            exp2_per_bus[pname].append({"h_bus": hb, "dP_crit_A": dP_A, "dP_crit_B": dP_B, "dP_crit_worst": dP_w})
            row_str += f"{pname[:7]}={dP_w:.3f} | "
        print(row_str)

    # Case I (Proportional headroom: sum_i h_i = H_tot fixed identically across all policies!)
    H_tot_grid = [0.60, 0.90, 1.20, 1.80, 3.00]
    exp2_prop = {p: [] for p in sat_policies}
    print("\n  [2B] Fixed total system headroom H_tot allocated proportionally to GFM weights (sum h_i = H_tot):")
    for Ht in H_tot_grid:
        row_str = f"    H_tot={Ht:.2f}: "
        for pname in sat_policies:
            M, Dg = build_MD(policies[pname])
            h_vec = np.zeros(n12)
            for idx, b in enumerate(gen_buses):
                h_vec[b] = policies[pname][idx] * Ht
            dP_A = bisect_dP_crit(M, D_base, Dg, h_vec, P12, K12_pre, K12_post, th_pre, fault_bus=0)
            dP_B = bisect_dP_crit(M, D_base, Dg, h_vec, P12, K12_pre, K12_post, th_pre, fault_bus=6)
            dP_w = min(dP_A, dP_B)
            exp2_prop[pname].append({"H_tot": Ht, "dP_crit_A": dP_A, "dP_crit_B": dP_B, "dP_crit_worst": dP_w})
            row_str += f"{pname[:7]}={dP_w:.3f} | "
        print(row_str)

    # Horizon convergence check at h_bus = 0.24 (uniform vs tie_gateways, t_end=26 vs t_end=52)
    M_u, Dg_u = build_MD(policies["uniform_6_buses"])
    h_u = np.zeros(n12)
    for b in gen_buses:
        h_u[b] = 0.24
    lo_h, hi_h = 0.0, 2.4
    for _ in range(11):
        mid = 0.5 * (lo_h + hi_h)
        Pp = P12.copy()
        Pp[6] -= mid
        r = simulate_swing_saturated(M_u, D_base, Dg_u, h_u, P12, K12_pre, Pp, K12_post, th_pre, t_fault=1.0, dt=0.004, t_end=52.0)
        if r["synchronized"]:
            lo_h = mid
        else:
            hi_h = mid
    dP_u_t52 = 0.5 * (lo_h + hi_h)
    dP_u_t26 = [x["dP_crit_B"] for x in exp2_per_bus["uniform_6_buses"] if abs(x["h_bus"] - 0.24) < 1e-9][0]
    print(f"\n  Horizon convergence check (uniform, h_bus=0.24, Bus 6): t=26 -> {dP_u_t26:.4f} vs t=52 -> {dP_u_t52:.4f} (|diff|={abs(dP_u_t26-dP_u_t52):.2e})")

    # ------------------------------------------------------------------
    # Figure 1: Fixed-Budget Placement — Linear Damping vs Nonlinear Fault Survival
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.2), dpi=150)

    p_order = ["area_A_only", "area_B_only", "tie_gateways_2_6", "fiedler_weighted", "uniform_6_buses"]
    p_labels = ["Area A\nOnly", "Area B\nOnly", "Tie-Line\nGateways\n{2, 6}", "Fiedler\nWeighted", "Uniform\n6 Buses"]
    x = np.arange(len(p_order))
    w_bar = 0.26

    ax1.bar(x - w_bar, [exp1_summary[p]["dP_crit_areaA"] for p in p_order], w_bar, color="#1f77b4", edgecolor="black", label="Fault in Area A (Bus 0)")
    ax1.bar(x,         [exp1_summary[p]["dP_crit_areaB"] for p in p_order], w_bar, color="#ff7f0e", edgecolor="black", label="Fault in Area B (Bus 6)")
    ax1.bar(x + w_bar, [exp1_summary[p]["dP_crit_worst"] for p in p_order], w_bar, color="#2ca02c", edgecolor="black", label="Worst-Case $\\min(\\Delta P_A, \\Delta P_B)$")
    ax1.set_xticks(x)
    ax1.set_xticklabels(p_labels, fontsize=8.8)
    ax1.set_ylabel("Max surviving generator-loss contingency $\\Delta P_{\\mathrm{crit}}$ (pu)")
    ax1.set_title("Unsaturated Fixed-Budget GFM Placement ($M_{\\mathrm{tot}}=5.4, D_{\\mathrm{tot}}=4.8$)", fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.28)
    ax1.legend(loc="upper left", fontsize=8.0)

    # Panel 2: Linear H2 angle norm vs nonlinear worst-case survival
    ax2.bar(x - 0.18, [-exp1_summary[p]["alpha_dom"] for p in p_order], 0.36, color="#9467bd", edgecolor="black", label="Linear inter-area damping $-\\alpha_{\\mathrm{dom}}$ (s$^{-1}$)")
    ax2_r = ax2.twinx()
    ax2_r.plot(x, [exp1_summary[p]["h2_angle_norm"] for p in p_order], "D-", color="#d62728", lw=2.3, ms=7, label="Linear $H_2$ angle norm (lower = better)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(p_labels, fontsize=8.8)
    ax2.set_ylabel("Slowest oscillatory damping $-\\alpha_{\\mathrm{dom}}$", color="#9467bd")
    ax2_r.set_ylabel("Linear $H_2$ angle synchronization norm", color="#d62728")
    ax2.set_title("Linear Damping & $H_2$ Norm Across Placement Policies", fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.28)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.0)

    fig.suptitle("Grid Stability Entry 001 (Part 1): Optimal Spatial Allocation of Limited GFM Inverters", fontsize=12.0, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp001_gfm_placement.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Semiconductor Headroom Saturation Cliff (Per-Bus vs Proportional Cap)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2), dpi=150)

    style = {
        "tie_gateways_2_6": ("o-", "#d62728", "Tie-line gateways {2, 6}"),
        "fiedler_weighted": ("s--", "#ff7f0e", "Fiedler-weighted (all 6)"),
        "uniform_6_buses":  ("D-", "#2ca02c", "Uniform distributed (all 6)"),
    }
    for pname in sat_policies:
        fmt, col, lbl = style[pname]
        ax1.plot([r["h_bus"] for r in exp2_per_bus[pname]], [r["dP_crit_worst"] for r in exp2_per_bus[pname]],
                 fmt, color=col, lw=2.3, ms=6, label=lbl)
        ax2.plot([r["H_tot"] for r in exp2_prop[pname]], [r["dP_crit_worst"] for r in exp2_prop[pname]],
                 fmt, color=col, lw=2.3, ms=6, label=lbl)

    ax1.set_xlabel("Per-bus inverter headroom ceiling $h_{\\mathrm{bus}}$ (pu)")
    ax1.set_ylabel("Worst-case surviving fault $\\Delta P_{\\mathrm{crit}}$ (pu)")
    ax1.set_title("Case A: Equal Per-Bus Inverter Rating (Concentrated Clips Early!)", fontweight="bold")
    ax1.grid(True, alpha=0.28)
    ax1.legend(loc="lower right", fontsize=8.2)

    ax2.set_xlabel("Fixed total system droop headroom $H_{\\mathrm{tot}} = \\sum h_i$ (pu)")
    ax2.set_ylabel("Worst-case surviving fault $\\Delta P_{\\mathrm{crit}}$ (pu)")
    ax2.set_title("Case B: Fixed Total Headroom Allocated Proportionally", fontweight="bold")
    ax2.grid(True, alpha=0.28)
    ax2.legend(loc="lower right", fontsize=8.2)

    fig.suptitle("Grid Stability Entry 001 (Part 2): Inverter Semiconductor Headroom Saturation Cliff", fontsize=12.0, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp001_headroom_saturation.png")
    plt.close(fig)

    metrics = {
        "lyapunov_saturation_energy_err": res_econ["energy_balance_rel_err"],
        "fiedler_weights": [float(x) for x in w_fiedler],
        "exp1_unsaturated_placement": exp1_summary,
        "exp2_per_bus_headroom_cliff": exp2_per_bus,
        "exp2_proportional_headroom_cliff": exp2_prop,
        "horizon_check_uniform_h024": {"t26": dP_u_t26, "t52": dP_u_t52, "abs_diff": float(abs(dP_u_t26 - dP_u_t52))},
    }
    with open(RESULTS_DIR / "exp001_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved grid-stability Entry 001 plots and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
