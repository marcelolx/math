"""Entry 000 (plasma-nozzles): Quasi-1D Paraxial Magnetic Nozzle & Alfvén Detachment Baseline.

Models a collisionless/fluid plasma expanding from a sonic throat (z = 0, M_0 = 1) along an
axisymmetric diverging solenoid/dipole magnetic nozzle with flux-tube area A(z):
    B(z) * A(z) = B_0 * A_0  (magnetic flux conservation)
    A(z) / A_0  = (R(z) / R_0)^2 = (1 + (z / L_n)^2)^(3/2)   [current-loop / solenoid scaling]

Governing unsteady quasi-1D conservative fluid equations (dimensionless: rho_0=1, c_s0=1, R_0=1):
    d(A * rho)/dt         + d(A * rho * u)/dz                = 0
    d(A * rho * u)/dt     + d(A * (rho * u^2 + p))/dz        = p * dA/dz
    d(A * E_tot)/dt       + d(A * u * (E_tot + p))/dz        = 0
where E_tot = p / (gamma - 1) + 0.5 * rho * u^2, and p_0 = 1 / gamma so c_s0 = sqrt(gamma*p_0/rho_0) = 1.

Exact analytical invariants along the supersonic branch (M(z) >= 1):
1. Mass conservation:      rho(z) * u(z) * A(z) = m_dot = 1
2. Isentropic relation:    p(z) / p_0 = (rho(z) / rho_0)^gamma
3. Mach-Area relation:     A(z) = (1/M) * ((2 + (gamma-1)*M^2) / (gamma+1))^((gamma+1)/(2*(gamma-1)))
4. Exact thrust balance:   F_stream(z) = m_dot*u(z) + p(z)*A(z) = F_0 + int_0^z p(z')*(dA/dz') dz'
5. Ideal-MHD Alfven Mach:  M_A(z)^2 = M_A0^2 * u(z) * A(z),  where M_A0^2 = (gamma / 2) * beta_0
   => Exact Alfven detachment surface z_A where M_A(z_A) = 1 (Arefiev-Breizman 2005).

Paraxial 2D divergence loss trade-off (solenoid flux tube R(z) = (1 + (z/L_n)^2)^(3/4)):
- Wall flare angle tan(theta_w(z)) = dR/dz; paraxial conical plume efficiency eta_div(z) = 0.5*(1 + cos(theta_w(z))).
- Net axial thrust at Alfven detachment z_A:
      F_net(beta_0) = eta_div(z_A) * F_stream(z_A)
  Balances two competing losses:
  * High beta_0 (weak B_0): premature detachment (z_A -> 0) -> no magnetic expansion thrust!
  * Low  beta_0 (strong B_0): late detachment (z_A >> L_n) -> severe dipole flare divergence loss!
  Predicts a sharp optimal throat plasma beta_0*(L_n/R_0, gamma) maximizing specific impulse gain!

Experiments:
1. Verify unsteady conservative HLL finite-volume PDE solver against exact analytical Mach-area,
   density, velocity, pressure, Alfven Mach number, and exact momentum/thrust balance (grid N=100, 200, 400).
2. Map the Alfven detachment location z_A(beta_0) and discover the optimal throat plasma beta_0*
   across nozzle aspect ratios L_n/R_0 in {1.5, 2.5, 4.0} and polytropic indices gamma in {5/3, 1.3, 1.12}.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# 1. Magnetic nozzle geometry & exact analytical quasi-1D solution
# ---------------------------------------------------------------------------

def nozzle_geometry(z: np.ndarray | float, L_n: float = 2.5) -> tuple:
    """Axisymmetric solenoid/dipole-like magnetic flux tube (R_0 = 1, A_0 = 1).
    B(z) / B_0 = (1 + (z/L_n)^2)^(-3/2) => A(z) = (1 + (z/L_n)^2)^(3/2), R(z) = (1 + (z/L_n)^2)^(3/4).
    """
    s = 1.0 + (z / L_n) ** 2
    A = s ** 1.5
    dAdz = 3.0 * (z / (L_n * L_n)) * np.sqrt(s)
    R = s ** 0.75
    dRdz = 1.5 * (z / (L_n * L_n)) * (s ** (-0.25))
    cos_th = 1.0 / np.sqrt(1.0 + dRdz * dRdz)
    eta_div = 0.5 * (1.0 + cos_th)  # Exact paraxial conical plume efficiency
    return A, dAdz, R, dRdz, cos_th, eta_div


def area_from_mach(M: float, gamma: float) -> float:
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    return (1.0 / M) * (((2.0 + gm1 * M * M) / gp1) ** (gp1 / (2.0 * gm1)))


def exact_supersonic_profile(z: np.ndarray, gamma: float = 5.0 / 3.0, L_n: float = 2.5) -> dict:
    """Exact steady isentropic supersonic plasma expansion from sonic throat (M_0=1, rho_0=1, c_s0=1)."""
    A, dAdz, R, dRdz, cos_th, eta_div = nozzle_geometry(z, L_n)
    M = np.empty_like(z, dtype=float)
    for i, ai in enumerate(A):
        if ai <= 1.0 + 1e-12:
            M[i] = 1.0
        else:
            M[i] = brentq(lambda m: area_from_mach(m, gamma) - ai, 1.0, 1e4, xtol=1e-13)

    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    # T/T_0 = (gamma + 1) / (2 + (gamma - 1)*M^2)
    T_ratio = gp1 / (2.0 + gm1 * M * M)
    rho = T_ratio ** (1.0 / gm1)
    p = (1.0 / gamma) * (T_ratio ** (gamma / gm1))
    c_s = np.sqrt(T_ratio)
    u = M * c_s
    F_stream = rho * u * u * A + p * A
    return {
        "z": z, "A": A, "dAdz": dAdz, "R": R, "cos_th": cos_th, "eta_div": eta_div,
        "M": M, "rho": rho, "u": u, "p": p, "c_s": c_s, "F_stream": F_stream,
    }


# ---------------------------------------------------------------------------
# 2. Unsteady conservative 1D HLL finite-volume plasma nozzle solver (RK2)
# ---------------------------------------------------------------------------

def solve_unsteady_nozzle_fv(
    z_max: float = 6.0,
    n_cells: int = 200,
    gamma: float = 5.0 / 3.0,
    L_n: float = 2.5,
    cfl: float = 0.40,
    t_end: float = 12.0,
) -> dict:
    """Solve unsteady quasi-1D conservative Euler equations to steady state with HLL flux + RK2.

    State vector per cell j: U_cons = [A*rho, A*rho*u, A*E].
    Throat inflow (z=0): slightly supersonic sonic-throat boundary (u_in = 1.0002*c_s0) so all
    characteristics enter the domain cleanly without reflection; outflow (z=z_max): supersonic extrapolation.
    """
    dz = z_max / n_cells
    z_c = (np.arange(n_cells) + 0.5) * dz
    z_f = np.arange(n_cells + 1) * dz

    A_c, dAdz_c, _, _, _, _ = nozzle_geometry(z_c, L_n)
    A_f, _, _, _, _, _ = nozzle_geometry(z_f, L_n)

    # Exact inflow state at z = 0 (sonic throat M = 1)
    rho_in = 1.0
    u_in = 1.0002
    p_in = 1.0 / gamma
    E_in = p_in / (gamma - 1.0) + 0.5 * rho_in * u_in * u_in

    # Start from a uniform (non-exact!) initial guess so solver genuinely relaxes to steady state!
    rho = np.full(n_cells, rho_in)
    u = np.full(n_cells, u_in)
    p = np.full(n_cells, p_in)
    E = p / (gamma - 1.0) + 0.5 * rho * u * u

    Q = np.vstack([A_c * rho, A_c * rho * u, A_c * E])  # shape (3, n_cells)

    def prim_from_Q(Q_cur: np.ndarray):
        r = np.maximum(Q_cur[0] / A_c, 1e-12)
        vel = Q_cur[1] / np.maximum(Q_cur[0], 1e-12)
        Et = Q_cur[2] / A_c
        pr = np.maximum((gamma - 1.0) * (Et - 0.5 * r * vel * vel), 1e-12)
        return r, vel, pr, Et

    def hll_rhs(Q_cur: np.ndarray) -> tuple[np.ndarray, float]:
        r, vel, pr, Et = prim_from_Q(Q_cur)
        cs = np.sqrt(gamma * pr / r)
        max_lam = float(np.max(np.abs(vel) + cs))

        # Ghost cells (2nd-order MUSCL with minmod limiter in primitive variables!)
        r_g = np.empty(n_cells + 2)
        v_g = np.empty(n_cells + 2)
        p_g = np.empty(n_cells + 2)
        r_g[1:-1], v_g[1:-1], p_g[1:-1] = r, vel, pr
        # Left ghost = sonic throat inflow
        r_g[0], v_g[0], p_g[0] = rho_in, u_in, p_in
        # Right ghost = supersonic zero-gradient outflow
        r_g[-1], v_g[-1], p_g[-1] = r[-1], vel[-1], pr[-1]

        def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))

        def recon(q_g: np.ndarray):
            dq = np.diff(q_g)
            sl = np.zeros(n_cells + 2)
            sl[1:-1] = minmod(dq[:-1], dq[1:])
            qL = q_g[:-1] + 0.5 * sl[:-1]  # left state at faces 0..n_cells
            qR = q_g[1:] - 0.5 * sl[1:]    # right state at faces 0..n_cells
            return qL, qR

        rL, rR = recon(r_g)
        vL, vR = recon(v_g)
        pL, pR = recon(p_g)
        rL, rR = np.maximum(rL, 1e-12), np.maximum(rR, 1e-12)
        pL, pR = np.maximum(pL, 1e-12), np.maximum(pR, 1e-12)

        EL = pL / (gamma - 1.0) + 0.5 * rL * vL * vL
        ER = pR / (gamma - 1.0) + 0.5 * rR * vR * vR
        cL = np.sqrt(gamma * pL / rL)
        cR = np.sqrt(gamma * pR / rR)

        SL = np.minimum(vL - cL, vR - cR)
        SR = np.maximum(vL + cL, vR + cR)

        UL = np.vstack([rL, rL * vL, EL])
        UR = np.vstack([rR, rR * vR, ER])
        FL = np.vstack([rL * vL, rL * vL * vL + pL, vL * (EL + pL)])
        FR = np.vstack([rR * vR, rR * vR * vR + pR, vR * (ER + pR)])

        denom = np.maximum(SR - SL, 1e-12)
        F_hll = (SR * FL - SL * FR + SL * SR * (UR - UL)) / denom
        F_face = np.where(SL >= 0.0, FL, np.where(SR <= 0.0, FR, F_hll))
        # Multiply by face area A_f
        AF = F_face * A_f[None, :]

        dQ = -(AF[:, 1:] - AF[:, :-1]) / dz
        # Well-balanced nozzle wall force: p_j * (A_{j+1/2} - A_{j-1/2}) / dz!
        dQ[1] += pr * (A_f[1:] - A_f[:-1]) / dz
        return dQ, max_lam

    t = 0.0
    steps = 0
    while t < t_end:
        dQ1, max_lam = hll_rhs(Q)
        dt = min(cfl * dz / max_lam, t_end - t)
        Q1 = Q + dt * dQ1
        dQ2, _ = hll_rhs(Q1)
        Q = 0.5 * (Q + Q1 + dt * dQ2)
        t += dt
        steps += 1

    r, vel, pr, _ = prim_from_Q(Q)
    cs = np.sqrt(gamma * pr / r)
    M = vel / cs
    mdot = r * vel * A_c
    F_stream = r * vel * vel * A_c + pr * A_c
    # Integrated magnetic wall force int_0^z p dA
    dA_cell = A_f[1:] - A_f[:-1]
    F_wall_cum = np.cumsum(pr * dA_cell)
    F_0 = rho_in * u_in * u_in + p_in
    thrust_bal_err = float(np.max(np.abs((F_stream - F_0) - (F_wall_cum - 0.5 * pr * dA_cell))) / np.max(F_stream))

    return {
        "z": z_c, "A": A_c, "rho": r, "u": vel, "p": pr, "M": M,
        "mdot": mdot, "F_stream": F_stream,
        "mdot_rel_err": float(np.max(np.abs(mdot - rho_in * u_in)) / (rho_in * u_in)),
        "thrust_balance_rel_err": thrust_bal_err,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# 3. Alfven detachment & optimal throat plasma beta_0* analysis
# ---------------------------------------------------------------------------

def analyze_detachment_optimum(
    beta0_grid: np.ndarray,
    gamma: float = 5.0 / 3.0,
    L_n: float = 2.5,
    z_max: float = 120.0,
    n_z: int = 5000,
) -> dict:
    """Compute exact Alfven detachment surface z_A(beta_0), stream thrust, divergence loss, and net thrust."""
    z = np.linspace(0.0, z_max, n_z)
    ex = exact_supersonic_profile(z, gamma=gamma, L_n=L_n)
    F_0 = float(ex["F_stream"][0])  # = 1 + 1/gamma (sonic throat reference thrust)
    uA = ex["u"] * ex["A"]          # Monotonically increasing from 1.0 at z=0

    z_A_arr = np.empty_like(beta0_grid)
    M_A_detach_M = np.empty_like(beta0_grid)
    F_stream_ratio = np.empty_like(beta0_grid)
    eta_div_arr = np.empty_like(beta0_grid)
    F_net_ratio = np.empty_like(beta0_grid)

    for i, b0 in enumerate(beta0_grid):
        M_A0_sq = 0.5 * gamma * b0
        if M_A0_sq >= 1.0:
            # Already super-Alfvenic at throat => immediate detachment at z_A = 0
            z_A_arr[i] = 0.0
            M_A_detach_M[i] = 1.0
            F_stream_ratio[i] = 1.0
            eta_div_arr[i] = 1.0
            F_net_ratio[i] = 1.0
            continue
        target_uA = 1.0 / M_A0_sq
        if target_uA > uA[-1]:
            z_A_arr[i] = z_max
            idx = n_z - 1
            M_A_detach_M[i] = float(ex["M"][idx])
            F_stream_ratio[i] = float(ex["F_stream"][idx] / F_0)
            eta_div_arr[i] = float(ex["eta_div"][idx])
            F_net_ratio[i] = F_stream_ratio[i] * eta_div_arr[i]
        else:
            zA = float(np.interp(target_uA, uA, z))
            z_A_arr[i] = zA
            M_A_detach_M[i] = float(np.interp(zA, z, ex["M"]))
            F_stream_ratio[i] = float(np.interp(zA, z, ex["F_stream"]) / F_0)
            eta_div_arr[i] = float(np.interp(zA, z, ex["eta_div"]))
            F_net_ratio[i] = F_stream_ratio[i] * eta_div_arr[i]

    idx_opt = int(np.argmax(F_net_ratio))
    return {
        "gamma": gamma,
        "L_n": L_n,
        "F_throat": F_0,
        "beta0_opt": float(beta0_grid[idx_opt]),
        "M_A0_opt": float(np.sqrt(0.5 * gamma * beta0_grid[idx_opt])),
        "z_A_opt": float(z_A_arr[idx_opt]),
        "M_detach_opt": float(M_A_detach_M[idx_opt]),
        "F_stream_gain_opt": float(F_stream_ratio[idx_opt]),
        "eta_div_opt": float(eta_div_arr[idx_opt]),
        "F_net_gain_opt": float(F_net_ratio[idx_opt]),
        "arrays": {
            "z_A": z_A_arr, "M_detach": M_A_detach_M,
            "F_stream_ratio": F_stream_ratio, "eta_div": eta_div_arr, "F_net_ratio": F_net_ratio,
        },
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    gamma_base = 5.0 / 3.0  # Monatomic fusion plasma (adiabatic)
    L_n_base = 2.5

    print("=== Entry 000 (plasma-nozzles): Quasi-1D Paraxial Magnetic Nozzle & Alfven Detachment ===")
    print(f"Baseline monatomic plasma: gamma = {gamma_base:.4f}, nozzle length scale L_n/R_0 = {L_n_base}\n")

    # ------------------------------------------------------------------
    # Experiment 1: Unsteady FV Solver vs Exact Analytical Nozzle (Grid Convergence N=100, 200, 400)
    # ------------------------------------------------------------------
    print("--- Experiment 1: Unsteady 2nd-Order HLL FV Solver vs Exact Analytical Solution ---")
    conv_rows = []
    fv_200 = None
    ex_200 = None
    for N in (100, 200, 400):
        fv = solve_unsteady_nozzle_fv(z_max=6.0, n_cells=N, gamma=gamma_base, L_n=L_n_base)
        ex = exact_supersonic_profile(fv["z"], gamma=gamma_base, L_n=L_n_base)
        err_M = float(np.max(np.abs(fv["M"] - ex["M"])) / np.max(ex["M"]))
        err_u = float(np.max(np.abs(fv["u"] - ex["u"])) / np.max(ex["u"]))
        err_F = float(np.max(np.abs(fv["F_stream"] - ex["F_stream"])) / np.max(ex["F_stream"]))
        print(
            f"  N={N:3d} ({fv['steps']:4d} steps): rel err M={err_M:.2e}, u={err_u:.2e}, F={err_F:.2e} | "
            f"mdot err={fv['mdot_rel_err']:.2e}, thrust-bal err={fv['thrust_balance_rel_err']:.2e}"
        )
        conv_rows.append({
            "N": N, "rel_err_M": err_M, "rel_err_u": err_u, "rel_err_F": err_F,
            "mdot_rel_err": fv["mdot_rel_err"], "thrust_balance_rel_err": fv["thrust_balance_rel_err"],
        })
        if N == 200:
            fv_200, ex_200 = fv, ex

    # ------------------------------------------------------------------
    # Experiment 2: Alfven Detachment vs Dipole Divergence Trade-Off (Optimal beta_0*)
    # ------------------------------------------------------------------
    print("\n--- Experiment 2: Optimal Throat Plasma Beta_0* (Alfven Detachment vs Dipole Divergence) ---")
    beta0_grid = np.logspace(-3.5, 0.08, 320)
    opt_table = []
    curves_Ln = {}
    for Ln_val in (1.5, 2.5, 4.0):
        res = analyze_detachment_optimum(beta0_grid, gamma=gamma_base, L_n=Ln_val)
        curves_Ln[Ln_val] = res
        print(
            f"  gamma=5/3, L_n/R_0={Ln_val:.1f}: beta_0*={res['beta0_opt']:.4f} (M_A0*={res['M_A0_opt']:.3f}) -> "
            f"z_A*/R_0={res['z_A_opt']:.2f}, M(z_A)={res['M_detach_opt']:.2f}, "
            f"F_stream/F_0={res['F_stream_gain_opt']:.3f}, eta_div={res['eta_div_opt']:.3f} -> "
            f"NET THRUST GAIN={res['F_net_gain_opt']:.3f}x (+{100*(res['F_net_gain_opt']-1):.1f}%)"
        )
        opt_table.append({k: v for k, v in res.items() if k != "arrays"})

    curves_gamma = {}
    for g_val, glabel in [(5.0 / 3.0, "5/3 (adiabatic)"), (1.30, "1.30 (partial heat)"), (1.12, "1.12 (near-isothermal)")]:
        res = analyze_detachment_optimum(beta0_grid, gamma=g_val, L_n=L_n_base)
        curves_gamma[glabel] = res
        if abs(g_val - 5.0 / 3.0) > 1e-4:
            print(
                f"  gamma={g_val:.2f}, L_n/R_0={L_n_base:.1f}: beta_0*={res['beta0_opt']:.4f} (M_A0*={res['M_A0_opt']:.3f}) -> "
                f"z_A*/R_0={res['z_A_opt']:.2f}, M(z_A)={res['M_detach_opt']:.2f}, "
                f"eta_div={res['eta_div_opt']:.3f} -> NET THRUST GAIN={res['F_net_gain_opt']:.3f}x (+{100*(res['F_net_gain_opt']-1):.1f}%)"
            )
            opt_table.append({k: v for k, v in res.items() if k != "arrays"})

    # ------------------------------------------------------------------
    # Figure 1: FV PDE Verification vs Exact Analytical Nozzle + Thrust Balance
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2), dpi=150)

    ax1.plot(ex_200["z"], ex_200["M"], "-", color="#1f77b4", lw=2.2, label="Exact Mach $M(z)$")
    ax1.plot(fv_200["z"][::6], fv_200["M"][::6], "o", color="#1f77b4", ms=4.5, mfc="none", label="HLL FV PDE ($N=200$)")
    ax1.plot(ex_200["z"], ex_200["u"], "-", color="#d62728", lw=2.0, label="Exact velocity $u(z)/c_{s0}$")
    ax1.plot(fv_200["z"][::6], fv_200["u"][::6], "s", color="#d62728", ms=4.0, mfc="none")
    # Overlay Alfven Mach number at optimal beta_0* for L_n = 2.5
    b_opt_25 = curves_Ln[2.5]["beta0_opt"]
    M_A_opt = np.sqrt(0.5 * gamma_base * b_opt_25 * ex_200["u"] * ex_200["A"])
    ax1.plot(ex_200["z"], M_A_opt, "--", color="#2ca02c", lw=2.0, label=f"Alfvén Mach $M_A(z)$ at $\\beta_0^*={b_opt_25:.3f}$")
    ax1.axhline(1.0, color="#2ca02c", ls=":", lw=1.2)
    ax1.axvline(curves_Ln[2.5]["z_A_opt"], color="#2ca02c", ls=":", lw=1.3, label=f"Detachment $z_A^*={curves_Ln[2.5]['z_A_opt']:.2f} R_0$")
    ax1.set_xlabel("Axial distance $z / R_0$")
    ax1.set_ylabel("Dimensionless profile ($M, u/c_{s0}, M_A$)")
    ax1.set_title("Quasi-1D Magnetic Nozzle Verification ($\\gamma=5/3, L_n=2.5 R_0$)", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8.0)

    F0 = float(ex_200["F_stream"][0])
    ax2.plot(ex_200["z"], ex_200["F_stream"] / F0, "-", color="#1f77b4", lw=2.2, label="1D stream thrust $F_{\\mathrm{stream}}/F_0$")
    ax2.plot(fv_200["z"][::6], fv_200["F_stream"][::6] / F0, "o", color="#1f77b4", ms=4.5, mfc="none", label="HLL FV PDE ($N=200$)")
    ax2.plot(ex_200["z"], ex_200["eta_div"], "--", color="#ff7f0e", lw=2.0, label="Dipole plume efficiency $\\eta_{\\mathrm{div}}(z)$")
    ax2.plot(ex_200["z"], (ex_200["F_stream"] / F0) * ex_200["eta_div"], "-", color="#d62728", lw=2.3,
             label="Net axial thrust $\\eta_{\\mathrm{div}} F_{\\mathrm{stream}}/F_0$")
    ax2.axvline(curves_Ln[2.5]["z_A_opt"], color="#2ca02c", ls=":", lw=1.4, label=f"Optimal detachment $z_A^*={curves_Ln[2.5]['z_A_opt']:.2f} R_0$")
    ax2.set_xlabel("Axial distance $z / R_0$")
    ax2.set_ylabel("Normalized thrust / efficiency")
    ax2.set_title("Magnetic Expansion Gain vs. Dipole Divergence Loss", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", fontsize=8.0)

    fig.suptitle("Plasma Nozzles Entry 000 (Part 1): Conservative HLL PDE Verification & Thrust Peak", fontsize=12.2, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "baseline_nozzle_verification.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Optimal Throat Plasma Beta_0* Across Nozzle Geometry & Polytropic Index
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.3), dpi=150)

    for Ln_val, col in [(1.5, "#d62728"), (2.5, "#1f77b4"), (4.0, "#2ca02c")]:
        c = curves_Ln[Ln_val]
        ax1.semilogx(beta0_grid, c["arrays"]["F_net_ratio"], color=col, lw=2.2,
                     label=f"$L_n/R_0={Ln_val:.1f}$ ($\\beta_0^*={c['beta0_opt']:.3f}$, gain={c['F_net_gain_opt']:.3f}x)")
        ax1.scatter([c["beta0_opt"]], [c["F_net_gain_opt"]], color=col, edgecolors="black", s=55, zorder=5)
    ax1.axhline(1.0, color="black", ls=":", lw=1.1)
    ax1.set_xlabel("Throat plasma beta $\\beta_0 = 2\\mu_0 p_0 / B_0^2$")
    ax1.set_ylabel("Net axial thrust gain $F_{\\mathrm{net}}(z_A) / F_{\\mathrm{throat}}$")
    ax1.set_title("Optimal $\\beta_0^*$ vs. Nozzle Aspect Ratio ($\\gamma = 5/3$)", fontweight="bold")
    ax1.grid(True, which="both", alpha=0.28)
    ax1.legend(loc="lower left", fontsize=8.1)

    for (glabel, res), col in zip(curves_gamma.items(), ["#1f77b4", "#ff7f0e", "#9467bd"]):
        ax2.semilogx(beta0_grid, res["arrays"]["F_net_ratio"], color=col, lw=2.2,
                     label=f"$\\gamma$ = {glabel} ($\\beta_0^*={res['beta0_opt']:.3f}$, gain={res['F_net_gain_opt']:.3f}x)")
        ax2.scatter([res["beta0_opt"]], [res["F_net_gain_opt"]], color=col, edgecolors="black", s=55, zorder=5)
    ax2.axhline(1.0, color="black", ls=":", lw=1.1)
    ax2.set_xlabel("Throat plasma beta $\\beta_0 = 2\\mu_0 p_0 / B_0^2$")
    ax2.set_ylabel("Net axial thrust gain $F_{\\mathrm{net}}(z_A) / F_{\\mathrm{throat}}$")
    ax2.set_title("Effect of Electron Thermal Conduction (Lower $\\gamma$, $L_n=2.5 R_0$)", fontweight="bold")
    ax2.grid(True, which="both", alpha=0.28)
    ax2.legend(loc="lower left", fontsize=8.1)

    fig.suptitle("Plasma Nozzles Entry 000 (Part 2): Ideal-MHD Alfvén Detachment Optimum $\\beta_0^*$", fontsize=12.2, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "baseline_detachment_optimum.png")
    plt.close(fig)

    metrics = {
        "parameters": {"gamma_base": gamma_base, "L_n_base": L_n_base, "z_max_fv": 6.0},
        "exp1_fv_convergence": conv_rows,
        "exp2_detachment_optima": opt_table,
    }
    with open(RESULTS_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved plasma-nozzles Entry 000 plots and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
