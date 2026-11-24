"""Entry 002 (plasma-nozzles): Self-Consistent Diamagnetic Magnetic Nozzle Inflation & Detachment.

Upgrades Entry 000's rigid vacuum-dipole nozzle to a self-consistent slender-flux-tube
ideal-MHD nozzle where plasma pressure p(z) radially balances the external vacuum magnetic
field B_vac(z), expelling internal flux and inflating the plume cross-section A(z):

    p(z) + B(z)^2 / 2 = B_vac(z)^2 / 2,        B(z) * A(z) = Phi_B = const

Exact unsteady 1D conservative compliant-flux-tube MHD equations (zero energy source term!):
    d(rho*A)/dt   + d(rho*u*A)/dz               = 0
    d(rho*u*A)/dt + d((rho*u^2 + p)*A)/dz       = p * dA/dz
    d(E_tot)/dt   + d((0.5*rho*u^2 + gamma/(gamma-1)*p)*u*A)/dz = 0

where the exact conserved total energy per unit length (plasma + internal magnetic + external
field displacement energy) is:
    E_tot = (0.5*rho*u^2 + p/(gamma-1) + B^2/2 + B_vac(z)^2/2) * A

Exact primitive recovery from (Q0, Q1, Q2) = (rho*A, rho*u*A, E_tot) is a single closed-form
quadratic equation in A (zero Newton iterations!).
Exact magnetoacoustic characteristic speed is the Roberts-Webb tube speed:
    c_T = c_s * v_A / sqrt(c_s^2 + v_A^2)

Exact steady self-consistent acceleration law:
    (u^2 / c_T^2 - 1) * (du / u) = -(B_vac^2 / B^2) * (dB_vac / B_vac)
Choking occurs at the magnetic throat (dB_vac/dz = 0) at u_0 = c_T0 < min(c_s0, v_A0)!

Experiments:
1. Exact steady self-consistent ODE/algebraic solution vs. unsteady 2nd-order MUSCL-minmod
   HLL conservative FV compliant-flux-tube MHD PDE (N = 100, 200, 400 grid convergence +
   exact mass, enthalpy, magnetic flux, and magnetic thrust integral verification).
2. Diamagnetic inflation law A(z) ~ B_vac(z)^(-2/gamma) vs. rigid vacuum A ~ B_vac^(-1)
   and upstream shift of the Alfven detachment surface (M_A = 1) across beta_vac,0.
3. Self-consistent net thrust efficiency (accounting for BOTH self-consistent magnetic thrust
   integral p dA AND actual inflated streamline divergence dr_self/dz) compared head-to-head
   against Entry 000's rigid-field baseline for adiabatic (gamma=5/3) and near-isothermal
   (gamma=1.12) plasmas and throat slenderness ratios R_0/L_B.
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
# 1. Vacuum magnetic field profile & exact steady self-consistent solution
# ---------------------------------------------------------------------------

def vacuum_dipole_B(z: np.ndarray | float, L_B: float = 1.0) -> np.ndarray | float:
    """Paraxial solenoid / dipole magnetic field B_vac(z) with B_vac(0) = 1, B_vac'(0) = 0."""
    return (1.0 + (z / L_B) ** 2) ** (-1.5)


def throat_state_self_consistent(beta_vac0: float, gamma: float) -> dict:
    """Exact choked throat state (z = 0, A_0 = 1, rho_0 = 1, B_vac0 = 1, c_s0 = 1).

    In Alfvén units (mu_0 = 1, B_vac0 = 1):
        p_vac0 = 0.5
        p_0    = 0.5 * beta_vac0
        B_0    = sqrt(1 - beta_vac0)      (exact radial pressure balance p_0 + B_0^2/2 = 1/2)
        rho_0  = gamma * p_0              (sets c_s0 = sqrt(gamma*p_0/rho_0) = 1.0!)
        v_A0   = B_0 / sqrt(rho_0) = sqrt(2*(1 - beta_vac0)/(gamma*beta_vac0))
        c_T0   = c_s0 * v_A0 / sqrt(c_s0^2 + v_A0^2)   (Roberts-Webb choked tube speed!)
        u_0    = c_T0
    """
    if not (0.0 < beta_vac0 < 1.0):
        raise ValueError("beta_vac0 must lie strictly in (0, 1) for magnetic confinement.")
    p_0 = 0.5 * beta_vac0
    B_0 = float(np.sqrt(1.0 - beta_vac0))
    rho_0 = gamma * p_0  # c_s0 = 1.0 exactly
    c_s0 = 1.0
    v_A0 = B_0 / np.sqrt(rho_0)
    c_T0 = (c_s0 * v_A0) / np.sqrt(c_s0 * c_s0 + v_A0 * v_A0)
    u_0 = c_T0
    A_0 = 1.0
    Phi_B = B_0 * A_0
    mdot = rho_0 * u_0 * A_0
    H_0 = 0.5 * u_0 * u_0 + (gamma / (gamma - 1.0)) * (p_0 / rho_0)
    K_s = p_0 / (rho_0 ** gamma)  # isentropic constant p = K_s * rho^gamma
    beta_in0 = beta_vac0 / (1.0 - beta_vac0)
    return {
        "beta_vac0": beta_vac0, "beta_in0": beta_in0, "gamma": gamma,
        "p_0": p_0, "B_0": B_0, "rho_0": rho_0, "c_s0": c_s0,
        "v_A0": v_A0, "c_T0": c_T0, "u_0": u_0, "A_0": A_0,
        "Phi_B": Phi_B, "mdot": mdot, "H_0": H_0, "K_s": K_s,
    }


def solve_exact_steady_self_consistent(
    z: np.ndarray,
    beta_vac0: float,
    gamma: float,
    L_B: float = 1.0,
) -> dict:
    """Exact steady super-tube (u >= c_T) self-consistent diamagnetic nozzle solution.

    For any downstream station z >= 0 with B_vac(z) <= 1, parameterize by density rho in (0, rho_0]:
        p(rho) = K_s * rho^gamma
        u(rho) = sqrt(2 * (H_0 - gamma/(gamma-1) * K_s * rho^(gamma-1)))
        A(rho) = mdot / (rho * u(rho))
        B_in(rho) = Phi_B / A(rho)
        p_tot(rho) = p(rho) + 0.5 * B_in(rho)^2
    Along the super-tube branch (rho < rho_0), p_tot(rho) is strictly increasing with rho!
    Proof: d(p_tot)/drho = c_s^2 + B^2/rho * (1 - u^2/c_s^2) ... wait:
    Actually d(p_tot)/du = -rho*u + B^2 * (dA/du)/A ... let's check sign directly:
    At throat u = c_T: d(p_tot)/drho = 0 (maximum of p_tot = 0.5 = B_vac(0)^2/2!).
    For rho < rho_0 (u > c_T): p_tot(rho) decreases monotonically from 0.5 down to 0 as rho -> 0!
    Hence brentq(p_tot(rho) - 0.5*B_vac(z)^2, 1e-12, rho_0) yields the EXACT unique solution!
    """
    st = throat_state_self_consistent(beta_vac0, gamma)
    rho_0 = st["rho_0"]
    Phi_B = st["Phi_B"]
    mdot = st["mdot"]
    H_0 = st["H_0"]
    K_s = st["K_s"]

    B_vac = np.asarray(vacuum_dipole_B(z, L_B), dtype=float)
    n = len(z)
    rho = np.empty(n)
    u = np.empty(n)
    p = np.empty(n)
    A = np.empty(n)
    B = np.empty(n)
    c_s = np.empty(n)
    v_A = np.empty(n)
    c_T = np.empty(n)

    def eval_from_rho(r: float) -> tuple[float, float, float, float, float]:
        pr = K_s * (r ** gamma)
        h_rem = max(2.0 * (H_0 - (gamma / (gamma - 1.0)) * (pr / r)), 1e-30)
        ur = float(np.sqrt(h_rem))
        Ar = mdot / (r * ur)
        Br = Phi_B / Ar
        ptot = pr + 0.5 * Br * Br
        return pr, ur, Ar, Br, ptot

    for i in range(n):
        target = 0.5 * B_vac[i] * B_vac[i]
        if target >= 0.5 - 1e-14:
            r_sol = rho_0
        else:
            # Find safe lower bracket where ptot < target
            lo = 1e-10 * rho_0
            while eval_from_rho(lo)[4] > target:
                lo *= 0.1
            r_sol = brentq(lambda r: eval_from_rho(r)[4] - target, lo, rho_0 * (1.0 - 1e-12), xtol=1e-13, rtol=1e-13)
        pr, ur, Ar, Br, _ = eval_from_rho(r_sol)
        rho[i] = r_sol
        u[i] = ur
        p[i] = pr
        A[i] = Ar
        B[i] = Br
        c_s[i] = np.sqrt(gamma * pr / r_sol)
        v_A[i] = Br / np.sqrt(r_sol)
        c_T[i] = (c_s[i] * v_A[i]) / np.sqrt(c_s[i] ** 2 + v_A[i] ** 2)

    M = u / c_s
    M_A = u / v_A
    M_T = u / c_T
    beta_in = 2.0 * p / np.maximum(B * B, 1e-30)
    F_mom = (rho * u * u + p) * A
    return {
        "z": z, "B_vac": B_vac, "rho": rho, "u": u, "p": p, "A": A, "B": B,
        "c_s": c_s, "v_A": v_A, "c_T": c_T, "M": M, "M_A": M_A, "M_T": M_T,
        "beta_in": beta_in, "F_mom": F_mom, "throat": st,
    }


def solve_exact_steady_rigid(
    z: np.ndarray,
    beta_vac0: float,
    gamma: float,
    L_B: float = 1.0,
) -> dict:
    """Rigid vacuum-field baseline (Entry 000) with identical throat gas state (p_0, rho_0, c_s0=1)
    and rigid area A_vac(z) = 1 / B_vac(z), choked at M = 1 (u_0 = c_s0 = 1).
    """
    p_0 = 0.5 * beta_vac0
    rho_0 = gamma * p_0
    B_vac = np.asarray(vacuum_dipole_B(z, L_B), dtype=float)
    A = 1.0 / B_vac
    B = B_vac.copy()

    # Exact supersonic Mach-area relation
    gp1, gm1 = gamma + 1.0, gamma - 1.0
    exp_a = gp1 / (2.0 * gm1)
    M = np.empty_like(A)
    for i, a_val in enumerate(A):
        if a_val <= 1.0 + 1e-13:
            M[i] = 1.0
        else:
            f = lambda m: (1.0 / m) * ((2.0 / gp1) * (1.0 + 0.5 * gm1 * m * m)) ** exp_a - a_val
            hi = 2.0
            while f(hi) < 0.0:
                hi *= 2.0
            M[i] = brentq(f, 1.0 + 1e-12, hi, xtol=1e-13)

    tr = (1.0 + 0.5 * gm1) / (1.0 + 0.5 * gm1 * M * M)
    rho = rho_0 * (tr ** (1.0 / gm1))
    p = p_0 * (tr ** (gamma / gm1))
    c_s = np.sqrt(gamma * p / rho)
    u = M * c_s
    v_A = B / np.sqrt(rho)
    M_A = u / v_A
    F_mom = (rho * u * u + p) * A
    return {
        "z": z, "B_vac": B_vac, "rho": rho, "u": u, "p": p, "A": A, "B": B,
        "c_s": c_s, "v_A": v_A, "M": M, "M_A": M_A, "F_mom": F_mom,
    }


# ---------------------------------------------------------------------------
# 2. Unsteady 2nd-Order MUSCL-Minmod HLL Conservative FV Compliant-Tube MHD PDE
# ---------------------------------------------------------------------------

def prim_from_cons_compliant(
    Q: np.ndarray,
    B_vac: np.ndarray,
    Phi_B: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact closed-form quadratic primitive recovery from Q = [rho*A, rho*u*A, E_tot]."""
    Q0 = np.maximum(Q[0], 1e-14)
    u = Q[1] / Q0
    R = Q[2] - 0.5 * Q[1] * u

    gm1 = gamma - 1.0
    c2 = (gamma / gm1) * 0.5 * B_vac * B_vac
    c0 = ((gamma - 2.0) / gm1) * 0.5 * Phi_B * Phi_B  # negative for gamma < 2!
    disc = np.maximum(R * R - 4.0 * c2 * c0, 0.0)
    A = (R + np.sqrt(disc)) / (2.0 * c2)
    A_min = Phi_B / np.maximum(B_vac, 1e-12) * (1.0 + 1e-12)
    A = np.maximum(A, A_min)

    rho = Q0 / A
    B = Phi_B / A
    p = np.maximum(0.5 * (B_vac * B_vac - B * B), 1e-14)
    c_s = np.sqrt(gamma * p / rho)
    v_A = B / np.sqrt(rho)
    c_T = (c_s * v_A) / np.sqrt(c_s * c_s + v_A * v_A)
    return rho, u, p, A, B, c_T


def cons_from_prim_compliant(
    rho: np.ndarray,
    u: np.ndarray,
    p: np.ndarray,
    A: np.ndarray,
    B_vac: np.ndarray,
    Phi_B: float,
    gamma: float,
) -> np.ndarray:
    B = Phi_B / A
    E_tot = (0.5 * rho * u * u + p / (gamma - 1.0) + 0.5 * B * B + 0.5 * B_vac * B_vac) * A
    return np.vstack([rho * A, rho * u * A, E_tot])


def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a * b > 0.0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def solve_unsteady_compliant_mhd_pde(
    beta_vac0: float,
    gamma: float,
    L_B: float = 1.0,
    z_max: float = 4.0,
    n_cells: int = 200,
    cfl: float = 0.40,
    t_end: float = 14.0,
) -> dict:
    """2nd-order MUSCL-minmod HLL conservative FV solver for unsteady compliant-flux-tube MHD.

    Starts slightly super-tube at z_in = 0.08*L_B (M_T > 1) so the inlet is cleanly
    super-characteristic and marches a mismatched initial guess to the exact steady MHD nozzle.
    """
    z_in = 0.08 * L_B
    dz = (z_max - z_in) / n_cells
    z_c = z_in + (np.arange(n_cells) + 0.5) * dz
    z_f = z_in + np.arange(n_cells + 1) * dz

    B_vac_c = np.asarray(vacuum_dipole_B(z_c, L_B), dtype=float)
    B_vac_f = np.asarray(vacuum_dipole_B(z_f, L_B), dtype=float)
    B_vac_ext = np.empty(n_cells + 4)
    B_vac_ext[2:-2] = B_vac_c
    B_vac_ext[0:2] = vacuum_dipole_B(np.array([z_c[0] - 2 * dz, z_c[0] - dz]), L_B)
    B_vac_ext[-2:] = vacuum_dipole_B(np.array([z_c[-1] + dz, z_c[-1] + 2 * dz]), L_B)

    # Exact inlet ghost states (super-tube M_T > 1 => all 3 characteristics enter domain!)
    z_ghost_in = np.array([z_c[0] - 2 * dz, z_c[0] - dz])
    ex_in = solve_exact_steady_self_consistent(z_ghost_in, beta_vac0, gamma, L_B)
    Phi_B = ex_in["throat"]["Phi_B"]
    Q_in_ghost = cons_from_prim_compliant(ex_in["rho"], ex_in["u"], ex_in["p"], ex_in["A"], ex_in["B_vac"], Phi_B, gamma)

    # Non-trivial perturbed initial condition (far from steady solution!)
    ex_ref = solve_exact_steady_self_consistent(z_c, beta_vac0, gamma, L_B)
    rho_init = ex_ref["rho"] * (1.0 + 0.12 * np.sin(np.pi * (z_c - z_in) / (z_max - z_in)))
    u_init = np.full_like(z_c, ex_ref["u"][0] * 1.15)
    # Recompute A, p self-consistently from perturbed p_init = 0.5*B_vac^2 * (p_ref/(0.5*B_vac^2))
    p_init = np.minimum(ex_ref["p"] * 1.10, 0.49 * B_vac_c * B_vac_c)
    A_init = Phi_B / np.sqrt(np.maximum(B_vac_c * B_vac_c - 2.0 * p_init, 1e-12))
    Q = cons_from_prim_compliant(rho_init, u_init, p_init, A_init, B_vac_c, Phi_B, gamma)

    def flux_from_prim(rho, u, p, A):
        F0 = rho * u * A
        F1 = (rho * u * u + p) * A
        F2 = (0.5 * rho * u * u + (gamma / (gamma - 1.0)) * p) * u * A
        return np.vstack([F0, F1, F2])

    def rhs(Q_cur: np.ndarray) -> tuple[np.ndarray, float]:
        # Build ghost-padded primitive arrays (reconstruct rho, u, p cleanly; get A from exact B_vac_f!)
        Q_pad = np.empty((3, n_cells + 4))
        Q_pad[:, 2:-2] = Q_cur
        Q_pad[:, 0:2] = Q_in_ghost
        # Outflow: zero-gradient on rho, u, and pressure fraction p / (0.5*B_vac^2)
        rho_c, u_c, p_c, A_c, _, cT_c = prim_from_cons_compliant(Q_cur, B_vac_c, Phi_B, gamma)
        max_spd = float(np.max(np.abs(u_c) + cT_c))

        # Pad primitives for 2nd-order MUSCL reconstruction
        rho_pad = np.empty(n_cells + 4)
        u_pad = np.empty(n_cells + 4)
        pf_pad = np.empty(n_cells + 4)  # dimensionless beta_loc = 2*p / B_vac^2 in (0, 1)

        rho_pad[2:-2] = rho_c
        u_pad[2:-2] = u_c
        pf_pad[2:-2] = np.clip(2.0 * p_c / (B_vac_c * B_vac_c), 1e-10, 1.0 - 1e-8)

        rho_pad[0:2] = ex_in["rho"]
        u_pad[0:2] = ex_in["u"]
        pf_pad[0:2] = 2.0 * ex_in["p"] / (ex_in["B_vac"] ** 2)

        # Linearly extrapolate outflow (super-tube outflow, all characteristics leave!)
        rho_pad[-2] = max(2.0 * rho_c[-1] - rho_c[-2], 1e-12)
        rho_pad[-1] = max(2.0 * rho_pad[-2] - rho_c[-1], 1e-12)
        u_pad[-2] = 2.0 * u_c[-1] - u_c[-2]
        u_pad[-1] = 2.0 * u_pad[-2] - u_c[-1]
        pf_pad[-2] = np.clip(2.0 * pf_pad[-3] - pf_pad[-4], 1e-10, 1.0 - 1e-8)
        pf_pad[-1] = np.clip(2.0 * pf_pad[-2] - pf_pad[-3], 1e-10, 1.0 - 1e-8)

        # MUSCL-minmod slopes on cells 1..n_cells+2
        def recon(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            dl = v[1:-1] - v[:-2]
            dr = v[2:] - v[1:-1]
            s = minmod(dl, dr)
            vL = v[1:-2] + 0.5 * s[:-1]  # left state at face i+1/2 (n_cells+1 faces)
            vR = v[2:-1] - 0.5 * s[1:]   # right state at face i+1/2
            return vL, vR

        rhoL, rhoR = recon(rho_pad)
        uL, uR = recon(u_pad)
        pfL, pfR = recon(pf_pad)

        pfL = np.clip(pfL, 1e-10, 1.0 - 1e-8)
        pfR = np.clip(pfR, 1e-10, 1.0 - 1e-8)
        pL = 0.5 * pfL * B_vac_f * B_vac_f
        pR = 0.5 * pfR * B_vac_f * B_vac_f
        AL = Phi_B / np.sqrt(np.maximum(B_vac_f * B_vac_f * (1.0 - pfL), 1e-14))
        AR = Phi_B / np.sqrt(np.maximum(B_vac_f * B_vac_f * (1.0 - pfR), 1e-14))
        BL = Phi_B / AL
        BR = Phi_B / AR

        csL = np.sqrt(gamma * pL / np.maximum(rhoL, 1e-14))
        csR = np.sqrt(gamma * pR / np.maximum(rhoR, 1e-14))
        vAL = BL / np.sqrt(np.maximum(rhoL, 1e-14))
        vAR = BR / np.sqrt(np.maximum(rhoR, 1e-14))
        cTL = (csL * vAL) / np.sqrt(csL * csL + vAL * vAL)
        cTR = (csR * vAR) / np.sqrt(csR * csR + vAR * vAR)

        QL = cons_from_prim_compliant(rhoL, uL, pL, AL, B_vac_f, Phi_B, gamma)
        QR = cons_from_prim_compliant(rhoR, uR, pR, AR, B_vac_f, Phi_B, gamma)
        FL = flux_from_prim(rhoL, uL, pL, AL)
        FR = flux_from_prim(rhoR, uR, pR, AR)

        sL = np.minimum(uL - cTL, uR - cTR)
        sR = np.maximum(uL + cTL, uR + cTR)
        F_hll = np.where(
            sL >= 0.0, FL,
            np.where(sR <= 0.0, FR, (sR * FL - sL * FR + sL * sR * (QR - QL)) / np.maximum(sR - sL, 1e-14)),
        )

        # Well-balanced momentum source p * dA/dz using face-averaged A!
        A_face = 0.5 * (AL + AR)
        dA_dz = (A_face[1:] - A_face[:-1]) / dz
        src = np.zeros_like(Q_cur)
        src[1] = p_c * dA_dz

        dQ = -(F_hll[:, 1:] - F_hll[:, :-1]) / dz + src
        return dQ, max_spd

    # SSP-RK2 time marching to steady state
    t = 0.0
    n_iter = 0
    while t < t_end:
        dQ1, max_spd = rhs(Q)
        dt = min(cfl * dz / max(max_spd, 1e-6), t_end - t)
        Q1 = Q + dt * dQ1
        dQ2, _ = rhs(Q1)
        Q = 0.5 * (Q + Q1 + dt * dQ2)
        t += dt
        n_iter += 1

    rho_f, u_f, p_f, A_f, B_f, cT_f = prim_from_cons_compliant(Q, B_vac_c, Phi_B, gamma)
    ex = solve_exact_steady_self_consistent(z_c, beta_vac0, gamma, L_B)

    err_A = float(np.sqrt(np.mean((A_f - ex["A"]) ** 2)) / np.sqrt(np.mean(ex["A"] ** 2)))
    err_u = float(np.sqrt(np.mean((u_f - ex["u"]) ** 2)) / np.sqrt(np.mean(ex["u"] ** 2)))
    err_p = float(np.sqrt(np.mean((p_f - ex["p"]) ** 2)) / np.sqrt(np.mean(ex["p"] ** 2)))
    err_B = float(np.sqrt(np.mean((B_f - ex["B"]) ** 2)) / np.sqrt(np.mean(ex["B"] ** 2)))

    mdot_arr = rho_f * u_f * A_f
    H_arr = 0.5 * u_f * u_f + (gamma / (gamma - 1.0)) * (p_f / rho_f)
    F_arr = (rho_f * u_f * u_f + p_f) * A_f
    F_mag_int = float(np.trapezoid(p_f, A_f))
    F_diff = float(F_arr[-1] - F_arr[0])
    thrust_err = abs(F_diff - F_mag_int) / max(abs(F_diff), 1e-12)

    return {
        "z": z_c, "rho": rho_f, "u": u_f, "p": p_f, "A": A_f, "B": B_f, "c_T": cT_f,
        "exact": ex, "n_iter": n_iter,
        "l2_rel_err_A": err_A, "l2_rel_err_u": err_u, "l2_rel_err_p": err_p, "l2_rel_err_B": err_B,
        "mdot_rel_drift": float(np.max(np.abs(mdot_arr - ex["throat"]["mdot"])) / ex["throat"]["mdot"]),
        "H_rel_drift": float(np.max(np.abs(H_arr - ex["throat"]["H_0"])) / ex["throat"]["H_0"]),
        "thrust_balance_rel_err": float(thrust_err),
    }


# ---------------------------------------------------------------------------
# 3. Self-consistent vs. rigid Alfvén detachment & net axial thrust efficiency
# ---------------------------------------------------------------------------

def evaluate_nozzle_performance(
    beta_vac0: float,
    gamma: float,
    r0_over_LB: float = 0.20,
    self_consistent: bool = True,
    L_B: float = 1.0,
) -> dict:
    """Compute exact Alfven detachment point (M_A = 1) and net axial thrust efficiency.

    Accounts for BOTH:
      1. Magnetic nozzle thrust amplification F_tot(z_det) / F_tot(0)
      2. Conical streamline divergence efficiency eta_div = 0.5 * (1 + cos(theta_edge)),
         where tan(theta_edge) = dr/dz = (r_0 / 2) * (1/sqrt(A)) * (dA/dz) evaluated using
         the ACTUAL radius r(z) = r_0 * sqrt(A(z)) at detachment z_det!
    """
    z_grid = np.concatenate([
        np.linspace(0.0, 2.0, 400),
        np.linspace(2.01, 15.0, 600),
        np.linspace(15.1, 60.0, 400),
    ]) * L_B

    if self_consistent:
        sol = solve_exact_steady_self_consistent(z_grid, beta_vac0, gamma, L_B)
    else:
        sol = solve_exact_steady_rigid(z_grid, beta_vac0, gamma, L_B)

    M_A = sol["M_A"]
    if M_A[0] >= 1.0:
        # Already super-Alfvenic at throat (possible only in rigid model for high beta)
        idx = 0
        z_det = 0.0
    else:
        cross = np.where(M_A >= 1.0)[0]
        if len(cross) == 0:
            idx = len(z_grid) - 1
            z_det = float(z_grid[-1])
        else:
            idx = int(cross[0])
            # Linear interpolation for exact crossing
            frac = (1.0 - M_A[idx - 1]) / max(M_A[idx] - M_A[idx - 1], 1e-12)
            z_det = float(z_grid[idx - 1] + frac * (z_grid[idx] - z_grid[idx - 1]))

    # Re-evaluate finely around z_det for accurate dr/dz!
    eps = max(1e-4 * L_B, 1e-4 * z_det)
    z_loc = np.array([max(0.0, z_det - eps), z_det, z_det + eps])
    loc = solve_exact_steady_self_consistent(z_loc, beta_vac0, gamma, L_B) if self_consistent else solve_exact_steady_rigid(z_loc, beta_vac0, gamma, L_B)

    r_loc = r0_over_LB * L_B * np.sqrt(loc["A"])
    dr_dz = float((r_loc[2] - r_loc[0]) / max(z_loc[2] - z_loc[0], 1e-12))
    theta_edge = float(np.arctan(dr_dz))
    eta_div = 0.5 * (1.0 + float(np.cos(theta_edge)))

    F0 = float(sol["F_mom"][0])
    F_det = float(loc["F_mom"][1])
    thrust_amp = F_det / F0
    net_axial_gain = eta_div * thrust_amp

    # Also compute absolute specific impulse normalized by sonic speed c_s0:
    # I_sp_eff = eta_div * F_det / (mdot * c_s0) — fair apples-to-apples comparison across
    # rigid (choked at u_0 = c_s0) vs. self-consistent (choked at u_0 = c_T0 < c_s0)!
    mdot = float(sol["rho"][0] * sol["u"][0] * sol["A"][0])
    isp_eff_det = eta_div * F_det / mdot
    # Reference unexpanded choked gasdynamic sonic nozzle (M=1, F_0_gas = (rho_0*c_s0^2 + p_0)*A_0 = (1 + 1/gamma)*mdot_gas*c_s0)
    isp_ref_sonic = 1.0 + 1.0 / gamma
    net_gain_vs_sonic = isp_eff_det / isp_ref_sonic

    return {
        "beta_vac0": beta_vac0,
        "z_det": z_det,
        "A_det": float(loc["A"][1]),
        "M_det": float(loc["M"][1]),
        "beta_in_det": float(2.0 * loc["p"][1] / max(loc["B"][1] ** 2, 1e-30)),
        "theta_edge_deg": float(np.degrees(theta_edge)),
        "eta_div": eta_div,
        "thrust_amp_vs_own_throat": thrust_amp,
        "net_axial_gain_vs_own_throat": net_axial_gain,
        "isp_eff": isp_eff_det,
        "net_gain_vs_sonic_nozzle": net_gain_vs_sonic,
    }


# ---------------------------------------------------------------------------
# 4. Main experiments & figures
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Entry 002 (plasma-nozzles): Self-Consistent Diamagnetic Nozzle Inflation & Detachment ===\n")

    # ------------------------------------------------------------------
    # Experiment 1: Exact Steady vs. Unsteady Compliant-Flux-Tube MHD PDE (3 Grids)
    # ------------------------------------------------------------------
    print("--- Experiment 1: Unsteady 2nd-Order MUSCL-HLL Compliant-Tube MHD PDE Verification ---")
    beta_ver = 0.25
    gamma_ver = 5.0 / 3.0
    grid_conv = {}
    pde_200 = None
    for nc in (100, 200, 400):
        res = solve_unsteady_compliant_mhd_pde(beta_ver, gamma_ver, L_B=1.0, z_max=4.0, n_cells=nc, t_end=14.0)
        if nc == 200:
            pde_200 = res
        print(
            f"  N={nc:3d} (beta_vac0={beta_ver}, gamma=5/3): "
            f"L2(A)={res['l2_rel_err_A']:.2e}, L2(u)={res['l2_rel_err_u']:.2e}, "
            f"L2(p)={res['l2_rel_err_p']:.2e}, L2(B)={res['l2_rel_err_B']:.2e} | "
            f"mdot_err={res['mdot_rel_drift']:.2e}, H_err={res['H_rel_drift']:.2e}, "
            f"thrust_bal_err={res['thrust_balance_rel_err']:.2e}"
        )
        grid_conv[str(nc)] = {
            "l2_rel_err_A": res["l2_rel_err_A"],
            "l2_rel_err_u": res["l2_rel_err_u"],
            "l2_rel_err_p": res["l2_rel_err_p"],
            "l2_rel_err_B": res["l2_rel_err_B"],
            "mdot_rel_drift": res["mdot_rel_drift"],
            "H_rel_drift": res["H_rel_drift"],
            "thrust_balance_rel_err": res["thrust_balance_rel_err"],
        }
    rate_A = np.log2(grid_conv["100"]["l2_rel_err_A"] / grid_conv["200"]["l2_rel_err_A"])
    rate_A2 = np.log2(grid_conv["200"]["l2_rel_err_A"] / grid_conv["400"]["l2_rel_err_A"])
    print(f"  Observed convergence order on inflated area A(z): 100->200 = {rate_A:.2f}, 200->400 = {rate_A2:.2f}\n")

    # ------------------------------------------------------------------
    # Experiment 2: Diamagnetic Inflation Scaling A ~ B_vac^(-2/gamma) & Detachment Shift
    # ------------------------------------------------------------------
    print("--- Experiment 2: Diamagnetic Inflation Law A ~ B_vac^(-2/gamma) ---")
    z_prof = np.linspace(0.0, 8.0, 500)
    sc_ad = solve_exact_steady_self_consistent(z_prof, 0.25, 5.0 / 3.0)
    rg_ad = solve_exact_steady_rigid(z_prof, 0.25, 5.0 / 3.0)
    sc_iso = solve_exact_steady_self_consistent(z_prof, 0.15, 1.12)
    rg_iso = solve_exact_steady_rigid(z_prof, 0.15, 1.12)

    # Fit far-field log-log exponent d(ln A) / d(ln(1/B_vac)) at z in [4, 8]
    mask_far = z_prof >= 4.0
    exp_ad_fit = float(np.polyfit(np.log(1.0 / sc_ad["B_vac"][mask_far]), np.log(sc_ad["A"][mask_far]), 1)[0])
    exp_iso_fit = float(np.polyfit(np.log(1.0 / sc_iso["B_vac"][mask_far]), np.log(sc_iso["A"][mask_far]), 1)[0])
    print(f"  Adiabatic (gamma=5/3, beta_vac0=0.25): far-field exponent = {exp_ad_fit:.4f} (asymptote 2/gamma = {2.0/(5.0/3.0):.4f}, rigid = 1.0000)")
    print(f"    At z=6.0: A_self = {np.interp(6.0, z_prof, sc_ad['A']):.1f} vs A_rigid = {np.interp(6.0, z_prof, rg_ad['A']):.1f} ({np.interp(6.0, z_prof, sc_ad['A'])/np.interp(6.0, z_prof, rg_ad['A']):.2f}x inflation)")
    print(f"  Near-isothermal (gamma=1.12, beta_vac0=0.15): far-field exponent = {exp_iso_fit:.4f} (asymptote 2/gamma = {2.0/1.12:.4f}, rigid = 1.0000)")
    print(f"    At z=6.0: A_self = {np.interp(6.0, z_prof, sc_iso['A']):.1f} vs A_rigid = {np.interp(6.0, z_prof, rg_iso['A']):.1f} ({np.interp(6.0, z_prof, sc_iso['A'])/np.interp(6.0, z_prof, rg_iso['A']):.2f}x inflation)\n")

    # ------------------------------------------------------------------
    # Experiment 3: Self-Consistent vs. Rigid Thrust Efficiency Sweep Across beta_vac0
    # ------------------------------------------------------------------
    print("--- Experiment 3: Self-Consistent vs. Rigid Detachment & Net Thrust Efficiency ---")
    betas_ad = np.logspace(-2.6, -0.06, 50)    # beta_vac0 in [0.0025, 0.87]
    betas_iso = np.logspace(-4.2, -0.12, 55)   # beta_vac0 in [6.3e-5, 0.76] (uncensored for R0/LB=0.10!)

    perf_summary = {}
    curves = {}
    for label, g_val, b_arr in [("adiabatic_53", 5.0 / 3.0, betas_ad), ("near_iso_112", 1.12, betas_iso)]:
        sc_list = [evaluate_nozzle_performance(float(b), g_val, r0_over_LB=0.20, self_consistent=True) for b in b_arr]
        rg_list = [evaluate_nozzle_performance(float(b), g_val, r0_over_LB=0.20, self_consistent=False) for b in b_arr]

        i_sc = int(np.argmax([x["net_gain_vs_sonic_nozzle"] for x in sc_list]))
        i_rg = int(np.argmax([x["net_gain_vs_sonic_nozzle"] for x in rg_list]))

        opt_sc = sc_list[i_sc]
        opt_rg = rg_list[i_rg]
        # Also compare at the rigid optimum beta!
        sc_at_rg_opt = sc_list[i_rg]

        # Verify exact detachment beta identity: beta_in(z_det) == 2 / (gamma * M_det^2)
        beta_id_err = abs(opt_sc["beta_in_det"] - 2.0 / (g_val * opt_sc["M_det"] ** 2))
        # Find high-beta tube-choking zero-gain crossover (where self-consistent drops below sonic nozzle!)
        gains_sc = np.array([x["net_gain_vs_sonic_nozzle"] - 1.0 for x in sc_list])
        neg_idx = np.where(gains_sc < 0.0)[0]
        beta_choke_zero = float(b_arr[neg_idx[0]]) if len(neg_idx) > 0 else None

        print(f"  [{label}] (R_0/L_B = 0.20):")
        print(
            f"    Rigid (Entry 000) opt:  beta_vac0*={opt_rg['beta_vac0']:.5f} | z_det={opt_rg['z_det']:.3f} | "
            f"theta={opt_rg['theta_edge_deg']:.1f} deg | net gain vs sonic = +{(opt_rg['net_gain_vs_sonic_nozzle']-1)*100:.2f}%"
        )
        print(
            f"    Self-consistent opt:    beta_vac0*={opt_sc['beta_vac0']:.5f} | z_det={opt_sc['z_det']:.3f} | "
            f"theta={opt_sc['theta_edge_deg']:.1f} deg | net gain vs sonic = +{(opt_sc['net_gain_vs_sonic_nozzle']-1)*100:.2f}%"
        )
        print(
            f"    Detachment identity check |beta_in_det - 2/(gamma*M_det^2)| = {beta_id_err:.2e} | "
            f"High-beta tube-choking zero-gain crossover beta_vac0 = {beta_choke_zero}"
        )

        # Slenderness sensitivity R_0/L_B in {0.10, 0.20, 0.35}
        slend_summary = {}
        for r0_lb in (0.10, 0.20, 0.35):
            sc_r = [evaluate_nozzle_performance(float(b), g_val, r0_over_LB=r0_lb, self_consistent=True) for b in b_arr]
            rg_r = [evaluate_nozzle_performance(float(b), g_val, r0_over_LB=r0_lb, self_consistent=False) for b in b_arr]
            os_r = sc_r[int(np.argmax([x["net_gain_vs_sonic_nozzle"] for x in sc_r]))]
            or_r = rg_r[int(np.argmax([x["net_gain_vs_sonic_nozzle"] for x in rg_r]))]
            slend_summary[f"R0_LB_{r0_lb:.2f}"] = {
                "rigid_beta_opt": or_r["beta_vac0"],
                "rigid_gain_pct": (or_r["net_gain_vs_sonic_nozzle"] - 1.0) * 100.0,
                "sc_beta_opt": os_r["beta_vac0"],
                "sc_gain_pct": (os_r["net_gain_vs_sonic_nozzle"] - 1.0) * 100.0,
                "delta_gain_pct_pts": (os_r["net_gain_vs_sonic_nozzle"] - or_r["net_gain_vs_sonic_nozzle"]) * 100.0,
            }

        perf_summary[label] = {
            "rigid_opt": opt_rg,
            "self_consistent_opt": opt_sc,
            "self_consistent_at_rigid_opt": sc_at_rg_opt,
            "detachment_beta_identity_err": float(beta_id_err),
            "high_beta_zero_gain_crossover": beta_choke_zero,
            "slenderness_sweep": slend_summary,
        }
        curves[label] = (b_arr, sc_list, rg_list)

    # ------------------------------------------------------------------
    # Figure 1: Exact Steady vs Unsteady Compliant-Tube MHD PDE + Diamagnetic Inflation
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.3), dpi=150)

    ex2 = pde_200["exact"]
    ax1.plot(ex2["z"], ex2["A"], "-", color="#1f77b4", lw=2.4, label="Exact self-consistent $A(z)$")
    ax1.plot(pde_200["z"][::7], pde_200["A"][::7], "o", color="#d62728", ms=4.6, mfc="none", mew=1.3, label="Unsteady MUSCL-HLL PDE ($N=200$)")
    ax1.plot(ex2["z"], 1.0 / ex2["B_vac"], "--", color="#555555", lw=1.8, label="Rigid vacuum $A_{\\mathrm{vac}} = 1/B_{\\mathrm{vac}}$")
    ax1_r = ax1.twinx()
    ax1_r.plot(ex2["z"], ex2["u"], "-", color="#2ca02c", lw=2.0, label="Exact velocity $u(z)/c_{s0}$")
    ax1_r.plot(pde_200["z"][::7], pde_200["u"][::7], "s", color="#ff7f0e", ms=4.0, mfc="none", mew=1.2, label="PDE velocity $u(z)/c_{s0}$")
    ax1.set_xlabel("Axial coordinate $z / L_B$")
    ax1.set_ylabel("Inflated plume area $A(z) / A_0$", color="#1f77b4")
    ax1_r.set_ylabel("Exhaust velocity $u(z) / c_{s0}$", color="#2ca02c")
    ax1.set_title("Unsteady Compliant-Tube MHD PDE vs. Exact Solution\n($\\beta_{\\mathrm{vac},0} = 0.25, \\gamma = 5/3$)", fontweight="bold")
    ax1.grid(True, alpha=0.28)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.6)

    # Panel 2: Log-log inflation scaling A vs 1/B_vac
    invB = 1.0 / sc_ad["B_vac"]
    ax2.loglog(invB[1:], sc_iso["A"][1:], "-", color="#d62728", lw=2.4,
               label=f"Self-consistent $\\gamma=1.12$ (slope {exp_iso_fit:.2f}, theory {2/1.12:.2f})")
    ax2.loglog(invB[1:], sc_ad["A"][1:], "-", color="#1f77b4", lw=2.4,
               label=f"Self-consistent $\\gamma=5/3$ (slope {exp_ad_fit:.2f}, theory {2/(5/3):.2f})")
    ax2.loglog(invB[1:], invB[1:], "--", color="black", lw=1.8, label="Rigid vacuum $A = B_{\\mathrm{vac}}^{-1}$ (slope 1.00)")
    ax2.set_xlabel("Vacuum magnetic expansion ratio $B_{\\mathrm{vac},0} / B_{\\mathrm{vac}}(z)$")
    ax2.set_ylabel("Actual plasma plume area ratio $A(z) / A_0$")
    ax2.set_title("Diamagnetic Nozzle Inflation Scaling: $A \\propto B_{\\mathrm{vac}}^{-2/\\gamma}$", fontweight="bold")
    ax2.grid(True, which="both", alpha=0.28)
    ax2.legend(loc="upper left", fontsize=8.0)

    fig.suptitle("Plasma Nozzles Entry 002 (Part 1): Self-Consistent Compliant-Flux-Tube MHD Verification", fontsize=12.0, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp002_diamagnetic_inflation.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Upstream Alfvén Detachment Shift & Net Thrust Efficiency Comparison
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.3), dpi=150)

    for label, col, name in [("adiabatic_53", "#1f77b4", "Adiabatic $\\gamma=5/3$"),
                             ("near_iso_112", "#d62728", "Near-isothermal $\\gamma=1.12$")]:
        b_arr, sc_list, rg_list = curves[label]
        ax1.loglog(b_arr, [x["z_det"] for x in rg_list], "--", color=col, lw=1.8, alpha=0.8, label=f"{name} (rigid Entry 000)")
        ax1.loglog(b_arr, [x["z_det"] for x in sc_list], "-", color=col, lw=2.5, label=f"{name} (self-consistent)")

        ax2.semilogx(b_arr, [(x["net_gain_vs_sonic_nozzle"] - 1.0) * 100.0 for x in rg_list],
                     "--", color=col, lw=1.8, alpha=0.8, label=f"{name} (rigid Entry 000)")
        ax2.semilogx(b_arr, [(x["net_gain_vs_sonic_nozzle"] - 1.0) * 100.0 for x in sc_list],
                     "-", color=col, lw=2.5, label=f"{name} (self-consistent)")

    ax1.set_xlabel("Vacuum throat plasma beta $\\beta_{\\mathrm{vac},0} = 2\\mu_0 p_0 / B_{\\mathrm{vac},0}^2$")
    ax1.set_ylabel("Alfvén detachment location $z_{\\mathrm{det}} / L_B$ ($M_A = 1$)")
    ax1.set_title("Diamagnetic Field Expulsion Pulls Detachment Upstream", fontweight="bold")
    ax1.grid(True, which="both", alpha=0.28)
    ax1.legend(loc="lower left", fontsize=7.9)

    ax2.axhline(0.0, color="black", lw=1.0, ls=":")
    ax2.set_xlabel("Vacuum throat plasma beta $\\beta_{\\mathrm{vac},0} = 2\\mu_0 p_0 / B_{\\mathrm{vac},0}^2$")
    ax2.set_ylabel("Net axial specific-impulse gain over sonic nozzle (%)")
    ax2.set_title("Net Thrust Gain: Rigid vs. Self-Consistent Nozzle ($R_0/L_B=0.20$)", fontweight="bold")
    ax2.grid(True, which="both", alpha=0.28)
    ax2.legend(loc="upper right", fontsize=7.9)

    fig.suptitle("Plasma Nozzles Entry 002 (Part 2): Self-Consistent Detachment & Thrust Optimization", fontsize=12.0, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp002_detachment_and_thrust.png")
    plt.close(fig)

    metrics = {
        "exp1_pde_grid_convergence": grid_conv,
        "exp1_convergence_orders_A": {"100_to_200": float(rate_A), "200_to_400": float(rate_A2)},
        "exp2_diamagnetic_inflation_exponents": {
            "adiabatic_53_fit": exp_ad_fit,
            "adiabatic_53_asymptote": 2.0 / (5.0 / 3.0),
            "near_iso_112_fit": exp_iso_fit,
            "near_iso_112_asymptote": 2.0 / 1.12,
        },
        "exp3_performance_comparison": perf_summary,
    }
    with open(RESULTS_DIR / "exp002_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved plasma-nozzles Entry 002 plots and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
