"""
Entry 017: 3D Adiabatic No-Slip Baffle Wakes, Core-Bypass Flow & Full All-Mode
Quench in the Metric-Corrected 3D Smooth Curved Bell Combustor
==============================================================================

Builds directly on hardened Entry 016 (commit 9dee502) to answer the two open
3D physics questions left by Peer Reviews #12-#13:

1. Exp 1 -- Converged 3D Adiabatic No-Slip Partial-Span Baffle Wakes & 3D
   Core-Bypass Law:
   - Upgrades the metric-corrected 3D cylindrical NS solver with adiabatic
     no-slip baffle walls (mass & total energy conservation ~1e-16 on wavy
     curved bells) and an exact 5-fold sector-symmetric Local Time-Stepping
     (LTS) steady-wake relaxer (converging rhs_max < 1e-8-1e-10 ~9x faster).
   - Verifies exact 1-sector -> full 5-sector 3D tiling equilibrium (zero
     wake-drift floor on full 3D global-stepped runs with coupled feed ODE
     and cavities, closing Peer Review #8's wake-drift trap).
   - Compares full-span (r_b = 0.0) vs. partial-span (r_b = 0.42) adiabatic
     no-slip wakes in straight cylinders (20x12x40 & refined 24x16x50) and
     smooth curved bells (32x10x40 & refined 36x12x40), quantifying the
     purely 3D radial core-bypass jet (u_core > M_c, halved shroud deficit).

2. Exp 2 -- Exact 3D No-Slip Wake Penalty Audit (Slip vs. Converged No-Slip):
   - Measures clean linear (A_eq = 1e-4, nonlinear=False) cold & reacting 1T
     and 2T damping/growth rates and FFT frequencies for slip vs. converged
     no-slip wakes across straight full-span, straight partial-span, smooth
     bell partial-span, and refined 36x12x40 smooth bell grids.

3. Exp 3 -- Closing the Smooth-Bell 1T Subcritical Triggering Gap & The
   Sector-Clipped Port Law:
   - Proves Entry 016's bell_tri50_1T (5.49%) was linearly stable
     (alpha_1T = -0.058) and survived solely via subcritical triggering.
   - Switching the 1.0% Nodal Mid-Ring from axisymmetric (n_cav=1) to
     5-sector segmented (n_cav=5, ZERO extra volume, beta_tot = 5.0%) boosts
     linear bell 1T margin 1.71x (-0.058 -> -0.100).
   - Isolates the Gaussian baffle-leakage trap (widening sigma_o without
     sector clipping smears ports through solid baffle walls and worsens 1T
     from decaying 3.22% to locked 5.40%) and proves sector-clipped wide-mouth
     ports (|dtheta| < pi/N_b) unlock genuine finite-amplitude quench.
   - Maps exact A_eq = 0.015..0.060 triggering thresholds across tri50_ax,
     tri50_seg, tri50_seg_wide (clipped), and tri60_seg (clipped).

4. Exp 4 -- Full All-6-Mode True-3D Smooth Curved Bell Nonlinear Audit (Slip
   & Converged No-Slip Wakes + Refined 36x12x40 Confirmation):
   - Tests spin_1T, stand_1T, spin_2T, 1R, 1L (plus k=0 feed chugging) in the
     metric-corrected smooth curved bell (Re = 500, psi = 0.25, n_p = 0.34,
     omega_hp = 0.50, check_valve = True) out to t = 140-160 under BOTH slip
     and converged 3D no-slip baffle wakes, plus refined 36x12x40 checks.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import jn_zeros, jnp_zeros, jv, jvp

GAMMA = 1.25
P_BAR = 1.0 / GAMMA
RHO_BAR = 1.0
L_A = 1.5
X_F = 0.25
SIG_F = 0.08
TAU_BAR = 0.16
Q_S = 0.045
KAPPA_NL = 4.5
PR = 0.72

S11 = float(jnp_zeros(1, 1)[0])
S21 = float(jnp_zeros(2, 1)[0])
S02 = float(jnp_zeros(0, 2)[0])
R0_1R = float(jn_zeros(0, 1)[0] / S02)

WH_1T_RET = 1.7175
WH_2T_RET = 3.376
WH_1L_TGT = 2.09

TAU_INJ_DEF = 1.0
TAU_F_DEF = 5.0

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# 1. Geometry & 3D Cylindrical FV Helpers
# ---------------------------------------------------------------------------

def isentropic_area_ratio(M: float) -> float:
    gp = GAMMA + 1.0
    gm = GAMMA - 1.0
    return (1.0 / M) * ((2.0 / gp) * (1.0 + 0.5 * gm * M * M)) ** (gp / (2.0 * gm))


def nozzle_R_smooth_3d(x: np.ndarray, M_c: float, L_n: float = 0.60, s_max: float = 0.28):
    R_t = float(np.sqrt(1.0 / isentropic_area_ratio(M_c)))
    k_t = 0.5 * (np.pi ** 2) * (1.0 - R_t) / (L_n * L_n)
    R = np.ones_like(x, dtype=float)
    dR = np.zeros_like(x, dtype=float)
    x_t = L_A + L_n
    m1 = (x > L_A) & (x <= x_t)
    s1 = (x[m1] - L_A) / L_n
    R[m1] = R_t + 0.5 * (1.0 - R_t) * (1.0 + np.cos(np.pi * s1))
    dR[m1] = -0.5 * np.pi * (1.0 - R_t) * np.sin(np.pi * s1) / L_n
    m2 = x > x_t
    y = x[m2] - x_t
    u = k_t * y / s_max
    R[m2] = R_t + (s_max * s_max / k_t) * (np.logaddexp(u, -u) - np.log(2.0))
    dR[m2] = s_max * np.tanh(u)
    return R, dR


def van_leer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = a * b
    return np.where(ab > 0.0, 2.0 * ab / np.where(ab > 0.0, a + b, 1.0), 0.0)


def phi_flame(q: np.ndarray) -> np.ndarray:
    x = q / Q_S
    return Q_S * np.tanh(x + KAPPA_NL * (x ** 3))


def hllc_3d(rL, vnL, vt1L, vt2L, pL, rR, vnR, vt1R, vt2R, pR) -> np.ndarray:
    cL = np.sqrt(GAMMA * pL / rL); cR = np.sqrt(GAMMA * pR / rR)
    EL = pL / (GAMMA - 1.0) + 0.5 * rL * (vnL * vnL + vt1L * vt1L + vt2L * vt2L)
    ER = pR / (GAMMA - 1.0) + 0.5 * rR * (vnR * vnR + vt1R * vt1R + vt2R * vt2R)
    sL = np.minimum(vnL - cL, vnR - cR); sR = np.maximum(vnL + cL, vnR + cR)
    den = rL * (sL - vnL) - rR * (sR - vnR)
    den = np.where(np.abs(den) > 1e-14, den, 1e-14)
    sS = (pR - pL + rL * vnL * (sL - vnL) - rR * vnR * (sR - vnR)) / den
    FL = np.stack([rL * vnL, rL * vnL * vnL + pL, rL * vnL * vt1L, rL * vnL * vt2L, vnL * (EL + pL)], axis=0)
    FR = np.stack([rR * vnR, rR * vnR * vnR + pR, rR * vnR * vt1R, rR * vnR * vt2R, vnR * (ER + pR)], axis=0)
    dLS = np.where(np.abs(sL - sS) > 1e-14, sL - sS, 1e-14)
    dLn = np.where(np.abs(sL - vnL) > 1e-14, sL - vnL, 1e-14)
    fL = rL * (sL - vnL) / dLS
    UsL = np.stack([fL, fL * sS, fL * vt1L, fL * vt2L, fL * (EL / rL + (sS - vnL) * (sS + pL / (rL * dLn)))], axis=0)
    FsL = FL + sL * (UsL - np.stack([rL, rL * vnL, rL * vt1L, rL * vt2L, EL], axis=0))
    dRS = np.where(np.abs(sR - sS) > 1e-14, sR - sS, 1e-14)
    dRn = np.where(np.abs(sR - vnR) > 1e-14, sR - vnR, 1e-14)
    fR = rR * (sR - vnR) / dRS
    UsR = np.stack([fR, fR * sS, fR * vt1R, fR * vt2R, fR * (ER / rR + (sS - vnR) * (sS + pR / (rR * dRn)))], axis=0)
    FsR = FR + sR * (UsR - np.stack([rR, rR * vnR, rR * vt1R, rR * vt2R, ER], axis=0))
    return np.where(sL >= 0.0, FL, np.where(sS >= 0.0, FsL, np.where(sR >= 0.0, FsR, FR)))


def inlet_choked_3d(rho_i, ux_i, p_i, m_in: float, m_port: np.ndarray | None = None):
    c_i = np.sqrt(GAMMA * p_i / rho_i)
    Jm = ux_i - (2.0 / (GAMMA - 1.0)) * c_i
    c_g = np.maximum(c_i.copy(), 0.2)
    a_exp = 2.0 / (GAMMA - 1.0)
    m_tgt = m_in if m_port is None else (m_in - m_port)
    for _ in range(7):
        u_g = Jm + a_exp * c_g
        f = (c_g ** a_exp) * u_g - m_tgt
        df = a_exp * (c_g ** (a_exp - 1.0)) * u_g + (c_g ** a_exp) * a_exp
        c_g = np.maximum(c_g - f / np.where(np.abs(df) > 1e-12, df, 1e-12), 0.15)
    ux_g = Jm + a_exp * c_g
    rho_g = c_g ** a_exp
    p_g = P_BAR * (rho_g ** GAMMA)
    return rho_g, ux_g, np.zeros_like(ux_g), np.zeros_like(ux_g), p_g


def outlet_mc_choked_3d(rho_i, ux_i, ur_i, ut_i, p_i, M_c: float):
    c_i = np.sqrt(GAMMA * p_i / rho_i)
    Jp = ux_i + (2.0 / (GAMMA - 1.0)) * c_i
    c_g = np.maximum(Jp / (M_c + 2.0 / (GAMMA - 1.0)), 0.15)
    ux_g = M_c * c_g
    rat = c_g / np.maximum(c_i, 1e-10)
    rho_g = rho_i * (rat ** (2.0 / (GAMMA - 1.0)))
    p_g = p_i * (rat ** (2.0 * GAMMA / (GAMMA - 1.0)))
    return rho_g, ux_g, ur_i.copy(), ut_i.copy(), p_g


def build_3d_geom(nx: int, nr: int, nth: int, M_c: float = 0.12, L_n: float = 0.60, L_d: float = 0.45, mode: str = "straight"):
    L_tot = L_A if mode in ("straight", "periodic_bump", "closed") else (L_A + L_n + L_d)
    dx = L_tot / nx; deta = 1.0 / nr; dth = 2.0 * np.pi / nth
    xf = np.linspace(0.0, L_tot, nx + 1); xc = 0.5 * (xf[:-1] + xf[1:])
    etaf = np.linspace(0.0, 1.0, nr + 1); etac = 0.5 * (etaf[:-1] + etaf[1:])
    thc = (np.arange(nth) + 0.5) * dth

    def R_and_dR(xv):
        if mode in ("straight", "closed"):
            return np.ones_like(xv), np.zeros_like(xv)
        if mode == "periodic_bump":
            s = np.sin(np.pi * xv / L_tot); c = np.cos(np.pi * xv / L_tot)
            R = np.sqrt(1.0 - 0.35 * s * s)
            dR = -0.35 * (np.pi / L_tot) * s * c / R
            return R, dR
        return nozzle_R_smooth_3d(xv, M_c, L_n)

    Rf, dRf = R_and_dR(xf)
    Rm, dRm = R_and_dR(xc)
    int_R = (dx / 6.0) * (Rf[:-1] + 4.0 * Rm + Rf[1:])
    int_R2 = (dx / 6.0) * (Rf[:-1] ** 2 + 4.0 * Rm ** 2 + Rf[1:] ** 2)
    int_R3 = (dx / 6.0) * (Rf[:-1] ** 3 + 4.0 * Rm ** 3 + Rf[1:] ** 3)
    d_eta2 = etaf[1:] ** 2 - etaf[:-1] ** 2
    d_eta = etaf[1:] - etaf[:-1]

    V = 0.5 * int_R2[:, None] * d_eta2[None, :] * dth
    r_bar = (2.0 / 3.0) * ((etaf[1:] ** 3 - etaf[:-1] ** 3) / d_eta2)[None, :] * (int_R3 / int_R2)[:, None]
    Sx = 0.5 * (Rf[:, None] ** 2) * d_eta2[None, :] * dth
    r_xf = (2.0 / 3.0) * ((etaf[1:] ** 3 - etaf[:-1] ** 3) / d_eta2)[None, :] * Rf[:, None]
    S_er = int_R[:, None] * etaf[None, :] * dth
    S_ex = 0.5 * (Rf[1:] ** 2 - Rf[:-1] ** 2)[:, None] * (etaf[None, :] ** 2) * dth
    Sm = np.sqrt(S_er ** 2 + S_ex ** 2)
    nxe = np.where(Sm > 0.0, -S_ex / np.where(Sm > 0.0, Sm, 1.0), 0.0)
    nre = np.where(Sm > 0.0, S_er / np.where(Sm > 0.0, Sm, 1.0), 1.0)
    r_ef = (int_R2 / int_R)[:, None] * etaf[None, :]
    S_th = int_R[:, None] * d_eta[None, :]
    hoop = S_er[:, 1:] - S_er[:, :-1]
    r_thf = (0.5 * d_eta2[None, :] * int_R2[:, None]) / S_th
    dA_in = np.broadcast_to(Sx[0, :, None], (nr, nth)).copy()
    eta_Rp_c = dRm[:, None] * etac[None, :]
    eta_Rp_xf = dRf[:, None] * etac[None, :]
    eta_Rp_ef = dRm[:, None] * etaf[None, :]
    return {
        "xc": xc, "xf": xf, "etac": etac, "etaf": etaf, "thc": thc, "dx": dx, "deta": deta, "dth": dth,
        "L_tot": L_tot, "Rm": Rm, "Rf": Rf, "dA_in": dA_in,
        "V": V[:, :, None], "r_bar": r_bar[:, :, None], "Sx": Sx[:, :, None], "r_xf": r_xf[:, :, None],
        "Sm": Sm[:, :, None], "nxe": nxe[:, :, None], "nre": nre[:, :, None], "r_ef": r_ef[:, :, None],
        "S_th": S_th[:, :, None], "r_thf": r_thf[:, :, None], "hoop": hoop[:, :, None],
        "eta_Rp_c": eta_Rp_c[:, :, None], "eta_Rp_xf": eta_Rp_xf[:, :, None], "eta_Rp_ef": eta_Rp_ef[:, :, None],
    }


# ---------------------------------------------------------------------------
# 2. Steady Base Solvers: Axisymmetric Smooth Bell & 5-Fold 3D No-Slip Wake (LTS)
# ---------------------------------------------------------------------------

def relax_axi_3d_bell_base(
    nx: int, nr: int, M_c: float = 0.12, L_n: float = 0.60, L_d: float = 0.45,
    Re: float = 500.0, t_relax: float = 160.0,
) -> dict:
    """Relax 2D axisymmetric transonic 3D smooth bell steady base with exact mapped-grid metrics."""
    g = build_3d_geom(nx, nr, 1, M_c=M_c, L_n=L_n, L_d=L_d, mode="bell_smooth")
    V = g["V"][:, :, 0]; Sx = g["Sx"][:, :, 0]; Sm = g["Sm"][:, :, 0]
    nxe = g["nxe"][:, :, 0]; nre = g["nre"][:, :, 0]; hoop = g["hoop"][:, :, 0]
    rb = g["r_bar"][:, :, 0]; r_ef = g["r_ef"][:, :, 0]
    eta_Rp_c = g["eta_Rp_c"][:, :, 0]; eta_Rp_xf = g["eta_Rp_xf"][:, :, 0]
    xc, Rm, dx, deta = g["xc"], g["Rm"], g["dx"], g["deta"]
    mu = 0.0 if not np.isfinite(Re) else (1.0 / Re)
    k_th = 0.0 if not np.isfinite(Re) else (GAMMA / (PR * (GAMMA - 1.0) * Re))

    A_t = 1.0 / isentropic_area_ratio(M_c); x_t = L_A + L_n
    M1d = np.array([
        1.0 if Rm[i] ** 2 / A_t <= 1.0 + 1e-14 else (
            brentq(lambda mm: isentropic_area_ratio(mm) - Rm[i] ** 2 / A_t, 1e-6, 1.0 - 1e-12)
            if xc[i] < x_t else
            brentq(lambda mm: isentropic_area_ratio(mm) - Rm[i] ** 2 / A_t, 1.0 + 1e-12, 12.0)
        ) for i in range(nx)
    ])
    fac_c = 1.0 + 0.5 * (GAMMA - 1.0) * M_c * M_c
    T1d = fac_c / (1.0 + 0.5 * (GAMMA - 1.0) * M1d * M1d)
    r1d = T1d ** (1.0 / (GAMMA - 1.0)); p1d = P_BAR * (T1d ** (GAMMA / (GAMMA - 1.0))); u1d = M1d * np.sqrt(T1d)
    r = np.broadcast_to(r1d[:, None], (nx, nr)).copy()
    u = np.broadcast_to(u1d[:, None], (nx, nr)).copy()
    v = np.zeros((nx, nr)); pr = np.broadcast_to(p1d[:, None], (nx, nr)).copy()
    Q = np.stack([V * r, V * r * u, V * r * v, np.zeros_like(V), V * (pr / (GAMMA - 1.0) + 0.5 * r * (u * u + v * v))], axis=0)

    def rhs_axi(Qs):
        r_s = np.maximum(Qs[0] / V, 1e-10); u_s = Qs[1] / (V * r_s); v_s = Qs[2] / (V * r_s)
        pr_s = np.maximum((GAMMA - 1.0) * (Qs[4] / V - 0.5 * r_s * (u_s * u_s + v_s * v_s)), 1e-10)
        P = np.stack([r_s, u_s, v_s, np.zeros_like(r_s), pr_s], axis=0)
        Px = np.empty((5, nx + 4, nr)); Px[:, 2:-2] = P
        ri, ui, vi, wi, pi = inlet_choked_3d(r_s[0], u_s[0], pr_s[0], M_c)
        for k in (0, 1):
            Px[0, k] = ri; Px[1, k] = ui; Px[2, k] = vi; Px[3, k] = wi; Px[4, k] = pi
        Px[:, -2] = P[:, -1]; Px[:, -1] = P[:, -1]
        sx = van_leer(Px[:, 1:-1] - Px[:, :-2], Px[:, 2:] - Px[:, 1:-1])
        PLx = Px[:, 1:-2] + 0.5 * sx[:, :-1]; PRx = Px[:, 2:-1] - 0.5 * sx[:, 1:]
        Fx = hllc_3d(np.maximum(PLx[0], 1e-10), PLx[1], PLx[2], PLx[3], np.maximum(PLx[4], 1e-10),
                     np.maximum(PRx[0], 1e-10), PRx[1], PRx[2], PRx[3], np.maximum(PRx[4], 1e-10))
        FxQ = np.stack([Sx * Fx[0], Sx * Fx[1], Sx * Fx[2], np.zeros_like(Sx), Sx * Fx[4]], axis=0)

        Pe = np.empty((5, nx, nr + 4)); Pe[:, :, 2:-2] = P
        Pe[:, :, 1] = P[:, :, 0]; Pe[2, :, 1] *= -1.0
        Pe[:, :, 0] = P[:, :, 1]; Pe[2, :, 0] *= -1.0
        Pe[:, :, nr + 2] = P[:, :, nr - 1]; Pe[:, :, nr + 3] = P[:, :, nr - 1]
        se = van_leer(Pe[:, :, 1:-1] - Pe[:, :, :-2], Pe[:, :, 2:] - Pe[:, :, 1:-1])
        PLe = Pe[:, :, 1:-2] + 0.5 * se[:, :, :-1]; PRe = Pe[:, :, 2:-1] - 0.5 * se[:, :, 1:]
        vnL = nxe * PLe[1] + nre * PLe[2]; vt1L = -nre * PLe[1] + nxe * PLe[2]
        vnR = nxe * PRe[1] + nre * PRe[2]; vt1R = -nre * PRe[1] + nxe * PRe[2]
        vnR[:, -1] = -vnL[:, -1]; vt1R[:, -1] = vt1L[:, -1]
        PRe[0, :, -1] = PLe[0, :, -1]; PRe[4, :, -1] = PLe[4, :, -1]
        Fe_n = hllc_3d(np.maximum(PLe[0], 1e-10), vnL, vt1L, PLe[3], np.maximum(PLe[4], 1e-10),
                       np.maximum(PRe[0], 1e-10), vnR, vt1R, PRe[3], np.maximum(PRe[4], 1e-10))
        FeQ = np.stack([Sm * Fe_n[0], Sm * (nxe * Fe_n[1] - nre * Fe_n[2]), Sm * (nre * Fe_n[1] + nxe * Fe_n[2]),
                        np.zeros_like(Sm), Sm * Fe_n[4]], axis=0)
        FeQ[:, :, 0] = 0.0
        res = -(FxQ[:, 1:] - FxQ[:, :-1]) - (FeQ[:, :, 1:] - FeQ[:, :, :-1])
        res[2] += pr_s * hoop

        if mu > 0.0:
            dr_c = deta * Rm[:, None]
            u_x = Px[1, 1:-1]; v_x = Px[2, 1:-1]; T_x = Px[4, 1:-1] / np.maximum(Px[0, 1:-1], 1e-10)
            u_e = Pe[1, :, 1:-1]; v_e = Pe[2, :, 1:-1]; T_e = Pe[4, :, 1:-1] / np.maximum(Pe[0, :, 1:-1], 1e-10)
            du_dr_c = (u_e[:, 2:] - u_e[:, :-2]) / (2.0 * dr_c)
            dv_dr_c = (v_e[:, 2:] - v_e[:, :-2]) / (2.0 * dr_c)
            dT_dr_c = (T_e[:, 2:] - T_e[:, :-2]) / (2.0 * dr_c)
            du_dx_c = (u_x[2:] - u_x[:-2]) / (2.0 * dx) - eta_Rp_c * du_dr_c
            dv_dx_c = (v_x[2:] - v_x[:-2]) / (2.0 * dx) - eta_Rp_c * dv_dr_c
            dT_dx_c = (T_x[2:] - T_x[:-2]) / (2.0 * dx) - eta_Rp_c * dT_dr_c
            def avg_x(a):
                o = np.empty((nx + 1, nr)); o[1:-1] = 0.5 * (a[:-1] + a[1:]); o[0] = a[0]; o[-1] = a[-1]; return o
            du_dr_xf = avg_x(du_dr_c); dv_dr_xf = avg_x(dv_dr_c); dT_dr_xf = avg_x(dT_dr_c)
            du_dx_f = (u_x[1:] - u_x[:-1]) / dx - eta_Rp_xf * du_dr_xf
            dv_dx_f = (v_x[1:] - v_x[:-1]) / dx - eta_Rp_xf * dv_dr_xf
            dT_dx_f = (T_x[1:] - T_x[:-1]) / dx - eta_Rp_xf * dT_dr_xf
            div_r_xf = avg_x(dv_dr_c + v_s / rb)
            uf_x = 0.5 * (u_x[:-1] + u_x[1:]); vf_x = 0.5 * (v_x[:-1] + v_x[1:])
            txx_xf = mu * ((4.0 / 3.0) * du_dx_f - (2.0 / 3.0) * div_r_xf)
            txr_xf = mu * (du_dr_xf + dv_dx_f)
            qx_xf = -k_th * dT_dx_f
            res[1] += Sx[1:] * txx_xf[1:] - Sx[:-1] * txx_xf[:-1]
            res[2] += Sx[1:] * txr_xf[1:] - Sx[:-1] * txr_xf[:-1]
            VFe = Sx * (uf_x * txx_xf + vf_x * txr_xf - qx_xf)
            res[4] += VFe[1:] - VFe[:-1]

            du_dr_ef = (u_e[:, 1:] - u_e[:, :-1]) / dr_c
            dv_dr_ef = (v_e[:, 1:] - v_e[:, :-1]) / dr_c
            dT_dr_ef = (T_e[:, 1:] - T_e[:, :-1]) / dr_c
            du_dx_ef = np.zeros((nx, nr + 1)); dv_dx_ef = np.zeros((nx, nr + 1)); dT_dx_ef = np.zeros((nx, nr + 1))
            du_dx_ef[:, 1:-1] = 0.5 * (du_dx_c[:, :-1] + du_dx_c[:, 1:])
            dv_dx_ef[:, 1:-1] = 0.5 * (dv_dx_c[:, :-1] + dv_dx_c[:, 1:])
            dT_dx_ef[:, 1:-1] = 0.5 * (dT_dx_c[:, :-1] + dT_dx_c[:, 1:])
            vf_e = 0.5 * (v_e[:, :-1] + v_e[:, 1:]); uf_e = 0.5 * (u_e[:, :-1] + u_e[:, 1:])
            vr_over_r_ef = np.where(r_ef > 1e-12, vf_e / np.where(r_ef > 1e-12, r_ef, 1.0), 0.0)
            txx_ef = mu * ((4.0 / 3.0) * du_dx_ef - (2.0 / 3.0) * (dv_dr_ef + vr_over_r_ef))
            trr_ef = mu * ((4.0 / 3.0) * dv_dr_ef - (2.0 / 3.0) * (du_dx_ef + vr_over_r_ef))
            txr_ef = mu * (du_dr_ef + dv_dx_ef)
            qx_ef = -k_th * dT_dx_ef; qr_ef = -k_th * dT_dr_ef
            vx_ef = Sm * (nxe * txx_ef + nre * txr_ef)
            vr_ef = Sm * (nxe * txr_ef + nre * trr_ef)
            ve_ef = Sm * (uf_e * (nxe * txx_ef + nre * txr_ef) + vf_e * (nxe * txr_ef + nre * trr_ef) - (nxe * qx_ef + nre * qr_ef))
            vx_ef[:, 0] = 0.0; vx_ef[:, -1] = 0.0; vr_ef[:, 0] = 0.0; vr_ef[:, -1] = 0.0
            ve_ef[:, 0] = 0.0; ve_ef[:, -1] = 0.0
            res[1] += vx_ef[:, 1:] - vx_ef[:, :-1]; res[2] += vr_ef[:, 1:] - vr_ef[:, :-1]
            res[4] += ve_ef[:, 1:] - ve_ef[:, :-1]
            ttt_c = mu * ((4.0 / 3.0) * (v_s / rb) - (2.0 / 3.0) * (du_dx_c + dv_dr_c))
            res[2] -= ttt_c * hoop
        return res

    t = 0.0; last_rhs = 1.0
    while t < t_relax - 1e-12:
        r_s = np.maximum(Q[0] / V, 1e-10); u_s = Q[1] / (V * r_s); v_s = Q[2] / (V * r_s)
        pr_s = np.maximum((GAMMA - 1.0) * (Q[4] / V - 0.5 * r_s * (u_s * u_s + v_s * v_s)), 1e-10)
        c_s = np.sqrt(GAMMA * pr_s / r_s)
        dr_eff = deta * Rm[:, None]
        inv_dt = ((np.abs(u_s) + c_s) / dx + (np.abs(v_s) + c_s) / dr_eff) / 0.30
        if mu > 0.0:
            nu_eff = max(4.0 / 3.0, GAMMA / PR) * mu / r_s
            inv_dt = inv_dt + nu_eff * (1.0 / (dx * dx) + 1.0 / (dr_eff * dr_eff)) / 0.40
        dt = min(1.0 / float(np.max(inv_dt)), t_relax - t)
        k1 = rhs_axi(Q); Q1 = Q + dt * k1
        k2 = rhs_axi(Q1); Q = 0.5 * (Q + Q1 + dt * k2)
        t += dt; last_rhs = float(np.max(np.abs(k1 / V)))
        if t > 20.0 and last_rhs < 1e-9:
            break

    r_s = Q[0] / V; u_s = Q[1] / (V * r_s); v_s = Q[2] / (V * r_s)
    pr_s = (GAMMA - 1.0) * (Q[4] / V - 0.5 * r_s * (u_s * u_s + v_s * v_s))
    c_s = np.sqrt(GAMMA * pr_s / r_s); M2d = np.sqrt(u_s * u_s + v_s * v_s) / c_s
    it_t = int(np.argmin(np.abs(xc - x_t)))
    return {
        "prim_axi": np.stack([r_s, u_s, v_s, np.zeros_like(r_s), pr_s], axis=0),
        "rhs_max": last_rhs,
        "M_throat_center": float(M2d[it_t, 0]), "M_throat_wall": float(M2d[it_t, -1]),
        "M_exit_center": float(M2d[-1, 0]), "M_exit_wall": float(M2d[-1, -1]),
    }


def relax_3d_noslip_wake_lts(
    nx: int, nr: int, nth: int, M_c: float = 0.12, L_n: float = 0.60, L_d: float = 0.45,
    mode: str = "straight", Re: float = 500.0,
    n_baffles: int = 5, L_b: float = 0.45, r_b: float = 0.42,
    prim_axi_base: np.ndarray | None = None, max_iter: int = 26000, tol: float = 1e-9,
) -> dict:
    """Exact 1-sector (2*pi/N_b) 3D steady adiabatic no-slip baffle wake solver using Local Time-Stepping (LTS).
    For odd N_b and even nth_sec = nth // N_b, centerline reflection theta -> theta + pi maps
    exactly to np.roll(..., nth_sec // 2, axis=3) on the 5-fold symmetric field."""
    assert nth % n_baffles == 0
    nth_s = nth // n_baffles
    assert (n_baffles % 2 == 0) or (nth_s % 2 == 0), "Odd n_baffles requires even nth // n_baffles"
    g = build_3d_geom(nx, nr, nth, M_c=M_c, L_n=L_n, L_d=L_d, mode=mode)
    V, rb = g["V"], g["r_bar"]
    xc, etac, dx, deta, dth = g["xc"], g["etac"], g["dx"], g["deta"], g["dth"]
    mu = 1.0 / Re
    k_th = GAMMA / (PR * (GAMMA - 1.0) * Re)
    dr_c = deta * g["Rm"][:, None, None]
    r_ef, r_xf, r_thf = g["r_ef"], g["r_xf"], g["r_thf"]
    nxe, nre = g["nxe"], g["nre"]
    Sx, Sm, S_th, hoop = g["Sx"], g["Sm"], g["S_th"], g["hoop"]
    eta_Rp_c, eta_Rp_xf = g["eta_Rp_c"], g["eta_Rp_xf"]

    ix_b = xc <= L_b; jr_b = etac >= r_b
    b_xr = ix_b[:, None] & jr_b[None, :]
    b_face = np.zeros((nx, nr, nth_s + 1), dtype=bool)
    b_face[b_xr, 0] = True; b_face[b_xr, nth_s] = True
    b_sz = np.zeros((nx, nr, nth_s + 2), dtype=bool)
    b_sz[b_xr, 0] = True; b_sz[b_xr, 1] = True; b_sz[b_xr, nth_s] = True; b_sz[b_xr, nth_s + 1] = True

    if prim_axi_base is not None:
        P_b = np.broadcast_to(prim_axi_base[:, :, :, None], (5, nx, nr, nth_s)).copy()
    else:
        P_b = np.zeros((5, nx, nr, nth_s))
        P_b[0] = RHO_BAR; P_b[1] = M_c; P_b[4] = P_BAR

    def to_Q(r_s, u_s, v_s, w_s, p_s):
        E_s = p_s / (GAMMA - 1.0) + 0.5 * r_s * (u_s * u_s + v_s * v_s + w_s * w_s)
        return np.stack([V * r_s, V * r_s * u_s, V * r_s * v_s, V * rb * r_s * w_s, V * E_s], axis=0)

    def from_Q(Qs):
        r_s = np.maximum(Qs[0] / V, 1e-10)
        u_s = Qs[1] / (V * r_s); v_s = Qs[2] / (V * r_s); w_s = Qs[3] / (V * rb * r_s)
        p_s = np.maximum((GAMMA - 1.0) * (Qs[4] / V - 0.5 * r_s * (u_s * u_s + v_s * v_s + w_s * w_s)), 1e-10)
        return r_s, u_s, v_s, w_s, p_s, np.sqrt(GAMMA * p_s / r_s)

    Q = to_Q(P_b[0], P_b[1], P_b[2], P_b[3], P_b[4])

    def rhs_sec(Qs):
        r_s, u_s, v_s, w_s, p_s, _ = from_Q(Qs)
        P = np.stack([r_s, u_s, v_s, w_s, p_s], axis=0)
        Px = np.empty((5, nx + 4, nr, nth_s)); Px[:, 2:-2] = P
        ri, ui, vi, wi, pi = inlet_choked_3d(r_s[0], u_s[0], p_s[0], M_c)
        for k in (0, 1):
            Px[0, k] = ri; Px[1, k] = ui; Px[2, k] = vi; Px[3, k] = wi; Px[4, k] = pi
        if mode == "straight":
            ro, uo, vo, wo, po = outlet_mc_choked_3d(r_s[-1], u_s[-1], v_s[-1], w_s[-1], p_s[-1], M_c)
            for k in (0, 1):
                Px[0, -(k + 1)] = ro; Px[1, -(k + 1)] = uo; Px[2, -(k + 1)] = vo; Px[3, -(k + 1)] = wo; Px[4, -(k + 1)] = po
        else:
            Px[:, -2] = P[:, -1]; Px[:, -1] = P[:, -1]
        sx = van_leer(Px[:, 1:-1] - Px[:, :-2], Px[:, 2:] - Px[:, 1:-1])
        PLx = Px[:, 1:-2] + 0.5 * sx[:, :-1]; PRx = Px[:, 2:-1] - 0.5 * sx[:, 1:]
        Fx = hllc_3d(np.maximum(PLx[0], 1e-10), PLx[1], PLx[2], PLx[3], np.maximum(PLx[4], 1e-10),
                     np.maximum(PRx[0], 1e-10), PRx[1], PRx[2], PRx[3], np.maximum(PRx[4], 1e-10))
        FxQ = np.stack([Sx * Fx[0], Sx * Fx[1], Sx * Fx[2], Sx * r_xf * Fx[3], Sx * Fx[4]], axis=0)

        Pe = np.empty((5, nx, nr + 4, nth_s)); Pe[:, :, 2:-2] = P
        roll_shift = (nth_s // 2) if (n_baffles % 2 == 1) else 0
        opp = np.roll(P[:, :, 0:2, :], roll_shift, axis=3)
        Pe[:, :, 1] = opp[:, :, 0]; Pe[2:4, :, 1] *= -1.0
        Pe[:, :, 0] = opp[:, :, 1]; Pe[2:4, :, 0] *= -1.0
        Pe[:, :, nr + 2] = P[:, :, nr - 1]; Pe[:, :, nr + 3] = P[:, :, nr - 1]
        se = van_leer(Pe[:, :, 1:-1] - Pe[:, :, :-2], Pe[:, :, 2:] - Pe[:, :, 1:-1])
        PLe = Pe[:, :, 1:-2] + 0.5 * se[:, :, :-1]; PRe = Pe[:, :, 2:-1] - 0.5 * se[:, :, 1:]
        vnL = nxe * PLe[1] + nre * PLe[2]; vt1L = -nre * PLe[1] + nxe * PLe[2]
        vnR = nxe * PRe[1] + nre * PRe[2]; vt1R = -nre * PRe[1] + nxe * PRe[2]
        vnR[:, -1] = -vnL[:, -1]; vt1R[:, -1] = vt1L[:, -1]
        PRe[0, :, -1] = PLe[0, :, -1]; PRe[3, :, -1] = PLe[3, :, -1]; PRe[4, :, -1] = PLe[4, :, -1]
        Fe_n = hllc_3d(np.maximum(PLe[0], 1e-10), vnL, vt1L, PLe[3], np.maximum(PLe[4], 1e-10),
                       np.maximum(PRe[0], 1e-10), vnR, vt1R, PRe[3], np.maximum(PRe[4], 1e-10))
        Fe_x = nxe * Fe_n[1] - nre * Fe_n[2]; Fe_r = nre * Fe_n[1] + nxe * Fe_n[2]
        FeQ = np.stack([Sm * Fe_n[0], Sm * Fe_x, Sm * Fe_r, Sm * r_ef * Fe_n[3], Sm * Fe_n[4]], axis=0)
        FeQ[:, :, 0] = 0.0

        Pt = np.concatenate([P[:, :, :, -2:], P, P[:, :, :, :2]], axis=3)
        st = van_leer(Pt[:, :, :, 1:-1] - Pt[:, :, :, :-2], Pt[:, :, :, 2:] - Pt[:, :, :, 1:-1])
        st = np.where(b_sz[None, :, :, :], 0.0, st)
        PLt = Pt[:, :, :, 1:-2] + 0.5 * st[:, :, :, :-1]; PRt = Pt[:, :, :, 2:-1] - 0.5 * st[:, :, :, 1:]
        Gt = hllc_3d(np.maximum(PLt[0], 1e-10), PLt[3], PLt[1], PLt[2], np.maximum(PLt[4], 1e-10),
                     np.maximum(PRt[0], 1e-10), PRt[3], PRt[1], PRt[2], np.maximum(PRt[4], 1e-10))
        GtQ_L = np.stack([S_th * Gt[0], S_th * Gt[2], S_th * Gt[3], S_th * r_thf * Gt[1], S_th * Gt[4]], axis=0)
        GtQ_R = GtQ_L.copy()
        GL_b = hllc_3d(np.maximum(PLt[0], 1e-10), PLt[3], PLt[1], PLt[2], np.maximum(PLt[4], 1e-10),
                       np.maximum(PLt[0], 1e-10), -PLt[3], PLt[1], PLt[2], np.maximum(PLt[4], 1e-10))
        GR_b = hllc_3d(np.maximum(PRt[0], 1e-10), -PRt[3], PRt[1], PRt[2], np.maximum(PRt[4], 1e-10),
                       np.maximum(PRt[0], 1e-10), PRt[3], PRt[1], PRt[2], np.maximum(PRt[4], 1e-10))
        m3 = b_face[None, :, :, :]
        GtQ_L = np.where(m3, np.stack([0.0 * S_th * GL_b[0], S_th * GL_b[2], S_th * GL_b[3], S_th * r_thf * GL_b[1], 0.0 * S_th * GL_b[4]], axis=0), GtQ_L)
        GtQ_R = np.where(m3, np.stack([0.0 * S_th * GR_b[0], S_th * GR_b[2], S_th * GR_b[3], S_th * r_thf * GR_b[1], 0.0 * S_th * GR_b[4]], axis=0), GtQ_R)

        res = -(FxQ[:, 1:] - FxQ[:, :-1]) - (FeQ[:, :, 1:] - FeQ[:, :, :-1]) - (GtQ_L[:, :, :, 1:] - GtQ_R[:, :, :, :-1])
        res[2] += (p_s + r_s * w_s * w_s) * hoop

        u_x = Px[1, 1:-1]; v_x = Px[2, 1:-1]; w_x = Px[3, 1:-1]; T_x = Px[4, 1:-1] / np.maximum(Px[0, 1:-1], 1e-10)
        u_e = Pe[1, :, 1:-1]; v_e = Pe[2, :, 1:-1]; w_e = Pe[3, :, 1:-1]; T_e = Pe[4, :, 1:-1] / np.maximum(Pe[0, :, 1:-1], 1e-10)
        u_t = Pt[1, :, :, 1:-1]; v_t = Pt[2, :, :, 1:-1]; w_t = Pt[3, :, :, 1:-1]; T_t = Pt[4, :, :, 1:-1] / np.maximum(Pt[0, :, :, 1:-1], 1e-10)

        du_dr_c = (u_e[:, 2:] - u_e[:, :-2]) / (2.0 * dr_c); dv_dr_c = (v_e[:, 2:] - v_e[:, :-2]) / (2.0 * dr_c)
        dw_dr_c = (w_e[:, 2:] - w_e[:, :-2]) / (2.0 * dr_c); dT_dr_c = (T_e[:, 2:] - T_e[:, :-2]) / (2.0 * dr_c)
        du_dx_c = (u_x[2:] - u_x[:-2]) / (2.0 * dx) - eta_Rp_c * du_dr_c
        dv_dx_c = (v_x[2:] - v_x[:-2]) / (2.0 * dx) - eta_Rp_c * dv_dr_c
        dw_dx_c = (w_x[2:] - w_x[:-2]) / (2.0 * dx) - eta_Rp_c * dw_dr_c
        dT_dx_c = (T_x[2:] - T_x[:-2]) / (2.0 * dx) - eta_Rp_c * dT_dr_c
        du_dt_c = (u_t[:, :, 2:] - u_t[:, :, :-2]) / (2.0 * dth)
        dv_dt_c = (v_t[:, :, 2:] - v_t[:, :, :-2]) / (2.0 * dth)
        dw_dt_c = (w_t[:, :, 2:] - w_t[:, :, :-2]) / (2.0 * dth)
        mL_b = b_face[:, :, :-1]; mR_b = b_face[:, :, 1:]
        du_dt_c = np.where(mL_b, (u_t[:, :, 2:] - u_t[:, :, 1:-1]) / dth, np.where(mR_b, (u_t[:, :, 1:-1] - u_t[:, :, :-2]) / dth, du_dt_c))
        dv_dt_c = np.where(mL_b, (v_t[:, :, 2:] - v_t[:, :, 1:-1]) / dth, np.where(mR_b, (v_t[:, :, 1:-1] - v_t[:, :, :-2]) / dth, dv_dt_c))
        dw_dt_c = np.where(mL_b, (w_t[:, :, 2:] - w_t[:, :, 1:-1]) / dth, np.where(mR_b, (w_t[:, :, 1:-1] - w_t[:, :, :-2]) / dth, dw_dt_c))

        def avg_x(arr):
            out = np.empty((nx + 1, nr, nth_s)); out[1:-1] = 0.5 * (arr[:-1] + arr[1:]); out[0] = arr[0]; out[-1] = arr[-1]; return out
        du_dr_xf = avg_x(du_dr_c); dv_dr_xf = avg_x(dv_dr_c); dw_dr_xf = avg_x(dw_dr_c); dT_dr_xf = avg_x(dT_dr_c)
        du_dx_f = (u_x[1:] - u_x[:-1]) / dx - eta_Rp_xf * du_dr_xf
        dv_dx_f = (v_x[1:] - v_x[:-1]) / dx - eta_Rp_xf * dv_dr_xf
        dw_dx_f = (w_x[1:] - w_x[:-1]) / dx - eta_Rp_xf * dw_dr_xf
        dT_dx_f = (T_x[1:] - T_x[:-1]) / dx - eta_Rp_xf * dT_dr_xf
        div_perp_xf = avg_x(dv_dr_c + (v_s + dw_dt_c) / rb)
        du_dt_xf = avg_x(du_dt_c)
        uf_x = 0.5 * (u_x[:-1] + u_x[1:]); vf_x = 0.5 * (v_x[:-1] + v_x[1:]); wf_x = 0.5 * (w_x[:-1] + w_x[1:])
        txx_xf = mu * ((4.0 / 3.0) * du_dx_f - (2.0 / 3.0) * div_perp_xf)
        txr_xf = mu * (du_dr_xf + dv_dx_f); txt_xf = mu * (du_dt_xf / r_xf + dw_dx_f); qx_xf = -k_th * dT_dx_f
        res[1] += Sx[1:] * txx_xf[1:] - Sx[:-1] * txx_xf[:-1]
        res[2] += Sx[1:] * txr_xf[1:] - Sx[:-1] * txr_xf[:-1]
        res[3] += (Sx * r_xf * txt_xf)[1:] - (Sx * r_xf * txt_xf)[:-1]
        VFx_e = Sx * (uf_x * txx_xf + vf_x * txr_xf + wf_x * txt_xf - qx_xf)
        res[4] += VFx_e[1:] - VFx_e[:-1]

        du_dr_ef = (u_e[:, 1:] - u_e[:, :-1]) / dr_c; dv_dr_ef = (v_e[:, 1:] - v_e[:, :-1]) / dr_c
        dw_dr_ef = (w_e[:, 1:] - w_e[:, :-1]) / dr_c; dT_dr_ef = (T_e[:, 1:] - T_e[:, :-1]) / dr_c
        def avg_e(arr):
            out = np.zeros((nx, nr + 1, nth_s)); out[:, 1:-1] = 0.5 * (arr[:, :-1] + arr[:, 1:]); return out
        du_dx_ef = avg_e(du_dx_c); dv_dx_ef = avg_e(dv_dx_c); dw_dx_ef = avg_e(dw_dx_c); dT_dx_ef = avg_e(dT_dx_c)
        du_dt_ef = avg_e(du_dt_c); dv_dt_ef = avg_e(dv_dt_c); dw_dt_ef = avg_e(dw_dt_c)
        uf_e = 0.5 * (u_e[:, :-1] + u_e[:, 1:]); vf_e = 0.5 * (v_e[:, :-1] + v_e[:, 1:]); wf_e = 0.5 * (w_e[:, :-1] + w_e[:, 1:])
        inv_ref = np.where(r_ef > 1e-12, 1.0 / np.where(r_ef > 1e-12, r_ef, 1.0), 0.0)
        div_ef = du_dx_ef + dv_dr_ef + (vf_e + dw_dt_ef) * inv_ref
        txx_ef = mu * (2.0 * du_dx_ef - (2.0 / 3.0) * div_ef); trr_ef = mu * (2.0 * dv_dr_ef - (2.0 / 3.0) * div_ef)
        txr_ef = mu * (du_dr_ef + dv_dx_ef); txt_ef = mu * (du_dt_ef * inv_ref + dw_dx_ef)
        trt_ef = mu * (dw_dr_ef - wf_e * inv_ref + dv_dt_ef * inv_ref)
        qx_ef = -k_th * dT_dx_ef; qr_ef = -k_th * dT_dr_ef
        fx_v = nxe * txx_ef + nre * txr_ef; fr_v = nxe * txr_ef + nre * trr_ef; ft_v = nxe * txt_ef + nre * trt_ef
        fe_v = uf_e * fx_v + vf_e * fr_v + wf_e * ft_v - (nxe * qx_ef + nre * qr_ef)
        VFe_u = Sm * fx_v; VFe_v = Sm * fr_v; VFe_w = Sm * r_ef * ft_v; VFe_e = Sm * fe_v
        VFe_u[:, 0] = 0.0; VFe_u[:, -1] = 0.0; VFe_v[:, 0] = 0.0; VFe_v[:, -1] = 0.0
        VFe_w[:, 0] = 0.0; VFe_w[:, -1] = 0.0; VFe_e[:, 0] = 0.0; VFe_e[:, -1] = 0.0
        res[1] += VFe_u[:, 1:] - VFe_u[:, :-1]; res[2] += VFe_v[:, 1:] - VFe_v[:, :-1]
        res[3] += VFe_w[:, 1:] - VFe_w[:, :-1]; res[4] += VFe_e[:, 1:] - VFe_e[:, :-1]

        du_dt_tf = (u_t[:, :, 1:] - u_t[:, :, :-1]) / dth; dv_dt_tf = (v_t[:, :, 1:] - v_t[:, :, :-1]) / dth
        dw_dt_tf = (w_t[:, :, 1:] - w_t[:, :, :-1]) / dth; dT_dt_tf = (T_t[:, :, 1:] - T_t[:, :, :-1]) / dth
        def avg_t(arr):
            a_ext = np.concatenate([arr[:, :, -1:], arr, arr[:, :, :1]], axis=2)
            return 0.5 * (a_ext[:, :, :-1] + a_ext[:, :, 1:])
        du_dx_tf = avg_t(du_dx_c); dv_dr_tf = avg_t(dv_dr_c); dw_dr_tf = avg_t(dw_dr_c); dw_dx_tf = avg_t(dw_dx_c)
        uf_t = 0.5 * (u_t[:, :, :-1] + u_t[:, :, 1:]); vf_t = 0.5 * (v_t[:, :, :-1] + v_t[:, :, 1:]); wf_t = 0.5 * (w_t[:, :, :-1] + w_t[:, :, 1:])
        inv_rt = 1.0 / r_thf
        div_tf = du_dx_tf + dv_dr_tf + (vf_t + dw_dt_tf) * inv_rt
        txt_tf = mu * (du_dt_tf * inv_rt + dw_dx_tf); trt_tf = mu * (dw_dr_tf - wf_t * inv_rt + dv_dt_tf * inv_rt)
        ttt_tf = mu * (2.0 * (dw_dt_tf + vf_t) * inv_rt - (2.0 / 3.0) * div_tf); qt_tf = -k_th * dT_dt_tf * inv_rt
        VG_uL = S_th * txt_tf; VG_vL = S_th * trt_tf; VG_wL = S_th * r_thf * ttt_tf
        VG_eL = S_th * (uf_t * txt_tf + vf_t * trt_tf + wf_t * ttt_tf - qt_tf)
        VG_uR, VG_vR, VG_wR, VG_eR = VG_uL.copy(), VG_vL.copy(), VG_wL.copy(), VG_eL.copy()
        tau_xL = mu * (-u_t[:, :, :-1]) / (0.5 * r_thf * dth)
        tau_xR = mu * (u_t[:, :, 1:]) / (0.5 * r_thf * dth)
        tau_rL = mu * (-v_t[:, :, :-1]) / (0.5 * r_thf * dth)
        tau_rR = mu * (v_t[:, :, 1:]) / (0.5 * r_thf * dth)
        VG_uL = np.where(b_face, S_th * tau_xL, VG_uL); VG_uR = np.where(b_face, S_th * tau_xR, VG_uR)
        VG_vL = np.where(b_face, S_th * tau_rL, VG_vL); VG_vR = np.where(b_face, S_th * tau_rR, VG_vR)
        VG_wL = np.where(b_face, 0.0, VG_wL); VG_wR = np.where(b_face, 0.0, VG_wR)
        VG_eL = np.where(b_face, 0.0, VG_eL); VG_eR = np.where(b_face, 0.0, VG_eR)
        res[1] += VG_uL[:, :, 1:] - VG_uR[:, :, :-1]; res[2] += VG_vL[:, :, 1:] - VG_vR[:, :, :-1]
        res[3] += VG_wL[:, :, 1:] - VG_wR[:, :, :-1]; res[4] += VG_eL[:, :, 1:] - VG_eR[:, :, :-1]
        div_c = du_dx_c + dv_dr_c + (v_s + dw_dt_c) / rb
        res[2] -= mu * (2.0 * (dw_dt_c + v_s) / rb - (2.0 / 3.0) * div_c) * hoop
        return res

    dr_eff = deta * g["Rm"][:, None, None]
    last_rhs = 1.0
    for it in range(1, max_iter + 1):
        r_c, u_c, v_c, w_c, p_c, c_c = from_Q(Q)
        lam_adv = (np.abs(u_c) + c_c) / dx + (np.abs(v_c) + c_c) / dr_eff + (np.abs(w_c) + c_c) / (rb * dth)
        nu_eff = max(4.0 / 3.0, GAMMA / PR) * mu / r_c
        lam_visc = nu_eff * (1.0 / (dx * dx) + 1.0 / (dr_eff * dr_eff) + 1.0 / ((rb * dth) ** 2))
        dt_loc = 1.0 / (lam_adv / 0.25 + lam_visc / 0.38)
        k1 = rhs_sec(Q); Q1 = Q + dt_loc * k1
        k2 = rhs_sec(Q1); Q = 0.5 * (Q + Q1 + dt_loc * k2)
        if it % 250 == 0:
            last_rhs = float(np.max(np.abs(k1 / V)))
            if last_rhs < tol:
                break

    r_c, u_c, v_c, w_c, p_c, _ = from_Q(Q)
    P_sec = np.stack([r_c, u_c, v_c, w_c, p_c], axis=0)
    P_full = np.tile(P_sec, (1, 1, 1, n_baffles))

    # 3D wake & core-bypass diagnostics inside chamber (x <= L_A)
    ch = xc <= L_A
    ix_tip = int(np.argmin(np.abs(xc - L_b)))
    ix_ds = int(np.argmin(np.abs(xc - (L_b + 0.15))))
    u_ch = u_c[ch]; r_ch = r_c[ch]; v_ch = v_c[ch]; p_ch = p_c[ch]
    V_s = np.broadcast_to(V[ch], u_ch.shape)
    shroud_m = etac >= r_b
    core_m = etac < r_b
    u_shroud_min = float(np.min(u_ch[:, shroud_m, :]))
    u_ds_min = float(np.min(u_c[ix_ds, shroud_m, :]))
    u_tip_min = float(np.min(u_c[ix_tip, shroud_m, :]))
    u_shroud_mean = float(np.sum(u_ch[:, shroud_m, :] * V_s[:, shroud_m, :]) / np.sum(V_s[:, shroud_m, :]))
    if np.any(core_m):
        u_core_max = float(np.max(u_ch[:, core_m, :]))
        u_core_tip_mean = float(np.sum(u_c[ix_tip, core_m, :] * Sx[ix_tip, core_m, :]) / (np.sum(Sx[ix_tip, core_m, :]) * nth_s))
        u_core_mean = float(np.sum(u_ch[:, core_m, :] * V_s[:, core_m, :]) / np.sum(V_s[:, core_m, :]))
        m_core_in = float(np.sum(r_c[0, core_m, :] * u_c[0, core_m, :] * Sx[0, core_m, :]))
        m_core_tip = float(np.sum(r_c[ix_tip, core_m, :] * u_c[ix_tip, core_m, :] * Sx[ix_tip, core_m, :]))
        core_bypass_pct = float((m_core_tip - m_core_in) / max(m_core_in, 1e-12) * 100.0)
        # Reference axisymmetric core mass flux at x=L_b (for curved bell, separate geometric contraction!)
        if prim_axi_base is not None:
            m_core_tip_axi = float(np.sum(prim_axi_base[0, ix_tip, core_m, None] * prim_axi_base[1, ix_tip, core_m, None] * Sx[ix_tip, core_m, :])) * nth_s
            core_bypass_vs_axi_pct = float((m_core_tip - m_core_tip_axi) / max(m_core_tip_axi, 1e-12) * 100.0)
        else:
            core_bypass_vs_axi_pct = core_bypass_pct
    else:
        u_core_max = float(np.max(u_ch)); u_core_tip_mean = float(np.mean(u_c[ix_tip])); u_core_mean = u_shroud_mean
        core_bypass_pct = 0.0; core_bypass_vs_axi_pct = 0.0

    # Save 2D slices for plotting: (x, r) mid-sector & near-baffle, and (r, theta) at x=L_b+0.05
    u_sec_2d_xr_wall = u_c[:, :, 0].copy()
    u_sec_2d_xr_mid = u_c[:, :, nth_s // 2].copy()
    u_sec_2d_rt_tip = u_c[ix_tip, :, :].copy()
    v_sec_2d_xr_mid = v_c[:, :, nth_s // 2].copy()

    return {
        "prim_3d": P_full,
        "rhs_max": last_rhs,
        "iters": it,
        "u_min_ch": float(np.min(u_ch)),
        "u_shroud_min": u_shroud_min,
        "u_tip_min": u_tip_min,
        "u_ds_min": u_ds_min,
        "u_shroud_mean": u_shroud_mean,
        "u_core_mean": u_core_mean,
        "u_core_tip_mean": u_core_tip_mean,
        "u_core_max": u_core_max,
        "core_bypass_pct": core_bypass_pct,
        "core_bypass_vs_axi_pct": core_bypass_vs_axi_pct,
        "u_xr_wall": u_sec_2d_xr_wall,
        "u_xr_mid": u_sec_2d_xr_mid,
        "v_xr_mid": v_sec_2d_xr_mid,
        "u_rt_tip": u_sec_2d_rt_tip,
        "xc": xc, "etac": etac, "th_sec": (np.arange(nth_s) + 0.5) * dth,
    }


# ---------------------------------------------------------------------------
# 3. Full 3D Cylindrical NS Solver (Slip & Adiabatic No-Slip Baffles + Feed + Sector-Clipped Cavities)
# ---------------------------------------------------------------------------

def simulate_3d_ns(
    nx: int = 20, nr: int = 12, nth: int = 40, t_end: float = 30.0,
    M_c: float = 0.12, L_n: float = 0.60, L_d: float = 0.45, mode: str = "straight",
    Re: float = 500.0,
    init_type: str = "spin_1T", A_eq: float = 1e-4,
    n_p: float = 0.0, omega_hp: float = 0.50,
    feed_spec: dict | None = None,
    n_baffles: int = 0, L_b: float = 0.45, r_b: float = 0.42, baffle_noslip: bool = False,
    cav_spec: list[dict] | None = None,
    prim_base: np.ndarray | None = None, cfl: float = 0.28, record_stride: int = 14,
) -> dict:
    """Metric-corrected 3D cylindrical NS solver supporting:
    - 2D axisymmetric (5, nx, nr) OR full 3D wake (5, nx, nr, nth) steady base flows
    - Slip or adiabatic no-slip partial/full-span baffles
    - Exact equilibrium ghost-inlet pressure baseline for cavities and Crocco-Cheng feed ODE
    - Optional sector-clipped wide-mouth cavity port footprints (|dtheta| < pi / n_cav)
    - Complex modal amplitudes (c1_re, c2_re) for exact FFT frequency measurement."""
    g = build_3d_geom(nx, nr, nth, M_c=M_c, L_n=L_n, L_d=L_d, mode=mode)
    V, rb = g["V"], g["r_bar"]
    xc, etac, thc, dx, deta, dth = g["xc"], g["etac"], g["thc"], g["dx"], g["deta"], g["dth"]
    X3, E3, T3 = np.meshgrid(xc, etac, thc, indexing="ij")
    ch_mask = xc <= L_A
    ch_env = np.where(X3 <= L_A, 1.0, 0.0)
    mu = 0.0 if not np.isfinite(Re) else (1.0 / Re)
    k_th = 0.0 if not np.isfinite(Re) else (GAMMA / (PR * (GAMMA - 1.0) * Re))

    V_full = np.broadcast_to(V, (nx, nr, nth))
    V_ch = V_full[ch_mask]
    w_ch = V_ch / float(np.sum(V_ch))
    X_ch, E_ch, T_ch = X3[ch_mask], E3[ch_mask], T3[ch_mask]
    phi_1T = jv(1, S11 * E_ch) / (np.sqrt(0.5 * (1.0 - 1.0 / S11**2)) * abs(jv(1, S11)))
    phi_2T = jv(2, S21 * E_ch) / (np.sqrt(0.5 * (1.0 - 4.0 / S21**2)) * abs(jv(2, S21)))
    phi_1R = jv(0, S02 * E_ch) / abs(jv(0, S02))
    phi_1L = np.sqrt(2.0) * np.cos(np.pi * X_ch / L_A)

    has_baffles = (n_baffles > 0) and (L_b > 0.0) and (r_b < 0.999)
    baffle_th_face = np.zeros((nx, nr, nth + 1), dtype=bool)
    baffle_slope_zero = np.zeros((nx, nr, nth + 2), dtype=bool)
    if has_baffles:
        ix_b = xc <= L_b; jr_b = etac >= r_b
        b_xr = ix_b[:, None] & jr_b[None, :]
        for kf in [int(round(k * nth / n_baffles)) % nth for k in range(n_baffles)]:
            baffle_th_face[b_xr, kf] = True
            if kf == 0:
                baffle_th_face[b_xr, nth] = True
            k_left = (kf - 1) % nth; k_right = kf
            baffle_slope_zero[b_xr, k_left + 1] = True
            baffle_slope_zero[b_xr, k_right + 1] = True
            if k_left == nth - 1:
                baffle_slope_zero[b_xr, 0] = True
            if k_right == 0:
                baffle_slope_zero[b_xr, nth + 1] = True

    if prim_base is not None:
        if prim_base.ndim == 3:
            P_b = np.broadcast_to(prim_base[:, :, :, None], (5, nx, nr, nth)).copy()
        else:
            P_b = prim_base.copy()
    else:
        ux0 = 0.0 if mode == "closed" or init_type == "quiescent" else M_c
        P_b = np.zeros((5, nx, nr, nth))
        P_b[0] = RHO_BAR; P_b[1] = ux0; P_b[4] = P_BAR
    rho_b, ux_b, ur_b, ut_b, p_b = P_b[0], P_b[1], P_b[2], P_b[3], P_b[4]
    rho = rho_b.copy(); ux = ux_b.copy(); ur = ur_b.copy(); ut = ut_b.copy(); p = p_b.copy()

    if init_type in ("quiescent", "steady_mean"):
        pass
    elif init_type == "periodic_3d_wave":
        dp = A_eq * (np.cos(2.0 * np.pi * X3 / g["L_tot"] + T3) * jv(1, S11 * E3) + 0.4 * jv(0, S02 * E3))
        rho += dp; p += dp
        ux += 0.2 * A_eq * np.sin(2.0 * np.pi * X3 / g["L_tot"])
        ur += 0.3 * A_eq * E3 * (1.0 - E3) * np.cos(T3)
        ut += 0.4 * A_eq * E3 * np.sin(T3)
    elif init_type == "bulk_k0":
        dp = A_eq * ch_env
        rho += dp; p += dp
    elif init_type in ("spin_1T", "stand_1T"):
        j1 = jv(1, S11 * E3) / abs(jv(1, S11))
        j1p = jvp(1, S11 * E3) / abs(jv(1, S11))
        if init_type == "spin_1T":
            dp = A_eq * j1 * np.cos(T3) * ch_env
            ur += A_eq * j1p * np.sin(T3) * ch_env
            ut += A_eq * (j1 / np.maximum(S11 * E3, 1e-12)) * np.cos(T3) * ch_env
        else:
            dp = A_eq * np.sqrt(2.0) * j1 * np.cos(T3) * ch_env
        rho += dp; p += dp
    elif init_type == "spin_2T":
        j2 = jv(2, S21 * E3) / abs(jv(2, S21))
        j2p = jvp(2, S21 * E3) / abs(jv(2, S21))
        dp = A_eq * j2 * np.cos(2.0 * T3) * ch_env
        ur += A_eq * j2p * np.sin(2.0 * T3) * ch_env
        ut += A_eq * (2.0 * j2 / np.maximum(S21 * E3, 1e-12)) * np.cos(2.0 * T3) * ch_env
        rho += dp; p += dp
    elif init_type == "1R":
        dp = A_eq * jv(0, S02 * E3) * ch_env
        rho += dp; p += dp
    elif init_type == "1L":
        dp = A_eq * np.cos(np.pi * X3 / L_A) * ch_env
        rho += dp; p += dp
    else:
        raise ValueError(f"Unknown init_type={init_type}")

    # Exact steady ghost-inlet pressure baseline so cavities & feed ODE vanish identically at P_b!
    dA_in = g["dA_in"]
    A_in_tot = float(np.sum(dA_in))
    if mode in ("straight", "bell_smooth"):
        _, _, _, _, p_in_base = inlet_choked_3d(rho_b[0], ux_b[0], p_b[0], M_c)
    else:
        p_in_base = p_b[0].copy()
    p_face_base = float(np.sum(p_in_base * dA_in) / A_in_tot)

    has_feed = feed_spec is not None and np.isfinite(feed_spec.get("psi", np.inf))
    feed_state = np.full(5, M_c, dtype=float)
    f_psi = float(feed_spec["psi"]) if has_feed else np.inf
    f_tau_inj = float(feed_spec.get("tau_inj", TAU_INJ_DEF)) if has_feed else 1.0
    f_tau_f = float(feed_spec.get("tau_f", TAU_F_DEF)) if has_feed else 5.0
    f_nl = bool(feed_spec.get("nonlinear", True)) if has_feed else True
    f_check_valve = bool(feed_spec.get("check_valve", True)) if has_feed else True

    cavs = []
    if cav_spec:
        Cd = 0.65
        E_in, T_in = np.meshgrid(etac, thc, indexing="ij")
        for cf in cav_spec:
            nc = int(cf.get("n_cav", 5))
            beta_k = float(cf["beta"]); wH = float(cf["wH"]); xiH = float(cf["xiH"])
            sig_o = float(cf.get("sigma_o", 0.05)); nl_drag = bool(cf.get("nonlinear", True))
            r_lo, r_hi = float(cf.get("r_lo", 0.0)), float(cf.get("r_hi", 1.0))
            sector_clip = bool(cf.get("sector_clip", False))
            V_k = beta_k * (np.pi * L_A) / nc
            A_ok = (sig_o * np.pi) / nc
            r_nl = (1.0 / (2.0 * (Cd ** 2) * (A_ok ** 2))) if nl_drag else 0.0
            W_ports = np.zeros((nc, nr, nth))
            r_mask = ((E_in >= r_lo) & (E_in <= r_hi)).astype(float)
            if np.sum(r_mask * dA_in) <= 0.0:
                j_near = int(np.argmin(np.abs(etac - 0.5 * (r_lo + r_hi))))
                r_mask[j_near, :] = 1.0
            if nc == 1:
                W_ports[0] = r_mask / float(np.sum(r_mask * dA_in))
            else:
                if "sig_th" in cf:
                    sig_th = float(cf["sig_th"])
                elif sector_clip:
                    sig_th = max(1.5 * dth, 0.22 * (2.0 * np.pi / nc))
                else:
                    sig_th = max(1.5 * dth, 0.45 * (2.0 * np.pi / nc) * min(1.0, sig_o * 4.0))
                half_sec = np.pi / nc
                for k in range(nc):
                    th_k = (k + 0.5) * (2.0 * np.pi / nc)
                    d_ang = np.angle(np.exp(1j * (T_in - th_k)))
                    wk = r_mask * np.exp(-0.5 * (d_ang / sig_th) ** 2)
                    if sector_clip:
                        wk = np.where(np.abs(d_ang) < half_sec, wk, 0.0)
                    W_ports[k] = wk / float(np.sum(wk * dA_in))
            cavs.append({"V_k": V_k, "wH": wH, "xiH": xiH, "r_nl": r_nl, "W": W_ports, "Pc": np.zeros(nc), "qt": np.zeros(nc)})

    has_flame = abs(n_p) > 0.0
    wf = np.where(ch_mask, np.exp(-0.5 * ((xc - X_F) / SIG_F) ** 2), 0.0)
    wf /= float(np.sum(wf) * dx)
    wf_3d = wf[:, None, None]
    fp = np.zeros((5, nr, nth))
    inv_ts = 4.0 / TAU_BAR

    def to_Q(r_s, u_s, v_s, w_s, p_s):
        E_s = p_s / (GAMMA - 1.0) + 0.5 * r_s * (u_s * u_s + v_s * v_s + w_s * w_s)
        return np.stack([V * r_s, V * r_s * u_s, V * r_s * v_s, V * rb * r_s * w_s, V * E_s], axis=0)

    def from_Q(Qs):
        r_s = np.maximum(Qs[0] / V, 1e-10)
        u_s = Qs[1] / (V * r_s); v_s = Qs[2] / (V * r_s); w_s = Qs[3] / (V * rb * r_s)
        p_s = np.maximum((GAMMA - 1.0) * (Qs[4] / V - 0.5 * r_s * (u_s * u_s + v_s * v_s + w_s * w_s)), 1e-10)
        return r_s, u_s, v_s, w_s, p_s, np.sqrt(GAMMA * p_s / r_s)

    Q = to_Q(rho, ux, ur, ut, p); Q0 = Q.copy()
    init_tot = np.array([float(np.sum(Q[k])) for k in range(5)])
    max_cons = np.zeros(5)

    dr_c = deta * g["Rm"][:, None, None]
    r_ef = g["r_ef"]; r_xf = g["r_xf"]; r_thf = g["r_thf"]
    nxe, nre = g["nxe"], g["nre"]
    Sx, Sm, S_th, hoop = g["Sx"], g["Sm"], g["S_th"], g["hoop"]
    eta_Rp_c, eta_Rp_xf = g["eta_Rp_c"], g["eta_Rp_xf"]

    def compute_rhs(Qs, fp_s, cav_states, f_s):
        r_s, u_s, v_s, w_s, p_s, _ = from_Q(Qs)
        P = np.stack([r_s, u_s, v_s, w_s, p_s], axis=0)

        port_flux = None
        if cavs:
            port_flux = np.zeros((nr, nth))
            for (_, qt_c), cf in zip(cav_states, cavs):
                port_flux += np.tensordot(qt_c, cf["W"], axes=(0, 0))

        if has_feed:
            floor_val = 0.0 if f_check_valve else (0.05 * M_c)
            m_in_now = float(np.maximum(f_s[4], floor_val))
        else:
            m_in_now = M_c

        Px = np.empty((5, nx + 4, nr, nth)); Px[:, 2:-2] = P
        p_in_face = p_s[0]
        if mode == "periodic_bump":
            Px[:, 0:2] = P[:, -2:]; Px[:, -2:] = P[:, 0:2]
        elif mode == "closed" or init_type == "quiescent":
            for k in (0, 1):
                Px[:, k] = P[:, 0]; Px[1, k] = -P[1, 0]
                Px[:, -(k + 1)] = P[:, -1]; Px[1, -(k + 1)] = -P[1, -1]
        elif mode == "straight":
            ri, ui, vi, wi, pi = inlet_choked_3d(r_s[0], u_s[0], p_s[0], m_in_now, m_port=port_flux)
            p_in_face = pi
            ro, uo, vo, wo, po = outlet_mc_choked_3d(r_s[-1], u_s[-1], v_s[-1], w_s[-1], p_s[-1], M_c)
            for k in (0, 1):
                Px[0, k] = ri; Px[1, k] = ui; Px[2, k] = vi; Px[3, k] = wi; Px[4, k] = pi
                Px[0, -(k + 1)] = ro; Px[1, -(k + 1)] = uo; Px[2, -(k + 1)] = vo; Px[3, -(k + 1)] = wo; Px[4, -(k + 1)] = po
        else:
            ri, ui, vi, wi, pi = inlet_choked_3d(r_s[0], u_s[0], p_s[0], m_in_now, m_port=port_flux)
            p_in_face = pi
            for k in (0, 1):
                Px[0, k] = ri; Px[1, k] = ui; Px[2, k] = vi; Px[3, k] = wi; Px[4, k] = pi
            Px[:, -2] = P[:, -1]; Px[:, -1] = P[:, -1]

        sx = van_leer(Px[:, 1:-1] - Px[:, :-2], Px[:, 2:] - Px[:, 1:-1])
        PLx = Px[:, 1:-2] + 0.5 * sx[:, :-1]; PRx = Px[:, 2:-1] - 0.5 * sx[:, 1:]
        if mode == "closed" or init_type == "quiescent":
            PLx[0, 0] = PRx[0, 0]; PLx[1, 0] = -PRx[1, 0]; PLx[2, 0] = PRx[2, 0]; PLx[3, 0] = PRx[3, 0]; PLx[4, 0] = PRx[4, 0]
            PRx[0, -1] = PLx[0, -1]; PRx[1, -1] = -PLx[1, -1]; PRx[2, -1] = PLx[2, -1]; PRx[3, -1] = PLx[3, -1]; PRx[4, -1] = PLx[4, -1]
        Fx = hllc_3d(np.maximum(PLx[0], 1e-10), PLx[1], PLx[2], PLx[3], np.maximum(PLx[4], 1e-10),
                     np.maximum(PRx[0], 1e-10), PRx[1], PRx[2], PRx[3], np.maximum(PRx[4], 1e-10))
        FxQ = np.stack([Sx * Fx[0], Sx * Fx[1], Sx * Fx[2], Sx * r_xf * Fx[3], Sx * Fx[4]], axis=0)

        Pe = np.empty((5, nx, nr + 4, nth)); Pe[:, :, 2:-2] = P
        opp = np.roll(P[:, :, 0:2, :], nth // 2, axis=3)
        Pe[:, :, 1] = opp[:, :, 0]; Pe[2:4, :, 1] *= -1.0
        Pe[:, :, 0] = opp[:, :, 1]; Pe[2:4, :, 0] *= -1.0
        Pe[:, :, nr + 2] = P[:, :, nr - 1]; Pe[:, :, nr + 3] = P[:, :, nr - 1]
        se = van_leer(Pe[:, :, 1:-1] - Pe[:, :, :-2], Pe[:, :, 2:] - Pe[:, :, 1:-1])
        PLe = Pe[:, :, 1:-2] + 0.5 * se[:, :, :-1]; PRe = Pe[:, :, 2:-1] - 0.5 * se[:, :, 1:]
        vnL = nxe * PLe[1] + nre * PLe[2]; vt1L = -nre * PLe[1] + nxe * PLe[2]
        vnR = nxe * PRe[1] + nre * PRe[2]; vt1R = -nre * PRe[1] + nxe * PRe[2]
        vnR[:, -1] = -vnL[:, -1]; vt1R[:, -1] = vt1L[:, -1]
        PRe[0, :, -1] = PLe[0, :, -1]; PRe[3, :, -1] = PLe[3, :, -1]; PRe[4, :, -1] = PLe[4, :, -1]
        Fe_n = hllc_3d(np.maximum(PLe[0], 1e-10), vnL, vt1L, PLe[3], np.maximum(PLe[4], 1e-10),
                       np.maximum(PRe[0], 1e-10), vnR, vt1R, PRe[3], np.maximum(PRe[4], 1e-10))
        Fe_x = nxe * Fe_n[1] - nre * Fe_n[2]; Fe_r = nre * Fe_n[1] + nxe * Fe_n[2]
        FeQ = np.stack([Sm * Fe_n[0], Sm * Fe_x, Sm * Fe_r, Sm * r_ef * Fe_n[3], Sm * Fe_n[4]], axis=0)
        FeQ[:, :, 0] = 0.0

        Pt = np.concatenate([P[:, :, :, -2:], P, P[:, :, :, :2]], axis=3)
        st = van_leer(Pt[:, :, :, 1:-1] - Pt[:, :, :, :-2], Pt[:, :, :, 2:] - Pt[:, :, :, 1:-1])
        if has_baffles:
            st = np.where(baffle_slope_zero[None, :, :, :], 0.0, st)
        PLt = Pt[:, :, :, 1:-2] + 0.5 * st[:, :, :, :-1]; PRt = Pt[:, :, :, 2:-1] - 0.5 * st[:, :, :, 1:]
        Gt = hllc_3d(np.maximum(PLt[0], 1e-10), PLt[3], PLt[1], PLt[2], np.maximum(PLt[4], 1e-10),
                     np.maximum(PRt[0], 1e-10), PRt[3], PRt[1], PRt[2], np.maximum(PRt[4], 1e-10))
        GtQ_L = np.stack([S_th * Gt[0], S_th * Gt[2], S_th * Gt[3], S_th * r_thf * Gt[1], S_th * Gt[4]], axis=0)
        GtQ_R = GtQ_L.copy()
        if has_baffles:
            GL_b = hllc_3d(np.maximum(PLt[0], 1e-10), PLt[3], PLt[1], PLt[2], np.maximum(PLt[4], 1e-10),
                           np.maximum(PLt[0], 1e-10), -PLt[3], PLt[1], PLt[2], np.maximum(PLt[4], 1e-10))
            GR_b = hllc_3d(np.maximum(PRt[0], 1e-10), -PRt[3], PRt[1], PRt[2], np.maximum(PRt[4], 1e-10),
                           np.maximum(PRt[0], 1e-10), PRt[3], PRt[1], PRt[2], np.maximum(PRt[4], 1e-10))
            m3 = baffle_th_face[None, :, :, :]
            GtQ_L = np.where(m3, np.stack([0.0 * S_th * GL_b[0], S_th * GL_b[2], S_th * GL_b[3], S_th * r_thf * GL_b[1], 0.0 * S_th * GL_b[4]], axis=0), GtQ_L)
            GtQ_R = np.where(m3, np.stack([0.0 * S_th * GR_b[0], S_th * GR_b[2], S_th * GR_b[3], S_th * r_thf * GR_b[1], 0.0 * S_th * GR_b[4]], axis=0), GtQ_R)

        res = -(FxQ[:, 1:] - FxQ[:, :-1]) - (FeQ[:, :, 1:] - FeQ[:, :, :-1]) - (GtQ_L[:, :, :, 1:] - GtQ_R[:, :, :, :-1])
        res[2] += (p_s + r_s * w_s * w_s) * hoop

        if mu > 0.0:
            u_x = Px[1, 1:-1]; v_x = Px[2, 1:-1]; w_x = Px[3, 1:-1]; T_x = Px[4, 1:-1] / np.maximum(Px[0, 1:-1], 1e-10)
            u_e = Pe[1, :, 1:-1]; v_e = Pe[2, :, 1:-1]; w_e = Pe[3, :, 1:-1]; T_e = Pe[4, :, 1:-1] / np.maximum(Pe[0, :, 1:-1], 1e-10)
            u_t = Pt[1, :, :, 1:-1]; v_t = Pt[2, :, :, 1:-1]; w_t = Pt[3, :, :, 1:-1]; T_t = Pt[4, :, :, 1:-1] / np.maximum(Pt[0, :, :, 1:-1], 1e-10)

            du_dr_c = (u_e[:, 2:] - u_e[:, :-2]) / (2.0 * dr_c); dv_dr_c = (v_e[:, 2:] - v_e[:, :-2]) / (2.0 * dr_c)
            dw_dr_c = (w_e[:, 2:] - w_e[:, :-2]) / (2.0 * dr_c); dT_dr_c = (T_e[:, 2:] - T_e[:, :-2]) / (2.0 * dr_c)
            du_dx_c = (u_x[2:] - u_x[:-2]) / (2.0 * dx) - eta_Rp_c * du_dr_c
            dv_dx_c = (v_x[2:] - v_x[:-2]) / (2.0 * dx) - eta_Rp_c * dv_dr_c
            dw_dx_c = (w_x[2:] - w_x[:-2]) / (2.0 * dx) - eta_Rp_c * dw_dr_c
            dT_dx_c = (T_x[2:] - T_x[:-2]) / (2.0 * dx) - eta_Rp_c * dT_dr_c
            du_dt_c = (u_t[:, :, 2:] - u_t[:, :, :-2]) / (2.0 * dth)
            dv_dt_c = (v_t[:, :, 2:] - v_t[:, :, :-2]) / (2.0 * dth)
            dw_dt_c = (w_t[:, :, 2:] - w_t[:, :, :-2]) / (2.0 * dth)
            if has_baffles:
                mL_b = baffle_th_face[:, :, :-1]; mR_b = baffle_th_face[:, :, 1:]
                du_dt_c = np.where(mL_b, (u_t[:, :, 2:] - u_t[:, :, 1:-1]) / dth, np.where(mR_b, (u_t[:, :, 1:-1] - u_t[:, :, :-2]) / dth, du_dt_c))
                dv_dt_c = np.where(mL_b, (v_t[:, :, 2:] - v_t[:, :, 1:-1]) / dth, np.where(mR_b, (v_t[:, :, 1:-1] - v_t[:, :, :-2]) / dth, dv_dt_c))
                dw_dt_c = np.where(mL_b, (w_t[:, :, 2:] - w_t[:, :, 1:-1]) / dth, np.where(mR_b, (w_t[:, :, 1:-1] - w_t[:, :, :-2]) / dth, dw_dt_c))

            def avg_x(arr):
                out = np.empty((nx + 1, nr, nth))
                out[1:-1] = 0.5 * (arr[:-1] + arr[1:])
                if mode == "periodic_bump":
                    out[0] = 0.5 * (arr[-1] + arr[0]); out[-1] = out[0]
                else:
                    out[0] = arr[0]; out[-1] = arr[-1]
                return out
            du_dr_xf = avg_x(du_dr_c); dv_dr_xf = avg_x(dv_dr_c); dw_dr_xf = avg_x(dw_dr_c); dT_dr_xf = avg_x(dT_dr_c)
            du_dx_f = (u_x[1:] - u_x[:-1]) / dx - eta_Rp_xf * du_dr_xf
            dv_dx_f = (v_x[1:] - v_x[:-1]) / dx - eta_Rp_xf * dv_dr_xf
            dw_dx_f = (w_x[1:] - w_x[:-1]) / dx - eta_Rp_xf * dw_dr_xf
            dT_dx_f = (T_x[1:] - T_x[:-1]) / dx - eta_Rp_xf * dT_dr_xf
            div_perp_xf = avg_x(dv_dr_c + (v_s + dw_dt_c) / rb)
            du_dt_xf = avg_x(du_dt_c)
            uf_x = 0.5 * (u_x[:-1] + u_x[1:]); vf_x = 0.5 * (v_x[:-1] + v_x[1:]); wf_x = 0.5 * (w_x[:-1] + w_x[1:])
            txx_xf = mu * ((4.0 / 3.0) * du_dx_f - (2.0 / 3.0) * div_perp_xf)
            txr_xf = mu * (du_dr_xf + dv_dx_f)
            txt_xf = mu * (du_dt_xf / r_xf + dw_dx_f)
            qx_xf = -k_th * dT_dx_f
            if mode == "closed" or init_type == "quiescent":
                txx_xf[0] = 0.0; txx_xf[-1] = 0.0; txr_xf[0] = 0.0; txr_xf[-1] = 0.0
                txt_xf[0] = 0.0; txt_xf[-1] = 0.0; qx_xf[0] = 0.0; qx_xf[-1] = 0.0
            VFx_u = Sx * txx_xf; VFx_v = Sx * txr_xf; VFx_w = Sx * r_xf * txt_xf
            VFx_e = Sx * (uf_x * txx_xf + vf_x * txr_xf + wf_x * txt_xf - qx_xf)
            res[1] += VFx_u[1:] - VFx_u[:-1]; res[2] += VFx_v[1:] - VFx_v[:-1]
            res[3] += VFx_w[1:] - VFx_w[:-1]; res[4] += VFx_e[1:] - VFx_e[:-1]

            du_dr_ef = (u_e[:, 1:] - u_e[:, :-1]) / dr_c; dv_dr_ef = (v_e[:, 1:] - v_e[:, :-1]) / dr_c
            dw_dr_ef = (w_e[:, 1:] - w_e[:, :-1]) / dr_c; dT_dr_ef = (T_e[:, 1:] - T_e[:, :-1]) / dr_c
            def avg_e(arr):
                out = np.zeros((nx, nr + 1, nth)); out[:, 1:-1] = 0.5 * (arr[:, :-1] + arr[:, 1:]); return out
            du_dx_ef = avg_e(du_dx_c); dv_dx_ef = avg_e(dv_dx_c); dw_dx_ef = avg_e(dw_dx_c); dT_dx_ef = avg_e(dT_dx_c)
            du_dt_ef = avg_e(du_dt_c); dv_dt_ef = avg_e(dv_dt_c); dw_dt_ef = avg_e(dw_dt_c)
            uf_e = 0.5 * (u_e[:, :-1] + u_e[:, 1:]); vf_e = 0.5 * (v_e[:, :-1] + v_e[:, 1:]); wf_e = 0.5 * (w_e[:, :-1] + w_e[:, 1:])
            inv_ref = np.where(r_ef > 1e-12, 1.0 / np.where(r_ef > 1e-12, r_ef, 1.0), 0.0)
            div_ef = du_dx_ef + dv_dr_ef + (vf_e + dw_dt_ef) * inv_ref
            txx_ef = mu * (2.0 * du_dx_ef - (2.0 / 3.0) * div_ef)
            trr_ef = mu * (2.0 * dv_dr_ef - (2.0 / 3.0) * div_ef)
            txr_ef = mu * (du_dr_ef + dv_dx_ef)
            txt_ef = mu * (du_dt_ef * inv_ref + dw_dx_ef)
            trt_ef = mu * (dw_dr_ef - wf_e * inv_ref + dv_dt_ef * inv_ref)
            qx_ef = -k_th * dT_dx_ef; qr_ef = -k_th * dT_dr_ef
            fx_v = nxe * txx_ef + nre * txr_ef; fr_v = nxe * txr_ef + nre * trr_ef; ft_v = nxe * txt_ef + nre * trt_ef
            fe_v = uf_e * fx_v + vf_e * fr_v + wf_e * ft_v - (nxe * qx_ef + nre * qr_ef)
            VFe_u = Sm * fx_v; VFe_v = Sm * fr_v; VFe_w = Sm * r_ef * ft_v; VFe_e = Sm * fe_v
            VFe_u[:, 0] = 0.0; VFe_u[:, -1] = 0.0; VFe_v[:, 0] = 0.0; VFe_v[:, -1] = 0.0
            VFe_w[:, 0] = 0.0; VFe_w[:, -1] = 0.0; VFe_e[:, 0] = 0.0; VFe_e[:, -1] = 0.0
            res[1] += VFe_u[:, 1:] - VFe_u[:, :-1]; res[2] += VFe_v[:, 1:] - VFe_v[:, :-1]
            res[3] += VFe_w[:, 1:] - VFe_w[:, :-1]; res[4] += VFe_e[:, 1:] - VFe_e[:, :-1]

            du_dt_tf = (u_t[:, :, 1:] - u_t[:, :, :-1]) / dth; dv_dt_tf = (v_t[:, :, 1:] - v_t[:, :, :-1]) / dth
            dw_dt_tf = (w_t[:, :, 1:] - w_t[:, :, :-1]) / dth; dT_dt_tf = (T_t[:, :, 1:] - T_t[:, :, :-1]) / dth
            def avg_t(arr):
                a_ext = np.concatenate([arr[:, :, -1:], arr, arr[:, :, :1]], axis=2)
                return 0.5 * (a_ext[:, :, :-1] + a_ext[:, :, 1:])
            du_dx_tf = avg_t(du_dx_c); dv_dr_tf = avg_t(dv_dr_c); dw_dr_tf = avg_t(dw_dr_c); dw_dx_tf = avg_t(dw_dx_c)
            uf_t = 0.5 * (u_t[:, :, :-1] + u_t[:, :, 1:]); vf_t = 0.5 * (v_t[:, :, :-1] + v_t[:, :, 1:]); wf_t = 0.5 * (w_t[:, :, :-1] + w_t[:, :, 1:])
            inv_rt = 1.0 / r_thf
            div_tf = du_dx_tf + dv_dr_tf + (vf_t + dw_dt_tf) * inv_rt
            txt_tf = mu * (du_dt_tf * inv_rt + dw_dx_tf)
            trt_tf = mu * (dw_dr_tf - wf_t * inv_rt + dv_dt_tf * inv_rt)
            ttt_tf = mu * (2.0 * (dw_dt_tf + vf_t) * inv_rt - (2.0 / 3.0) * div_tf)
            qt_tf = -k_th * dT_dt_tf * inv_rt
            VG_uL = S_th * txt_tf; VG_vL = S_th * trt_tf; VG_wL = S_th * r_thf * ttt_tf
            VG_eL = S_th * (uf_t * txt_tf + vf_t * trt_tf + wf_t * ttt_tf - qt_tf)
            VG_uR, VG_vR, VG_wR, VG_eR = VG_uL.copy(), VG_vL.copy(), VG_wL.copy(), VG_eL.copy()
            if has_baffles:
                if baffle_noslip:
                    tau_xL = mu * (-u_t[:, :, :-1]) / (0.5 * r_thf * dth)
                    tau_xR = mu * (u_t[:, :, 1:]) / (0.5 * r_thf * dth)
                    tau_rL = mu * (-v_t[:, :, :-1]) / (0.5 * r_thf * dth)
                    tau_rR = mu * (v_t[:, :, 1:]) / (0.5 * r_thf * dth)
                    VG_uL = np.where(baffle_th_face, S_th * tau_xL, VG_uL)
                    VG_uR = np.where(baffle_th_face, S_th * tau_xR, VG_uR)
                    VG_vL = np.where(baffle_th_face, S_th * tau_rL, VG_vL)
                    VG_vR = np.where(baffle_th_face, S_th * tau_rR, VG_vR)
                else:
                    VG_uL = np.where(baffle_th_face, 0.0, VG_uL); VG_uR = np.where(baffle_th_face, 0.0, VG_uR)
                    VG_vL = np.where(baffle_th_face, 0.0, VG_vL); VG_vR = np.where(baffle_th_face, 0.0, VG_vR)
                VG_wL = np.where(baffle_th_face, 0.0, VG_wL); VG_wR = np.where(baffle_th_face, 0.0, VG_wR)
                VG_eL = np.where(baffle_th_face, 0.0, VG_eL); VG_eR = np.where(baffle_th_face, 0.0, VG_eR)
            res[1] += VG_uL[:, :, 1:] - VG_uR[:, :, :-1]; res[2] += VG_vL[:, :, 1:] - VG_vR[:, :, :-1]
            res[3] += VG_wL[:, :, 1:] - VG_wR[:, :, :-1]; res[4] += VG_eL[:, :, 1:] - VG_eR[:, :, :-1]

            div_c = du_dx_c + dv_dr_c + (v_s + dw_dt_c) / rb
            ttt_c = mu * (2.0 * (dw_dt_c + v_s) / rb - (2.0 / 3.0) * div_c)
            res[2] -= ttt_c * hoop

        dfp = np.zeros_like(fp_s)
        if has_flame:
            p_sense = np.sum(wf_3d * (p_s - p_b), axis=0) * dx
            dfp[0] = omega_hp * (p_sense - fp_s[0])
            dfp[1] = inv_ts * ((p_sense - fp_s[0]) - fp_s[1])
            for rr in range(1, 4):
                dfp[rr + 1] = inv_ts * (fp_s[rr] - fp_s[rr + 1])
            q_eff = n_p * phi_flame(fp_s[4])
            res[4] += V * (wf_3d * q_eff[None, :, :]) / (GAMMA - 1.0)

        dcav = []
        if cavs:
            dp_in = p_in_face - p_in_base
            for cf, (Pc_c, qt_c) in zip(cavs, cav_states):
                wH = cf["wH"]; xiH = cf["xiH"]; r_nl = cf["r_nl"]; Vk = cf["V_k"]
                p_bar_k = np.tensordot(cf["W"], dp_in * dA_in, axes=([1, 2], [0, 1]))
                dPc = qt_c / Vk
                dqt = (wH * wH) * Vk * (p_bar_k - Pc_c - r_nl * np.abs(qt_c) * qt_c) - 2.0 * xiH * wH * qt_c
                dcav.append((dPc, dqt))

        dfeed = np.zeros(5, dtype=float)
        if has_feed:
            p_face_avg = float(np.sum(p_in_face * dA_in) / A_in_tot)
            dp_rel = (p_face_avg - p_face_base) / P_BAR
            m_cur = max(0.0, f_s[0]) if f_check_valve else f_s[0]
            m_res = (0.5 * ((m_cur * abs(m_cur)) / M_c - M_c)) if f_nl else (m_cur - M_c)
            rhs_m0 = (-m_res - (M_c / (2.0 * f_psi)) * dp_rel) / f_tau_inj
            if f_check_valve and m_cur <= 0.0 and rhs_m0 < 0.0:
                rhs_m0 = 0.0
            dfeed[0] = rhs_m0
            rf = 4.0 / f_tau_f
            dfeed[1] = rf * (m_cur - f_s[1])
            dfeed[2] = rf * (f_s[1] - f_s[2])
            dfeed[3] = rf * (f_s[2] - f_s[3])
            dfeed[4] = rf * (f_s[3] - f_s[4])

        return res, dfp, dcav, dfeed

    # Measure initial full-grid residual (verifies 1-sector -> 5-sector tiling!)
    c0_init = [(cf["Pc"], cf["qt"]) for cf in cavs]
    rhs0, _, _, _ = compute_rhs(Q, fp, c0_init, feed_state)
    init_rhs_max = float(np.max(np.abs(rhs0 / V)))

    t = 0.0; step = 0
    ts, e_ch, e_1t, e_2t, e_1r, e_1l, e_mean = [], [], [], [], [], [], []
    c1_re, c2_re, p_mean_signed, p_pk = [], [], [], []
    w1_th = np.exp(-1j * T_ch); w2_th = np.exp(-2j * T_ch)
    dr_eff = deta * g["Rm"][:, None, None]

    while t < t_end - 1e-12:
        r_c, u_c, v_c, w_c, p_c, c_c = from_Q(Q)
        lam_adv = (np.abs(u_c) + c_c) / dx + (np.abs(v_c) + c_c) / dr_eff + (np.abs(w_c) + c_c) / (rb * dth)
        inv_dt = lam_adv / cfl
        if mu > 0.0:
            nu_eff = max(4.0 / 3.0, GAMMA / PR) * mu / r_c
            inv_dt = inv_dt + nu_eff * (1.0 / (dx * dx) + 1.0 / (dr_eff * dr_eff) + 1.0 / ((rb * dth) ** 2)) / 0.40
        dt = min(1.0 / float(np.max(inv_dt)), t_end - t)

        if step % record_stride == 0:
            dp_c = (p_c - p_b)[ch_mask]; du_c = (u_c - ux_b)[ch_mask]; dv_c = (v_c - ur_b)[ch_mask]; dw_c = (w_c - ut_b)[ch_mask]
            ts.append(t)
            e_ch.append(0.5 * float(np.sum(w_ch * (dp_c * dp_c + du_c * du_c + dv_c * dv_c + dw_c * dw_c))))
            c1 = complex(np.sum(w_ch * dp_c * phi_1T * w1_th))
            c2 = complex(np.sum(w_ch * dp_c * phi_2T * w2_th))
            c0 = float(np.sum(w_ch * dp_c * phi_1R))
            c1l = float(np.sum(w_ch * dp_c * phi_1L))
            cmn = float(np.sum(w_ch * dp_c))
            e_1t.append(2.0 * abs(c1) ** 2); e_2t.append(2.0 * abs(c2) ** 2)
            e_1r.append(c0 * c0); e_1l.append(c1l * c1l); e_mean.append(cmn * cmn)
            c1_re.append(float(c1.real)); c2_re.append(float(c2.real))
            p_mean_signed.append(cmn)
            p_pk.append(float(np.max(np.abs(dp_c)) / P_BAR * 100.0))
            if mode == "periodic_bump":
                tot = np.array([float(np.sum(Q[k])) for k in range(5)])
                max_cons = np.maximum(max_cons, np.abs(tot - init_tot) / np.maximum(np.abs(init_tot), 1.0))

        c0_s = [(cf["Pc"], cf["qt"]) for cf in cavs]
        k1Q, k1f, k1c, k1fd = compute_rhs(Q, fp, c0_s, feed_state)
        Q1 = Q + dt * k1Q; fp1 = fp + dt * k1f; fd1 = feed_state + dt * k1fd
        if f_check_valve:
            fd1[0] = max(0.0, fd1[0])
        c1_s = [(s0 + dt * ds1, v0 + dt * dv1) for (s0, v0), (ds1, dv1) in zip(c0_s, k1c)]
        k2Q, k2f, k2c, k2fd = compute_rhs(Q1, fp1, c1_s, fd1)
        Q = 0.5 * (Q + Q1 + dt * k2Q); fp = 0.5 * (fp + fp1 + dt * k2f)
        feed_state = 0.5 * (feed_state + fd1 + dt * k2fd)
        if f_check_valve:
            feed_state[0] = max(0.0, feed_state[0])
        for cf, (s0, v0), (s1, v1), (ds2, dv2) in zip(cavs, c0_s, c1_s, k2c):
            cf["Pc"] = 0.5 * (s0 + s1 + dt * ds2); cf["qt"] = 0.5 * (v0 + v1 + dt * dv2)
        t += dt; step += 1

    return {
        "t": np.array(ts), "e_ch": np.array(e_ch), "e_1t": np.array(e_1t), "e_2t": np.array(e_2t),
        "e_1r": np.array(e_1r), "e_1l": np.array(e_1l), "e_mean": np.array(e_mean),
        "c1_re": np.array(c1_re), "c2_re": np.array(c2_re),
        "p_mean_signed": np.array(p_mean_signed), "p_max_pct": np.array(p_pk),
        "init_rhs_max": init_rhs_max,
        "quiescent_err": float(np.max(np.abs(Q - Q0))),
        "feed_quiescent_err": float(np.max(np.abs(feed_state - M_c))),
        "periodic_cons_err": {"mass": float(max_cons[0]), "ang_mom": float(max_cons[3]), "energy": float(max_cons[4])},
    }


# ---------------------------------------------------------------------------
# 4. Parallel Execution, Caching & Diagnostics
# ---------------------------------------------------------------------------

def _wake_worker(args):
    lbl, kw = args
    t0 = time.perf_counter()
    res = relax_3d_noslip_wake_lts(**kw)
    return lbl, res, time.perf_counter() - t0


def _sim_worker(args):
    lbl, kw = args
    t0 = time.perf_counter()
    res = simulate_3d_ns(**kw)
    return lbl, res, time.perf_counter() - t0


def _cache_dir() -> Path | None:
    c = os.environ.get("EXP017_CACHE_DIR", "").strip()
    if not c:
        return None
    p = Path(c); p.mkdir(parents=True, exist_ok=True)
    return p


def run_cached_pool(worker_fn, tasks: list[tuple[str, dict]], prefix: str = "", max_workers: int = 8) -> dict:
    cdir = _cache_dir()
    out = {}; pending = []
    for lbl, kw in tasks:
        fp = (cdir / f"{prefix}{lbl}.pkl") if cdir is not None else None
        if fp is not None and fp.exists():
            with open(fp, "rb") as fh:
                out[lbl] = pickle.load(fh)
        else:
            pending.append((lbl, kw))
    if not pending:
        return out
    nw = min(max_workers, max(1, (os.cpu_count() or 2) - 2), len(pending))
    total = len(tasks); done = len(out)
    with ProcessPoolExecutor(max_workers=nw) as ex:
        futs = [ex.submit(worker_fn, item) for item in pending]
        for fut in as_completed(futs):
            lbl, res, dt_s = fut.result()
            out[lbl] = res; done += 1
            if cdir is not None:
                tmp = cdir / f"{prefix}{lbl}.pkl.tmp"
                with open(tmp, "wb") as fh:
                    pickle.dump(res, fh)
                tmp.replace(cdir / f"{prefix}{lbl}.pkl")
            print(f"    [{done:2d}/{total:2d}] {lbl} ({dt_s:.1f}s)", flush=True)
    return out


def fit_alpha(res: dict, t_lo: float, t_hi: float, key: str = "e_ch") -> float:
    t = res["t"]; m = (t >= t_lo) & (t <= t_hi)
    return float(np.polyfit(t[m], 0.5 * np.log(np.maximum(res[key][m], 1e-30)), 1)[0])


def fit_freq_fft(res: dict, t_lo: float, t_hi: float, key: str = "c1_re") -> float:
    t = res["t"]; m = (t >= t_lo) & (t <= t_hi)
    tt = t[m]; ss = res[key][m] - np.mean(res[key][m])
    tu = np.linspace(tt[0], tt[-1], 1024)
    su = np.interp(tu, tt, ss) * np.hanning(len(tu))
    spec = np.abs(np.fft.rfft(su, n=16384))
    freqs = 2.0 * np.pi * np.fft.rfftfreq(16384, d=tu[1] - tu[0])
    idx = int(np.argmax(spec[1:]) + 1)
    return float(freqs[idx])


def summarize_lin(res: dict, t_lo: float = 12.0, t_hi: float | None = None, mode_key: str = "e_1t", freq_key: str = "c1_re") -> dict:
    t = res["t"]
    th = float(t[-1]) if t_hi is None else min(t_hi, float(t[-1]))
    return {
        "alpha_ch": fit_alpha(res, t_lo, th, "e_ch"),
        "alpha_mode": fit_alpha(res, t_lo, th, mode_key),
        "omega": fit_freq_fft(res, max(6.0, t_lo - 4.0), th, freq_key),
        "e_ch_end": float(res["e_ch"][-1]),
        "e_mode_end": float(res[mode_key][-1]),
        "e_1l_end": float(res["e_1l"][-1]),
        "e_mean_end": float(res["e_mean"][-1]),
    }


def summarize_nl(res: dict) -> dict:
    t = res["t"]; ml = t >= 0.85 * t[-1]
    return {
        "t_end": float(t[-1]),
        "p_peak_late": float(np.mean(res["p_max_pct"][ml])),
        "max_abs_pbar_pct": float(np.max(np.abs(res["p_mean_signed"][ml])) / P_BAR * 100.0),
        "e_wave_late": float(np.mean(res["e_ch"][ml])),
        "e_1t_late": float(np.mean(res["e_1t"][ml])),
        "e_2t_late": float(np.mean(res["e_2t"][ml])),
        "e_1r_late": float(np.mean(res["e_1r"][ml])),
        "e_1l_late": float(np.mean(res["e_1l"][ml])),
        "e_mean_late": float(np.mean(res["e_mean"][ml])),
        "slope_tail": fit_alpha(res, 0.65 * t[-1], t[-1], "e_ch"),
        "slope_1t_tail": fit_alpha(res, 0.65 * t[-1], t[-1], "e_1t"),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp017_metrics.json"
    t_start = time.perf_counter()

    print("=" * 98)
    print("Entry 017: 3D Adiabatic No-Slip Baffle Wakes, Core-Bypass & Full All-Mode 3D Smooth Bell Quench")
    print("=" * 98, flush=True)

    # ------------------------------------------------------------------
    # Step 0: Axisymmetric Smooth Bell Bases + Converged 3D No-Slip Wakes (LTS, parallel!)
    # ------------------------------------------------------------------
    print("\n--- Step 0: Relaxing Axisymmetric Bell Bases & 5-Fold 3D No-Slip Wakes (LTS) ---", flush=True)
    axi_32 = relax_axi_3d_bell_base(32, 10, M_c=0.12, L_n=0.60, L_d=0.45, Re=500.0, t_relax=160.0)
    axi_36 = relax_axi_3d_bell_base(36, 12, M_c=0.12, L_n=0.60, L_d=0.45, Re=500.0, t_relax=160.0)
    print(f"  axi_32: rhs={axi_32['rhs_max']:.2e}, M_exit={axi_32['M_exit_center']:.3f}/{axi_32['M_exit_wall']:.3f}", flush=True)
    print(f"  axi_36: rhs={axi_36['rhs_max']:.2e}, M_exit={axi_36['M_exit_center']:.3f}/{axi_36['M_exit_wall']:.3f}", flush=True)

    wake_tasks = [
        ("wake_str_rb42_20",  dict(nx=20, nr=12, nth=40, mode="straight",    Re=500.0, n_baffles=5, L_b=0.45, r_b=0.42, max_iter=30000, tol=1e-9)),
        ("wake_str_rb00_20",  dict(nx=20, nr=12, nth=40, mode="straight",    Re=500.0, n_baffles=5, L_b=0.45, r_b=0.00, max_iter=30000, tol=1e-9)),
        ("wake_str_rb42_ref", dict(nx=24, nr=16, nth=50, mode="straight",    Re=500.0, n_baffles=5, L_b=0.45, r_b=0.42, max_iter=30000, tol=1e-9)),
        ("wake_bell_rb42_32", dict(nx=32, nr=10, nth=40, mode="bell_smooth", Re=500.0, n_baffles=5, L_b=0.45, r_b=0.42, prim_axi_base=axi_32["prim_axi"], max_iter=30000, tol=1e-9)),
        ("wake_bell_rb00_32", dict(nx=32, nr=10, nth=40, mode="bell_smooth", Re=500.0, n_baffles=5, L_b=0.45, r_b=0.00, prim_axi_base=axi_32["prim_axi"], max_iter=30000, tol=1e-9)),
        ("wake_bell_rb42_36", dict(nx=36, nr=12, nth=40, mode="bell_smooth", Re=500.0, n_baffles=5, L_b=0.45, r_b=0.42, prim_axi_base=axi_36["prim_axi"], max_iter=32000, tol=1e-9)),
    ]
    wakes = run_cached_pool(_wake_worker, wake_tasks, prefix="w_", max_workers=6)
    # Fix cached u_core_tip_mean if loaded from older wake pickle (Sx shape (nx,nr,1) vs nth_s)
    for k, w in wakes.items():
        if w["u_core_tip_mean"] > 0.5:
            nth_s = len(w["th_sec"])
            w["u_core_tip_mean"] = float(w["u_core_tip_mean"] / nth_s)
        print(f"  {k:18s}: rhs={w['rhs_max']:.2e} (it={w['iters']}), shroud_u_min={w['u_shroud_min']:.4f}, core_max={w['u_core_max']:.4f}, bypass_vs_axi={w['core_bypass_vs_axi_pct']:+.2f}%", flush=True)

    # ------------------------------------------------------------------
    # Cavity Architectures
    # ------------------------------------------------------------------
    # Entry 016 baseline (axisymmetric 1.0% mid-ring, 5.0% total)
    cav_tri50_ax = [
        {"beta": 0.0285, "wH": WH_1T_RET, "xiH": 0.25, "sigma_o": 0.14, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": True, "sector_clip": False},
        {"beta": 0.0080, "wH": WH_2T_RET, "xiH": 0.24, "sigma_o": 0.08, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": True, "sector_clip": False},
        {"beta": 0.0035, "wH": S02,       "xiH": 0.14, "sigma_o": 0.04, "n_cav": 1, "r_lo": 0.00, "r_hi": 0.26, "nonlinear": True, "sector_clip": False},
        {"beta": 0.0100, "wH": WH_1L_TGT, "xiH": 0.14, "sigma_o": 0.08, "n_cav": 1, "r_lo": 0.44, "r_hi": 0.74, "nonlinear": True, "sector_clip": False},
    ]
    # Zero-volume upgrade: n_cav=5 segmented Nodal Mid-Ring (5.0% total, same sigma_o=0.14)
    cav_tri50_seg = [dict(c) for c in cav_tri50_ax]
    cav_tri50_seg[3]["n_cav"] = 5
    # Unclipped wide-mouth trap (sigma_o=0.24 smears Gaussian tail through baffle walls!)
    cav_tri50_unclip = [dict(c) for c in cav_tri50_seg]
    cav_tri50_unclip[0]["sigma_o"] = 0.24; cav_tri50_unclip[3]["sigma_o"] = 0.14
    # Sector-clipped wide-mouth 5.0% (ports strictly inside |dtheta| < pi/5, sigma_o=0.24 shroud / 0.14 mid!)
    cav_tri50_clip = [dict(c, sector_clip=True) for c in cav_tri50_unclip]
    # Sector-clipped wide-mouth 6.0% (+1.0% shroud 1T -> 3.85% shroud 1T, 6.0% total)
    cav_tri60_clip = [dict(c) for c in cav_tri50_clip]
    cav_tri60_clip[0]["beta"] = 0.0385; cav_tri60_clip[0]["sigma_o"] = 0.26

    def to_lin(clist):
        return [dict(c, nonlinear=False) for c in clist]

    f25_lin = {"psi": 0.25, "nonlinear": False, "check_valve": True}
    f25_nl  = {"psi": 0.25, "nonlinear": True,  "check_valve": True}

    # ------------------------------------------------------------------
    # Phase 1: Conservation, Quiescent-Wake Floor & Exact Linear Slip vs. No-Slip Wake Audit
    # ------------------------------------------------------------------
    print("\n--- Phase 1: Conservation, Wake-Drift Floor & Linear Slip vs. No-Slip Audit ---", flush=True)
    p1_tasks = [
        # 1A: Exact adiabatic no-slip 5-baffle conservation on wavy curved bell + wake-drift equilibrium checks!
        ("cons_noslip_wavy",       dict(nx=16, nr=10, nth=40, t_end=2.0,  mode="periodic_bump", Re=500.0, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True, init_type="periodic_3d_wave", A_eq=0.06, record_stride=15)),
        ("drift_str_noslip_eq",    dict(nx=20, nr=12, nth=40, t_end=40.0, mode="straight",      Re=500.0, feed_spec=f25_nl, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True, cav_spec=cav_tri50_clip, prim_base=wakes["wake_str_rb42_20"]["prim_3d"],  init_type="steady_mean", A_eq=0.0, n_p=0.34, omega_hp=0.50, record_stride=24)),
        ("drift_bell_noslip_eq",   dict(nx=32, nr=10, nth=40, t_end=40.0, mode="bell_smooth",   Re=500.0, feed_spec=f25_nl, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True, cav_spec=cav_tri50_clip, prim_base=wakes["wake_bell_rb42_32"]["prim_3d"], init_type="steady_mean", A_eq=0.0, n_p=0.34, omega_hp=0.50, record_stride=24)),
        # 2A: Straight cylinder (20x12x40) Cold & Hot Linear 1T / 2T: Full-Span (r_b=0) vs Partial-Span (r_b=0.42), Slip vs No-Slip
        ("lin_str_cold_rb00_slip_1T", dict(nx=20, nr=12, nth=40, t_end=32.0, mode="straight", Re=500.0, feed_spec=f25_lin, n_baffles=5, L_b=0.45, r_b=0.00, baffle_noslip=False, init_type="spin_1T", A_eq=1e-4, n_p=0.0, record_stride=8)),
        ("lin_str_cold_rb00_ns_1T",   dict(nx=20, nr=12, nth=40, t_end=32.0, mode="straight", Re=500.0, feed_spec=f25_lin, n_baffles=5, L_b=0.45, r_b=0.00, baffle_noslip=True,  prim_base=wakes["wake_str_rb00_20"]["prim_3d"], init_type="spin_1T", A_eq=1e-4, n_p=0.0, record_stride=8)),
        ("lin_str_cold_rb42_slip_1T", dict(nx=20, nr=12, nth=40, t_end=32.0, mode="straight", Re=500.0, feed_spec=f25_lin, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=False, init_type="spin_1T", A_eq=1e-4, n_p=0.0, record_stride=8)),
        ("lin_str_cold_rb42_ns_1T",   dict(nx=20, nr=12, nth=40, t_end=32.0, mode="straight", Re=500.0, feed_spec=f25_lin, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True,  prim_base=wakes["wake_str_rb42_20"]["prim_3d"], init_type="spin_1T", A_eq=1e-4, n_p=0.0, record_stride=8)),
        ("lin_str_cold_rb42_slip_2T", dict(nx=20, nr=12, nth=40, t_end=32.0, mode="straight", Re=500.0, feed_spec=f25_lin, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=False, init_type="spin_2T", A_eq=1e-4, n_p=0.0, record_stride=8)),
        ("lin_str_cold_rb42_ns_2T",   dict(nx=20, nr=12, nth=40, t_end=32.0, mode="straight", Re=500.0, feed_spec=f25_lin, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True,  prim_base=wakes["wake_str_rb42_20"]["prim_3d"], init_type="spin_2T", A_eq=1e-4, n_p=0.0, record_stride=8)),
        ("lin_str_ref_rb42_slip_1T",  dict(nx=24, nr=16, nth=50, t_end=28.0, mode="straight", Re=500.0, feed_spec=f25_lin, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=False, init_type="spin_1T", A_eq=1e-4, n_p=0.0, record_stride=10)),
        ("lin_str_ref_rb42_ns_1T",    dict(nx=24, nr=16, nth=50, t_end=28.0, mode="straight", Re=500.0, feed_spec=f25_lin, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True,  prim_base=wakes["wake_str_rb42_ref"]["prim_3d"], init_type="spin_1T", A_eq=1e-4, n_p=0.0, record_stride=10)),
        # 2B: Smooth Curved Bell (32x10x40) Cold & Hot Linear 1T / 2T: Bare, Slip vs No-Slip, Tri50_ax vs Seg vs Clipped vs Tri60!
        ("lin_bell_cold_bare_1T",     dict(nx=32, nr=10, nth=40, t_end=35.0, mode="bell_smooth", Re=500.0, feed_spec=f25_lin, prim_base=axi_32["prim_axi"], init_type="spin_1T", A_eq=1e-4, n_p=0.0, record_stride=8)),
        ("lin_bell_cold_rb42_slip_1T",dict(nx=32, nr=10, nth=40, t_end=35.0, mode="bell_smooth", Re=500.0, feed_spec=f25_lin, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=False, prim_base=axi_32["prim_axi"], init_type="spin_1T", A_eq=1e-4, n_p=0.0, record_stride=8)),
        ("lin_bell_cold_rb42_ns_1T",  dict(nx=32, nr=10, nth=40, t_end=35.0, mode="bell_smooth", Re=500.0, feed_spec=f25_lin, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True,  prim_base=wakes["wake_bell_rb42_32"]["prim_3d"], init_type="spin_1T", A_eq=1e-4, n_p=0.0, record_stride=8)),
        ("lin_bell_cold_rb00_ns_1T",  dict(nx=32, nr=10, nth=40, t_end=35.0, mode="bell_smooth", Re=500.0, feed_spec=f25_lin, n_baffles=5, L_b=0.45, r_b=0.00, baffle_noslip=True,  prim_base=wakes["wake_bell_rb00_32"]["prim_3d"], init_type="spin_1T", A_eq=1e-4, n_p=0.0, record_stride=8)),
        ("lin_bell_cold_rb42_slip_2T",dict(nx=32, nr=10, nth=40, t_end=35.0, mode="bell_smooth", Re=500.0, feed_spec=f25_lin, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=False, prim_base=axi_32["prim_axi"], init_type="spin_2T", A_eq=1e-4, n_p=0.0, record_stride=8)),
        ("lin_bell_cold_rb42_ns_2T",  dict(nx=32, nr=10, nth=40, t_end=35.0, mode="bell_smooth", Re=500.0, feed_spec=f25_lin, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True,  prim_base=wakes["wake_bell_rb42_32"]["prim_3d"], init_type="spin_2T", A_eq=1e-4, n_p=0.0, record_stride=8)),
        ("lin_bell_hot_bare_1T",      dict(nx=32, nr=10, nth=40, t_end=40.0, mode="bell_smooth", Re=500.0, feed_spec=f25_lin, prim_base=axi_32["prim_axi"], init_type="spin_1T", A_eq=1e-4, n_p=0.34, omega_hp=0.50, record_stride=10)),
        ("lin_bell_hot_baf_slip_1T",  dict(nx=32, nr=10, nth=40, t_end=40.0, mode="bell_smooth", Re=500.0, feed_spec=f25_lin, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=False, prim_base=axi_32["prim_axi"], init_type="spin_1T", A_eq=1e-4, n_p=0.34, omega_hp=0.50, record_stride=10)),
        ("lin_bell_hot_baf_ns_1T",    dict(nx=32, nr=10, nth=40, t_end=40.0, mode="bell_smooth", Re=500.0, feed_spec=f25_lin, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True,  prim_base=wakes["wake_bell_rb42_32"]["prim_3d"], init_type="spin_1T", A_eq=1e-4, n_p=0.34, omega_hp=0.50, record_stride=10)),
        ("lin_bell_hot_tri50ax_1T",   dict(nx=32, nr=10, nth=40, t_end=42.0, mode="bell_smooth", Re=500.0, feed_spec=f25_lin, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=False, cav_spec=to_lin(cav_tri50_ax),     prim_base=axi_32["prim_axi"], init_type="spin_1T", A_eq=1e-4, n_p=0.34, omega_hp=0.50, record_stride=10)),
        ("lin_bell_hot_tri50seg_1T",  dict(nx=32, nr=10, nth=40, t_end=42.0, mode="bell_smooth", Re=500.0, feed_spec=f25_lin, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=False, cav_spec=to_lin(cav_tri50_seg),    prim_base=axi_32["prim_axi"], init_type="spin_1T", A_eq=1e-4, n_p=0.34, omega_hp=0.50, record_stride=10)),
        ("lin_bell_hot_tri50unc_1T",  dict(nx=32, nr=10, nth=40, t_end=42.0, mode="bell_smooth", Re=500.0, feed_spec=f25_lin, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=False, cav_spec=to_lin(cav_tri50_unclip), prim_base=axi_32["prim_axi"], init_type="spin_1T", A_eq=1e-4, n_p=0.34, omega_hp=0.50, record_stride=10)),
        ("lin_bell_hot_tri50clp_1T",  dict(nx=32, nr=10, nth=40, t_end=42.0, mode="bell_smooth", Re=500.0, feed_spec=f25_lin, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=False, cav_spec=to_lin(cav_tri50_clip),   prim_base=axi_32["prim_axi"], init_type="spin_1T", A_eq=1e-4, n_p=0.34, omega_hp=0.50, record_stride=10)),
        ("lin_bell_hot_tri50clp_ns",  dict(nx=32, nr=10, nth=40, t_end=42.0, mode="bell_smooth", Re=500.0, feed_spec=f25_lin, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True,  cav_spec=to_lin(cav_tri50_clip),   prim_base=wakes["wake_bell_rb42_32"]["prim_3d"], init_type="spin_1T", A_eq=1e-4, n_p=0.34, omega_hp=0.50, record_stride=10)),
        ("lin_bell_hot_tri60clp_ns",  dict(nx=32, nr=10, nth=40, t_end=42.0, mode="bell_smooth", Re=500.0, feed_spec=f25_lin, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True,  cav_spec=to_lin(cav_tri60_clip),   prim_base=wakes["wake_bell_rb42_32"]["prim_3d"], init_type="spin_1T", A_eq=1e-4, n_p=0.34, omega_hp=0.50, record_stride=10)),
        ("lin_bell36_tri50clp_slip",  dict(nx=36, nr=12, nth=40, t_end=40.0, mode="bell_smooth", Re=500.0, feed_spec=f25_lin, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=False, cav_spec=to_lin(cav_tri50_clip),   prim_base=axi_36["prim_axi"], init_type="spin_1T", A_eq=1e-4, n_p=0.34, omega_hp=0.50, record_stride=12)),
        ("lin_bell36_tri50clp_ns",    dict(nx=36, nr=12, nth=40, t_end=40.0, mode="bell_smooth", Re=500.0, feed_spec=f25_lin, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True,  cav_spec=to_lin(cav_tri50_clip),   prim_base=wakes["wake_bell_rb42_36"]["prim_3d"], init_type="spin_1T", A_eq=1e-4, n_p=0.34, omega_hp=0.50, record_stride=12)),
    ]
    p1_sorted = sorted(p1_tasks, key=lambda kv: kv[1]["nx"] * kv[1]["nr"] * kv[1]["nth"] * kv[1]["t_end"], reverse=True)
    p1_res = run_cached_pool(_sim_worker, p1_sorted, prefix="p1_", max_workers=10)

    # ------------------------------------------------------------------
    # Phase 2: Nonlinear Smooth-Bell Port-Clipping, Amplitude Threshold & Full All-6-Mode Slip + No-Slip Shootout
    # ------------------------------------------------------------------
    print("\n--- Phase 2: Nonlinear Smooth-Bell Port-Clipping, Thresholds & All-6-Mode Shootout ---", flush=True)
    p2_dict = {
        # Exp 3: Port-clipping trap & amplitude threshold bracket (bell 32x10x40, Re=500, psi=0.25, n_p=0.34)
        "nl_bell_bare_1T":         dict(nx=32, nr=10, nth=40, t_end=100.0, mode="bell_smooth", Re=500.0, feed_spec=f25_nl, prim_base=axi_32["prim_axi"], init_type="spin_1T", A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "nl_bell_tri50ax_1T":      dict(nx=32, nr=10, nth=40, t_end=120.0, mode="bell_smooth", Re=500.0, feed_spec=f25_nl, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=False, cav_spec=cav_tri50_ax,     prim_base=axi_32["prim_axi"], init_type="spin_1T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "nl_bell_tri50unc_1T":     dict(nx=32, nr=10, nth=40, t_end=120.0, mode="bell_smooth", Re=500.0, feed_spec=f25_nl, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=False, cav_spec=cav_tri50_unclip, prim_base=axi_32["prim_axi"], init_type="spin_1T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "nl_bell_tri50seg_1T":     dict(nx=32, nr=10, nth=40, t_end=140.0, mode="bell_smooth", Re=500.0, feed_spec=f25_nl, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=False, cav_spec=cav_tri50_seg,    prim_base=axi_32["prim_axi"], init_type="spin_1T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "nl_bell_tri50ax_A02":     dict(nx=32, nr=10, nth=40, t_end=90.0,  mode="bell_smooth", Re=500.0, feed_spec=f25_nl, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=False, cav_spec=cav_tri50_ax,     prim_base=axi_32["prim_axi"], init_type="spin_1T",  A_eq=0.02, n_p=0.34, omega_hp=0.50, record_stride=20),
        "nl_bell_tri50clp_A06":    dict(nx=32, nr=10, nth=40, t_end=120.0, mode="bell_smooth", Re=500.0, feed_spec=f25_nl, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=False, cav_spec=cav_tri50_clip,   prim_base=axi_32["prim_axi"], init_type="spin_1T",  A_eq=0.06, n_p=0.34, omega_hp=0.50, record_stride=24),
        # Exp 4A: 5.0% Sector-Clipped Tri-Band (tri50_clip) -- ALL 5 MODES under SLIP baffles (t=120-140)
        "nl_bell_clp50_slip_1T":   dict(nx=32, nr=10, nth=40, t_end=140.0, mode="bell_smooth", Re=500.0, feed_spec=f25_nl, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=False, cav_spec=cav_tri50_clip, prim_base=axi_32["prim_axi"], init_type="spin_1T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "nl_bell_clp50_slip_st1T": dict(nx=32, nr=10, nth=40, t_end=140.0, mode="bell_smooth", Re=500.0, feed_spec=f25_nl, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=False, cav_spec=cav_tri50_clip, prim_base=axi_32["prim_axi"], init_type="stand_1T", A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "nl_bell_clp50_slip_2T":   dict(nx=32, nr=10, nth=40, t_end=120.0, mode="bell_smooth", Re=500.0, feed_spec=f25_nl, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=False, cav_spec=cav_tri50_clip, prim_base=axi_32["prim_axi"], init_type="spin_2T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "nl_bell_clp50_slip_1R":   dict(nx=32, nr=10, nth=40, t_end=140.0, mode="bell_smooth", Re=500.0, feed_spec=f25_nl, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=False, cav_spec=cav_tri50_clip, prim_base=axi_32["prim_axi"], init_type="1R",       A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "nl_bell_clp50_slip_1L":   dict(nx=32, nr=10, nth=40, t_end=140.0, mode="bell_smooth", Re=500.0, feed_spec=f25_nl, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=False, cav_spec=cav_tri50_clip, prim_base=axi_32["prim_axi"], init_type="1L",       A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        # Exp 4B: 5.0% Sector-Clipped Tri-Band (tri50_clip) -- ALL 5 MODES under CONVERGED 3D NO-SLIP WAKES (t=120-140)!
        "nl_bell_clp50_ns_1T":     dict(nx=32, nr=10, nth=40, t_end=140.0, mode="bell_smooth", Re=500.0, feed_spec=f25_nl, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True,  cav_spec=cav_tri50_clip, prim_base=wakes["wake_bell_rb42_32"]["prim_3d"], init_type="spin_1T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "nl_bell_clp50_ns_st1T":   dict(nx=32, nr=10, nth=40, t_end=140.0, mode="bell_smooth", Re=500.0, feed_spec=f25_nl, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True,  cav_spec=cav_tri50_clip, prim_base=wakes["wake_bell_rb42_32"]["prim_3d"], init_type="stand_1T", A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "nl_bell_clp50_ns_2T":     dict(nx=32, nr=10, nth=40, t_end=120.0, mode="bell_smooth", Re=500.0, feed_spec=f25_nl, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True,  cav_spec=cav_tri50_clip, prim_base=wakes["wake_bell_rb42_32"]["prim_3d"], init_type="spin_2T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "nl_bell_clp50_ns_1R":     dict(nx=32, nr=10, nth=40, t_end=140.0, mode="bell_smooth", Re=500.0, feed_spec=f25_nl, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True,  cav_spec=cav_tri50_clip, prim_base=wakes["wake_bell_rb42_32"]["prim_3d"], init_type="1R",       A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "nl_bell_clp50_ns_1L":     dict(nx=32, nr=10, nth=40, t_end=140.0, mode="bell_smooth", Re=500.0, feed_spec=f25_nl, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True,  cav_spec=cav_tri50_clip, prim_base=wakes["wake_bell_rb42_32"]["prim_3d"], init_type="1L",       A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        # Exp 4C: 6.0% Sector-Clipped Tri-Band (tri60_clip) under Converged No-Slip Wakes (1T, st1T, A_eq=0.06)
        "nl_bell_clp60_ns_1T":     dict(nx=32, nr=10, nth=40, t_end=120.0, mode="bell_smooth", Re=500.0, feed_spec=f25_nl, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True,  cav_spec=cav_tri60_clip, prim_base=wakes["wake_bell_rb42_32"]["prim_3d"], init_type="spin_1T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "nl_bell_clp60_ns_st1T":   dict(nx=32, nr=10, nth=40, t_end=120.0, mode="bell_smooth", Re=500.0, feed_spec=f25_nl, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True,  cav_spec=cav_tri60_clip, prim_base=wakes["wake_bell_rb42_32"]["prim_3d"], init_type="stand_1T", A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "nl_bell_clp60_ns_A06":    dict(nx=32, nr=10, nth=40, t_end=120.0, mode="bell_smooth", Re=500.0, feed_spec=f25_nl, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True,  cav_spec=cav_tri60_clip, prim_base=wakes["wake_bell_rb42_32"]["prim_3d"], init_type="spin_1T",  A_eq=0.06, n_p=0.34, omega_hp=0.50, record_stride=24),
        # Exp 4D: Refined 36x12x40 Smooth Bell Verification (Converged No-Slip Wakes!)
        "nl_bell36_clp50_ns_1T":   dict(nx=36, nr=12, nth=40, t_end=120.0, mode="bell_smooth", Re=500.0, feed_spec=f25_nl, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True,  cav_spec=cav_tri50_clip, prim_base=wakes["wake_bell_rb42_36"]["prim_3d"], init_type="spin_1T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=28),
        "nl_bell36_clp50_ns_st1T": dict(nx=36, nr=12, nth=40, t_end=120.0, mode="bell_smooth", Re=500.0, feed_spec=f25_nl, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True,  cav_spec=cav_tri50_clip, prim_base=wakes["wake_bell_rb42_36"]["prim_3d"], init_type="stand_1T", A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=28),
        "nl_bell36_clp50_ns_1L":   dict(nx=36, nr=12, nth=40, t_end=120.0, mode="bell_smooth", Re=500.0, feed_spec=f25_nl, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True,  cav_spec=cav_tri50_clip, prim_base=wakes["wake_bell_rb42_36"]["prim_3d"], init_type="1L",       A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=28),
    }
    p2_sorted = sorted(p2_dict.items(), key=lambda kv: kv[1]["nx"] * kv[1]["nr"] * kv[1]["nth"] * kv[1]["t_end"], reverse=True)
    p2_res = run_cached_pool(_sim_worker, p2_sorted, prefix="p2_", max_workers=10)

    # ------------------------------------------------------------------
    # Assemble Summaries & Print Report
    # ------------------------------------------------------------------
    wake_summary = {
        k: {
            "rhs_max": w["rhs_max"], "iters": w["iters"],
            "u_min_ch": w["u_min_ch"], "u_shroud_min": w["u_shroud_min"],
            "u_tip_min": w["u_tip_min"], "u_ds_min": w["u_ds_min"],
            "u_shroud_mean": w["u_shroud_mean"], "u_core_mean": w["u_core_mean"],
            "u_core_tip_mean": w["u_core_tip_mean"], "u_core_max": w["u_core_max"],
            "core_bypass_pct": w["core_bypass_pct"],
            "core_bypass_vs_axi_pct": w["core_bypass_vs_axi_pct"],
        } for k, w in wakes.items()
    }

    lin_summary = {}
    for k, r in p1_res.items():
        if k.startswith("lin_"):
            mk = "e_2t" if "_2T" in k else "e_1t"
            fk = "c2_re" if "_2T" in k else "c1_re"
            lin_summary[k] = summarize_lin(r, t_lo=12.0, mode_key=mk, freq_key=fk)

    nl_summary = {k: summarize_nl(r) for k, r in p2_res.items()}

    # Wake penalty ratios: (alpha_ns - alpha_slip) / |alpha_slip| (positive => less damping / penalty!)
    def wake_pen(k_slip, k_ns, field="alpha_mode"):
        a_s = lin_summary[k_slip][field]; a_n = lin_summary[k_ns][field]
        return {"alpha_slip": a_s, "alpha_noslip": a_n, "delta": a_n - a_s,
                "pct_change_mag": float((abs(a_n) - abs(a_s)) / max(abs(a_s), 1e-12) * 100.0)}

    penalties = {
        "str_cold_rb00_1T":   wake_pen("lin_str_cold_rb00_slip_1T", "lin_str_cold_rb00_ns_1T", "alpha_ch"),
        "str_cold_rb42_1T":   wake_pen("lin_str_cold_rb42_slip_1T", "lin_str_cold_rb42_ns_1T", "alpha_ch"),
        "str_cold_rb42_2T":   wake_pen("lin_str_cold_rb42_slip_2T", "lin_str_cold_rb42_ns_2T", "alpha_ch"),
        "str_ref_rb42_1T":    wake_pen("lin_str_ref_rb42_slip_1T",  "lin_str_ref_rb42_ns_1T",  "alpha_ch"),
        "bell_cold_rb42_1T":  wake_pen("lin_bell_cold_rb42_slip_1T","lin_bell_cold_rb42_ns_1T","alpha_ch"),
        "bell_cold_rb42_2T":  wake_pen("lin_bell_cold_rb42_slip_2T","lin_bell_cold_rb42_ns_2T","alpha_ch"),
        "bell_hot_baf_1T":    wake_pen("lin_bell_hot_baf_slip_1T",  "lin_bell_hot_baf_ns_1T",  "alpha_mode"),
        "bell_hot_tri50clp":  wake_pen("lin_bell_hot_tri50clp_1T",  "lin_bell_hot_tri50clp_ns","alpha_mode"),
        "bell36_tri50clp":    wake_pen("lin_bell36_tri50clp_slip",  "lin_bell36_tri50clp_ns",  "alpha_mode"),
    }

    print("\n--- Summary: Conservation & Wake-Drift Floor ---")
    print(f"  cons_noslip_wavy: {p1_res['cons_noslip_wavy']['periodic_cons_err']}")
    print(f"  drift_str_noslip_eq:  init_rhs={p1_res['drift_str_noslip_eq']['init_rhs_max']:.2e}, Q_err={p1_res['drift_str_noslip_eq']['quiescent_err']:.2e}, e_ch_end={p1_res['drift_str_noslip_eq']['e_ch'][-1]:.2e}")
    print(f"  drift_bell_noslip_eq: init_rhs={p1_res['drift_bell_noslip_eq']['init_rhs_max']:.2e}, Q_err={p1_res['drift_bell_noslip_eq']['quiescent_err']:.2e}, e_ch_end={p1_res['drift_bell_noslip_eq']['e_ch'][-1]:.2e}")

    print("\n--- Summary: Linear Slip vs. No-Slip Wake Audit ---")
    for k, v in sorted(lin_summary.items()):
        print(f"  {k:28s}: alpha_ch={v['alpha_ch']:+.5f}, alpha_mode={v['alpha_mode']:+.5f}, omega={v['omega']:.4f}, e_mode_end={v['e_mode_end']:.2e}")
    print("\n--- Summary: 3D Wake Penalty Breakdown ---")
    for k, v in penalties.items():
        print(f"  {k:20s}: slip={v['alpha_slip']:+.5f} -> noslip={v['alpha_noslip']:+.5f} (delta={v['delta']:+.5f}, |alpha| change={v['pct_change_mag']:+.2f}%)")

    print("\n--- Summary: Nonlinear Smooth-Bell Shootout ---")
    for k, v in nl_summary.items():
        print(f"  {k:24s} (t={v['t_end']:5.1f}): P={v['p_peak_late']:6.3f}%, E_wave={v['e_wave_late']:.3e}, E_1T={v['e_1t_late']:.3e}, E_1L={v['e_1l_late']:.3e}, E_mn={v['e_mean_late']:.2e}, slope={v['slope_tail']:+.4f}")

    # ------------------------------------------------------------------
    # Save Metrics JSON
    # ------------------------------------------------------------------
    metrics = {
        "meta": {"entry": "017", "Re": 500.0, "M_c": 0.12, "psi": 0.25, "n_p": 0.34, "omega_hp": 0.50,
                 "runtime_sec": round(time.perf_counter() - t_start, 1)},
        "exp1_conservation_and_drift": {
            "cons_noslip_wavy": p1_res["cons_noslip_wavy"]["periodic_cons_err"],
            "drift_str_noslip_eq": {
                "init_rhs_max": p1_res["drift_str_noslip_eq"]["init_rhs_max"],
                "quiescent_err": p1_res["drift_str_noslip_eq"]["quiescent_err"],
                "feed_quiescent_err": p1_res["drift_str_noslip_eq"]["feed_quiescent_err"],
                "e_ch_end": float(p1_res["drift_str_noslip_eq"]["e_ch"][-1]),
            },
            "drift_bell_noslip_eq": {
                "init_rhs_max": p1_res["drift_bell_noslip_eq"]["init_rhs_max"],
                "quiescent_err": p1_res["drift_bell_noslip_eq"]["quiescent_err"],
                "feed_quiescent_err": p1_res["drift_bell_noslip_eq"]["feed_quiescent_err"],
                "e_ch_end": float(p1_res["drift_bell_noslip_eq"]["e_ch"][-1]),
            },
            "axi_bases": {
                "axi_32": {k: v for k, v in axi_32.items() if k != "prim_axi"},
                "axi_36": {k: v for k, v in axi_36.items() if k != "prim_axi"},
            },
            "wakes": wake_summary,
        },
        "exp2_linear_wake_audit": {"runs": lin_summary, "penalties": penalties},
        "exp3_4_nonlinear_bell_shootout": nl_summary,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nSaved metrics -> {out_path}", flush=True)

    # ------------------------------------------------------------------
    # Figures (3 clean, non-overlapping publication PNGs)
    # ------------------------------------------------------------------
    # Figure 1: 3D Steady No-Slip Baffle Wake & Core-Bypass Flow + Linear Wake Penalty Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7), constrained_layout=True)

    # Panel A: (r, theta) cross-section of u_x at baffle trailing edge x = L_b in 32x10x40 smooth bell
    wb = wakes["wake_bell_rb42_32"]
    u_rt = wb["u_rt_tip"]  # (nr, nth_s)
    # Tile 2 sectors for visual clarity across a baffle wall
    u_rt2 = np.concatenate([u_rt, u_rt], axis=1)
    th2 = np.degrees(np.concatenate([wb["th_sec"], wb["th_sec"] + 2.0 * np.pi / 5.0]))
    im = axes[0].pcolormesh(th2, wb["etac"], u_rt2 / 0.12, shading="auto", cmap="RdYlBu_r", vmin=0.55, vmax=1.20)
    axes[0].axhline(0.42, color="k", ls="--", lw=1.4, label=r"Baffle inner edge $r_b=0.42$")
    axes[0].axvline(72.0, color="k", ls="-", lw=2.2, label="No-slip baffle blade")
    axes[0].set_xlabel(r"Azimuth $\theta$ (deg, 2 sectors of $72^\circ$)")
    axes[0].set_ylabel(r"Normalized radius $\eta = r / R(x)$")
    axes[0].set_title(f"(a) 3D Smooth-Bell No-Slip Wake at $x=L_b$\n($u_{{\\min}}/\\bar{{M}}_c={wb['u_tip_min']/0.12:.2f}$, core bypass {wb['core_bypass_vs_axi_pct']:+.1f}%)")
    axes[0].legend(loc="lower right", fontsize=8.0, framealpha=0.92)
    cb = fig.colorbar(im, ax=axes[0], pad=0.02); cb.set_label(r"$u_x / \bar{M}_c$")

    # Panel B: Radial profiles of sector-averaged u_x(r) at x=L_b: full-span (r_b=0) vs partial-span (r_b=0.42) in straight & bell
    ws42 = wakes["wake_str_rb42_20"]; ws00 = wakes["wake_str_rb00_20"]; wb00 = wakes["wake_bell_rb00_32"]
    ix_s = int(np.argmin(np.abs(ws42["xc"] - 0.45)))
    ix_b = int(np.argmin(np.abs(wb["xc"] - 0.45)))
    axes[1].plot(ws42["etac"], np.mean(ws42["prim_3d"][1, ix_s, :, :], axis=1) / 0.12, "o-", color="#1d4ed8", lw=2.0, ms=4, label=r"Straight partial ($r_b=0.42$, sector mean)")
    axes[1].plot(ws42["etac"], ws42["u_xr_wall"][ix_s, :] / 0.12, "s--", color="#60a5fa", lw=1.5, ms=3.5, label=r"Straight partial (wall-adjacent cell)")
    axes[1].plot(ws00["etac"], ws00["u_xr_wall"][ix_s, :] / 0.12, "^:", color="#dc2626", lw=1.7, ms=3.5, label=r"Straight full-span ($r_b=0.0$, wall cell)")
    axes[1].plot(wb["etac"],   np.mean(wb["prim_3d"][1, ix_b, :, :], axis=1) / 0.12, "D-", color="#059669", lw=2.0, ms=4, label=r"Bell partial ($r_b=0.42$, sector mean)")
    axes[1].axvline(0.42, color="#374151", ls="--", lw=1.2)
    axes[1].axhline(1.0, color="#9ca3af", ls=":", lw=1.0)
    axes[1].text(0.18, 1.13, "Open core\nbypass jet", ha="center", fontsize=8.5, color="#1d4ed8", fontweight="bold")
    axes[1].text(0.72, 0.64, "Shroud wake\ndeficit", ha="center", fontsize=8.5, color="#dc2626", fontweight="bold")
    axes[1].set_xlabel(r"Normalized radius $\eta = r / R(x)$")
    axes[1].set_ylabel(r"Axial velocity $u_x / \bar{M}_c$ at $x = L_b$")
    axes[1].set_title("(b) 3D Core-Bypass Acceleration vs. Full-Span\n(Partial-span diverts shroud drag into core)")
    axes[1].set_ylim(0.50, 1.24); axes[1].grid(True, alpha=0.28)
    axes[1].legend(loc="lower left", fontsize=7.6, framealpha=0.92)

    # Panel C: Slip vs Converged No-Slip linear damping/growth across configurations
    cats = ["Str cold\n$r_b=0$ 1T", "Str cold\n$r_b=0.42$ 1T", "Str ref\n$r_b=0.42$ 1T", "Bell cold\n$r_b=0.42$ 1T", "Bell hot\nTri50 1T", "Bell 36x12\nTri50 1T"]
    slip_vals = [
        -lin_summary["lin_str_cold_rb00_slip_1T"]["alpha_ch"],
        -lin_summary["lin_str_cold_rb42_slip_1T"]["alpha_ch"],
        -lin_summary["lin_str_ref_rb42_slip_1T"]["alpha_ch"],
        -lin_summary["lin_bell_cold_rb42_slip_1T"]["alpha_ch"],
        -lin_summary["lin_bell_hot_tri50clp_1T"]["alpha_mode"],
        -lin_summary["lin_bell36_tri50clp_slip"]["alpha_mode"],
    ]
    ns_vals = [
        -lin_summary["lin_str_cold_rb00_ns_1T"]["alpha_ch"],
        -lin_summary["lin_str_cold_rb42_ns_1T"]["alpha_ch"],
        -lin_summary["lin_str_ref_rb42_ns_1T"]["alpha_ch"],
        -lin_summary["lin_bell_cold_rb42_ns_1T"]["alpha_ch"],
        -lin_summary["lin_bell_hot_tri50clp_ns"]["alpha_mode"],
        -lin_summary["lin_bell36_tri50clp_ns"]["alpha_mode"],
    ]
    x_pos = np.arange(len(cats)); w_bar = 0.36
    axes[2].bar(x_pos - w_bar / 2, slip_vals, w_bar, color="#2563eb", label="Slip baffles")
    axes[2].bar(x_pos + w_bar / 2, ns_vals,   w_bar, color="#ea580c", label="Converged 3D no-slip wake")
    for xi, sv, nv in zip(x_pos, slip_vals, ns_vals):
        pct = (nv - sv) / max(abs(sv), 1e-12) * 100.0
        axes[2].text(xi, max(sv, nv) + 0.004, f"{pct:+.1f}%", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    axes[2].set_xticks(x_pos); axes[2].set_xticklabels(cats, fontsize=7.6)
    axes[2].set_ylabel(r"Linear $1T$ damping margin $-\alpha_{1T}$")
    axes[2].set_title("(c) 3D No-Slip Wake Effect (Slip vs. No-Slip)\n(Re = 500, $A_0=10^{-4}$, rhs $< 10^{-8}$)")
    axes[2].set_ylim(0.0, max(max(slip_vals), max(ns_vals)) * 1.24)
    axes[2].grid(True, axis="y", alpha=0.28)
    axes[2].legend(loc="upper right", fontsize=8.0, framealpha=0.92)
    fig.savefig(RESULTS_DIR / "exp017_3d_noslip_wakes.png", dpi=180)
    plt.close(fig)

    # Figure 2: Closing the Bell 1T Gap -- Linear Margin, Port-Clipping Trap & Subcritical Thresholds
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)

    # Panel A: Linear 1T decay envelopes in the hot 3D smooth bell (Re=500, n_p=0.34, psi=0.25)
    for key, lbl, col, ls in [
        ("lin_bell_hot_bare_1T",     f"Bare bell ({lin_summary['lin_bell_hot_bare_1T']['alpha_mode']:+.4f})", "#dc2626", "-"),
        ("lin_bell_hot_baf_slip_1T", f"Baffles only ({lin_summary['lin_bell_hot_baf_slip_1T']['alpha_mode']:+.4f})", "#ea580c", "--"),
        ("lin_bell_hot_tri50ax_1T",  f"Entry 016 Tri50 ax ({lin_summary['lin_bell_hot_tri50ax_1T']['alpha_mode']:+.4f})", "#6b7280", ":"),
        ("lin_bell_hot_tri50unc_1T", f"Tri50 seg unclipped ({lin_summary['lin_bell_hot_tri50unc_1T']['alpha_mode']:+.4f})", "#9333ea", "-."),
        ("lin_bell_hot_tri50seg_1T", f"Tri50 seg 0.14 ({lin_summary['lin_bell_hot_tri50seg_1T']['alpha_mode']:+.4f})", "#2563eb", "-"),
        ("lin_bell_hot_tri50clp_1T", f"Tri50 clipped wide ({lin_summary['lin_bell_hot_tri50clp_1T']['alpha_mode']:+.4f})", "#059669", "-"),
        ("lin_bell_hot_tri50clp_ns", f"Tri50 clipped + no-slip ({lin_summary['lin_bell_hot_tri50clp_ns']['alpha_mode']:+.4f})", "#047857", "--"),
    ]:
        r = p1_res[key]
        axes[0].semilogy(r["t"], np.maximum(r["e_1t"], 1e-16), color=col, ls=ls, lw=1.9, label=lbl)
    axes[0].set_xlabel(r"Time $t$")
    axes[0].set_ylabel(r"Linear $1T$ modal energy $E_{1T}(t)$ ($A_0=10^{-4}$)")
    axes[0].set_title("(a) Smooth-Bell Linear $1T$ Margin Jump\n(Zero-volume $n_{\\rm cav}=1\\to 5$ + sector clipping)")
    axes[0].set_ylim(1e-15, 3e-6); axes[0].grid(True, alpha=0.28)
    axes[0].legend(loc="lower left", fontsize=7.6, framealpha=0.92)

    # Panel B: Nonlinear A_eq=0.04 & 0.06 1T evolution proving the Gaussian port-smearing trap vs sector-clipped quench!
    for key, lbl, col, ls in [
        ("nl_bell_bare_1T",       f"Bare bell ({nl_summary['nl_bell_bare_1T']['p_peak_late']:.2f}%)", "#dc2626", "-"),
        ("nl_bell_tri50ax_1T",    f"Entry 016 Tri50 ax ({nl_summary['nl_bell_tri50ax_1T']['p_peak_late']:.2f}%)", "#6b7280", ":"),
        ("nl_bell_tri50unc_1T",   f"Unclipped wide 0.24 trap ({nl_summary['nl_bell_tri50unc_1T']['p_peak_late']:.2f}%)", "#9333ea", "-."),
        ("nl_bell_tri50seg_1T",   f"Tri50 seg 0.14 ({nl_summary['nl_bell_tri50seg_1T']['p_peak_late']:.2f}%)", "#2563eb", "--"),
        ("nl_bell_clp50_slip_1T", f"Tri50 clipped wide slip ({nl_summary['nl_bell_clp50_slip_1T']['p_peak_late']:.3f}%)", "#059669", "-"),
        ("nl_bell_clp50_ns_1T",   f"Tri50 clipped wide no-slip ({nl_summary['nl_bell_clp50_ns_1T']['p_peak_late']:.3f}%)", "#047857", "--"),
        ("nl_bell_tri50clp_A06",  f"Tri50 clipped A=0.06 ({nl_summary['nl_bell_tri50clp_A06']['p_peak_late']:.3f}%)", "#0d9488", ":"),
        ("nl_bell_clp60_ns_A06",  f"Tri60 clipped no-slip A=0.06 ({nl_summary['nl_bell_clp60_ns_A06']['p_peak_late']:.3f}%)", "#111827", "-"),
    ]:
        r = p2_res[key]
        axes[1].semilogy(r["t"], np.maximum(r["e_ch"], 1e-13), color=col, ls=ls, lw=1.9, label=lbl)
    axes[1].set_xlabel(r"Time $t$")
    axes[1].set_ylabel(r"Chamber wave energy $E_{\rm wave}(t)$")
    axes[1].set_title("(b) Nonlinear Bell $1T$ Bomb Shootout ($A_0=0.04, 0.06$)\n(Sector clipping prevents blade leakage & unlocks quench)")
    axes[1].set_ylim(1e-11, 2e-2); axes[1].grid(True, alpha=0.28)
    axes[1].legend(loc="lower left", fontsize=7.5, framealpha=0.92)
    fig.savefig(RESULTS_DIR / "exp017_bell_1T_sector_clip_quench.png", dpi=180)
    plt.close(fig)

    # Figure 3: Full All-6-Mode Smooth Curved Bell Quench (Slip vs Converged No-Slip + 36x12x40 Refinement)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)

    # Panel A: Slip vs Converged No-Slip 3D Wakes across ALL 5 modes at 5.0% Tri-Band (tri50_clip)
    for key, lbl, col, ls in [
        ("nl_bell_bare_1T",         f"Bare bell 1T ({nl_summary['nl_bell_bare_1T']['p_peak_late']:.2f}%)", "#dc2626", "-"),
        ("nl_bell_clp50_slip_1T",   f"5% Clipped slip 1T ({nl_summary['nl_bell_clp50_slip_1T']['p_peak_late']:.3f}%)", "#1d4ed8", "-"),
        ("nl_bell_clp50_ns_1T",     f"5% Clipped no-slip 1T ({nl_summary['nl_bell_clp50_ns_1T']['p_peak_late']:.3f}%)", "#1d4ed8", "--"),
        ("nl_bell_clp50_ns_st1T",   f"5% Clipped no-slip st1T ({nl_summary['nl_bell_clp50_ns_st1T']['p_peak_late']:.3f}%)", "#0891b2", "--"),
        ("nl_bell_clp50_ns_2T",     f"5% Clipped no-slip 2T ({nl_summary['nl_bell_clp50_ns_2T']['p_peak_late']:.3f}%)", "#7c3aed", "-."),
        ("nl_bell_clp50_ns_1R",     f"5% Clipped no-slip 1R ({nl_summary['nl_bell_clp50_ns_1R']['p_peak_late']:.3f}%)", "#d97706", ":"),
        ("nl_bell_clp50_ns_1L",     f"5% Clipped no-slip 1L ({nl_summary['nl_bell_clp50_ns_1L']['p_peak_late']:.3f}%)", "#059669", "-"),
    ]:
        r = p2_res[key]
        axes[0].semilogy(r["t"], np.maximum(r["e_ch"], 1e-13), color=col, ls=ls, lw=1.9, label=lbl)
    axes[0].set_xlabel(r"Time $t$")
    axes[0].set_ylabel(r"Chamber wave energy $E_{\rm wave}(t)$")
    axes[0].set_title("(a) All-5-Mode Smooth-Bell Quench (5.0% Tri-Band)\n(Re = 500, $\\psi=0.25$, Slip & Converged No-Slip Wakes)")
    axes[0].set_ylim(1e-11, 2e-2); axes[0].grid(True, alpha=0.28)
    axes[0].legend(loc="lower left", fontsize=7.6, framealpha=0.92)

    # Panel B: Refined 36x12x40 Converged No-Slip Bell Verification + 6.0% Margin & Wake-Drift Floor
    for key, lbl, col, ls in [
        ("nl_bell_bare_1T",         f"Bare bell 32x10x40 1T ({nl_summary['nl_bell_bare_1T']['p_peak_late']:.2f}%)", "#dc2626", "-"),
        ("nl_bell36_clp50_ns_1T",   f"Refined 36x12x40 no-slip 1T ({nl_summary['nl_bell36_clp50_ns_1T']['p_peak_late']:.3f}%)", "#1d4ed8", "-"),
        ("nl_bell36_clp50_ns_st1T", f"Refined 36x12x40 no-slip st1T ({nl_summary['nl_bell36_clp50_ns_st1T']['p_peak_late']:.3f}%)", "#0891b2", "--"),
        ("nl_bell36_clp50_ns_1L",   f"Refined 36x12x40 no-slip 1L ({nl_summary['nl_bell36_clp50_ns_1L']['p_peak_late']:.3f}%)", "#059669", "-."),
        ("nl_bell_clp60_ns_1T",     f"6.0% Clipped no-slip 1T ({nl_summary['nl_bell_clp60_ns_1T']['p_peak_late']:.3f}%)", "#111827", ":"),
        ("nl_bell_clp60_ns_st1T",   f"6.0% Clipped no-slip st1T ({nl_summary['nl_bell_clp60_ns_st1T']['p_peak_late']:.3f}%)", "#4b5563", "--"),
    ]:
        r = p2_res[key]
        axes[1].semilogy(r["t"], np.maximum(r["e_ch"], 1e-13), color=col, ls=ls, lw=1.9, label=lbl)
    # Also show wake-drift equilibrium floor
    rd = p1_res["drift_bell_noslip_eq"]
    axes[1].semilogy(rd["t"], np.maximum(rd["e_ch"], 1e-18), color="#9ca3af", ls=":", lw=1.4, label=f"No-slip wake drift floor ({rd['e_ch'][-1]:.1e})")
    axes[1].set_xlabel(r"Time $t$")
    axes[1].set_ylabel(r"Chamber wave energy $E_{\rm wave}(t)$")
    axes[1].set_title("(b) Refined 36x12x40 Converged No-Slip Audit\n(& 6.0% Margin + Zero Wake-Drift Floor)")
    axes[1].set_ylim(1e-16, 2e-2); axes[1].grid(True, alpha=0.28)
    axes[1].legend(loc="lower right", fontsize=7.5, framealpha=0.92)
    fig.savefig(RESULTS_DIR / "exp017_all6mode_noslip_bell_quench.png", dpi=180)
    plt.close(fig)

    print(f"Done in {time.perf_counter() - t_start:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
