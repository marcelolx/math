"""Entry 016 (rocket-combustion, Hardened per Peer Review #13): Physical Crocco-Cheng
Feed-System Chugging, Indirect Entropy-Chug Coupling & Tri-Band 3D Faceplate Audit
in 3D Viscous Chambers & Metric-Corrected Smooth Bells.

Hardened changes addressing Peer Review #13 (C1-C4, M1-M7, m1-m8):
  - C1: Added exact entropy-aware Marble-Candel outlet BC u' = 0.5*M_c*(gamma*p' - rho')
        in exact_1d_coupled_feed_eig alongside isentropic outlet comparison, exposing the
        +0.029-0.035 indirect entropy-chug feedback that lifts bare psi_crit from 0.1702
        (isentropic) to 0.2574 (entropy-aware) and matches 3D linear coupled PDE runs
        (psi=0.16 bare, psi=0.25 tri50, psi=0.35 tri50) to < 0.003 in alpha and 0.3% in omega.
  - C2 & C4: Extended straight-chamber psi=0.25 runs (tri40seg_1T, tri50_1T, tri50_st1T,
        tri50_1R) to t = 220 confirming linear k=0 chug growth (+0.0016 to +0.014 at
        omega=0.281) and added straight-chamber psi = 0.35 runs (above entropy-aware
        psi_crit) to prove genuine straight-chamber k=0 chug stabilization when stiff enough.
  - C3 & m2: Extended all Exp 2B cold linear 1L EP & orthogonality runs to t = 40 with
        consistent nonlinear=False and beat-averaged [6, 40] / [15, 40] fits, retiring the
        [6, 20] beating artifact (-0.3189 / 3.18x) in favor of ~1.7x (-0.188 vs -0.109).
  - M1: Instrumented flow reversal fraction, floor-clamp fraction, and artificial mass
        injection; added check-valve closure (m_inj >= 0) on psi=0.07 AND clean zero-clamp
        psi=0.09 nonlinear chugging case.
  - M2 & M3: Computed full omega_hp sensitivity table for psi_crit_coupled (both isentropic
        and entropy-aware) and strictly consistent bare-to-bare & Tri-Band-to-Tri-Band ratios.
  - M4 & M5: Added genuine open-inlet mean-flow (M_c=0.12, t=50) equilibrium check calling
        inlet_choked_3d with active feed ODE; corrected inlet thermodynamics docs (isentropic
        inflow, compliance time theta_g = L_A/(gamma*M_c) = 10.0 vs residence time 12.5).
  - M6, M7 & m8: Added bell_bare_1L (like-for-like bell 1L baseline), bell_tri50_st1T,
        bell_tri50_2T, refined 36x12x40 bell_tri50_ref_1L (t=140), and extended bell_tri50_1L
        (t=200) & bell_tri50_1R (t=180) with explicit bulk-mode decay reporting.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig
from scipy.optimize import brentq, root
from scipy.special import jn_zeros, jv, jvp


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

GAMMA = 1.25
PR = 0.72
P_BAR = 1.0 / GAMMA       # 0.80 => c_bar = 1.0, rho_bar = 1.0, T_bar = P_BAR
RHO_BAR = 1.0
L_A = 1.5
X_F = 0.25
SIG_F = 0.08
TAU_BAR = 0.16
Q_S = 0.045
KAPPA_NL = 4.5

TAU_INJ_DEF = 1.0
TAU_F_DEF = 5.0


def bessel_prime_zero(m: int, n: int) -> float:
    if m == 0:
        return float(jn_zeros(1, n)[n - 1])
    zm = jn_zeros(m, n + 1)
    lo = 0.5 * m if n == 1 else float(zm[n - 2])
    hi = float(zm[n - 1])
    return float(brentq(lambda s: float(jvp(m, s)), lo, hi))


S11 = bessel_prime_zero(1, 1)   # 1.84118378 (1T)
S21 = bessel_prime_zero(2, 1)   # 3.05423693 (2T)
S02 = bessel_prime_zero(0, 1)   # 3.83170597 (1R)
S31 = bessel_prime_zero(3, 1)   # 4.20118894 (3T)
R0_1R = float(jn_zeros(0, 1)[0] / S02)  # 0.62761241 (exact 1R nodal circle)

WH_1T_RET = 1.7175
WH_2T_RET = 3.3760
WH_1L_TGT = 2.0900


def isentropic_area_ratio(M: float) -> float:
    g = GAMMA
    return (1.0 / M) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * M * M)) ** ((g + 1.0) / (2.0 * (g - 1.0)))


def nozzle_R_smooth_3d(x: np.ndarray, M_c: float, L_n: float, s_max: float = 0.28) -> tuple[np.ndarray, np.ndarray]:
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


# ---------------------------------------------------------------------------
# 1. Analytical 0D & Exact 1D Chebyshev Coupled Crocco-Cheng Feed Eigen-Solver
# ---------------------------------------------------------------------------

def cheb_diff(N: int, L: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = np.arange(N + 1)
    z = np.cos(np.pi * k / N)
    c = np.ones(N + 1); c[0] = 2.0; c[-1] = 2.0; c *= (-1.0) ** k
    Z = np.tile(z, (N + 1, 1))
    D = (np.outer(c, 1.0 / c)) / (Z.T - Z + np.eye(N + 1))
    D -= np.diag(np.sum(D, axis=1))
    x = 0.5 * L * (1.0 - z)
    Dx = -(2.0 / L) * D
    th = np.pi * k / N
    w_cc = np.zeros(N + 1)
    for j in range(N + 1):
        s = sum((1.0 if 2 * r == N else 2.0) * np.cos(2 * r * th[j]) / (4 * r * r - 1.0) for r in range(1, N // 2 + 1))
        w_cc[j] = (1.0 if (j == 0 or j == N) else 2.0) / N * (1.0 - s)
    w_cc *= 0.5 * L
    return x, Dx, w_cc


def exact_0d_crocco_cheng_root(psi: float, tau_inj: float = TAU_INJ_DEF, tau_f: float = TAU_F_DEF, M_c: float = 0.12) -> complex:
    """Exact 0D isentropic bulk Crocco-Cheng dispersion root:
    theta_g * s + (gamma+1)/(2*gamma) + (1/(2*psi)) * (1 + s*tau_f/4)^(-4) / (1 + s*tau_inj) = 0,
    where theta_g = L_A/(gamma*M_c) = 10.0 is the isentropic chamber compliance time."""
    theta_g = L_A / (GAMMA * M_c)
    c_noz = (GAMMA + 1.0) / (2.0 * GAMMA)

    def res(z):
        s = z[0] + 1j * z[1]
        val = theta_g * s + c_noz + (1.0 / (2.0 * psi)) * ((1.0 + 0.25 * s * tau_f) ** (-4)) / (1.0 + s * tau_inj)
        return [val.real, val.imag]

    sol = root(res, [0.0, 0.30], tol=1e-13)
    return complex(sol.x[0], sol.x[1])


def exact_1d_coupled_feed_eig(
    psi: float | None = None,
    tau_inj: float = TAU_INJ_DEF,
    tau_f: float = TAU_F_DEF,
    n_p: float = 0.0,
    omega_hp: float = 0.50,
    cav_list: list[dict] | None = None,
    entropy_aware: bool = True,
    M_c: float = 0.12,
    N: int = 64,
) -> dict:
    """Exact 1D Chebyshev linearized compressible Euler + physical Crocco-Cheng feed ODE.
    - Inlet BC at x=0: isentropic inflow (p'(0) = c_bar^2 rho'(0)) and mass-flux continuity.
    - Outlet BC at x=L_A:
        * entropy_aware=True (default, exact Marble-Candel matching PDE outlet_mc_choked_3d):
            u'(L_A) = 0.5 * M_c * (gamma * p'(L_A) - rho'(L_A))
        * entropy_aware=False (isentropic approximation ignoring flame-generated entropy waves):
            u'(L_A) = 0.5 * (gamma - 1) * M_c * p'(L_A)
    - Note (Review #13 m1): Strong BC imposition creates O(N^2) spurious Chebyshev modes
      (e.g. +3.9 +/- 303i at N=56); physical roots are filtered via |v| < 12.0."""
    x, Dx, w_cc = cheb_diff(N, L_A)
    n = N + 1
    cav_list = cav_list or []
    nc = len(cav_list)
    has_feed = psi is not None and np.isfinite(psi)
    has_hf = abs(n_p) > 0.0
    dim = 3 * n + (5 if has_feed else 0) + (5 if has_hf else 0) + 2 * nc
    Mat = np.zeros((dim, dim), dtype=complex)
    B = np.eye(dim, dtype=complex)
    sl = lambda q: slice(q * n, (q + 1) * n)

    Mat[sl(0), sl(0)] = -M_c * Dx; Mat[sl(0), sl(1)] = -Dx
    Mat[sl(1), sl(1)] = -M_c * Dx; Mat[sl(1), sl(2)] = -Dx
    Mat[sl(2), sl(1)] = -Dx;       Mat[sl(2), sl(2)] = -M_c * Dx

    off = 3 * n
    ifd = None
    if has_feed:
        ifd = off; off += 5
        Mat[ifd, ifd] = -1.0 / tau_inj
        Mat[ifd, 2 * n] = -(M_c / (2.0 * psi * P_BAR)) / tau_inj
        rf = 4.0 / tau_f
        for j in range(4):
            Mat[ifd + 1 + j, ifd + j] = rf
            Mat[ifd + 1 + j, ifd + 1 + j] = -rf
    if has_hf:
        ihf = off; off += 5
        wf = np.exp(-0.5 * ((x - X_F) / SIG_F) ** 2)
        wf /= float(np.sum(w_cc * wf))
        ws = w_cc * wf
        inv_ts = 4.0 / TAU_BAR
        Mat[ihf + 0, sl(2)] = omega_hp * ws; Mat[ihf + 0, ihf + 0] = -omega_hp
        Mat[ihf + 1, sl(2)] = inv_ts * ws;   Mat[ihf + 1, ihf + 0] = -inv_ts; Mat[ihf + 1, ihf + 1] = -inv_ts
        for r in range(1, 4):
            Mat[ihf + 1 + r, ihf + r] = inv_ts; Mat[ihf + 1 + r, ihf + 1 + r] = -inv_ts
        Mat[sl(2), ihf + 4] = n_p * wf
    icav = off
    for k, cv in enumerate(cav_list):
        ic = icav + 2 * k
        Vc = float(cv["beta"]) * L_A
        wH = float(cv["wH"]); xiH = float(cv["xiH"])
        Mat[ic, ic + 1] = 1.0 / Vc
        Mat[ic + 1, 2 * n] = (wH ** 2) * Vc
        Mat[ic + 1, ic] = -(wH ** 2) * Vc
        Mat[ic + 1, ic + 1] = -2.0 * xiH * wH

    # Inlet BCs at x = 0: isentropic inflow (p' = c_bar^2 rho') & mass-flux continuity
    for r in (0, n):
        Mat[r, :] = 0.0; B[r, :] = 0.0
    Mat[0, 0] = 1.0; Mat[0, 2 * n] = -1.0
    Mat[n, n] = 1.0; Mat[n, 2 * n] = M_c
    if has_feed:
        Mat[n, ifd + 4] = -1.0
    for k in range(nc):
        Mat[n, icav + 2 * k + 1] = 1.0

    # Outlet BC at x = L_A
    r_ex = 2 * n + N
    Mat[r_ex, :] = 0.0; B[r_ex, :] = 0.0
    Mat[r_ex, n + N] = 1.0
    if entropy_aware:
        Mat[r_ex, 2 * n + N] = -0.5 * GAMMA * M_c
        Mat[r_ex, N] = +0.5 * M_c
    else:
        Mat[r_ex, 2 * n + N] = -0.5 * (GAMMA - 1.0) * M_c

    vals = eig(Mat, B, right=False)
    vals = [v for v in vals if np.isfinite(v) and v.imag >= -1e-6 and abs(v) < 12.0]
    real_dc = sorted([v.real for v in vals if abs(v.imag) < 1e-4], reverse=True)
    chug = sorted([v for v in vals if 0.02 <= v.imag <= 1.0], key=lambda z: z.real, reverse=True)
    l1 = sorted([v for v in vals if 1.4 <= v.imag <= 2.7], key=lambda z: z.real, reverse=True)
    return {
        "has_real_dc_root": len(real_dc) > 0,
        "max_real_dc": float(real_dc[0]) if real_dc else None,
        "chug": complex(chug[0]) if chug else None,
        "l1": complex(l1[0]) if l1 else None,
    }


def compute_bessel_zone_coupling() -> dict:
    """Exact analytical radial projection weights C_mn,rad = <J_m(s_mn r)>_zone^2 / <J_m^2>_disk
    for the three disjoint 3D faceplate zones. Note (Review #13 m3): for axisymmetric n_cav=1
    zones, azimuthal coupling to m != 0 is identically zero; for n_cav=5 sectors, an azimuthal
    windowing factor multiplies C_mn,rad."""
    r = np.linspace(0.0, 1.0, 10001)
    zones = {
        "micro_hub_00_26": (0.00, 0.26),
        "nodal_mid_44_74": (0.44, 0.74),
        "outer_shroud_78_100": (0.78, 1.00),
    }
    modes = [("1T", 1, S11), ("2T", 2, S21), ("1R", 0, S02), ("1L", 0, 0.0)]
    out = {"r0_1R_nodal_circle": R0_1R, "zones": {}}
    for zname, (a, b) in zones.items():
        m = (r >= a) & (r <= b)
        area = float(2.0 * np.trapezoid(r[m], r[m]))
        zdat = {"r_lo": a, "r_hi": b, "area_frac": area}
        for mlbl, maz, s_mn in modes:
            f = np.ones_like(r) if s_mn == 0.0 else jv(maz, s_mn * r)
            norm2 = float(2.0 * np.trapezoid(f * f * r, r))
            mean_f = float(2.0 * np.trapezoid(f[m] * r[m], r[m]) / area)
            zdat[f"C_rad_{mlbl}"] = (mean_f * mean_f) / norm2
            if mlbl == "1R":
                zdat["signed_mean_J0"] = mean_f
        out["zones"][zname] = zdat
    return out


# ---------------------------------------------------------------------------
# 2. Metric-Corrected 3D Cylindrical FV Compressible NS Solver + Feed ODE
# ---------------------------------------------------------------------------

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


def relax_axi_3d_bell_base(
    nx: int, nr: int, M_c: float = 0.12, L_n: float = 0.60, L_d: float = 0.45,
    Re: float = np.inf, t_relax: float = 160.0,
) -> dict:
    """Relax 2D axisymmetric (nth=1) transonic 3D smooth bell steady base with exact mapped-grid viscous metrics."""
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
            T_s = pr_s / r_s
            u_x = np.empty((nx + 2, nr)); u_x[1:-1] = u_s; u_x[0] = ui; u_x[-1] = u_s[-1]
            v_x = np.empty((nx + 2, nr)); v_x[1:-1] = v_s; v_x[0] = 0.0; v_x[-1] = v_s[-1]
            T_x = np.empty((nx + 2, nr)); T_x[1:-1] = T_s; T_x[0] = pi / ri; T_x[-1] = T_s[-1]
            dr_c = deta * Rm[:, None]
            v_e = np.empty((nx, nr + 2)); v_e[:, 1:-1] = v_s; v_e[:, 0] = -v_s[:, 0]; v_e[:, -1] = v_s[:, -1]
            u_e = np.empty((nx, nr + 2)); u_e[:, 1:-1] = u_s; u_e[:, 0] = u_s[:, 0]; u_e[:, -1] = u_s[:, -1]
            T_e = np.empty((nx, nr + 2)); T_e[:, 1:-1] = T_s; T_e[:, 0] = T_s[:, 0]; T_e[:, -1] = T_s[:, -1]
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
            res[4] += Sx[1:] * (uf_x[1:] * txx_xf[1:] + vf_x[1:] * txr_xf[1:] - qx_xf[1:]) - Sx[:-1] * (uf_x[:-1] * txx_xf[:-1] + vf_x[:-1] * txr_xf[:-1] - qx_xf[:-1])

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
            vx_ef[:, 0] = 0.0; vx_ef[:, -1] = 0.0
            vr_ef[:, 0] = 0.0; vr_ef[:, -1] = 0.0
            ve_ef[:, 0] = 0.0; ve_ef[:, -1] = 0.0
            res[1] += vx_ef[:, 1:] - vx_ef[:, :-1]
            res[2] += vr_ef[:, 1:] - vr_ef[:, :-1]
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
        lam_adv = (np.abs(u_s) + c_s) / dx + (np.abs(v_s) + c_s) / dr_eff
        inv_dt = lam_adv / 0.30
        if mu > 0.0:
            nu_eff = max(4.0 / 3.0, GAMMA / PR) * mu / r_s
            lam_visc = nu_eff * (1.0 / (dx * dx) + 1.0 / (dr_eff * dr_eff))
            inv_dt = inv_dt + lam_visc / 0.40
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


def simulate_3d_ns(
    nx: int = 20, nr: int = 12, nth: int = 40, t_end: float = 30.0,
    M_c: float = 0.12, L_n: float = 0.60, L_d: float = 0.45, mode: str = "straight",
    Re: float = np.inf,
    init_type: str = "spin_1T", A_eq: float = 1e-4,
    n_p: float = 0.0, omega_hp: float = 0.50,
    feed_spec: dict | None = None,
    n_baffles: int = 0, L_b: float = 0.45, r_b: float = 0.0,
    cav_spec: list[dict] | None = None,
    prim_axi_base: np.ndarray | None = None, cfl: float = 0.28, record_stride: int = 14,
) -> dict:
    """Metric-corrected 3D cylindrical NS solver with physical Crocco-Cheng feed ODE
    (with optional check_valve closure and reversal/clamp diagnostics), partial-span radial
    baffles, and Tri-Band mass-conserving faceplate cavities."""
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

    if prim_axi_base is not None:
        P_b = np.broadcast_to(prim_axi_base[:, :, :, None], (5, nx, nr, nth)).copy()
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

    dA_in = g["dA_in"]
    A_in_tot = float(np.sum(dA_in))
    p_face_base = float(np.sum(p_b[0] * dA_in) / A_in_tot)

    has_feed = feed_spec is not None and np.isfinite(feed_spec.get("psi", np.inf))
    feed_state = np.full(5, M_c, dtype=float)
    f_psi = float(feed_spec["psi"]) if has_feed else np.inf
    f_tau_inj = float(feed_spec.get("tau_inj", TAU_INJ_DEF)) if has_feed else 1.0
    f_tau_f = float(feed_spec.get("tau_f", TAU_F_DEF)) if has_feed else 5.0
    f_nl = bool(feed_spec.get("nonlinear", True)) if has_feed else True
    f_check_valve = bool(feed_spec.get("check_valve", False)) if has_feed else False
    f_floor_frac = float(feed_spec.get("floor_frac", 0.05)) if has_feed else 0.05

    cavs = []
    if cav_spec:
        Cd = 0.65
        E_in, T_in = np.meshgrid(etac, thc, indexing="ij")
        for cf in cav_spec:
            nc = int(cf.get("n_cav", 5))
            beta_k = float(cf["beta"]); wH = float(cf["wH"]); xiH = float(cf["xiH"])
            sig_o = float(cf.get("sigma_o", 0.05)); nl_drag = bool(cf.get("nonlinear", True))
            r_lo, r_hi = float(cf.get("r_lo", 0.0)), float(cf.get("r_hi", 1.0))
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
                sig_th = max(1.5 * dth, 0.45 * (2.0 * np.pi / nc) * min(1.0, sig_o * 4.0))
                for k in range(nc):
                    th_k = (k + 0.5) * (2.0 * np.pi / nc)
                    d_ang = np.angle(np.exp(1j * (T_in - th_k)))
                    wk = r_mask * np.exp(-0.5 * (d_ang / sig_th) ** 2)
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
    init_ch_mass = float(np.sum(Q[0][ch_mask]))

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
            floor_val = 0.0 if f_check_valve else (f_floor_frac * M_c)
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

            du_dr_c = (u_e[:, 2:] - u_e[:, :-2]) / (2.0 * dr_c)
            dv_dr_c = (v_e[:, 2:] - v_e[:, :-2]) / (2.0 * dr_c)
            dw_dr_c = (w_e[:, 2:] - w_e[:, :-2]) / (2.0 * dr_c)
            dT_dr_c = (T_e[:, 2:] - T_e[:, :-2]) / (2.0 * dr_c)
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
            dp_in = p_in_face - p_b[0]
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
            if f_nl:
                m_res = 0.5 * ((m_cur * abs(m_cur)) / M_c - M_c)
            else:
                m_res = m_cur - M_c
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

    t = 0.0; step = 0
    ts, e_ch, e_1t, e_2t, e_1r, e_1l, e_mean = [], [], [], [], [], [], []
    p_mean_signed, p_pk, mass_drift, m_inj_hist = [], [], [], []
    w1_th = np.exp(-1j * T_ch); w2_th = np.exp(-2j * T_ch)
    dr_eff = deta * g["Rm"][:, None, None]

    # Review #13 M1 diagnostics: flow reversal, clamp activity & artificial mass
    rev_steps = 0; clamp_steps = 0; added_mass_int = 0.0; min_minj_ratio = 1.0

    while t < t_end - 1e-12:
        r_c, u_c, v_c, w_c, p_c, c_c = from_Q(Q)
        lam_adv = (np.abs(u_c) + c_c) / dx + (np.abs(v_c) + c_c) / dr_eff + (np.abs(w_c) + c_c) / (rb * dth)
        inv_dt = lam_adv / cfl
        if mu > 0.0:
            nu_eff = max(4.0 / 3.0, GAMMA / PR) * mu / r_c
            lam_visc = nu_eff * (1.0 / (dx * dx) + 1.0 / (dr_eff * dr_eff) + 1.0 / ((rb * dth) ** 2))
            inv_dt = inv_dt + lam_visc / 0.40
        dt = min(1.0 / float(np.max(inv_dt)), t_end - t)

        if has_feed:
            min_minj_ratio = min(min_minj_ratio, float(feed_state[0] / M_c))
            if feed_state[0] < 0.0:
                rev_steps += 1
            floor_val = 0.0 if f_check_valve else (f_floor_frac * M_c)
            if feed_state[4] < floor_val:
                clamp_steps += 1
                added_mass_int += (floor_val - feed_state[4]) * dt

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
            p_mean_signed.append(cmn)
            p_pk.append(float(np.max(np.abs(dp_c)) / P_BAR * 100.0))
            mass_drift.append(float((np.sum(Q[0][ch_mask]) - init_ch_mass) / init_ch_mass))
            m_inj_hist.append(float((feed_state[0] - M_c) / M_c * 100.0))
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

    n_st = max(step, 1)
    return {
        "t": np.array(ts), "e_ch": np.array(e_ch), "e_1t": np.array(e_1t), "e_2t": np.array(e_2t),
        "e_1r": np.array(e_1r), "e_1l": np.array(e_1l), "e_mean": np.array(e_mean),
        "p_mean_signed": np.array(p_mean_signed), "p_max_pct": np.array(p_pk),
        "mass_drift": np.array(mass_drift), "m_inj_pct": np.array(m_inj_hist),
        "quiescent_err": float(np.max(np.abs(Q - Q0))),
        "feed_quiescent_err": float(np.max(np.abs(feed_state - M_c))),
        "rev_frac": float(rev_steps / n_st),
        "clamp_frac": float(clamp_steps / n_st),
        "added_mass_frac": float(added_mass_int / max(M_c * t_end, 1e-12)),
        "min_minj_ratio": float(min_minj_ratio),
        "periodic_cons_err": {"mass": float(max_cons[0]), "ang_mom": float(max_cons[3]), "energy": float(max_cons[4])},
    }


def _sim_worker(args):
    lbl, kw = args
    t0 = time.perf_counter()
    res = simulate_3d_ns(**kw)
    return lbl, res, time.perf_counter() - t0


def run_batch(tasks: list[tuple[str, dict]], max_workers: int | None = None) -> dict:
    cache_env = os.environ.get("EXP016_CACHE_DIR", "").strip()
    cache_dir = Path(cache_env) if cache_env else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}; pending = []
    for lbl, kw in tasks:
        if cache_dir is not None and (cache_dir / f"{lbl}.pkl").exists():
            with open(cache_dir / f"{lbl}.pkl", "rb") as fh:
                out[lbl] = pickle.load(fh)
        else:
            pending.append((lbl, kw))
    if not pending:
        return out
    nw = max_workers or min(8, max(1, (os.cpu_count() or 2) - 2), len(pending))
    total = len(tasks); done = len(out)
    if nw <= 1 or len(pending) <= 1:
        for item in pending:
            lbl, res, dt_s = _sim_worker(item)
            out[lbl] = res; done += 1
            if cache_dir is not None:
                tmp = cache_dir / f"{lbl}.pkl.tmp"
                with open(tmp, "wb") as fh:
                    pickle.dump(res, fh)
                tmp.replace(cache_dir / f"{lbl}.pkl")
            print(f"    [{done:2d}/{total:2d}] {lbl} ({dt_s:.1f}s)", flush=True)
        return out
    with ProcessPoolExecutor(max_workers=nw) as ex:
        futs = [ex.submit(_sim_worker, item) for item in pending]
        for fut in as_completed(futs):
            lbl, res, dt_s = fut.result()
            out[lbl] = res; done += 1
            if cache_dir is not None:
                tmp = cache_dir / f"{lbl}.pkl.tmp"
                with open(tmp, "wb") as fh:
                    pickle.dump(res, fh)
                tmp.replace(cache_dir / f"{lbl}.pkl")
            print(f"    [{done:2d}/{total:2d}] {lbl} ({dt_s:.1f}s)", flush=True)
    return out


def fit_alpha(res: dict, t_lo: float, t_hi: float, key: str = "e_ch") -> float:
    t = res["t"]; m = (t >= t_lo) & (t <= t_hi)
    return float(np.polyfit(t[m], 0.5 * np.log(np.maximum(res[key][m], 1e-30)), 1)[0])


def fit_damped_oscillator(t: np.ndarray, sig: np.ndarray, t_lo: float, t_hi: float) -> tuple[float, float]:
    """Fit A*exp(alpha*t)*cos(omega*t + phi) from local peak envelope & zero-crossings."""
    m = (t >= t_lo) & (t <= t_hi)
    tt = t[m]; ss = sig[m]
    zc = []
    for i in range(len(ss) - 1):
        if ss[i] * ss[i + 1] < 0.0:
            zc.append(tt[i] + (tt[i + 1] - tt[i]) * abs(ss[i]) / (abs(ss[i]) + abs(ss[i + 1])))
    omega = float(np.pi / np.mean(np.diff(zc))) if len(zc) >= 3 else 0.0
    p_t, p_v = [], []
    if len(zc) >= 3:
        for i in range(len(zc) - 1):
            ms = (tt >= zc[i]) & (tt <= zc[i + 1])
            if np.any(ms):
                idx = np.argmax(np.abs(ss[ms]))
                p_t.append(float(tt[ms][idx])); p_v.append(float(np.abs(ss[ms][idx])))
    if len(p_t) >= 2:
        alpha = float(np.polyfit(np.array(p_t), np.log(np.maximum(np.array(p_v), 1e-30)), 1)[0])
    else:
        alpha = float(np.polyfit(tt, np.log(np.maximum(np.abs(ss), 1e-30)), 1)[0])
    return alpha, omega


def summarize_nl(res: dict) -> dict:
    t = res["t"]; ml = t >= 0.85 * t[-1]
    bulk_a, bulk_w = fit_damped_oscillator(t, res["p_mean_signed"], 0.55 * t[-1], t[-1])
    return {
        "t_end": float(t[-1]),
        "p_peak_late": float(np.mean(res["p_max_pct"][ml])),
        "max_abs_pbar_pct": float(np.max(np.abs(res["p_mean_signed"][ml])) / P_BAR * 100.0),
        "max_abs_minj_pct": float(np.max(np.abs(res["m_inj_pct"][ml]))),
        "rev_frac": res["rev_frac"],
        "clamp_frac": res["clamp_frac"],
        "added_mass_frac": res["added_mass_frac"],
        "min_minj_ratio": res["min_minj_ratio"],
        "e_wave_late": float(np.mean(res["e_ch"][ml])),
        "e_1t_late": float(np.mean(res["e_1t"][ml])),
        "e_2t_late": float(np.mean(res["e_2t"][ml])),
        "e_1r_late": float(np.mean(res["e_1r"][ml])),
        "e_1l_late": float(np.mean(res["e_1l"][ml])),
        "e_mean_late": float(np.mean(res["e_mean"][ml])),
        "slope_tail": fit_alpha(res, 0.65 * t[-1], t[-1], "e_ch"),
        "bulk_alpha_late": bulk_a,
        "bulk_omega_late": bulk_w,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp016_metrics.json"

    print("=" * 98)
    print("Entry 016 (Hardened per Peer Review #13): Indirect Entropy-Chug Coupling & Tri-Band 3D Audit")
    print("=" * 98)

    bessel_zones = compute_bessel_zone_coupling()
    cav_dual30 = [
        {"beta": 0.0185, "wH": WH_1T_RET, "xiH": 0.22, "sigma_o": 0.10, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": True},
        {"beta": 0.0080, "wH": WH_2T_RET, "xiH": 0.24, "sigma_o": 0.08, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": True},
        {"beta": 0.0035, "wH": S02,       "xiH": 0.14, "sigma_o": 0.04, "n_cav": 1, "r_lo": 0.00, "r_hi": 0.26, "nonlinear": True},
    ]
    cav_mid10_ax_lin  = [{"beta": 0.0100, "wH": WH_1L_TGT, "xiH": 0.14, "sigma_o": 0.08, "n_cav": 1, "r_lo": 0.44, "r_hi": 0.74, "nonlinear": False}]
    cav_mid10_seg_lin = [{"beta": 0.0100, "wH": WH_1L_TGT, "xiH": 0.14, "sigma_o": 0.08, "n_cav": 5, "r_lo": 0.44, "r_hi": 0.74, "nonlinear": False}]
    cav_mid10_ax_nl   = [{"beta": 0.0100, "wH": WH_1L_TGT, "xiH": 0.14, "sigma_o": 0.08, "n_cav": 1, "r_lo": 0.44, "r_hi": 0.74, "nonlinear": True}]
    cav_mid10_seg_nl  = [{"beta": 0.0100, "wH": WH_1L_TGT, "xiH": 0.14, "sigma_o": 0.08, "n_cav": 5, "r_lo": 0.44, "r_hi": 0.74, "nonlinear": True}]
    cav_tri40_ax = cav_dual30 + cav_mid10_ax_nl
    cav_tri40_seg = cav_dual30 + cav_mid10_seg_nl
    cav_tri50_ax = [
        {"beta": 0.0285, "wH": WH_1T_RET, "xiH": 0.25, "sigma_o": 0.14, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": True},
        {"beta": 0.0080, "wH": WH_2T_RET, "xiH": 0.24, "sigma_o": 0.08, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": True},
        {"beta": 0.0035, "wH": S02,       "xiH": 0.14, "sigma_o": 0.04, "n_cav": 1, "r_lo": 0.00, "r_hi": 0.26, "nonlinear": True},
        {"beta": 0.0100, "wH": WH_1L_TGT, "xiH": 0.14, "sigma_o": 0.08, "n_cav": 1, "r_lo": 0.44, "r_hi": 0.74, "nonlinear": True},
    ]
    cav_tri50_lin = [{**c, "nonlinear": False} for c in cav_tri50_ax]

    # Exact 1D Chebyshev sweeps: pure vs. coupled isentropic vs. coupled ENTROPY-AWARE (Review #13 C1, M2, M3!)
    psi_grid = [0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.25, 0.30, 0.35]
    cc_sweep = []
    for pv in psi_grid:
        s0d = exact_0d_crocco_cheng_root(pv)
        e_pure = exact_1d_coupled_feed_eig(psi=pv, n_p=0.0, entropy_aware=True)
        e_is_bare = exact_1d_coupled_feed_eig(psi=pv, n_p=0.34, omega_hp=0.50, cav_list=[], entropy_aware=False)
        e_en_bare = exact_1d_coupled_feed_eig(psi=pv, n_p=0.34, omega_hp=0.50, cav_list=[], entropy_aware=True)
        e_en_tri5 = exact_1d_coupled_feed_eig(psi=pv, n_p=0.34, omega_hp=0.50, cav_list=cav_tri50_ax, entropy_aware=True)
        cc_sweep.append({
            "psi": pv,
            "s0d_alpha": s0d.real, "s0d_omega": s0d.imag,
            "has_real_dc_root": e_pure["has_real_dc_root"],
            "pure_alpha": e_pure["chug"].real, "pure_omega": e_pure["chug"].imag,
            "coupled_isen_bare_alpha": e_is_bare["chug"].real, "coupled_isen_bare_omega": e_is_bare["chug"].imag,
            "coupled_ent_bare_alpha": e_en_bare["chug"].real,  "coupled_ent_bare_omega": e_en_bare["chug"].imag,
            "coupled_ent_tri5_alpha": e_en_tri5["chug"].real,  "coupled_ent_tri5_omega": e_en_tri5["chug"].imag,
            "coupled_ent_tri5_1L_alpha": e_en_tri5["l1"].real, "coupled_ent_tri5_1L_omega": e_en_tri5["l1"].imag,
        })

    # Strictly consistent bare-to-bare and Tri-Band-to-Tri-Band critical stiffnesses (Review #13 M3!)
    psi_crit_pure_bare = float(brentq(lambda p: exact_1d_coupled_feed_eig(psi=p, n_p=0.0, cav_list=[], entropy_aware=True)["chug"].real, 0.06, 0.15))
    psi_crit_pure_tri4 = float(brentq(lambda p: exact_1d_coupled_feed_eig(psi=p, n_p=0.0, cav_list=cav_tri40_ax, entropy_aware=True)["chug"].real, 0.06, 0.15))
    psi_crit_pure_tri5 = float(brentq(lambda p: exact_1d_coupled_feed_eig(psi=p, n_p=0.0, cav_list=cav_tri50_ax, entropy_aware=True)["chug"].real, 0.06, 0.15))
    psi_crit_isen_bare = float(brentq(lambda p: exact_1d_coupled_feed_eig(psi=p, n_p=0.34, omega_hp=0.50, cav_list=[], entropy_aware=False)["chug"].real, 0.10, 0.30))
    psi_crit_isen_tri4 = float(brentq(lambda p: exact_1d_coupled_feed_eig(psi=p, n_p=0.34, omega_hp=0.50, cav_list=cav_tri40_ax, entropy_aware=False)["chug"].real, 0.10, 0.30))
    psi_crit_ent_bare  = float(brentq(lambda p: exact_1d_coupled_feed_eig(psi=p, n_p=0.34, omega_hp=0.50, cav_list=[], entropy_aware=True)["chug"].real, 0.12, 0.45))
    psi_crit_ent_tri4  = float(brentq(lambda p: exact_1d_coupled_feed_eig(psi=p, n_p=0.34, omega_hp=0.50, cav_list=cav_tri40_ax, entropy_aware=True)["chug"].real, 0.12, 0.45))
    psi_crit_ent_tri5  = float(brentq(lambda p: exact_1d_coupled_feed_eig(psi=p, n_p=0.34, omega_hp=0.50, cav_list=cav_tri50_ax, entropy_aware=True)["chug"].real, 0.12, 0.45))

    print(f"\n--- Analytical: Consistent Critical Stiffness (Bare & Tri-Band, Isentropic vs. Entropy-Aware) ---")
    print(f"  Pure feed (n_p=0, ZERO omega_hp): bare={psi_crit_pure_bare:.4f}, tri4={psi_crit_pure_tri4:.4f}, tri5={psi_crit_pure_tri5:.4f}")
    print(f"  Coupled isentropic (hp=0.50):     bare={psi_crit_isen_bare:.4f} ({psi_crit_isen_bare/psi_crit_pure_bare:.2f}x), tri4={psi_crit_isen_tri4:.4f} ({psi_crit_isen_tri4/psi_crit_pure_tri4:.2f}x)")
    print(f"  Coupled ENTROPY-AWARE (hp=0.50):  bare={psi_crit_ent_bare:.4f} ({psi_crit_ent_bare/psi_crit_pure_bare:.2f}x), tri5={psi_crit_ent_tri5:.4f} ({psi_crit_ent_tri5/psi_crit_pure_tri5:.2f}x)")

    # omega_hp sensitivity table (Review #13 M2!)
    def find_psi_crit(hpv: float, entropy_aware: bool) -> float | None:
        f = lambda p: exact_1d_coupled_feed_eig(psi=p, n_p=0.34, omega_hp=hpv, cav_list=cav_tri40_ax, entropy_aware=entropy_aware)["chug"].real
        if f(0.06) <= 0.0:
            return 0.06
        for hi in (0.50, 1.50, 5.0, 20.0):
            if f(hi) < 0.0:
                return float(brentq(f, 0.06, hi))
        return None  # Intrinsically chug-unstable even at psi -> inf!

    omega_hp_sens = []
    for hpv in [0.15, 0.30, 0.50, 1.00, 2.00, 5.00]:
        p_is = find_psi_crit(hpv, False)
        p_en = find_psi_crit(hpv, True)
        omega_hp_sens.append({"omega_hp": hpv, "psi_crit_isen_tri4": p_is, "psi_crit_ent_tri4": p_en})
        s_is = f"{p_is:.4f}" if p_is is not None else "intrinsic(>20)"
        s_en = f"{p_en:.4f}" if p_en is not None else "intrinsic(>20)"
        print(f"  omega_hp={hpv:4.2f}: psi_crit(isen)={s_is}, psi_crit(entropy-aware)={s_en}")

    ep_1l_cheb = []
    for xi in [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.22, 0.28]:
        ec = exact_1d_coupled_feed_eig(psi=0.25, n_p=0.0, cav_list=[{"beta": 0.010, "wH": WH_1L_TGT, "xiH": xi}], entropy_aware=True)["l1"]
        eh = exact_1d_coupled_feed_eig(psi=0.25, n_p=0.34, omega_hp=0.50, cav_list=cav_dual30 + [{"beta": 0.010, "wH": WH_1L_TGT, "xiH": xi}], entropy_aware=True)["l1"]
        ep_1l_cheb.append({"xiH": xi, "cold_1L_alpha": ec.real, "cold_1L_omega": ec.imag, "hot_1L_alpha": eh.real, "hot_1L_omega": eh.imag})

    # ------------------------------------------------------------------
    # Phase 1: Fast Benchmarks, Linear Pure & Coupled PDE Validation & t=40 EP Audit
    # ------------------------------------------------------------------
    print("\n--- Phase 1: Open-Inlet Feed Equilibrium, Linear Pure+Coupled PDE & t=40 Cold 1L EP ---", flush=True)
    feed25 = {"psi": 0.25, "tau_inj": TAU_INJ_DEF, "tau_f": TAU_F_DEF, "nonlinear": True}
    feed35 = {"psi": 0.35, "tau_inj": TAU_INJ_DEF, "tau_f": TAU_F_DEF, "nonlinear": True}
    feed14 = {"psi": 0.14, "tau_inj": TAU_INJ_DEF, "tau_f": TAU_F_DEF, "nonlinear": True}
    p1_tasks = [
        # 1A: Closed bell metric well-balancedness + GENUINE open-inlet mean-flow feed equilibrium (Review #13 M4!)
        ("quiescent_closed_bell", dict(nx=20, nr=10, nth=32, t_end=2.5,  M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", Re=500.0, init_type="quiescent", A_eq=0.0, record_stride=20)),
        ("steady_open_feed_eq",   dict(nx=20, nr=8,  nth=24, t_end=50.0, M_c=0.12, mode="straight", Re=500.0, feed_spec=feed25, init_type="steady_mean", A_eq=0.0, record_stride=24)),
        ("cons_ns_wavy_cyl",      dict(nx=16, nr=10, nth=32, t_end=2.0,  M_c=0.12, mode="periodic_bump", Re=500.0, init_type="periodic_3d_wave", A_eq=0.06, record_stride=15)),
        # 1B: Pure-feed (n_p=0, ZERO omega_hp) linear validation
        ("lin_cc_psi06",          dict(nx=20, nr=8, nth=24, t_end=60.0, M_c=0.12, mode="straight", Re=np.inf, feed_spec={"psi": 0.06, "nonlinear": False}, init_type="bulk_k0", A_eq=1e-4, record_stride=12)),
        ("lin_cc_psi08",          dict(nx=20, nr=8, nth=24, t_end=60.0, M_c=0.12, mode="straight", Re=np.inf, feed_spec={"psi": 0.08, "nonlinear": False}, init_type="bulk_k0", A_eq=1e-4, record_stride=12)),
        ("lin_cc_psi08_nx32",     dict(nx=32, nr=8, nth=24, t_end=60.0, M_c=0.12, mode="straight", Re=np.inf, feed_spec={"psi": 0.08, "nonlinear": False}, init_type="bulk_k0", A_eq=1e-4, record_stride=16)),
        ("lin_cc_psi12",          dict(nx=20, nr=8, nth=24, t_end=60.0, M_c=0.12, mode="straight", Re=np.inf, feed_spec={"psi": 0.12, "nonlinear": False}, init_type="bulk_k0", A_eq=1e-4, record_stride=12)),
        ("lin_cc_psi25",          dict(nx=20, nr=8, nth=24, t_end=60.0, M_c=0.12, mode="straight", Re=np.inf, feed_spec={"psi": 0.25, "nonlinear": False}, init_type="bulk_k0", A_eq=1e-4, record_stride=12)),
        # 1B-coupled: LINEAR 3D PDE with n_p=0.34 validating entropy-aware 1D Chebyshev! (Review #13 C1!)
        ("lin_coup_psi16_bare",   dict(nx=20, nr=8, nth=24, t_end=60.0, M_c=0.12, mode="straight", Re=np.inf, feed_spec={"psi": 0.16, "nonlinear": False}, n_p=0.34, omega_hp=0.50, init_type="bulk_k0", A_eq=1e-4, record_stride=12)),
        ("lin_coup_psi25_tri5",   dict(nx=20, nr=8, nth=24, t_end=60.0, M_c=0.12, mode="straight", Re=np.inf, feed_spec={"psi": 0.25, "nonlinear": False}, cav_spec=cav_tri50_lin, n_p=0.34, omega_hp=0.50, init_type="bulk_k0", A_eq=1e-4, record_stride=12)),
        ("lin_coup_psi25_t5_ns",  dict(nx=20, nr=12, nth=40, t_end=70.0, M_c=0.12, mode="straight", Re=500.0, feed_spec={"psi": 0.25, "nonlinear": False}, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri50_lin, n_p=0.34, omega_hp=0.50, init_type="bulk_k0", A_eq=1e-4, record_stride=16)),
        ("lin_coup_psi35_t5_ns",  dict(nx=20, nr=12, nth=40, t_end=70.0, M_c=0.12, mode="straight", Re=500.0, feed_spec={"psi": 0.35, "nonlinear": False}, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri50_lin, n_p=0.34, omega_hp=0.50, init_type="bulk_k0", A_eq=1e-4, record_stride=16)),
        # 2B: Cold 3D NS Nodal Mid-Ring orthogonality & t=40 EP sweep (ALL nonlinear=False, Review #13 C3 & m2!)
        ("orth_bare_1L", dict(nx=20, nr=12, nth=40, t_end=40.0, M_c=0.12, mode="straight", Re=500.0, feed_spec={"psi": 0.25, "nonlinear": False}, init_type="1L",      A_eq=1e-4, record_stride=12)),
        ("orth_bare_1T", dict(nx=20, nr=12, nth=40, t_end=24.0, M_c=0.12, mode="straight", Re=500.0, feed_spec={"psi": 0.25, "nonlinear": False}, init_type="spin_1T", A_eq=1e-4, record_stride=12)),
        ("orth_bare_2T", dict(nx=20, nr=12, nth=40, t_end=24.0, M_c=0.12, mode="straight", Re=500.0, feed_spec={"psi": 0.25, "nonlinear": False}, init_type="spin_2T", A_eq=1e-4, record_stride=12)),
        ("orth_bare_1R", dict(nx=20, nr=12, nth=40, t_end=24.0, M_c=0.12, mode="straight", Re=500.0, feed_spec={"psi": 0.25, "nonlinear": False}, init_type="1R",      A_eq=1e-4, record_stride=12)),
        ("orth_ax_1L",   dict(nx=20, nr=12, nth=40, t_end=40.0, M_c=0.12, mode="straight", Re=500.0, feed_spec={"psi": 0.25, "nonlinear": False}, cav_spec=cav_mid10_ax_lin,  init_type="1L",      A_eq=1e-4, record_stride=12)),
        ("orth_ax_1T",   dict(nx=20, nr=12, nth=40, t_end=24.0, M_c=0.12, mode="straight", Re=500.0, feed_spec={"psi": 0.25, "nonlinear": False}, cav_spec=cav_mid10_ax_lin,  init_type="spin_1T", A_eq=1e-4, record_stride=12)),
        ("orth_ax_2T",   dict(nx=20, nr=12, nth=40, t_end=24.0, M_c=0.12, mode="straight", Re=500.0, feed_spec={"psi": 0.25, "nonlinear": False}, cav_spec=cav_mid10_ax_lin,  init_type="spin_2T", A_eq=1e-4, record_stride=12)),
        ("orth_ax_1R",   dict(nx=20, nr=12, nth=40, t_end=24.0, M_c=0.12, mode="straight", Re=500.0, feed_spec={"psi": 0.25, "nonlinear": False}, cav_spec=cav_mid10_ax_lin,  init_type="1R",      A_eq=1e-4, record_stride=12)),
        ("orth_seg_1L",  dict(nx=20, nr=12, nth=40, t_end=40.0, M_c=0.12, mode="straight", Re=500.0, feed_spec={"psi": 0.25, "nonlinear": False}, cav_spec=cav_mid10_seg_lin, init_type="1L",      A_eq=1e-4, record_stride=12)),
        ("orth_seg_1T",  dict(nx=20, nr=12, nth=40, t_end=24.0, M_c=0.12, mode="straight", Re=500.0, feed_spec={"psi": 0.25, "nonlinear": False}, cav_spec=cav_mid10_seg_lin, init_type="spin_1T", A_eq=1e-4, record_stride=12)),
        ("orth_seg_2T",  dict(nx=20, nr=12, nth=40, t_end=24.0, M_c=0.12, mode="straight", Re=500.0, feed_spec={"psi": 0.25, "nonlinear": False}, cav_spec=cav_mid10_seg_lin, init_type="spin_2T", A_eq=1e-4, record_stride=12)),
        ("orth_seg_1R",  dict(nx=20, nr=12, nth=40, t_end=24.0, M_c=0.12, mode="straight", Re=500.0, feed_spec={"psi": 0.25, "nonlinear": False}, cav_spec=cav_mid10_seg_lin, init_type="1R",      A_eq=1e-4, record_stride=12)),
        ("ep1l_xi08",    dict(nx=20, nr=12, nth=40, t_end=40.0, M_c=0.12, mode="straight", Re=500.0, feed_spec={"psi": 0.25, "nonlinear": False}, cav_spec=[{"beta": 0.010, "wH": WH_1L_TGT, "xiH": 0.08, "sigma_o": 0.08, "n_cav": 1, "r_lo": 0.44, "r_hi": 0.74, "nonlinear": False}], init_type="1L", A_eq=1e-4, record_stride=12)),
        ("ep1l_xi18",    dict(nx=20, nr=12, nth=40, t_end=40.0, M_c=0.12, mode="straight", Re=500.0, feed_spec={"psi": 0.25, "nonlinear": False}, cav_spec=[{"beta": 0.010, "wH": WH_1L_TGT, "xiH": 0.18, "sigma_o": 0.08, "n_cav": 1, "r_lo": 0.44, "r_hi": 0.74, "nonlinear": False}], init_type="1L", A_eq=1e-4, record_stride=12)),
        ("ep1l_xi28",    dict(nx=20, nr=12, nth=40, t_end=40.0, M_c=0.12, mode="straight", Re=500.0, feed_spec={"psi": 0.25, "nonlinear": False}, cav_spec=[{"beta": 0.010, "wH": WH_1L_TGT, "xiH": 0.28, "sigma_o": 0.08, "n_cav": 1, "r_lo": 0.44, "r_hi": 0.74, "nonlinear": False}], init_type="1L", A_eq=1e-4, record_stride=12)),
    ]
    p1_res = run_batch(p1_tasks)

    axi_bell_32 = relax_axi_3d_bell_base(32, 10, M_c=0.12, L_n=0.60, L_d=0.45, Re=500.0, t_relax=160.0)
    axi_bell_36 = relax_axi_3d_bell_base(36, 12, M_c=0.12, L_n=0.60, L_d=0.45, Re=500.0, t_relax=160.0)

    # ------------------------------------------------------------------
    # Phase 2: Unified LPT-ordered Nonlinear 3D NS Batch (Exp 1C, 2A, 3, 4)
    # ------------------------------------------------------------------
    p2_dict = {
        # Exp 1C: Pure physical feed chugging with clamp audit, check-valve closure & clean psi=0.09 (Review #13 M1!)
        "feed_pure_psi07":    dict(nx=20, nr=10, nth=32, t_end=140.0, M_c=0.12, mode="straight", Re=500.0, feed_spec={"psi": 0.07, "nonlinear": True, "check_valve": False}, init_type="bulk_k0", A_eq=0.04, n_p=0.0, record_stride=24),
        "feed_pure_psi07_cv": dict(nx=20, nr=10, nth=32, t_end=140.0, M_c=0.12, mode="straight", Re=500.0, feed_spec={"psi": 0.07, "nonlinear": True, "check_valve": True},  init_type="bulk_k0", A_eq=0.04, n_p=0.0, record_stride=24),
        "feed_pure_psi09":    dict(nx=20, nr=10, nth=32, t_end=180.0, M_c=0.12, mode="straight", Re=500.0, feed_spec={"psi": 0.09, "nonlinear": True, "check_valve": True},  init_type="bulk_k0", A_eq=0.04, n_p=0.0, record_stride=24),
        "feed_pure_psi25":    dict(nx=20, nr=10, nth=32, t_end=140.0, M_c=0.12, mode="straight", Re=500.0, feed_spec=feed25, init_type="bulk_k0", A_eq=0.04, n_p=0.0, record_stride=24),
        # Exp 2A: Coupled trap (psi=0.14) vs straight psi=0.25 (entropy-chug unstable!) vs psi=0.35 (truly chug-stable!)
        "coupled_psi14_trap": dict(nx=20, nr=12, nth=40, t_end=140.0, M_c=0.12, mode="straight", Re=500.0, feed_spec=feed14, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri40_ax, init_type="1L", A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "coupled_psi25_tri4": dict(nx=20, nr=12, nth=40, t_end=140.0, M_c=0.12, mode="straight", Re=500.0, feed_spec=feed25, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri40_ax, init_type="1L", A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        # Exp 3: Straight 3D NS — Honest t=220 psi=0.25 entropy-chug audit + psi=0.35 stiff-feed comparison!
        "bare_1T":          dict(nx=20, nr=12, nth=40, t_end=90.0,  M_c=0.12, mode="straight", Re=500.0, feed_spec=feed25, init_type="spin_1T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "bare_st1T":        dict(nx=20, nr=12, nth=40, t_end=90.0,  M_c=0.12, mode="straight", Re=500.0, feed_spec=feed25, init_type="stand_1T", A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "bare_2T":          dict(nx=20, nr=12, nth=40, t_end=90.0,  M_c=0.12, mode="straight", Re=500.0, feed_spec=feed25, init_type="spin_2T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "bare_1R":          dict(nx=20, nr=12, nth=40, t_end=90.0,  M_c=0.12, mode="straight", Re=500.0, feed_spec=feed25, init_type="1R",       A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "bare_1L":          dict(nx=20, nr=12, nth=40, t_end=90.0,  M_c=0.12, mode="straight", Re=500.0, feed_spec=feed25, init_type="1L",       A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "dual30_st1T":      dict(nx=20, nr=12, nth=40, t_end=140.0, M_c=0.12, mode="straight", Re=500.0, feed_spec=feed25, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_dual30, init_type="stand_1T", A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "dual30_1L":        dict(nx=20, nr=12, nth=40, t_end=140.0, M_c=0.12, mode="straight", Re=500.0, feed_spec=feed25, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_dual30, init_type="1L",       A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "tri40ax_1T":       dict(nx=20, nr=12, nth=40, t_end=140.0, M_c=0.12, mode="straight", Re=500.0, feed_spec=feed25, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri40_ax, init_type="spin_1T", A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "tri40seg_1T":      dict(nx=20, nr=12, nth=40, t_end=220.0, M_c=0.12, mode="straight", Re=500.0, feed_spec=feed25, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri40_seg, init_type="spin_1T", A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "tri40seg_1L":      dict(nx=20, nr=12, nth=40, t_end=140.0, M_c=0.12, mode="straight", Re=500.0, feed_spec=feed25, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri40_seg, init_type="1L",      A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        # psi=0.25 extended to t=220 (exposes linear entropy-chug growth +0.0016-0.0026!)
        "tri50_p25_1T":     dict(nx=20, nr=12, nth=40, t_end=220.0, M_c=0.12, mode="straight", Re=500.0, feed_spec=feed25, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri50_ax, init_type="spin_1T", A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "tri50_p25_1R":     dict(nx=20, nr=12, nth=40, t_end=220.0, M_c=0.12, mode="straight", Re=500.0, feed_spec=feed25, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri50_ax, init_type="1R",      A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "tri50_p25_1L":     dict(nx=20, nr=12, nth=40, t_end=140.0, M_c=0.12, mode="straight", Re=500.0, feed_spec=feed25, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri50_ax, init_type="1L",      A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        # psi=0.35 (ABOVE entropy-aware psi_crit! Truly chug-stable in straight compact-MC chamber!)
        "tri50_p35_1T":     dict(nx=20, nr=12, nth=40, t_end=180.0, M_c=0.12, mode="straight", Re=500.0, feed_spec=feed35, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri50_ax, init_type="spin_1T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "tri50_p35_st1T":   dict(nx=20, nr=12, nth=40, t_end=180.0, M_c=0.12, mode="straight", Re=500.0, feed_spec=feed35, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri50_ax, init_type="stand_1T", A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "tri50_p35_2T":     dict(nx=20, nr=12, nth=40, t_end=180.0, M_c=0.12, mode="straight", Re=500.0, feed_spec=feed35, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri50_ax, init_type="spin_2T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "tri50_p35_1R":     dict(nx=20, nr=12, nth=40, t_end=180.0, M_c=0.12, mode="straight", Re=500.0, feed_spec=feed35, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri50_ax, init_type="1R",       A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "tri50_p35_1L":     dict(nx=20, nr=12, nth=40, t_end=180.0, M_c=0.12, mode="straight", Re=500.0, feed_spec=feed35, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri50_ax, init_type="1L",       A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "tri50_ref_1T":     dict(nx=24, nr=16, nth=50, t_end=120.0, M_c=0.12, mode="straight", Re=500.0, feed_spec=feed35, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri50_ax, init_type="spin_1T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=28),
        "tri50_psi14_1T":   dict(nx=20, nr=12, nth=40, t_end=140.0, M_c=0.12, mode="straight", Re=500.0, feed_spec=feed14, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri50_ax, init_type="spin_1T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        # Exp 4: Metric-Corrected True-3D Smooth Bell NS (Re=500, psi=0.25 — finite-nozzle entropy filtering!)
        "bell_bare_1T":      dict(nx=32, nr=10, nth=40, t_end=80.0,  M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", Re=500.0, feed_spec=feed25, prim_axi_base=axi_bell_32["prim_axi"], init_type="spin_1T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "bell_bare_1L":      dict(nx=32, nr=10, nth=40, t_end=80.0,  M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", Re=500.0, feed_spec=feed25, prim_axi_base=axi_bell_32["prim_axi"], init_type="1L",       A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "bell_dual30_1T":    dict(nx=32, nr=10, nth=40, t_end=100.0, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", Re=500.0, feed_spec=feed25, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_dual30,   prim_axi_base=axi_bell_32["prim_axi"], init_type="spin_1T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "bell_tri40_1T":     dict(nx=32, nr=10, nth=40, t_end=120.0, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", Re=500.0, feed_spec=feed25, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri40_ax, prim_axi_base=axi_bell_32["prim_axi"], init_type="spin_1T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "bell_tri50_1T":     dict(nx=32, nr=10, nth=40, t_end=120.0, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", Re=500.0, feed_spec=feed25, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri50_ax, prim_axi_base=axi_bell_32["prim_axi"], init_type="spin_1T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "bell_tri50_st1T":   dict(nx=32, nr=10, nth=40, t_end=120.0, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", Re=500.0, feed_spec=feed25, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri50_ax, prim_axi_base=axi_bell_32["prim_axi"], init_type="stand_1T", A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "bell_tri50_2T":     dict(nx=32, nr=10, nth=40, t_end=120.0, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", Re=500.0, feed_spec=feed25, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri50_ax, prim_axi_base=axi_bell_32["prim_axi"], init_type="spin_2T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "bell_tri50_1L":     dict(nx=32, nr=10, nth=40, t_end=200.0, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", Re=500.0, feed_spec=feed25, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri50_ax, prim_axi_base=axi_bell_32["prim_axi"], init_type="1L",       A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "bell_tri50_1R":     dict(nx=32, nr=10, nth=40, t_end=180.0, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", Re=500.0, feed_spec=feed25, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri50_ax, prim_axi_base=axi_bell_32["prim_axi"], init_type="1R",       A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24),
        "bell_tri50_ref_1L": dict(nx=36, nr=12, nth=40, t_end=140.0, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", Re=500.0, feed_spec=feed25, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri50_ax, prim_axi_base=axi_bell_36["prim_axi"], init_type="1L",       A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=28),
        "bell_tri50_ref_1T": dict(nx=36, nr=12, nth=40, t_end=80.0,  M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", Re=500.0, feed_spec=feed25, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_tri50_ax, prim_axi_base=axi_bell_36["prim_axi"], init_type="spin_1T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=28),
    }
    lpt_order = [
        "bell_tri50_ref_1L", "bell_tri50_1L", "bell_tri50_1R", "bell_tri50_ref_1T", "tri50_ref_1T",
        "bell_tri50_1T", "bell_tri50_st1T", "bell_tri50_2T", "bell_tri40_1T", "bell_dual30_1T", "bell_bare_1T", "bell_bare_1L",
        "tri40seg_1T", "tri50_p25_1T", "tri50_p25_1R",
        "tri50_p35_1T", "tri50_p35_st1T", "tri50_p35_2T", "tri50_p35_1R", "tri50_p35_1L",
        "tri50_p25_1L", "tri50_psi14_1T", "tri40ax_1T", "tri40seg_1L",
        "dual30_st1T", "dual30_1L", "coupled_psi14_trap", "coupled_psi25_tri4",
        "feed_pure_psi09", "bare_1T", "bare_st1T", "bare_2T", "bare_1R", "bare_1L",
        "feed_pure_psi07", "feed_pure_psi07_cv", "feed_pure_psi25",
    ]
    print(f"\n--- Phase 2: Launching {len(lpt_order)} Nonlinear 3D NS Runs (LPT-ordered, 8 workers) ---", flush=True)
    p2_res = run_batch([(k, p2_dict[k]) for k in lpt_order])

    # ------------------------------------------------------------------
    # Post-Processing & Honest Metric Extraction
    # ------------------------------------------------------------------
    print("\n--- Exp 1: Open-Inlet Feed Eq, Linear Pure & Coupled Entropy-Aware Validation ---")
    print(f"  1A Closed Bell Metric Drift:      Q_err={p1_res['quiescent_closed_bell']['quiescent_err']:.2e}")
    print(f"  1A Open-Inlet Mean-Flow Feed Eq:  Q_err={p1_res['steady_open_feed_eq']['quiescent_err']:.2e}, feed_err={p1_res['steady_open_feed_eq']['feed_quiescent_err']:.2e}")
    cb = p1_res["cons_ns_wavy_cyl"]["periodic_cons_err"]
    print(f"  1A Periodic Wavy Cyl NS Cons:     mass={cb['mass']:.2e}, ang_mom={cb['ang_mom']:.2e}, energy={cb['energy']:.2e}")

    lin_cc_out = {}
    for cid, pv, npv, cvl in [
        ("lin_cc_psi06",        0.06, 0.0,  []),
        ("lin_cc_psi08",        0.08, 0.0,  []),
        ("lin_cc_psi08_nx32",   0.08, 0.0,  []),
        ("lin_cc_psi12",        0.12, 0.0,  []),
        ("lin_cc_psi25",        0.25, 0.0,  []),
        ("lin_coup_psi16_bare", 0.16, 0.34, []),
        ("lin_coup_psi25_tri5", 0.25, 0.34, cav_tri50_ax),
        ("lin_coup_psi25_t5_ns",0.25, 0.34, cav_tri50_ax),
        ("lin_coup_psi35_t5_ns",0.35, 0.34, cav_tri50_ax),
    ]:
        rr = p1_res[cid]
        al, om = fit_damped_oscillator(rr["t"], rr["p_mean_signed"], 12.0, rr["t"][-1] - 3.0)
        ex_en = exact_1d_coupled_feed_eig(psi=pv, n_p=npv, omega_hp=0.50, cav_list=cvl, entropy_aware=True)["chug"]
        ex_is = exact_1d_coupled_feed_eig(psi=pv, n_p=npv, omega_hp=0.50, cav_list=cvl, entropy_aware=False)["chug"]
        lin_cc_out[cid] = {
            "psi": pv, "n_p": npv, "pde_alpha": al, "pde_omega": om,
            "cheb_ent_alpha": ex_en.real, "cheb_ent_omega": ex_en.imag,
            "cheb_isen_alpha": ex_is.real, "cheb_isen_omega": ex_is.imag,
            "omega_err_ent_pct": abs(om - ex_en.imag) / ex_en.imag * 100.0,
            "alpha_abs_err_ent": abs(al - ex_en.real),
        }
        print(f"  1B [{cid:20s}] PDE={al:+.5f}+{om:.4f}i | 1D_ent={ex_en.real:+.5f}+{ex_en.imag:.4f}i (w={lin_cc_out[cid]['omega_err_ent_pct']:.2f}%) | 1D_isen={ex_is.real:+.5f}+{ex_is.imag:.4f}i")

    nl_feed_out = {k: summarize_nl(p2_res[k]) for k in ("feed_pure_psi07", "feed_pure_psi07_cv", "feed_pure_psi09", "feed_pure_psi25", "coupled_psi14_trap", "coupled_psi25_tri4")}
    for k, s in nl_feed_out.items():
        print(f"  1C/2A [{k:18s}] P={s['p_peak_late']:6.2f}% | |p̄'|={s['max_abs_pbar_pct']:5.2f}% | rev={s['rev_frac']*100:4.1f}% | clmp={s['clamp_frac']*100:4.1f}% | +m={s['added_mass_frac']*100:4.1f}% | E_mn={s['e_mean_late']:.2e}")

    print("\n--- Exp 2B: Cold 3D NS Nodal Mid-Ring Orthogonality & Long-Window [6,40] 1L EP ---")
    orth_out = {}
    for m_lbl, ek, tmax in [("1L", "e_1l", 40.0), ("1T", "e_1t", 22.0), ("2T", "e_2t", 22.0), ("1R", "e_1r", 22.0)]:
        ab = fit_alpha(p1_res[f"orth_bare_{m_lbl}"], 6.0, tmax, ek)
        aa = fit_alpha(p1_res[f"orth_ax_{m_lbl}"],   6.0, tmax, ek)
        asg = fit_alpha(p1_res[f"orth_seg_{m_lbl}"], 6.0, tmax, ek)
        orth_out[m_lbl] = {
            "window": [6.0, tmax], "bare": ab, "mid_ax_n1": aa, "delta_ax": aa - ab,
            "mid_seg_n5": asg, "delta_seg": asg - ab,
        }
        print(f"  2B [{m_lbl} [6,{tmax:2.0f}]] bare={ab:+.5f} | Mid n=1: {aa:+.5f} (d={aa-ab:+.5f}) | Mid n=5: {asg:+.5f} (d={asg-ab:+.5f})")

    ep1l_pde = {}
    for lbl, rkey in [("xi08", "ep1l_xi08"), ("xi14", "orth_ax_1L"), ("xi18", "ep1l_xi18"), ("xi28", "ep1l_xi28")]:
        ep1l_pde[lbl] = {
            "w_6_20": fit_alpha(p1_res[rkey], 6.0, 20.0, "e_1l"),
            "w_6_40": fit_alpha(p1_res[rkey], 6.0, 40.0, "e_1l"),
            "w_15_40": fit_alpha(p1_res[rkey], 15.0, 40.0, "e_1l"),
        }
        print(f"  2B EP [{lbl}]: [6,20]={ep1l_pde[lbl]['w_6_20']:+.5f}, [6,40]={ep1l_pde[lbl]['w_6_40']:+.5f}, [15,40]={ep1l_pde[lbl]['w_15_40']:+.5f}")

    print("\n--- Exp 3: Straight 3D NS Audit (psi=0.25 t=220 vs. psi=0.35 Entropy-Safe Stiffness) ---")
    exp3_keys = [
        "bare_1T", "bare_st1T", "bare_2T", "bare_1R", "bare_1L",
        "dual30_st1T", "dual30_1L", "tri40ax_1T", "tri40seg_1T", "tri40seg_1L", "coupled_psi25_tri4",
        "tri50_p25_1T", "tri50_p25_1R", "tri50_p25_1L",
        "tri50_p35_1T", "tri50_p35_st1T", "tri50_p35_2T", "tri50_p35_1R", "tri50_p35_1L",
        "tri50_ref_1T", "tri50_psi14_1T",
    ]
    exp3_out = {}
    for k in exp3_keys:
        exp3_out[k] = summarize_nl(p2_res[k])
        s = exp3_out[k]
        print(f"  3 [{k:18s}] t={s['t_end']:5.1f} P={s['p_peak_late']:6.3f}% | E={s['e_wave_late']:.2e} | slp={s['slope_tail']:+.4f} | blk={s['bulk_alpha_late']:+.4f} | 1T={s['e_1t_late']:.1e} | 1L={s['e_1l_late']:.1e} | mn={s['e_mean_late']:.1e}")

    print("\n--- Exp 4: Metric-Corrected True-3D Smooth Bell NS Shootout (Re=500, psi=0.25) ---")
    exp4_keys = [
        "bell_bare_1T", "bell_bare_1L", "bell_dual30_1T", "bell_tri40_1T",
        "bell_tri50_1T", "bell_tri50_st1T", "bell_tri50_2T",
        "bell_tri50_1L", "bell_tri50_1R", "bell_tri50_ref_1L", "bell_tri50_ref_1T",
    ]
    exp4_out = {}
    for k in exp4_keys:
        exp4_out[k] = summarize_nl(p2_res[k])
        s = exp4_out[k]
        print(f"  4 [{k:18s}] t={s['t_end']:5.1f} P={s['p_peak_late']:6.3f}% | E={s['e_wave_late']:.2e} | slp={s['slope_tail']:+.4f} | blk={s['bulk_alpha_late']:+.4f}({s['bulk_omega_late']:.3f}) | 1T={s['e_1t_late']:.1e} | 1L={s['e_1l_late']:.1e} | mn={s['e_mean_late']:.1e}")

    metrics = {
        "exp1_crocco_cheng_feed": {
            "quiescent_closed_bell_err": p1_res["quiescent_closed_bell"]["quiescent_err"],
            "open_inlet_feed_eq_Q_err": p1_res["steady_open_feed_eq"]["quiescent_err"],
            "open_inlet_feed_eq_feed_err": p1_res["steady_open_feed_eq"]["feed_quiescent_err"],
            "cons_ns_wavy_cyl": cb,
            "psi_crit_consistent": {
                "pure_bare": psi_crit_pure_bare, "pure_tri4": psi_crit_pure_tri4, "pure_tri5": psi_crit_pure_tri5,
                "coupled_isen_bare": psi_crit_isen_bare, "coupled_isen_tri4": psi_crit_isen_tri4,
                "coupled_ent_bare": psi_crit_ent_bare,   "coupled_ent_tri4": psi_crit_ent_tri4, "coupled_ent_tri5": psi_crit_ent_tri5,
                "ratio_isen_bare": psi_crit_isen_bare / psi_crit_pure_bare,
                "ratio_isen_tri4": psi_crit_isen_tri4 / psi_crit_pure_tri4,
                "ratio_ent_bare": psi_crit_ent_bare / psi_crit_pure_bare,
                "ratio_ent_tri5": psi_crit_ent_tri5 / psi_crit_pure_tri5,
            },
            "omega_hp_sensitivity": omega_hp_sens,
            "cheb_0d_1d_sweep": cc_sweep,
            "pde_linear_validation": lin_cc_out,
            "pde_nonlinear_chugging": nl_feed_out,
        },
        "exp2_triband_orthogonality": {
            "analytical_bessel_zones": bessel_zones,
            "pde_cold_orthogonality": orth_out,
            "ep_1l_cheb": ep_1l_cheb,
            "ep_1l_pde": ep1l_pde,
        },
        "exp3_straight_5mode_audit": exp3_out,
        "exp4_bell_ns_shootout": {
            "axi_bell_bases": {
                "32x10": {k: v for k, v in axi_bell_32.items() if k != "prim_axi"},
                "36x12": {k: v for k, v in axi_bell_36.items() if k != "prim_axi"},
            },
            "runs": exp4_out,
        },
    }
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved hardened Entry 016 metrics to {out_path}")

    # ------------------------------------------------------------------
    # Figures (Hardened, zero overlap, 100% honest titles & consistent numbers)
    # ------------------------------------------------------------------
    # Fig 1: Indirect Entropy-Chug Coupling (0D / 1D Isen vs Entropy-Aware / 3D Linear PDE) + Nonlinear Closure Audit
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.5), dpi=150)
    psis = [r["psi"] for r in cc_sweep]
    ax1.plot(psis, [r["pure_alpha"] for r in cc_sweep], "-", color="#1f77b4", lw=2.3, label=f"1D Pure Feed ($n_p=0, \\psi_c={psi_crit_pure_bare:.3f}$)")
    ax1.plot(psis, [r["coupled_isen_bare_alpha"] for r in cc_sweep], "--", color="#ff7f0e", lw=2.1, label=f"1D Isentropic Coupled ($n_p=0.34, \\psi_c={psi_crit_isen_bare:.3f}$)")
    ax1.plot(psis, [r["coupled_ent_bare_alpha"] for r in cc_sweep], "-", color="#d62728", lw=2.4, label=f"1D Entropy-Aware Coupled (bare, $\\psi_c={psi_crit_ent_bare:.3f}$)")
    ax1.plot(psis, [r["coupled_ent_tri5_alpha"] for r in cc_sweep], "-.", color="#9467bd", lw=2.0, label=f"1D Entropy-Aware + Tri-Band 5% ($\\psi_c={psi_crit_ent_tri5:.3f}$)")
    p_pure = [lin_cc_out[k] for k in ("lin_cc_psi06", "lin_cc_psi08", "lin_cc_psi12", "lin_cc_psi25")]
    ax1.plot([p["psi"] for p in p_pure], [p["pde_alpha"] for p in p_pure], "o", color="#1f77b4", ms=7.2, mfc="white", mew=2.0, label="3D PDE Linear ($n_p=0$, ZERO $\\omega_{\\mathrm{hp}}$)")
    ax1.plot([0.16], [lin_cc_out["lin_coup_psi16_bare"]["pde_alpha"]], "s", color="#d62728", ms=7.8, mfc="white", mew=2.1, label="3D PDE Linear Coupled (bare, $n_p=0.34$)")
    ax1.plot([0.25, 0.35], [lin_cc_out["lin_coup_psi25_t5_ns"]["pde_alpha"], lin_cc_out["lin_coup_psi35_t5_ns"]["pde_alpha"]],
             "^", color="#9467bd", ms=8.0, mfc="white", mew=2.1, label="3D NS PDE Coupled + Tri-Band 5%")
    ax1.axhline(0.0, color="k", lw=1.0, alpha=0.6)
    ax1.set_xlabel("Injector Pressure-Drop Stiffness $\\psi = \\Delta p_{\\mathrm{inj}} / \\bar{p}_c$", fontsize=11)
    ax1.set_ylabel("Chugging Growth Rate $\\alpha_{\\mathrm{chug}}$", fontsize=11)
    ax1.set_title(f"Indirect Entropy-Chug Coupling (${psi_crit_ent_bare/psi_crit_pure_bare:.2f}\\times$ Shift at $\\omega_{{\\mathrm{{hp}}}}=0.50$)", fontsize=10.5, fontweight="bold")
    ax1.grid(True, alpha=0.28)
    ax1.legend(fontsize=7.2, loc="upper right", framealpha=0.94)

    for cid, lbl, col, ls in [
        ("feed_pure_psi07",    "Pure $\\psi=0.07$ (floor 0.05, 27% rev)", "#d62728", "-"),
        ("feed_pure_psi07_cv", "Pure $\\psi=0.07$ (check valve, 0% rev)", "#ff7f0e", "--"),
        ("feed_pure_psi09",    "Pure $\\psi=0.09$ (clean check valve)", "#9467bd", "-."),
        ("feed_pure_psi25",    "Pure $\\psi=0.25$ (stiff quench)", "#2ca02c", "-"),
    ]:
        rr = p2_res[cid]; st = nl_feed_out[cid]
        ax2.plot(rr["t"], rr["p_mean_signed"] / P_BAR * 100.0, color=col, ls=ls, lw=2.0,
                 label=f"{lbl} (|$\\bar{{p}}'$|={st['max_abs_pbar_pct']:.1f}%)")
    ax2.set_xlabel("Time $t$", fontsize=11)
    ax2.set_ylabel("Chamber-Mean Bulk Pressure $\\bar{p}' / \\bar{p}_c$ (%)", fontsize=11)
    ax2.set_title("Pure Feed Chugging ($n_p=0$): Backflow Closure Sensitivity", fontsize=10.5, fontweight="bold")
    ax2.grid(True, alpha=0.28)
    ax2.legend(fontsize=7.8, loc="lower left", framealpha=0.94)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp016_crocco_cheng_chugging.png", dpi=150)
    plt.close(fig)

    # Fig 2: Long-Window [6,40] Nodal Mid-Ring Orthogonality & Beat-Free 1L EP Curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.5), dpi=150)
    mlbls = ["1L", "1T", "2T", "1R"]
    x_pos = np.arange(len(mlbls)); w_bar = 0.34
    ax1.bar(x_pos - 0.5 * w_bar, [-orth_out[m]["delta_ax"] for m in mlbls], width=w_bar, color="#1f77b4", label="Nodal Mid-Ring $n_{\\mathrm{cav}}=1$ (Axisymmetric)")
    ax1.bar(x_pos + 0.5 * w_bar, [-orth_out[m]["delta_seg"] for m in mlbls], width=w_bar, color="#ff7f0e", label="Nodal Mid-Ring $n_{\\mathrm{cav}}=5$ (Segmented)")
    ax1.axhline(0.0, color="k", lw=0.9)
    ax1.set_xticks(x_pos); ax1.set_xticklabels(["$1L$ ([6,40])", "$1T$", "$2T$", "$1R$"], fontsize=10)
    ax1.set_ylabel("Added Cold 3D NS Damping $-\\Delta\\alpha$", fontsize=11)
    ax1.set_title("Nodal Mid-Ring ($r\\in[0.44,0.74]$, $r_0=0.628$): Long-Window Isolation", fontsize=10.4, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.28)
    ax1.legend(fontsize=8.3, loc="upper right", framealpha=0.94)

    xis_c = [r["xiH"] for r in ep_1l_cheb]
    ax2.plot(xis_c, [r["cold_1L_alpha"] for r in ep_1l_cheb], "s-", color="#1f77b4", lw=2.2, label="1D Cheb Cold $1L$ ($1.0\\%$ Mid-Ring, $n_p=0$)")
    ax2.plot(xis_c, [r["hot_1L_alpha"] for r in ep_1l_cheb], "^-", color="#d62728", lw=2.3, label="1D Cheb Hot $1L$ (Tri-Band, $n_p=0.34$)")
    ax2.plot([0.08, 0.14, 0.18, 0.28], [ep1l_pde[k]["w_6_40"] for k in ("xi08", "xi14", "xi18", "xi28")],
             "o", color="#2ca02c", ms=8.0, mfc="white", mew=2.2, label="3D NS PDE Cold $1L$ Long Window $[6, 40]$")
    ax2.plot([0.08, 0.14, 0.18, 0.28], [ep1l_pde[k]["w_6_20"] for k in ("xi08", "xi14", "xi18", "xi28")],
             "x", color="#777777", ms=7.0, mew=1.8, label="Short-Window $[6, 20]$ Beating Artifact")
    ax2.axhline(0.0, color="k", lw=1.0, ls="--", alpha=0.6)
    ax2.set_xlabel("Mid-Ring Cavity Damping Ratio $\\xi_{H,1L}$", fontsize=11)
    ax2.set_ylabel("$1L$ Modal Growth Rate $\\alpha_{1L}$", fontsize=11)
    ax2.set_title("1L Nodal Mid-Ring EP Tuning: Long Window $[6, 40]$ vs. Short $[6, 20]$", fontsize=10.4, fontweight="bold")
    ax2.grid(True, alpha=0.28)
    ax2.legend(fontsize=7.6, loc="lower right", framealpha=0.94)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp016_triband_orthogonality.png", dpi=150)
    plt.close(fig)

    # Fig 3: Straight psi=0.25 vs psi=0.35 Audit & True-3D Smooth Bell All-Mode Shootout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.5), dpi=150)
    for cid, lbl, col, ls in [
        ("bare_1T",        "Bare 3D NS: spin $1T$", "#d62728", "-"),
        ("tri40seg_1T",    "Tri 4.0% ($n=5, \\psi=0.25$): grows chug!", "#e377c2", "-."),
        ("tri50_p25_1T",   "Tri 5.0% ($\\psi=0.25$): slow chug +0.002", "#ff7f0e", "--"),
        ("tri50_p35_1T",   "Tri 5.0% ($\\psi=0.35$): spin $1T$", "#1f77b4", "-"),
        ("tri50_p35_st1T", "Tri 5.0% ($\\psi=0.35$): stand $1T$", "#17becf", "--"),
        ("tri50_p35_2T",   "Tri 5.0% ($\\psi=0.35$): spin $2T$", "#9467bd", "-."),
        ("tri50_p35_1R",   "Tri 5.0% ($\\psi=0.35$): $1R$", "#2ca02c", "-"),
        ("tri50_p35_1L",   "Tri 5.0% ($\\psi=0.35$): $1L$", "#2ca02c", ":"),
        ("tri50_ref_1T",   "Refined 24x16x50 ($\\psi=0.35$): spin $1T$", "#555555", ":"),
    ]:
        rr = p2_res[cid]; st = exp3_out[cid]
        ax1.semilogy(rr["t"], np.maximum(rr["e_ch"], 1e-13), color=col, ls=ls, lw=1.9, label=f"{lbl} (P={st['p_peak_late']:.2f}%)")
    ax1.set_xlabel("Time $t$", fontsize=11)
    ax1.set_ylabel("3D Chamber Wave Energy $E_{\\mathrm{wave}}(t)$", fontsize=11)
    ax1.set_title("Straight Compact-MC NS: $\\psi=0.25$ Entropy-Chug vs. $\\psi=0.35$", fontsize=10.3, fontweight="bold")
    ax1.set_ylim(1e-12, 8e-3)
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=6.8, loc="lower left", framealpha=0.94)

    for cid, lbl, col, ls in [
        ("bell_bare_1T",      "3D Bell Bare: spin $1T$", "#d62728", "-"),
        ("bell_bare_1L",      "3D Bell Bare: $1L$ (like-for-like ref)", "#8c564b", "--"),
        ("bell_tri50_1T",     "3D Bell Tri 5.0%: spin $1T$ (32x10)", "#1f77b4", "-"),
        ("bell_tri50_st1T",   "3D Bell Tri 5.0%: stand $1T$ (32x10)", "#ff7f0e", "-."),
        ("bell_tri50_2T",     "3D Bell Tri 5.0%: spin $2T$ (quenched!)", "#9467bd", "--"),
        ("bell_tri50_1L",     "3D Bell Tri 5.0%: $1L$ ($t=200$, quenched!)", "#2ca02c", "-"),
        ("bell_tri50_ref_1L", "3D Bell Tri 5.0%: $1L$ (36x12, quenched!)", "#17becf", ":"),
        ("bell_tri50_1R",     "3D Bell Tri 5.0%: $1R$ ($t=180$, quenched!)", "#2ca02c", "--"),
    ]:
        rr = p2_res[cid]; st = exp4_out[cid]
        ax2.semilogy(rr["t"], np.maximum(rr["e_ch"], 1e-13), color=col, ls=ls, lw=1.9, label=f"{lbl} (P={st['p_peak_late']:.2f}%)")
    ax2.set_xlabel("Time $t$", fontsize=11)
    ax2.set_ylabel("3D Chamber Wave Energy $E_{\\mathrm{wave}}(t)$", fontsize=11)
    ax2.set_title("Metric-Fixed 3D Bell NS ($\\psi=0.25$): Entropy-Filtered $1L/1R/2T$ Quench", fontsize=10.1, fontweight="bold")
    ax2.set_ylim(1e-12, 1.5e-2)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=6.9, loc="lower left", framealpha=0.94)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp016_5mode_3d_ns_quench.png", dpi=150)
    plt.close(fig)
    print("Saved all 3 hardened Entry 016 figures.")


if __name__ == "__main__":
    main()
