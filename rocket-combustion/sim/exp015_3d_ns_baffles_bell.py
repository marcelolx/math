"""Entry 015 (rocket-combustion, Hardened per Peer Review #12): Full 3D Cylindrical
Compressible Navier-Stokes, Partial-Span Shroud-Ring Baffles, Mass-Conserving Cavities
& the 3D Transverse -> Chugging -> 1L Mode-Migration Audit.

Hardened changes addressing Peer Review #12 (C1-C3, M1-M5, m1-m10):
  - M1: Exact mapped-grid curved-bell viscous x-derivatives d/dx|_r = d/dx|_eta - eta*R'*d/dr
        and full slanted eta-face heat-flux projection -(nxe*qx_ef + nre*qr_ef) in BOTH
        simulate_3d_ns and relax_axi_3d_bell_base (retaining ~1e-16 conservation).
  - M4: Added nr = 20, 28 curved-wall stress-free SK convergence runs against the exact
        3x3 stress-free adiabatic disk dispersion roots (-0.006537 1T, -0.041144 1R at Re=300),
        showing O(Re^-1) formula + O(Re^-3/2) shift and 0.25% (1T) / 0.77% (1R) convergence.
  - C1: Added matched Re=inf straight-cylinder PDE (48x12x48) + stress-free injector Chebyshev
        to honestly show the PDE does not resolve the faceplate Stokes layer (delta ~ 0.047)
        and the earlier 4.1-5.5% Re=500 match to no-slip Chebyshev was coincidental.
  - C2 & m5: Added m=-3 Bragg projection (e_3tm), plain Hann FFT (no moving-RMS bias), total-
        energy E_ch and [10,22] window sensitivity, inviscid Re=inf runs, and refined 36x18x50
        Exp 2 runs, replacing "2.57-2.90x viscous vortex shedding" with the robust E_ch boost
        (1.5-1.7x at r_b=0.42, 1.9-2.2x at r_b=0.58; present inviscid too) + m=-3 transfer.
  - C3: Extended Exp 3 to t = 180 recording signed mean pressure and chugging period T ~ 39.3,
        honestly showing mass-conserving m_dot does NOT stop k=0 chugging at omega_hp = 0.15,
        the 1D linear model (T = 55-57, stable) does not explain T ~ 39.3, and omega_hp = 0.50
        is a flame-response low-frequency cutoff that keeps both runs chugging-free to t = 180.
  - M2, M3, M5: Added matched bare_ns_st1T, extended hyb3d_1R to t = 180 (documenting linear
        1L growth +0.006), labeled mid-transient st1T/2T runs honestly, verified metric-corrected
        bell_hyb_1T out to t = 120, and reported throat-Mach grid sensitivity across 36x12/40x12/48x14.
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
from scipy.optimize import brentq
from scipy.special import jn_zeros, jv, jvp


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

GAMMA = 1.25
PR = 0.72
P_BAR = 1.0 / GAMMA       # 0.80 => c_bar = 1.0, rho_bar = 1.0, T_bar = P_BAR
RHO_BAR = 1.0
L_A = 1.5
X_F = 0.25
SIG_F = 0.08
OMEGA_HP = 0.15
TAU_BAR = 0.16
Q_S = 0.045
KAPPA_NL = 4.5

# Exact 3x3 stress-free adiabatic disk linear NS dispersion roots at Re = 300 (verified in Review #12 M4)
EXACT_SF_ROOT_1T_RE300 = -0.006537
EXACT_SF_ROOT_1R_RE300 = -0.041144


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


def exact_straight_bessel_ns_eig(
    M_c: float, s_mn: float, Re: float = np.inf, inlet_stressfree: bool = False, N: int = 80,
) -> complex:
    """Exact 1D Chebyshev compressible NS eigen-solver for a straight 3D cylinder (k_perp = s_mn).
    Optional `inlet_stressfree=True` replaces u_perp'(0)=0 with du_perp'/dx(0)=0 to isolate the
    unresolved injector-face Stokes layer (Review #12 C1)."""
    x, Dx, _ = cheb_diff(N, L_A)
    D2 = Dx @ Dx
    n = N + 1
    I = np.eye(n)
    dim = 4 * n
    Mat = np.zeros((dim, dim), dtype=complex)
    B = np.eye(dim, dtype=complex)
    sl = lambda q: slice(q * n, (q + 1) * n)
    mu = 0.0 if not np.isfinite(Re) else (1.0 / Re)

    Mat[sl(0), sl(0)] = -M_c * Dx
    Mat[sl(0), sl(1)] = -Dx
    Mat[sl(0), sl(2)] = -1j * s_mn * I
    Mat[sl(1), sl(1)] = -M_c * Dx + mu * ((4.0 / 3.0) * D2 - (s_mn * s_mn) * I)
    Mat[sl(1), sl(2)] = mu * (1.0 / 3.0) * (1j * s_mn) * Dx
    Mat[sl(1), sl(3)] = -Dx
    Mat[sl(2), sl(1)] = mu * (1.0 / 3.0) * (1j * s_mn) * Dx
    Mat[sl(2), sl(2)] = -M_c * Dx + mu * (D2 - (4.0 / 3.0) * (s_mn * s_mn) * I)
    Mat[sl(2), sl(3)] = -1j * s_mn * I
    lap = D2 - (s_mn * s_mn) * I
    Mat[sl(3), sl(0)] = -(1.0 / (PR * Re) if np.isfinite(Re) else 0.0) * lap
    Mat[sl(3), sl(1)] = -Dx
    Mat[sl(3), sl(2)] = -1j * s_mn * I
    Mat[sl(3), sl(3)] = -M_c * Dx + (GAMMA / (PR * Re) if np.isfinite(Re) else 0.0) * lap

    for r in (0, n, 2 * n):
        Mat[r, :] = 0.0; B[r, :] = 0.0
    Mat[0, 0] = 1.0; Mat[0, 3 * n] = -1.0
    Mat[n, n] = 1.0; Mat[n, 3 * n] = M_c
    if inlet_stressfree and np.isfinite(Re):
        Mat[2 * n, sl(2)] = Dx[0, :]
    else:
        Mat[2 * n, 2 * n] = 1.0
    if np.isfinite(Re):
        for r_ex_v, blk in [(2 * n + N, sl(2)), (N, None)]:
            Mat[r_ex_v, :] = 0.0; B[r_ex_v, :] = 0.0
            if blk is not None:
                Mat[r_ex_v, blk] = Dx[N, :]
            else:
                Mat[r_ex_v, sl(3)] = GAMMA * Dx[N, :]
                Mat[r_ex_v, sl(0)] = -Dx[N, :]
    r_ex = 3 * n + N
    Mat[r_ex, :] = 0.0; B[r_ex, :] = 0.0
    Mat[r_ex, n + N] = 1.0; Mat[r_ex, 3 * n + N] = -0.5 * (GAMMA - 1.0) * M_c
    vals = eig(Mat, B, right=False)
    w_lo, w_hi = (0.70 * s_mn, 1.35 * s_mn) if s_mn > 0 else (0.5, 2.6)
    cands = [v for v in vals if np.isfinite(v) and w_lo <= v.imag <= w_hi and -0.45 < v.real < 0.30]
    cands.sort(key=lambda z: z.real, reverse=True)
    return complex(cands[0])


def exact_1d_chugging_eig(n_p: float, omega_hp: float = OMEGA_HP, beta_tot: float = 0.0, M_c: float = 0.12, N: int = 48) -> tuple[complex | None, complex | None]:
    """Exact 1D Chebyshev eigen-solver for coupled k=0 low-frequency chugging & 1L modes
    (single lumped compliance beta_tot at omega_H=1.84 for chugging compliance limit)."""
    x, Dx, w_cc = cheb_diff(N, L_A)
    n = N + 1; has_c = beta_tot > 0.0
    dim = 3 * n + 5 + (2 if has_c else 0)
    Mat = np.zeros((dim, dim), dtype=complex); B = np.eye(dim, dtype=complex)
    sl = lambda q: slice(q * n, (q + 1) * n)
    Mat[sl(0), sl(0)] = -M_c * Dx; Mat[sl(0), sl(1)] = -Dx
    Mat[sl(1), sl(1)] = -M_c * Dx; Mat[sl(1), sl(2)] = -Dx
    Mat[sl(2), sl(1)] = -Dx;       Mat[sl(2), sl(2)] = -M_c * Dx
    wf = np.exp(-0.5 * ((x - X_F) / SIG_F) ** 2); wf /= float(np.sum(w_cc * wf))
    ws = w_cc * wf; inv_ts = 4.0 / TAU_BAR; ib = 3 * n
    Mat[ib + 0, sl(2)] = omega_hp * ws; Mat[ib + 0, ib + 0] = -omega_hp
    Mat[ib + 1, sl(2)] = inv_ts * ws;   Mat[ib + 1, ib + 0] = -inv_ts; Mat[ib + 1, ib + 1] = -inv_ts
    for r in range(1, 4):
        Mat[ib + 1 + r, ib + r] = inv_ts; Mat[ib + 1 + r, ib + 1 + r] = -inv_ts
    Mat[sl(2), ib + 4] = n_p * wf
    if has_c:
        ic = ib + 5; Vc = beta_tot * L_A; wH = 1.84; xiH = 0.22
        Mat[ic, ic + 1] = 1.0 / Vc
        Mat[ic + 1, 2 * n] = (wH ** 2) * Vc; Mat[ic + 1, ic] = -(wH ** 2) * Vc; Mat[ic + 1, ic + 1] = -2.0 * xiH * wH
    for r in (0, n):
        Mat[r, :] = 0.0; B[r, :] = 0.0
    Mat[0, 0] = 1.0; Mat[0, 2 * n] = -1.0
    Mat[n, n] = 1.0; Mat[n, 2 * n] = M_c
    if has_c:
        Mat[n, ib + 6] = 1.0
    r_ex = 2 * n + N
    Mat[r_ex, :] = 0.0; B[r_ex, :] = 0.0
    Mat[r_ex, n + N] = 1.0; Mat[r_ex, 2 * n + N] = -0.5 * (GAMMA - 1.0) * M_c
    vals = eig(Mat, B, right=False)
    chug = sorted([v for v in vals if np.isfinite(v) and 0.01 <= v.imag <= 0.60], key=lambda z: z.real, reverse=True)
    l1 = sorted([v for v in vals if np.isfinite(v) and 1.20 <= v.imag <= 2.60], key=lambda z: z.real, reverse=True)
    return (complex(chug[0]) if chug else None), (complex(l1[0]) if l1 else None)


# ---------------------------------------------------------------------------
# 2. 3D Cylindrical Conical-Frustum FV Compressible NS Solver (Metric-Corrected)
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


def inlet_choked_3d(rho_i, ux_i, p_i, M_c: float, m_port: np.ndarray | None = None, u_port: np.ndarray | None = None):
    c_i = np.sqrt(GAMMA * p_i / rho_i)
    Jm = ux_i - (2.0 / (GAMMA - 1.0)) * c_i
    c_g = np.maximum(c_i.copy(), 0.2)
    a_exp = 2.0 / (GAMMA - 1.0)
    if m_port is not None:
        m_tgt = M_c - m_port
        for _ in range(7):
            u_g = Jm + a_exp * c_g
            f = (c_g ** a_exp) * u_g - m_tgt
            df = a_exp * (c_g ** (a_exp - 1.0)) * u_g + (c_g ** a_exp) * a_exp
            c_g = np.maximum(c_g - f / np.where(np.abs(df) > 1e-12, df, 1e-12), 0.15)
    else:
        up = np.zeros_like(ux_i) if u_port is None else u_port
        for _ in range(7):
            u_g = Jm + a_exp * c_g
            f = (c_g ** a_exp) * (u_g + up) - M_c
            df = a_exp * (c_g ** (a_exp - 1.0)) * (u_g + up) + (c_g ** a_exp) * a_exp
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
    gcl_x = float(np.max(np.abs((Sx[1:] - Sx[:-1]) - (S_ex[:, 1:] - S_ex[:, :-1]))))
    gcl_r = float(np.max(np.abs((S_er[:, 1:] - S_er[:, :-1]) - hoop)))
    dA_in = np.broadcast_to(Sx[0, :, None], (nr, nth)).copy()
    # Exact mapped-grid metric factors eta * R'(x) for d/dx|_r = d/dx|_eta - eta*R'*d/dr (Review #12 M1!)
    eta_Rp_c = dRm[:, None] * etac[None, :]     # (nx, nr)
    eta_Rp_xf = dRf[:, None] * etac[None, :]    # (nx+1, nr)
    eta_Rp_ef = dRm[:, None] * etaf[None, :]    # (nx, nr+1)
    return {
        "xc": xc, "xf": xf, "etac": etac, "etaf": etaf, "thc": thc, "dx": dx, "deta": deta, "dth": dth,
        "L_tot": L_tot, "Rm": Rm, "Rf": Rf, "gcl_x": gcl_x, "gcl_r": gcl_r, "dA_in": dA_in,
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
    eta_Rp_c = g["eta_Rp_c"][:, :, 0]; eta_Rp_xf = g["eta_Rp_xf"][:, :, 0]; eta_Rp_ef = g["eta_Rp_ef"][:, :, 0]
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
            # Exact fixed-r axial derivatives: d/dx|_r = d/dx|_eta - eta*R'*d/dr (Review #12 M1!)
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
            qx_ef = -k_th * dT_dx_ef
            qr_ef = -k_th * dT_dr_ef
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
    nx: int = 24, nr: int = 12, nth: int = 48, t_end: float = 30.0,
    M_c: float = 0.12, L_n: float = 0.60, L_d: float = 0.45, mode: str = "straight",
    Re: float = np.inf,
    init_type: str = "spin_1T", A_eq: float = 1e-4,
    n_p: float = 0.0, omega_hp: float = OMEGA_HP,
    n_baffles: int = 0, L_b: float = 0.45, r_b: float = 0.0, baffle_noslip: bool = False,
    cav_spec: list[dict] | None = None, cav_mass_conserving: bool = True,
    prim_axi_base: np.ndarray | None = None, cfl: float = 0.28, record_stride: int = 12,
) -> dict:
    """Full 3D cylindrical (x, r, theta) conservative FV compressible NS solver with exact
    mapped-grid viscous metrics, 3D partial-span radial baffles, and m=-3 Bragg projection."""
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
    phi_3T = jv(3, S31 * E_ch) / (np.sqrt(0.5 * (1.0 - 9.0 / S31**2)) * abs(jv(3, S31)))
    phi_1L = np.sqrt(2.0) * np.cos(np.pi * X_ch / L_A)

    has_baffles = (n_baffles > 0) and (L_b > 0.0) and (r_b < 0.999)
    baffle_th_face = np.zeros((nx, nr, nth + 1), dtype=bool)
    baffle_slope_zero = np.zeros((nx, nr, nth + 2), dtype=bool)
    baffle_idx = []
    baffle_metal_frac = 0.0
    if has_baffles:
        ix_b = xc <= L_b
        jr_b = etac >= r_b
        b_xr = ix_b[:, None] & jr_b[None, :]
        baffle_idx = [int(round(k * nth / n_baffles)) % nth for k in range(n_baffles)]
        for kf in baffle_idx:
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
        baffle_metal_frac = float(np.sum(g["S_th"][b_xr, 0]) / max(np.sum(g["S_th"][ix_b, :, 0]), 1e-12))

    if prim_axi_base is not None:
        P_b = np.broadcast_to(prim_axi_base[:, :, :, None], (5, nx, nr, nth)).copy()
    else:
        ux0 = 0.0 if mode == "closed" or init_type == "quiescent" else M_c
        P_b = np.zeros((5, nx, nr, nth))
        P_b[0] = RHO_BAR; P_b[1] = ux0; P_b[4] = P_BAR
    rho_b, ux_b, ur_b, ut_b, p_b = P_b[0], P_b[1], P_b[2], P_b[3], P_b[4]
    rho = rho_b.copy(); ux = ux_b.copy(); ur = ur_b.copy(); ut = ut_b.copy(); p = p_b.copy()

    if init_type == "quiescent":
        pass
    elif init_type == "periodic_3d_wave":
        dp = A_eq * (np.cos(2.0 * np.pi * X3 / g["L_tot"] + T3) * jv(1, S11 * E3) + 0.4 * jv(0, S02 * E3))
        rho += dp; p += dp
        ux += 0.2 * A_eq * np.sin(2.0 * np.pi * X3 / g["L_tot"])
        ur += 0.3 * A_eq * E3 * (1.0 - E3) * np.cos(T3)
        ut += 0.4 * A_eq * E3 * np.sin(T3)
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
    elif init_type in ("spin_2T", "stand_2T"):
        j2 = jv(2, S21 * E3) / abs(jv(2, S21))
        j2p = jvp(2, S21 * E3) / abs(jv(2, S21))
        if init_type == "spin_2T":
            dp = A_eq * j2 * np.cos(2.0 * T3) * ch_env
            ur += A_eq * j2p * np.sin(2.0 * T3) * ch_env
            ut += A_eq * (2.0 * j2 / np.maximum(S21 * E3, 1e-12)) * np.cos(2.0 * T3) * ch_env
        else:
            dp = A_eq * np.sqrt(2.0) * j2 * np.cos(2.0 * T3) * ch_env
        rho += dp; p += dp
    elif init_type == "1R":
        j0 = jv(0, S02 * E3)
        dp = A_eq * j0 * ch_env
        rho += dp; p += dp
    elif init_type == "1L":
        dp = A_eq * np.cos(np.pi * X3 / L_A) * ch_env
        rho += dp; p += dp
    else:
        raise ValueError(f"Unknown init_type={init_type}")

    cavs = []
    if cav_spec:
        Cd = 0.65
        dA_in = g["dA_in"]
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
    max_cav_mass_err = 0.0

    dr_c = deta * g["Rm"][:, None, None]
    r_ef = g["r_ef"]; r_xf = g["r_xf"]; r_thf = g["r_thf"]
    nxe, nre = g["nxe"], g["nre"]
    Sx, Sm, S_th, hoop = g["Sx"], g["Sm"], g["S_th"], g["hoop"]
    eta_Rp_c, eta_Rp_xf = g["eta_Rp_c"], g["eta_Rp_xf"]

    def compute_rhs(Qs, fp_s, cav_states):
        r_s, u_s, v_s, w_s, p_s, _ = from_Q(Qs)
        P = np.stack([r_s, u_s, v_s, w_s, p_s], axis=0)

        port_flux = None
        if cavs:
            port_flux = np.zeros((nr, nth))
            for (_, qt_c), cf in zip(cav_states, cavs):
                port_flux += np.tensordot(qt_c, cf["W"], axes=(0, 0))

        Px = np.empty((5, nx + 4, nr, nth)); Px[:, 2:-2] = P
        p_in_face = p_s[0]
        if mode == "periodic_bump":
            Px[:, 0:2] = P[:, -2:]; Px[:, -2:] = P[:, 0:2]
        elif mode == "closed" or init_type == "quiescent":
            for k in (0, 1):
                Px[:, k] = P[:, 0]; Px[1, k] = -P[1, 0]
                Px[:, -(k + 1)] = P[:, -1]; Px[1, -(k + 1)] = -P[1, -1]
        elif mode == "straight":
            if cav_mass_conserving:
                ri, ui, vi, wi, pi = inlet_choked_3d(r_s[0], u_s[0], p_s[0], M_c, m_port=port_flux)
            else:
                ri, ui, vi, wi, pi = inlet_choked_3d(r_s[0], u_s[0], p_s[0], M_c, u_port=port_flux)
            p_in_face = pi
            ro, uo, vo, wo, po = outlet_mc_choked_3d(r_s[-1], u_s[-1], v_s[-1], w_s[-1], p_s[-1], M_c)
            for k in (0, 1):
                Px[0, k] = ri; Px[1, k] = ui; Px[2, k] = vi; Px[3, k] = wi; Px[4, k] = pi
                Px[0, -(k + 1)] = ro; Px[1, -(k + 1)] = uo; Px[2, -(k + 1)] = vo; Px[3, -(k + 1)] = wo; Px[4, -(k + 1)] = po
        else:
            if cav_mass_conserving:
                ri, ui, vi, wi, pi = inlet_choked_3d(r_s[0], u_s[0], p_s[0], M_c, m_port=port_flux)
            else:
                ri, ui, vi, wi, pi = inlet_choked_3d(r_s[0], u_s[0], p_s[0], M_c, u_port=port_flux)
            p_in_face = pi
            for k in (0, 1):
                Px[0, k] = ri; Px[1, k] = ui; Px[2, k] = vi; Px[3, k] = wi; Px[4, k] = pi
            Px[:, -2] = P[:, -1]; Px[:, -1] = P[:, -1]

        sx = van_leer(Px[:, 1:-1] - Px[:, :-2], Px[:, 2:] - Px[:, 1:-1])
        PLx = Px[:, 1:-2] + 0.5 * sx[:, :-1]; PRx = Px[:, 2:-1] - 0.5 * sx[:, 1:]
        if mode == "closed" or init_type == "quiescent":
            PLx[0, 0] = PRx[0, 0]; PLx[1, 0] = -PRx[1, 0]; PLx[2, 0] = PRx[2, 0]; PLx[3, 0] = PRx[3, 0]; PLx[4, 0] = PRx[4, 0]
            PRx[0, -1] = PLx[0, -1]; PRx[1, -1] = -PLx[1, -1]; PRx[2, -1] = PLx[2, -1]; PRx[3, -1] = PLx[3, -1]; PRx[4, -1] = PLx[4, -1]
        if mode == "closed" and cavs and cav_mass_conserving:
            Fx = hllc_3d(np.maximum(PLx[0], 1e-10), PLx[1], PLx[2], PLx[3], np.maximum(PLx[4], 1e-10),
                         np.maximum(PRx[0], 1e-10), PRx[1], PRx[2], PRx[3], np.maximum(PRx[4], 1e-10))
            Fx[0, 0] = -port_flux
            Fx[4, 0] = -port_flux * (GAMMA / (GAMMA - 1.0)) * P_BAR
            p_in_face = 0.5 * (PLx[4, 0] + PRx[4, 0])
        else:
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
            # Exact fixed-r axial derivatives d/dx|_r = d/dx|_eta - eta*R'*d/dr (Review #12 M1!)
            du_dx_c = (u_x[2:] - u_x[:-2]) / (2.0 * dx) - eta_Rp_c * du_dr_c
            dv_dx_c = (v_x[2:] - v_x[:-2]) / (2.0 * dx) - eta_Rp_c * dv_dr_c
            dw_dx_c = (w_x[2:] - w_x[:-2]) / (2.0 * dx) - eta_Rp_c * dw_dr_c
            dT_dx_c = (T_x[2:] - T_x[:-2]) / (2.0 * dx) - eta_Rp_c * dT_dr_c
            du_dt_c = (u_t[:, :, 2:] - u_t[:, :, :-2]) / (2.0 * dth)
            dv_dt_c = (v_t[:, :, 2:] - v_t[:, :, :-2]) / (2.0 * dth)
            dw_dt_c = (w_t[:, :, 2:] - w_t[:, :, :-2]) / (2.0 * dth)
            if has_baffles:
                mL_b = baffle_th_face[:, :, :-1]
                mR_b = baffle_th_face[:, :, 1:]
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
            res[1] += VFx_u[1:] - VFx_u[:-1]
            res[2] += VFx_v[1:] - VFx_v[:-1]
            res[3] += VFx_w[1:] - VFx_w[:-1]
            res[4] += VFx_e[1:] - VFx_e[:-1]

            du_dr_ef = (u_e[:, 1:] - u_e[:, :-1]) / dr_c
            dv_dr_ef = (v_e[:, 1:] - v_e[:, :-1]) / dr_c
            dw_dr_ef = (w_e[:, 1:] - w_e[:, :-1]) / dr_c
            dT_dr_ef = (T_e[:, 1:] - T_e[:, :-1]) / dr_c
            def avg_e(arr):
                out = np.zeros((nx, nr + 1, nth))
                out[:, 1:-1] = 0.5 * (arr[:, :-1] + arr[:, 1:])
                return out
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
            qx_ef = -k_th * dT_dx_ef
            qr_ef = -k_th * dT_dr_ef
            fx_v = nxe * txx_ef + nre * txr_ef
            fr_v = nxe * txr_ef + nre * trr_ef
            ft_v = nxe * txt_ef + nre * trt_ef
            fe_v = uf_e * fx_v + vf_e * fr_v + wf_e * ft_v - (nxe * qx_ef + nre * qr_ef)
            VFe_u = Sm * fx_v; VFe_v = Sm * fr_v; VFe_w = Sm * r_ef * ft_v; VFe_e = Sm * fe_v
            VFe_u[:, 0] = 0.0; VFe_u[:, -1] = 0.0
            VFe_v[:, 0] = 0.0; VFe_v[:, -1] = 0.0
            VFe_w[:, 0] = 0.0; VFe_w[:, -1] = 0.0
            VFe_e[:, 0] = 0.0; VFe_e[:, -1] = 0.0
            res[1] += VFe_u[:, 1:] - VFe_u[:, :-1]
            res[2] += VFe_v[:, 1:] - VFe_v[:, :-1]
            res[3] += VFe_w[:, 1:] - VFe_w[:, :-1]
            res[4] += VFe_e[:, 1:] - VFe_e[:, :-1]

            du_dt_tf = (u_t[:, :, 1:] - u_t[:, :, :-1]) / dth
            dv_dt_tf = (v_t[:, :, 1:] - v_t[:, :, :-1]) / dth
            dw_dt_tf = (w_t[:, :, 1:] - w_t[:, :, :-1]) / dth
            dT_dt_tf = (T_t[:, :, 1:] - T_t[:, :, :-1]) / dth
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
                    VG_uL = np.where(baffle_th_face, 0.0, VG_uL)
                    VG_uR = np.where(baffle_th_face, 0.0, VG_uR)
                    VG_vL = np.where(baffle_th_face, 0.0, VG_vL)
                    VG_vR = np.where(baffle_th_face, 0.0, VG_vR)
                VG_wL = np.where(baffle_th_face, 0.0, VG_wL)
                VG_wR = np.where(baffle_th_face, 0.0, VG_wR)
                VG_eL = np.where(baffle_th_face, 0.0, VG_eL)
                VG_eR = np.where(baffle_th_face, 0.0, VG_eR)
            res[1] += VG_uL[:, :, 1:] - VG_uR[:, :, :-1]
            res[2] += VG_vL[:, :, 1:] - VG_vR[:, :, :-1]
            res[3] += VG_wL[:, :, 1:] - VG_wR[:, :, :-1]
            res[4] += VG_eL[:, :, 1:] - VG_eR[:, :, :-1]

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
            dA_in = g["dA_in"]
            for cf, (Pc_c, qt_c) in zip(cavs, cav_states):
                wH = cf["wH"]; xiH = cf["xiH"]; r_nl = cf["r_nl"]; Vk = cf["V_k"]
                p_bar_k = np.tensordot(cf["W"], dp_in * dA_in, axes=([1, 2], [0, 1]))
                dPc = qt_c / Vk
                dqt = (wH * wH) * Vk * (p_bar_k - Pc_c - r_nl * np.abs(qt_c) * qt_c) - 2.0 * xiH * wH * qt_c
                dcav.append((dPc, dqt))
        return res, dfp, dcav

    t = 0.0; step = 0
    ts, e_ch, e_1t, e_2t, e_1r, e_3t, e_3tm, e_1l, e_mean = [], [], [], [], [], [], [], [], []
    c1_re, c2_re, c0_re, p_mean_signed, p_pk, mass_drift = [], [], [], [], [], []
    w1_th = np.exp(-1j * T_ch); w2_th = np.exp(-2j * T_ch); w3_th = np.exp(-3j * T_ch); w3m_th = np.exp(3j * T_ch)
    dr_eff = deta * g["Rm"][:, None, None]

    while t < t_end - 1e-12:
        r_c, u_c, v_c, w_c, p_c, c_c = from_Q(Q)
        lam_adv = (np.abs(u_c) + c_c) / dx + (np.abs(v_c) + c_c) / dr_eff + (np.abs(w_c) + c_c) / (rb * dth)
        inv_dt = lam_adv / cfl
        if mu > 0.0:
            nu_eff = max(4.0 / 3.0, GAMMA / PR) * mu / r_c
            lam_visc = nu_eff * (1.0 / (dx * dx) + 1.0 / (dr_eff * dr_eff) + 1.0 / ((rb * dth) ** 2))
            inv_dt = inv_dt + lam_visc / 0.40
        dt = min(1.0 / float(np.max(inv_dt)), t_end - t)

        if step % record_stride == 0:
            dp_c = (p_c - p_b)[ch_mask]; du_c = (u_c - ux_b)[ch_mask]; dv_c = (v_c - ur_b)[ch_mask]; dw_c = (w_c - ut_b)[ch_mask]
            ts.append(t)
            e_ch.append(0.5 * float(np.sum(w_ch * (dp_c * dp_c + du_c * du_c + dv_c * dv_c + dw_c * dw_c))))
            c1 = complex(np.sum(w_ch * dp_c * phi_1T * w1_th))
            c2 = complex(np.sum(w_ch * dp_c * phi_2T * w2_th))
            c0 = float(np.sum(w_ch * dp_c * phi_1R))
            c3 = complex(np.sum(w_ch * dp_c * phi_3T * w3_th))
            c3m = complex(np.sum(w_ch * dp_c * phi_3T * w3m_th))
            c1l = float(np.sum(w_ch * dp_c * phi_1L))
            cmn = float(np.sum(w_ch * dp_c))
            # Note: e_1t, e_2t, e_3t, e_3tm are pressure-only 2|c|^2 modal intensities; e_ch is total acoustic energy
            e_1t.append(2.0 * abs(c1) ** 2); e_2t.append(2.0 * abs(c2) ** 2)
            e_1r.append(c0 * c0); e_3t.append(2.0 * abs(c3) ** 2); e_3tm.append(2.0 * abs(c3m) ** 2)
            e_1l.append(c1l * c1l); e_mean.append(cmn * cmn); p_mean_signed.append(cmn)
            c1_re.append(c1.real); c2_re.append(c2.real); c0_re.append(c0)
            p_pk.append(float(np.max(np.abs(dp_c)) / P_BAR * 100.0))
            mass_drift.append(float((np.sum(Q[0][ch_mask]) - init_ch_mass) / init_ch_mass))
            if mode == "periodic_bump":
                tot = np.array([float(np.sum(Q[k])) for k in range(5)])
                max_cons = np.maximum(max_cons, np.abs(tot - init_tot) / np.maximum(np.abs(init_tot), 1.0))
            if mode == "closed" and cavs and cav_mass_conserving:
                m_cav = sum(float(np.sum(cf["Pc"]) * cf["V_k"]) for cf in cavs)
                max_cav_mass_err = max(max_cav_mass_err, abs((float(np.sum(Q[0])) + m_cav) - init_tot[0]) / init_tot[0])

        c0_s = [(cf["Pc"], cf["qt"]) for cf in cavs]
        k1Q, k1f, k1c = compute_rhs(Q, fp, c0_s)
        Q1 = Q + dt * k1Q; fp1 = fp + dt * k1f
        c1_s = [(s0 + dt * ds1, v0 + dt * dv1) for (s0, v0), (ds1, dv1) in zip(c0_s, k1c)]
        k2Q, k2f, k2c = compute_rhs(Q1, fp1, c1_s)
        Q = 0.5 * (Q + Q1 + dt * k2Q); fp = 0.5 * (fp + fp1 + dt * k2f)
        for cf, (s0, v0), (s1, v1), (ds2, dv2) in zip(cavs, c0_s, c1_s, k2c):
            cf["Pc"] = 0.5 * (s0 + s1 + dt * ds2); cf["qt"] = 0.5 * (v0 + v1 + dt * dv2)
        t += dt; step += 1

    return {
        "t": np.array(ts), "e_ch": np.array(e_ch), "e_1t": np.array(e_1t), "e_2t": np.array(e_2t),
        "e_1r": np.array(e_1r), "e_3t": np.array(e_3t), "e_3tm": np.array(e_3tm),
        "e_1l": np.array(e_1l), "e_mean": np.array(e_mean), "p_mean_signed": np.array(p_mean_signed),
        "c1_re": np.array(c1_re), "c2_re": np.array(c2_re), "c0_re": np.array(c0_re),
        "p_max_pct": np.array(p_pk), "mass_drift": np.array(mass_drift),
        "quiescent_err": float(np.max(np.abs(Q - Q0))),
        "gcl_x": g["gcl_x"], "gcl_r": g["gcl_r"],
        "baffle_metal_frac": baffle_metal_frac,
        "cav_closed_mass_err": max_cav_mass_err,
        "periodic_cons_err": {"mass": float(max_cons[0]), "mom_x": float(max_cons[1]),
                              "ang_mom": float(max_cons[3]), "energy": float(max_cons[4])},
    }


def _sim_worker(args):
    lbl, kw = args
    t0 = time.perf_counter()
    res = simulate_3d_ns(**kw)
    return lbl, res, time.perf_counter() - t0


def run_batch(tasks: list[tuple[str, dict]], max_workers: int | None = None) -> dict:
    cache_env = os.environ.get("EXP015_CACHE_DIR", "").strip()
    cache_dir = Path(cache_env) if cache_env else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    pending = []
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


def fit_alpha(res: dict, t_lo: float, t_hi: float, key: str = "e_1t") -> float:
    t = res["t"]; m = (t >= t_lo) & (t <= t_hi)
    return float(np.polyfit(t[m], 0.5 * np.log(np.maximum(res[key][m], 1e-30)), 1)[0])


def measure_plain_hann_peaks(t: np.ndarray, sig: np.ndarray, t_lo: float = 4.0, f_lo: float = 0.8, f_hi: float = 5.0) -> tuple[float, float | None, float]:
    """Plain Hann-windowed FFT peak detector (no moving-RMS bias, per Review #12 m5!).
    Returns (primary_omega, secondary_omega_or_None, secondary_over_primary_amp)."""
    m = t >= t_lo
    tu = np.linspace(t[m][0], t[m][-1], 1024)
    su = np.interp(tu, t[m], sig[m])
    su = (su - np.mean(su)) * np.hanning(len(su))
    nfft = 32768
    freqs = np.fft.rfftfreq(nfft, d=(tu[1] - tu[0])) * 2.0 * np.pi
    spec = np.abs(np.fft.rfft(su, n=nfft))
    mf = (freqs >= f_lo) & (freqs <= f_hi)
    fr = freqs[mf]; sp = spec[mf]
    # Local maxima
    loc = [i for i in range(1, len(sp) - 1) if sp[i] > sp[i - 1] and sp[i] > sp[i + 1]]
    loc.sort(key=lambda i: sp[i], reverse=True)
    w1 = float(fr[loc[0]])
    w2 = None; rat2 = 0.0
    for idx in loc[1:]:
        if abs(fr[idx] - w1) > 0.35:
            w2 = float(fr[idx]); rat2 = float(sp[idx] / sp[loc[0]])
            break
    return w1, w2, rat2


def measure_zero_crossing_period(t: np.ndarray, sig: np.ndarray, t_lo: float = 20.0) -> float | None:
    m = t >= t_lo; tt = t[m]; ss = sig[m]
    zc = []
    for i in range(len(ss) - 1):
        if ss[i] * ss[i + 1] < 0.0:
            zc.append(tt[i] + (tt[i + 1] - tt[i]) * abs(ss[i]) / (abs(ss[i]) + abs(ss[i + 1])))
    if len(zc) < 3:
        return None
    half_periods = np.diff(zc)
    return float(2.0 * np.mean(half_periods))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp015_metrics.json"

    print("=" * 96)
    print("Entry 015 (Hardened per Peer Review #12): 3D NS, Partial-Span Baffles & Mode-Migration Audit")
    print("=" * 96)

    # ------------------------------------------------------------------
    # Experiment 1: Metric-Corrected 3D NS Conservation, nr=14/20/28 SK Convergence & Honest Fixed-Re Audit
    # ------------------------------------------------------------------
    print("\n--- Experiment 1: 3D NS Conservation, SK Convergence (nr=14->28) & Honest Re=500 Audit ---")
    cav_closed_test = [{"beta": 0.02, "wH": S11, "xiH": 0.24, "sigma_o": 0.06, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": True}]
    exp1_tasks = [
        ("quiescent_3d_ns_bell", dict(nx=20, nr=10, nth=32, t_end=2.5, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", Re=500.0, init_type="quiescent", A_eq=0.0, record_stride=20)),
        ("cons_ns_bell_bare",    dict(nx=16, nr=10, nth=32, t_end=2.0, M_c=0.12, mode="periodic_bump", Re=500.0, init_type="periodic_3d_wave", A_eq=0.06, record_stride=15)),
        ("cons_ns_bell_b5_slip", dict(nx=16, nr=10, nth=30, t_end=2.0, M_c=0.12, mode="periodic_bump", Re=500.0, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=False, init_type="periodic_3d_wave", A_eq=0.06, record_stride=15)),
        ("cons_ns_bell_b5_nslp", dict(nx=16, nr=10, nth=30, t_end=2.0, M_c=0.12, mode="periodic_bump", Re=500.0, n_baffles=5, L_b=0.45, r_b=0.42, baffle_noslip=True,  init_type="periodic_3d_wave", A_eq=0.06, record_stride=15)),
        ("cons_closed_cav_mass", dict(nx=12, nr=10, nth=32, t_end=4.0, M_c=0.0,  mode="closed",        Re=500.0, cav_spec=cav_closed_test, cav_mass_conserving=True, init_type="spin_1T", A_eq=0.04, record_stride=10)),
        # SK radial refinement nr = 14, 20, 28 at Re = 300 (Review #12 M4!)
        ("sk_1T_nr14_inf",   dict(nx=6, nr=14, nth=48, t_end=14.0, M_c=0.0, mode="closed", Re=np.inf, init_type="spin_1T", A_eq=1e-4, record_stride=8)),
        ("sk_1T_nr14_re300", dict(nx=6, nr=14, nth=48, t_end=14.0, M_c=0.0, mode="closed", Re=300.0,  init_type="spin_1T", A_eq=1e-4, record_stride=8)),
        ("sk_1T_nr20_inf",   dict(nx=6, nr=20, nth=48, t_end=14.0, M_c=0.0, mode="closed", Re=np.inf, init_type="spin_1T", A_eq=1e-4, record_stride=10)),
        ("sk_1T_nr20_re300", dict(nx=6, nr=20, nth=48, t_end=14.0, M_c=0.0, mode="closed", Re=300.0,  init_type="spin_1T", A_eq=1e-4, record_stride=10)),
        ("sk_1T_nr28_inf",   dict(nx=6, nr=28, nth=48, t_end=14.0, M_c=0.0, mode="closed", Re=np.inf, init_type="spin_1T", A_eq=1e-4, record_stride=12)),
        ("sk_1T_nr28_re300", dict(nx=6, nr=28, nth=48, t_end=14.0, M_c=0.0, mode="closed", Re=300.0,  init_type="spin_1T", A_eq=1e-4, record_stride=12)),
        ("sk_1R_nr14_inf",   dict(nx=6, nr=14, nth=48, t_end=14.0, M_c=0.0, mode="closed", Re=np.inf, init_type="1R",      A_eq=1e-4, record_stride=8)),
        ("sk_1R_nr14_re300", dict(nx=6, nr=14, nth=48, t_end=14.0, M_c=0.0, mode="closed", Re=300.0,  init_type="1R",      A_eq=1e-4, record_stride=8)),
        ("sk_1R_nr20_inf",   dict(nx=6, nr=20, nth=48, t_end=14.0, M_c=0.0, mode="closed", Re=np.inf, init_type="1R",      A_eq=1e-4, record_stride=10)),
        ("sk_1R_nr20_re300", dict(nx=6, nr=20, nth=48, t_end=14.0, M_c=0.0, mode="closed", Re=300.0,  init_type="1R",      A_eq=1e-4, record_stride=10)),
        ("sk_1R_nr28_inf",   dict(nx=6, nr=28, nth=48, t_end=14.0, M_c=0.0, mode="closed", Re=np.inf, init_type="1R",      A_eq=1e-4, record_stride=12)),
        ("sk_1R_nr28_re300", dict(nx=6, nr=28, nth=48, t_end=14.0, M_c=0.0, mode="closed", Re=300.0,  init_type="1R",      A_eq=1e-4, record_stride=12)),
    ]
    rb_grid = [0.00, 0.25, 0.42, 0.58, 0.75]
    exp2_core_tasks = [
        ("bare_re500_1T", dict(nx=24, nr=12, nth=50, t_end=24.0, M_c=0.12, mode="straight", Re=500.0, init_type="spin_1T", A_eq=1e-4, record_stride=10)),
        ("bare_re500_2T", dict(nx=24, nr=12, nth=50, t_end=24.0, M_c=0.12, mode="straight", Re=500.0, init_type="spin_2T", A_eq=1e-4, record_stride=10)),
    ]
    for rb_v in rb_grid:
        tag = f"rb{int(round(rb_v*100)):02d}"
        exp2_core_tasks.append((f"{tag}_1T", dict(nx=24, nr=12, nth=50, t_end=24.0, M_c=0.12, mode="straight", Re=500.0, n_baffles=5, L_b=0.45, r_b=rb_v, init_type="spin_1T", A_eq=1e-4, record_stride=10)))
        exp2_core_tasks.append((f"{tag}_2T", dict(nx=24, nr=12, nth=50, t_end=24.0, M_c=0.12, mode="straight", Re=500.0, n_baffles=5, L_b=0.45, r_b=rb_v, init_type="spin_2T", A_eq=1e-4, record_stride=10)))

    # Phase 1: Fast benchmarks + Exp 2 core sweep (computes retuned cavity targets for Exp 4)
    phase1_res = run_batch(exp1_tasks + exp2_core_tasks)
    exp1_res = phase1_res
    rq = exp1_res["quiescent_3d_ns_bell"]
    cb = exp1_res["cons_ns_bell_bare"]["periodic_cons_err"]
    cs = exp1_res["cons_ns_bell_b5_slip"]["periodic_cons_err"]
    cn = exp1_res["cons_ns_bell_b5_nslp"]["periodic_cons_err"]
    cm = exp1_res["cons_closed_cav_mass"]["cav_closed_mass_err"]
    print(f"  1A Quiescent 3D NS Bell: drift={rq['quiescent_err']:.2e}")
    print(f"  1B Bare 3D NS Bell Cons:    mass={cb['mass']:.2e}, ang_mom={cb['ang_mom']:.2e}, energy={cb['energy']:.2e}")
    print(f"  1B B5 Slip/NoSl Cons:       mass={cs['mass']:.2e}/{cn['mass']:.2e}, energy={cs['energy']:.2e}/{cn['energy']:.2e}")
    print(f"  1B Closed Chamber+Cavity Mass Cons: err={cm:.2e}")

    sk_fac = 0.5 * (4.0 / 3.0 + (GAMMA - 1.0) / PR)
    sk_out = {}
    for m_lbl, m_az, s_mn, ek, ex_root in [
        ("1T", 1, S11, "e_1t", EXACT_SF_ROOT_1T_RE300),
        ("1R", 0, S02, "e_1r", EXACT_SF_ROOT_1R_RE300),
    ]:
        exact_bulk = -(s_mn * s_mn) * sk_fac / 300.0
        exact_sf_o1 = exact_bulk + ((2.0 * m_az * m_az / (300.0 * (s_mn * s_mn - m_az * m_az))) if m_az > 0 else 0.0)
        for nr_v in (14, 20, 28):
            a_inf = fit_alpha(exp1_res[f"sk_{m_lbl}_nr{nr_v}_inf"], 4.0, 13.0, ek)
            a_re = fit_alpha(exp1_res[f"sk_{m_lbl}_nr{nr_v}_re300"], 4.0, 13.0, ek)
            net_v = a_re - a_inf
            err_root = abs(net_v - ex_root) / abs(ex_root) * 100.0
            err_o1 = abs(net_v - exact_sf_o1) / abs(exact_sf_o1) * 100.0
            sk_out[f"{m_lbl}_nr{nr_v}_Re300"] = {
                "alpha_re": a_re, "alpha_inf": a_inf, "net_visc": net_v,
                "exact_o1_formula": exact_sf_o1, "exact_disk_root": ex_root,
                "err_vs_root_pct": err_root, "err_vs_o1_pct": err_o1,
            }
            print(f"  1C SK [{m_lbl} nr={nr_v:2d} Re=300]: net={net_v:+.5f} (inf={a_inf:+.5f}) vs root={ex_root:+.5f} ({err_root:.2f}%), O(1/Re)={exact_sf_o1:+.5f} ({err_o1:.2f}%)")

    # Precompute Exp 2 core targets + 1D spectral & axisymmetric bases
    wH_1T_ret, _, _ = measure_plain_hann_peaks(phase1_res["rb42_1T"]["t"], phase1_res["rb42_1T"]["c1_re"], 4.0, 1.0, 2.5)
    wH_2T_ret, _, _ = measure_plain_hann_peaks(phase1_res["rb42_2T"]["t"], phase1_res["rb42_2T"]["c2_re"], 4.0, 1.8, 4.0)
    ex_inf = exact_straight_bessel_ns_eig(0.12, S11, Re=np.inf, N=80)
    ex_ns_500 = exact_straight_bessel_ns_eig(0.12, S11, Re=500.0, inlet_stressfree=False, N=80)
    ex_sf_500 = exact_straight_bessel_ns_eig(0.12, S11, Re=500.0, inlet_stressfree=True,  N=80)
    ex_ns_1000 = exact_straight_bessel_ns_eig(0.12, S11, Re=1000.0, inlet_stressfree=False, N=80)
    axi_b_inf = relax_axi_3d_bell_base(40, 12, M_c=0.12, L_n=0.60, L_d=0.45, Re=np.inf, t_relax=160.0)
    axi_b_500_36 = relax_axi_3d_bell_base(36, 12, M_c=0.12, L_n=0.60, L_d=0.45, Re=500.0, t_relax=160.0)
    axi_b_500_40 = relax_axi_3d_bell_base(40, 12, M_c=0.12, L_n=0.60, L_d=0.45, Re=500.0, t_relax=160.0)
    axi_b_1000_40 = relax_axi_3d_bell_base(40, 12, M_c=0.12, L_n=0.60, L_d=0.45, Re=1000.0, t_relax=160.0)
    axi_b_500_48 = relax_axi_3d_bell_base(48, 14, M_c=0.12, L_n=0.60, L_d=0.45, Re=500.0, t_relax=160.0)

    cav_e014_dual = [
        {"beta": 0.0185, "wH": S11, "xiH": 0.22, "sigma_o": 0.10, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": True},
        {"beta": 0.0080, "wH": S21, "xiH": 0.24, "sigma_o": 0.08, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": True},
        {"beta": 0.0035, "wH": S02, "xiH": 0.14, "sigma_o": 0.04, "n_cav": 1, "r_lo": 0.00, "r_hi": 0.26, "nonlinear": True},
    ]
    cav_3d_retuned = [
        {"beta": 0.0185, "wH": wH_1T_ret, "xiH": 0.22, "sigma_o": 0.10, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": True},
        {"beta": 0.0080, "wH": wH_2T_ret, "xiH": 0.24, "sigma_o": 0.08, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": True},
        {"beta": 0.0035, "wH": S02,       "xiH": 0.14, "sigma_o": 0.04, "n_cav": 1, "r_lo": 0.00, "r_hi": 0.26, "nonlinear": True},
    ]

    # Phase 2: Unified LPT-ordered heavy batch (Exp 1D + Exp 2 robust + Exp 3 + Exp 4) across 8 cores
    exp1d_tasks = [
        ("bell_re500_48",  dict(nx=48, nr=14, nth=48, t_end=30.0, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", Re=500.0,  init_type="spin_1T", A_eq=1e-4, prim_axi_base=axi_b_500_48["prim_axi"], record_stride=18)),
        ("bell_re500_40",  dict(nx=40, nr=12, nth=48, t_end=30.0, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", Re=500.0,  init_type="spin_1T", A_eq=1e-4, prim_axi_base=axi_b_500_40["prim_axi"], record_stride=16)),
        ("bell_re1000_40", dict(nx=40, nr=12, nth=48, t_end=30.0, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", Re=1000.0, init_type="spin_1T", A_eq=1e-4, prim_axi_base=axi_b_1000_40["prim_axi"], record_stride=16)),
        ("bell_inf_40",    dict(nx=40, nr=12, nth=48, t_end=30.0, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", Re=np.inf, init_type="spin_1T", A_eq=1e-4, prim_axi_base=axi_b_inf["prim_axi"], record_stride=16)),
        ("st_re500_nx48",  dict(nx=48, nr=12, nth=48, t_end=24.0, M_c=0.12, mode="straight", Re=500.0,  init_type="spin_1T", A_eq=1e-4, record_stride=16)),
        ("st_re1000_nx48", dict(nx=48, nr=12, nth=48, t_end=24.0, M_c=0.12, mode="straight", Re=1000.0, init_type="spin_1T", A_eq=1e-4, record_stride=16)),
        ("st_inf_nx48",    dict(nx=48, nr=12, nth=48, t_end=24.0, M_c=0.12, mode="straight", Re=np.inf, init_type="spin_1T", A_eq=1e-4, record_stride=16)),
    ]
    exp2_robust_tasks = [
        ("bare_f36_2T",   dict(nx=36, nr=18, nth=50, t_end=24.0, M_c=0.12, mode="straight", Re=500.0, init_type="spin_2T", A_eq=1e-4, record_stride=16)),
        ("f36_rb00_2T",   dict(nx=36, nr=18, nth=50, t_end=24.0, M_c=0.12, mode="straight", Re=500.0, n_baffles=5, L_b=0.45, r_b=0.00, init_type="spin_2T", A_eq=1e-4, record_stride=16)),
        ("f36_rb42_2T",   dict(nx=36, nr=18, nth=50, t_end=24.0, M_c=0.12, mode="straight", Re=500.0, n_baffles=5, L_b=0.45, r_b=0.42, init_type="spin_2T", A_eq=1e-4, record_stride=16)),
        ("f36_rb58_2T",   dict(nx=36, nr=18, nth=50, t_end=24.0, M_c=0.12, mode="straight", Re=500.0, n_baffles=5, L_b=0.45, r_b=0.58, init_type="spin_2T", A_eq=1e-4, record_stride=16)),
        ("bare_inf_2T",   dict(nx=24, nr=12, nth=50, t_end=24.0, M_c=0.12, mode="straight", Re=np.inf, init_type="spin_2T", A_eq=1e-4, record_stride=10)),
        ("inf_rb00_2T",   dict(nx=24, nr=12, nth=50, t_end=24.0, M_c=0.12, mode="straight", Re=np.inf, n_baffles=5, L_b=0.45, r_b=0.00, init_type="spin_2T", A_eq=1e-4, record_stride=10)),
        ("inf_rb42_2T",   dict(nx=24, nr=12, nth=50, t_end=24.0, M_c=0.12, mode="straight", Re=np.inf, n_baffles=5, L_b=0.45, r_b=0.42, init_type="spin_2T", A_eq=1e-4, record_stride=10)),
        ("inf_rb58_2T",   dict(nx=24, nr=12, nth=50, t_end=24.0, M_c=0.12, mode="straight", Re=np.inf, n_baffles=5, L_b=0.45, r_b=0.58, init_type="spin_2T", A_eq=1e-4, record_stride=10)),
    ]
    exp3_tasks = [
        ("e014_legacy_1T",     dict(nx=20, nr=12, nth=48, t_end=180.0, M_c=0.12, mode="straight", Re=np.inf, init_type="spin_1T",  A_eq=0.04, n_p=0.30, omega_hp=0.15, cav_spec=cav_e014_dual, cav_mass_conserving=False, record_stride=24)),
        ("e015_mcons_1T",      dict(nx=20, nr=12, nth=48, t_end=180.0, M_c=0.12, mode="straight", Re=np.inf, init_type="spin_1T",  A_eq=0.04, n_p=0.30, omega_hp=0.15, cav_spec=cav_e014_dual, cav_mass_conserving=True,  record_stride=24)),
        ("e015_hp50_mcons_1T", dict(nx=20, nr=12, nth=48, t_end=180.0, M_c=0.12, mode="straight", Re=np.inf, init_type="spin_1T",  A_eq=0.04, n_p=0.30, omega_hp=0.50, cav_spec=cav_e014_dual, cav_mass_conserving=True,  record_stride=24)),
        ("e015_hp50_st1T",     dict(nx=20, nr=12, nth=48, t_end=180.0, M_c=0.12, mode="straight", Re=np.inf, init_type="stand_1T", A_eq=0.04, n_p=0.30, omega_hp=0.50, cav_spec=cav_e014_dual, cav_mass_conserving=True,  record_stride=24)),
    ]
    exp4_tasks = [
        ("bare_ns_1T",    dict(nx=20, nr=12, nth=50, t_end=90.0,  M_c=0.12, mode="straight", Re=500.0, init_type="spin_1T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24)),
        ("bare_ns_st1T",  dict(nx=20, nr=12, nth=50, t_end=90.0,  M_c=0.12, mode="straight", Re=500.0, init_type="stand_1T", A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24)),
        ("bare_ns_2T",    dict(nx=20, nr=12, nth=50, t_end=90.0,  M_c=0.12, mode="straight", Re=500.0, init_type="spin_2T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24)),
        ("bare_ns_1R",    dict(nx=20, nr=12, nth=50, t_end=90.0,  M_c=0.12, mode="straight", Re=500.0, init_type="1R",       A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24)),
        ("bare_ns_1L",    dict(nx=20, nr=12, nth=50, t_end=90.0,  M_c=0.12, mode="straight", Re=500.0, init_type="1L",       A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24)),
        ("hyb3d_1T",      dict(nx=20, nr=12, nth=50, t_end=120.0, M_c=0.12, mode="straight", Re=500.0, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_3d_retuned, cav_mass_conserving=True, init_type="spin_1T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24)),
        ("hyb3d_st1T",    dict(nx=20, nr=12, nth=50, t_end=120.0, M_c=0.12, mode="straight", Re=500.0, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_3d_retuned, cav_mass_conserving=True, init_type="stand_1T", A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24)),
        ("hyb3d_2T",      dict(nx=20, nr=12, nth=50, t_end=120.0, M_c=0.12, mode="straight", Re=500.0, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_3d_retuned, cav_mass_conserving=True, init_type="spin_2T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24)),
        ("hyb3d_1R",      dict(nx=20, nr=12, nth=50, t_end=180.0, M_c=0.12, mode="straight", Re=500.0, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_3d_retuned, cav_mass_conserving=True, init_type="1R",       A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24)),
        ("hyb3d_1L",      dict(nx=20, nr=12, nth=50, t_end=120.0, M_c=0.12, mode="straight", Re=500.0, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_3d_retuned, cav_mass_conserving=True, init_type="1L",       A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24)),
        ("hyb3d_hp15_2T", dict(nx=20, nr=12, nth=50, t_end=120.0, M_c=0.12, mode="straight", Re=500.0, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_3d_retuned, cav_mass_conserving=True, init_type="spin_2T",  A_eq=0.04, n_p=0.34, omega_hp=0.15, record_stride=24)),
        ("hyb3d_f16_1T",  dict(nx=20, nr=16, nth=50, t_end=120.0, M_c=0.12, mode="straight", Re=500.0, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_3d_retuned, cav_mass_conserving=True, init_type="spin_1T",  A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24)),
        ("bell_bare_1T",  dict(nx=36, nr=12, nth=50, t_end=80.0,  M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", Re=500.0, prim_axi_base=axi_b_500_36["prim_axi"], init_type="spin_1T", A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24)),
        ("bell_hyb_1T",   dict(nx=36, nr=12, nth=50, t_end=120.0, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", Re=500.0, n_baffles=5, L_b=0.45, r_b=0.42, cav_spec=cav_3d_retuned, cav_mass_conserving=True, prim_axi_base=axi_b_500_36["prim_axi"], init_type="spin_1T", A_eq=0.04, n_p=0.34, omega_hp=0.50, record_stride=24)),
    ]
    # LPT order: longest bell/refined runs first so all 8 cores stay saturated with zero tail waste
    lpt_priority = [
        "bell_hyb_1T", "bell_bare_1T", "bell_re500_48", "hyb3d_f16_1T", "hyb3d_1R",
        "bell_re500_40", "bell_re1000_40", "bell_inf_40",
        "f36_rb58_2T", "f36_rb42_2T", "f36_rb00_2T", "bare_f36_2T",
        "e014_legacy_1T", "e015_mcons_1T", "e015_hp50_mcons_1T", "e015_hp50_st1T",
        "hyb3d_1T", "hyb3d_st1T", "hyb3d_2T", "hyb3d_1L", "hyb3d_hp15_2T",
        "bare_ns_1T", "bare_ns_st1T", "bare_ns_2T", "bare_ns_1R", "bare_ns_1L",
        "st_re500_nx48", "st_re1000_nx48", "st_inf_nx48",
        "bare_inf_2T", "inf_rb00_2T", "inf_rb42_2T", "inf_rb58_2T",
    ]
    all_p2 = dict(exp1d_tasks + exp2_robust_tasks + exp3_tasks + exp4_tasks)
    p2_ordered = [(k, all_p2[k]) for k in lpt_priority]
    print(f"\n--- Phase 2: Launching {len(p2_ordered)} Heavy Runs (LPT-ordered, 8 workers) ---", flush=True)
    phase2_res = run_batch(p2_ordered)
    exp1d_res = phase2_res
    exp2_res = {**phase1_res, **phase2_res}
    exp3_res = phase2_res
    exp4_res = phase2_res

    exp1d_out = {
        "cheb_inviscid": ex_inf.real,
        "cheb_noslip_Re500": ex_ns_500.real,
        "cheb_stressfree_Re500": ex_sf_500.real,
        "cheb_noslip_Re1000": ex_ns_1000.real,
        "st_inf_nx48": fit_alpha(exp1d_res["st_inf_nx48"], 8.0, 22.0, "e_1t"),
        "st_re500_nx48": fit_alpha(exp1d_res["st_re500_nx48"], 8.0, 22.0, "e_1t"),
        "st_re1000_nx48": fit_alpha(exp1d_res["st_re1000_nx48"], 8.0, 22.0, "e_1t"),
        "bell_inf_40": fit_alpha(exp1d_res["bell_inf_40"], 12.0, 28.0, "e_1t"),
        "bell_re1000_40": fit_alpha(exp1d_res["bell_re1000_40"], 12.0, 28.0, "e_1t"),
        "bell_re500_40": fit_alpha(exp1d_res["bell_re500_40"], 12.0, 28.0, "e_1t"),
        "bell_re500_48": fit_alpha(exp1d_res["bell_re500_48"], 12.0, 28.0, "e_1t"),
        "axi_throat_sensitivity_Re500": {
            "36x12": {k: v for k, v in axi_b_500_36.items() if k != "prim_axi"},
            "40x12": {k: v for k, v in axi_b_500_40.items() if k != "prim_axi"},
            "48x14": {k: v for k, v in axi_b_500_48.items() if k != "prim_axi"},
        },
    }
    print(f"  1D Straight nx48: Re=inf={exp1d_out['st_inf_nx48']:+.5f}, Re=500={exp1d_out['st_re500_nx48']:+.5f} (Cheb inv={ex_inf.real:+.5f}, ns500={ex_ns_500.real:+.5f}, sf500={ex_sf_500.real:+.5f})")
    print(f"  1D Metric-fixed Bell (Ln=0.60): Re=inf={exp1d_out['bell_inf_40']:+.5f}, Re=1000={exp1d_out['bell_re1000_40']:+.5f}, Re=500(40x12)={exp1d_out['bell_re500_40']:+.5f}, (48x14)={exp1d_out['bell_re500_48']:+.5f}")

    # ------------------------------------------------------------------
    # Experiment 2 Summary: Partial-Span Baffles
    # ------------------------------------------------------------------
    print("\n--- Experiment 2: 3D Partial-Span Baffles (E_ch, Window, m=-3 Bragg, Inviscid & 36x18x50) ---")
    a_b1_m = fit_alpha(exp2_res["bare_re500_1T"], 7.0, 22.0, "e_1t")
    a_b1_e = fit_alpha(exp2_res["bare_re500_1T"], 7.0, 22.0, "e_ch")
    w_b1, _, _ = measure_plain_hann_peaks(exp2_res["bare_re500_1T"]["t"], exp2_res["bare_re500_1T"]["c1_re"], 4.0, 1.0, 2.5)
    a_b2_m = fit_alpha(exp2_res["bare_re500_2T"], 7.0, 22.0, "e_2t")
    a_b2_m10 = fit_alpha(exp2_res["bare_re500_2T"], 10.0, 22.0, "e_2t")
    a_b2_e = fit_alpha(exp2_res["bare_re500_2T"], 7.0, 22.0, "e_ch")
    a_b2_e10 = fit_alpha(exp2_res["bare_re500_2T"], 10.0, 22.0, "e_ch")
    w_b2, _, _ = measure_plain_hann_peaks(exp2_res["bare_re500_2T"]["t"], exp2_res["bare_re500_2T"]["c2_re"], 4.0, 1.8, 4.0)

    exp2_rows = []
    for rb_v in rb_grid:
        tag = f"rb{int(round(rb_v*100)):02d}"
        r1 = exp2_res[f"{tag}_1T"]; r2 = exp2_res[f"{tag}_2T"]
        a1_m = fit_alpha(r1, 7.0, 22.0, "e_1t"); a1_e = fit_alpha(r1, 7.0, 22.0, "e_ch")
        w1, _, _ = measure_plain_hann_peaks(r1["t"], r1["c1_re"], 4.0, 1.0, 2.5)
        a2_m = fit_alpha(r2, 7.0, 22.0, "e_2t"); a2_m10 = fit_alpha(r2, 10.0, 22.0, "e_2t")
        a2_e = fit_alpha(r2, 7.0, 22.0, "e_ch"); a2_e10 = fit_alpha(r2, 10.0, 22.0, "e_ch")
        w2_p, w2_s, w2_rat = measure_plain_hann_peaks(r2["t"], r2["c2_re"], 4.0, 1.8, 4.0)
        ml2 = r2["t"] >= 14.0
        bragg_m3_ratio = float(np.mean(r2["e_3tm"][ml2]) / max(np.mean(r2["e_2t"][ml2]), 1e-30))
        row = {
            "r_b": rb_v, "metal_frac": r1["baffle_metal_frac"],
            "alpha_1T": a1_m, "delta_1T": a1_m - a_b1_m, "delta_1T_Ech": a1_e - a_b1_e,
            "omega_1T": w1, "downshift_1T_pct": (w1 - w_b1) / w_b1 * 100.0,
            "alpha_2T_7_22": a2_m, "delta_2T_7_22": a2_m - a_b2_m,
            "alpha_2T_10_22": a2_m10, "delta_2T_10_22": a2_m10 - a_b2_m10,
            "alpha_2T_Ech_7_22": a2_e, "delta_2T_Ech_7_22": a2_e - a_b2_e,
            "alpha_2T_Ech_10_22": a2_e10, "delta_2T_Ech_10_22": a2_e10 - a_b2_e10,
            "omega_2T_primary": w2_p, "omega_2T_secondary": w2_s, "bimodal_amp_ratio": w2_rat,
            "bragg_m3_over_m2": bragg_m3_ratio,
        }
        exp2_rows.append(row)
        print(f"  2B r_b={rb_v:.2f}: dEch_2T[7,22]={row['delta_2T_Ech_7_22']:+.4f} ([10,22]={row['delta_2T_Ech_10_22']:+.4f}) | de2t[7,22]={row['delta_2T_7_22']:+.4f} ([10,22]={row['delta_2T_10_22']:+.4f}) | w2={w2_p:.3f}/{w2_s} ({w2_rat:.2f}) | m-3/m2={bragg_m3_ratio:.2f}")

    exp2_robust = {}
    for pfx, b_key in [("inf", "bare_inf_2T"), ("f36", "bare_f36_2T")]:
        b_e7 = fit_alpha(exp2_res[b_key], 7.0, 22.0, "e_ch"); b_m7 = fit_alpha(exp2_res[b_key], 7.0, 22.0, "e_2t")
        b_e10 = fit_alpha(exp2_res[b_key], 10.0, 22.0, "e_ch"); b_m10 = fit_alpha(exp2_res[b_key], 10.0, 22.0, "e_2t")
        d0_e7 = fit_alpha(exp2_res[f"{pfx}_rb00_2T"], 7.0, 22.0, "e_ch") - b_e7
        d0_m7 = fit_alpha(exp2_res[f"{pfx}_rb00_2T"], 7.0, 22.0, "e_2t") - b_m7
        d0_e10 = fit_alpha(exp2_res[f"{pfx}_rb00_2T"], 10.0, 22.0, "e_ch") - b_e10
        d0_m10 = fit_alpha(exp2_res[f"{pfx}_rb00_2T"], 10.0, 22.0, "e_2t") - b_m10
        sub = {"bare_e2t_7_22": b_m7, "bare_Ech_7_22": b_e7}
        for rb_v in (0.00, 0.42, 0.58):
            tag = f"rb{int(round(rb_v*100)):02d}"
            rr = exp2_res[f"{pfx}_{tag}_2T"]; ml = rr["t"] >= 14.0
            de7 = fit_alpha(rr, 7.0, 22.0, "e_ch") - b_e7
            dm7 = fit_alpha(rr, 7.0, 22.0, "e_2t") - b_m7
            de10 = fit_alpha(rr, 10.0, 22.0, "e_ch") - b_e10
            dm10 = fit_alpha(rr, 10.0, 22.0, "e_2t") - b_m10
            sub[tag] = {
                "ratio_Ech_7_22": de7 / d0_e7, "ratio_Ech_10_22": de10 / d0_e10,
                "ratio_e2t_7_22": dm7 / d0_m7, "ratio_e2t_10_22": dm10 / d0_m10,
                "bragg_m3_over_m2": float(np.mean(rr["e_3tm"][ml]) / max(np.mean(rr["e_2t"][ml]), 1e-30)),
            }
            print(f"  2C [{pfx} {tag}]: Ech ratio [7,22]={de7/d0_e7:.2f}x ([10,22]={de10/d0_e10:.2f}x) | e2t [7,22]={dm7/d0_m7:.2f}x ([10,22]={dm10/d0_m10:.2f}x) | m-3/m2={sub[tag]['bragg_m3_over_m2']:.2f}")
        exp2_robust[pfx] = sub

    # ------------------------------------------------------------------
    # Experiment 3 Summary: Honest t = 180 k = 0 Chugging Audit
    # ------------------------------------------------------------------
    print("\n--- Experiment 3: Honest t = 180 k = 0 Chugging & Flame-Cutoff Audit ---")
    chug_eigs = {}
    for np_v in (0.30, 0.34):
        for hp in (0.15, 0.50):
            for bt in (0.0, 0.03):
                c0, c1 = exact_1d_chugging_eig(np_v, hp, bt)
                key = f"np{int(round(np_v*100))}_hp{int(round(hp*100))}_b{int(round(bt*100))}"
                chug_eigs[key] = {
                    "chug_alpha": c0.real if c0 else None, "chug_omega": c0.imag if c0 else None,
                    "chug_period_1d": (2.0 * np.pi / c0.imag) if (c0 and c0.imag > 0) else None,
                    "1L_alpha": c1.real if c1 else None, "1L_omega": c1.imag if c1 else None,
                }
    exp3_runs = {}
    for cid, _ in exp3_tasks:
        rr = exp3_res[cid]; t = rr["t"]; m160 = t >= 160.0
        T_pde = measure_zero_crossing_period(t, rr["p_mean_signed"], 20.0)
        exp3_runs[cid] = {
            "p_peak_160_180": float(np.mean(rr["p_max_pct"][m160])),
            "max_abs_pbar_pct_160_180": float(np.max(np.abs(rr["p_mean_signed"][m160])) / P_BAR * 100.0),
            "max_abs_dm_pct_160_180": float(np.max(np.abs(rr["mass_drift"][m160])) * 100.0),
            "e_1t_160_180": float(np.mean(rr["e_1t"][m160])),
            "e_mean_160_180": float(np.mean(rr["e_mean"][m160])),
            "pde_chug_period": T_pde,
        }
        s = exp3_runs[cid]
        print(f"  3B(t=180) [{cid:18s}] P={s['p_peak_160_180']:5.2f}% | |p̄'|={s['max_abs_pbar_pct_160_180']:5.2f}% | |dm|={s['max_abs_dm_pct_160_180']:5.3f}% | E_1T={s['e_1t_160_180']:.2e} | E_mn={s['e_mean_160_180']:.2e} | T={T_pde}")

    # ------------------------------------------------------------------
    # Experiment 4 Summary: Full 3D NS All-Mode Audit
    # ------------------------------------------------------------------
    print("\n--- Experiment 4: Full 3D NS All-Mode Audit (omega_hp=0.50, Re=500, Metric-Fixed Bell) ---")
    exp4_out = {}
    for cid, _ in exp4_tasks:
        rr = exp4_res[cid]; t = rr["t"]; ml = t >= 0.85 * t[-1]
        exp4_out[cid] = {
            "t_end": float(t[-1]),
            "p_peak_late": float(np.mean(rr["p_max_pct"][ml])),
            "e_wave_late": float(np.mean(rr["e_ch"][ml])),
            "e_1t_late": float(np.mean(rr["e_1t"][ml])),
            "e_2t_late": float(np.mean(rr["e_2t"][ml])),
            "e_1r_late": float(np.mean(rr["e_1r"][ml])),
            "e_1l_late": float(np.mean(rr["e_1l"][ml])),
            "e_mean_late": float(np.mean(rr["e_mean"][ml])),
            "slope_tail": fit_alpha(rr, 0.65 * t[-1], t[-1], "e_ch"),
        }
        s = exp4_out[cid]
        print(f"  4 [{cid:14s}] t={s['t_end']:5.1f} P={s['p_peak_late']:6.3f}% | E={s['e_wave_late']:.2e} | slp={s['slope_tail']:+.4f} | 1T={s['e_1t_late']:.1e} | 2T={s['e_2t_late']:.1e} | 1L={s['e_1l_late']:.1e} | mn={s['e_mean_late']:.1e}")

    metrics = {
        "exp1_3d_ns_benchmarks": {
            "quiescent_3d_ns_bell_err": rq["quiescent_err"],
            "cons_ns_bell_bare": cb,
            "cons_ns_bell_b5_slip": cs,
            "cons_ns_bell_b5_noslip": cn,
            "closed_chamber_cavity_mass_err": cm,
            "stokes_kirchhoff_3d_refinement": sk_out,
            "fixed_Re_and_metric_bell_audit": exp1d_out,
        },
        "exp2_partial_span_baffles": {
            "bare_re500": {"alpha_1T": a_b1_m, "omega_1T": w_b1, "alpha_2T_7_22": a_b2_m, "alpha_2T_Ech_7_22": a_b2_e, "omega_2T": w_b2},
            "rb_sweep": exp2_rows,
            "robustness_inviscid_and_36x18": exp2_robust,
            "retuned_targets_rb42": {"wH_1T": wH_1T_ret, "wH_2T": wH_2T_ret},
        },
        "exp3_chugging_and_cutoff_audit": {
            "cheb_1d_chugging_eigs": chug_eigs,
            "pde_3d_t180_runs": exp3_runs,
        },
        "exp4_3d_ns_allmode_audit": exp4_out,
    }
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved hardened Entry 015 metrics to {out_path}")

    # ------------------------------------------------------------------
    # Figures (Hardened, zero legend overlap, 100% honest titles)
    # ------------------------------------------------------------------
    # Fig 1: Partial-Span Baffles — Robust Total-Energy E_ch Boost (Re=500, Re=inf, 36x18) + Bimodal 2T/m=-3 Spectrum
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.5), dpi=150)
    rbs = [r["r_b"] for r in exp2_rows]
    m_pct = [r["metal_frac"] * 100.0 for r in exp2_rows]
    d1_ret = [r["delta_1T"] / exp2_rows[0]["delta_1T"] * 100.0 for r in exp2_rows]
    d2_ech = [r["delta_2T_Ech_7_22"] / exp2_rows[0]["delta_2T_Ech_7_22"] * 100.0 for r in exp2_rows]
    ax1.plot(rbs, d2_ech, "^-", color="#d62728", lw=2.6, label="$2T$ Total-Energy $E_{\\mathrm{ch}}$ Gain (24x12x50, Re=500)")
    ax1.plot([0.0, 0.42, 0.58], [exp2_robust["f36"][k]["ratio_Ech_7_22"] * 100.0 for k in ("rb00", "rb42", "rb58")],
             "D--", color="#ff7f0e", lw=2.1, label="$2T$ Total-Energy $E_{\\mathrm{ch}}$ Gain (36x18x50, Re=500)")
    ax1.plot([0.0, 0.42, 0.58], [exp2_robust["inf"][k]["ratio_Ech_7_22"] * 100.0 for k in ("rb00", "rb42", "rb58")],
             "v:", color="#9467bd", lw=2.0, label="$2T$ Total-Energy $E_{\\mathrm{ch}}$ Gain (24x12x50, Re=$\\infty$)")
    ax1.plot(rbs, d1_ret, "s-", color="#1f77b4", lw=2.3, label="$1T$ Baffle Damping Retained ($\\Delta\\alpha / \\Delta\\alpha_{\\mathrm{full}}$)")
    ax1.plot(rbs, m_pct, "o:", color="#555555", lw=1.9, label="Blade Metal Area ($1 - r_b$)")
    ax1.axvline(0.42, color="#2ca02c", ls="-.", lw=1.5, alpha=0.8, label="Shroud-Ring $r_b=0.42$ (hub open)")
    ax1.set_xlabel("Inner Baffle Cutout Radius $r_b / R$", fontsize=11)
    ax1.set_ylabel("Percentage of Full-Span Value (%)", fontsize=11)
    ax1.set_title("3D Partial-Span Baffle Total-Energy $E_{\\mathrm{ch}}$ Law ($N_b=5, L_b=0.45$)", fontsize=10.8, fontweight="bold")
    ax1.set_ylim(0, 265)
    ax1.grid(True, alpha=0.28)
    ax1.legend(fontsize=7.6, loc="upper left", framealpha=0.94)

    ax2.plot(rbs, [r["bragg_m3_over_m2"] for r in exp2_rows], "s-", color="#d62728", lw=2.4, label="Bragg Transfer $E(m=-3) / E(m=+2)$ (24x12x50)")
    ax2.plot([0.0, 0.42, 0.58], [exp2_robust["f36"][k]["bragg_m3_over_m2"] for k in ("rb00", "rb42", "rb58")],
             "D--", color="#ff7f0e", lw=2.1, label="Bragg Transfer $E(m=-3) / E(m=+2)$ (36x18x50)")
    ax2.plot(rbs, [r["bimodal_amp_ratio"] for r in exp2_rows], "^-.", color="#1f77b4", lw=2.2, label="Bimodal FFT Secondary/Primary Peak Ratio ($c_{2,\\mathrm{re}}$)")
    ax2.axvline(0.42, color="#2ca02c", ls="-.", lw=1.5, alpha=0.8)
    ax2.set_xlabel("Inner Baffle Cutout Radius $r_b / R$", fontsize=11)
    ax2.set_ylabel("Modal Coupling / Bimodality Ratio", fontsize=11)
    ax2.set_title("Why Raw $e_{2T}$ Beats: $m=-3$ Bragg Scattering & Bimodality", fontsize=10.8, fontweight="bold")
    ax2.set_ylim(-0.02, 0.92)
    ax2.grid(True, alpha=0.28)
    ax2.legend(fontsize=8.1, loc="upper right", framealpha=0.94)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp015_3d_partial_span_baffle_law.png", dpi=150)
    plt.close(fig)

    # Fig 2: Honest t=180 k=0 Chugging Audit (m_dot still chugs at omega_hp=0.15; flame cutoff omega_hp=0.50 eliminates it)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.5), dpi=150)
    for cid, lbl, col, ls in [
        ("e014_legacy_1T",     "Legacy $u_{\\mathrm{port}}$ ($\\omega_{\\mathrm{hp}}=0.15$): chugs $T\\approx 39$", "#d62728", "-"),
        ("e015_mcons_1T",      "Mass-Cons. $\\dot{m}$ ($\\omega_{\\mathrm{hp}}=0.15$): STILL chugs!", "#1f77b4", "--"),
        ("e015_hp50_mcons_1T", "Flame Cutoff $\\omega_{\\mathrm{hp}}=0.50$: spin $1T$ (chug-free)", "#2ca02c", "-"),
        ("e015_hp50_st1T",     "Flame Cutoff $\\omega_{\\mathrm{hp}}=0.50$: stand $1T$ (decays!)", "#9467bd", "-."),
    ]:
        rr = exp3_res[cid]
        ax1.semilogy(rr["t"], np.maximum(rr["e_mean"], 1e-13), color=col, ls=ls, lw=2.1, label=lbl)
        ax2.plot(rr["t"], rr["mass_drift"] * 100.0, color=col, ls=ls, lw=2.1, label=f"{lbl} (P180={exp3_runs[cid]['p_peak_160_180']:.2f}%)")
    ax1.set_xlabel("Time $t$", fontsize=11)
    ax1.set_ylabel("Chamber-Mean $k=0$ Bulk Energy $E_{\\mathrm{mean}}(t)$", fontsize=11)
    ax1.set_title("Honest $t=180$ Chugging Audit: $\\dot{m}$ Alone Chugs; $\\omega_{\\mathrm{hp}}=0.50$ Suppresses", fontsize=10.3, fontweight="bold")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=7.6, loc="lower left", framealpha=0.94)

    ax2.set_xlabel("Time $t$", fontsize=11)
    ax2.set_ylabel("Chamber Total Mass Excursion $\\Delta M / M_0$ (%)", fontsize=11)
    ax2.set_title("Signed Mass Excursion Out to $t=180$ ($T_{\\mathrm{PDE}}\\approx 39.3$ vs 1D $56$)", fontsize=10.3, fontweight="bold")
    ax2.grid(True, alpha=0.28)
    ax2.legend(fontsize=7.3, loc="lower left", framealpha=0.94)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp015_mass_conserving_cavity_cure.png", dpi=150)
    plt.close(fig)

    # Fig 3: Full 3D NS All-Mode Audit + Metric-Corrected Smooth Bell (t=120) & Chugging Contrast
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.5), dpi=150)
    for cid, lbl, col, ls in [
        ("bare_ns_1T",   "Bare 3D NS: spin $1T$", "#d62728", "-"),
        ("bare_ns_st1T", "Bare 3D NS: stand $1T$", "#e377c2", ":"),
        ("bare_ns_2T",   "Bare 3D NS: spin $2T$", "#8c564b", "--"),
        ("bare_ns_1L",   "Bare 3D NS: $1L$", "#777777", ":"),
        ("hyb3d_1T",     "3D Hybrid: spin $1T$ (stationary)", "#1f77b4", "-"),
        ("hyb3d_st1T",   "3D Hybrid: stand $1T$ (growing to $1L$)", "#9467bd", "--"),
        ("hyb3d_2T",     "3D Hybrid: spin $2T$ (growing to $1L$)", "#17becf", "-."),
        ("hyb3d_1R",     "3D Hybrid: $1R$ (linear $1L$ +0.006 to $t=180$)", "#2ca02c", "-"),
        ("hyb3d_1L",     "3D Hybrid: $1L$ (survives)", "#2ca02c", ":"),
    ]:
        rr = exp4_res[cid]; st = exp4_out[cid]
        ax1.semilogy(rr["t"], np.maximum(rr["e_ch"], 1e-13), color=col, ls=ls, lw=1.9, label=f"{lbl} (P={st['p_peak_late']:.2f}%)")
    ax1.set_xlabel("Time $t$", fontsize=11)
    ax1.set_ylabel("3D Chamber Wave Energy $E_{\\mathrm{wave}}(t)$", fontsize=11)
    ax1.set_title("3D NS ($\\mathrm{Re}=500, \\omega_{\\mathrm{hp}}=0.50, n_p=0.34$): All-Mode Audit", fontsize=10.5, fontweight="bold")
    ax1.set_ylim(1e-10, 8e-3)
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=6.9, loc="lower right", framealpha=0.94)

    for cid, lbl, col, ls in [
        ("bell_bare_1T",  "True-3D Smooth Bell NS (Bare): spin $1T$", "#d62728", "-"),
        ("bell_hyb_1T",   "True-3D Smooth Bell NS (Hybrid, $t=120$): spin $1T$", "#1f77b4", "-"),
        ("hyb3d_1T",      "Straight 3D NS (Hybrid, 20x12x50, $t=120$)", "#2ca02c", "--"),
        ("hyb3d_f16_1T",  "Straight 3D NS (Hybrid, 20x16x50, $t=120$)", "#9467bd", "-."),
        ("hyb3d_hp15_2T", "Contrast: $\\omega_{\\mathrm{hp}}=0.15$ $2T$ (chugging!)", "#ff7f0e", ":"),
    ]:
        rr = exp4_res[cid]; st = exp4_out[cid]
        ax2.semilogy(rr["t"], np.maximum(rr["e_ch"], 1e-13), color=col, ls=ls, lw=2.1, label=f"{lbl} (P={st['p_peak_late']:.2f}%)")
    ax2.set_xlabel("Time $t$", fontsize=11)
    ax2.set_ylabel("3D Chamber Wave Energy $E_{\\mathrm{wave}}(t)$", fontsize=11)
    ax2.set_title("Metric-Fixed 3D Bell NS ($t=120$), Radial Check & Chugging Contrast", fontsize=10.3, fontweight="bold")
    ax2.set_ylim(1e-6, 1e-2)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=7.4, loc="lower left", framealpha=0.94)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp015_3d_ns_allmode_quench.png", dpi=150)
    plt.close(fig)
    print("Saved all 3 hardened Entry 015 figures.")


if __name__ == "__main__":
    main()
