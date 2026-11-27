"""Entry 014 (rocket-combustion, Hardened per Peer Review #11): Full 3D Cylindrical
(x, r, theta) Chamber & Transonic Bell Nozzle — Exact Bessel Cutoff Collapse (s_11,
s_21, s_02), True-3D Meridional Bell Damping, Chiral 1T -> 1R vs. 2T Cascade Selection
Rule, and Shroud-vs.-Hub Faceplate Cavity Placement Law.

Upgrades the rocket-combustion track (004-013) from the 2D thin-shell annulus
(k_theta = m / R(x)) to a true 3D cylindrical (x, r, theta) conservative finite-volume
compressible Euler solver on body-fitted conical-frustum bell-nozzle cells:
  - Discrete Geometric Conservation Law well-balancing (gcl_x ~ 2.6e-18, quiescent ~ 4.7e-18;
    Simpson quadrature metric consistency <= 2.5e-4, O(dtheta^2) ~ 0.07% hoop projection).
  - Exact discrete conservation of mass, axial angular momentum (r * rho * u_theta), and total
    energy (~1e-16) with singularity-free centerline (S(r=0) = 0, antipodal theta+pi MUSCL
    ghosts) and mirrored-face slanted bell wall slip (v_n,R = -v_n,L).

Hardened discoveries verified against Peer Review #11:
  1. 3D Bessel Cutoff Collapse vs. True-3D Meridional Bell Damping (Exp 1 & 2):
     - Straight-cylinder axial refinement (nx = 24 -> 48 -> 96) proves the coarse nx = 24
       straight 1T damping (-0.09333) is axial numerical vorticity damping that moves
       monotonically toward exact inviscid Chebyshev (-0.04239; late nx = 96: -0.0467).
     - On Entry 012's C^2 bell (N = 160 converged, single-branch fundamental 1R), inviscid
       quasi-3D WKB 1T damping collapses 4.77x harder than 2D thin-shell at L_n = 0.50
       (-0.0245 -> -0.0051), with bare Hopf at fixed tau_bar = 0.16 dropping 3.93x at
       M_c = 0.14 (and 3.36x at tau_bar = 0.40).
     - In contrast to the straight cylinder, the true-3D conical-frustum smooth bell
       exhibits Sauer/Hall curved sonic-line throat bowing (M_throat = 0.804 -> 1.376)
       and robust ~-0.042 to -0.049 cold 1T damping across both axial (nx = 40 -> 80) and
       radial (nr = 12 -> 20) refinement (~20x stronger than inviscid quasi-3D WKB -0.0023),
       isolating genuine 3D meridional/curved-throat physics.
  2. 3D Chiral 1T -> 1R vs. 1T -> 2T Nonlinear Cascade Selection Rule (Exp 3):
     - In 3D cylinders, 2T sits at 1.659 * omega_1T (detuned), whereas 1R sits at
       2.081 * omega_1T (within 3.9% of 2 * omega_1T).
     - Standing 1T pumps oscillating AC 1R ~600x stronger than spinning 1T in [6.4, 16]
       (600-970x across later windows; 427x in reacting 3D throughflow), following the exact
       quartic law E_1R,ac ~ A_0^3.99 for A_0 <= 0.02 (3.88 over full range), while spinning
       1T's smaller AC 1R also scales physically at second order (A_0^3.87).
  3. 3D Shroud (r=1) vs. Hub (r=0) Faceplate Placement & Dual-Zone Shootout (Exp 4):
     - Exact modal weights W_11(1)=1.418, W_21(1)=1.751 vs. W_02(0)=6.165 (and exact zero
       projection int_0^1 J_0(s_02 r) r dr = 0 for uniform-1R, J_1(0) = 0 for hub-1T) give
       0.5% shroud/uniform Delta-alpha = 1.462x (12x48) / 1.456x (20x64) with matched bare
       windows, and a 4-point xi_H sweep directly locates the 3D shroud EP peak near xi_H~0.24.
     - On refined 24x20x64, a 0.35% micro-hub beats a 5.7x larger 2.0% shroud on 1R by 2.4x
       on [7, 22] (1.3x on [14, 24]). In reacting 20x20x64 1R tests, shroud-only FAILS
       (+0.0046, 7.41% growing) while dual-zone 0.35% micro-hub quenches 1R (~0.11%, ~6,800x
       energy drop vs. bare). At t = 120 on 20x12x48 (n_p = 0.30), dual-zone 1T/stand-1T
       re-grow via untracked axial longitudinal (1L/2L) / bulk dynamics while 1T keeps
       decaying and the 1R standing daughter is cut ~53x below bare (~3.4x below shroud-only).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig
from scipy.optimize import brentq
from scipy.special import jn_zeros, jv, jvp


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

GAMMA = 1.25
P_BAR = 1.0 / GAMMA       # 0.80 => c_bar = 1.0, rho_bar = 1.0
RHO_BAR = 1.0
L_A = 1.5
L_D_STD = 0.25
X_F = 0.25
SIG_F = 0.08
OMEGA_HP = 0.15
TAU_BAR = 0.16            # Scaled to 3D Bessel frequencies so 1T (1.84), 2T (3.05), 1R (3.83) sit in active Rayleigh band
Q_S = 0.045
KAPPA_NL = 4.5


# ---------------------------------------------------------------------------
# 1. Exact 3D Cylindrical Bessel Constants & Modal Energy Weights
# ---------------------------------------------------------------------------

def bessel_prime_zero(m: int, n: int) -> float:
    """n-th positive zero of J'_m(s) = 0 (for m=0, n=1 gives 1R = 3.831706)."""
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
S12 = bessel_prime_zero(1, 2)   # 5.33144277 (1T1R)


def bessel_modal_weight(m: int, s_mn: float, r: np.ndarray | float) -> np.ndarray | float:
    """Normalized area energy density W_mn(r) = |J_m(s r)|^2 / [(1 - m^2/s^2) J_m(s)^2]."""
    jm_wall = float(jv(m, s_mn))
    norm = (1.0 - (m * m) / (s_mn * s_mn)) * (jm_wall * jm_wall)
    return (jv(m, s_mn * r) ** 2) / norm


# ---------------------------------------------------------------------------
# 2. Nozzle Contours & Exact Quasi-3D Bessel-Chebyshev Eigen-Solver
# ---------------------------------------------------------------------------

def isentropic_area_ratio(M: float) -> float:
    g = GAMMA
    return (1.0 / M) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * M * M)) ** ((g + 1.0) / (2.0 * (g - 1.0)))


def nozzle_R_c2_entry012(x: np.ndarray, M_c: float, L_n: float) -> tuple[np.ndarray, np.ndarray]:
    """Entry 012's C^2 quartic/rational area bell contour (R = sqrt(A)) for direct 3D vs 2D comparison."""
    A_t = 1.0 / isentropic_area_ratio(M_c)
    k_t = 12.0 * (1.0 - A_t) / (L_n * L_n)
    dA = 0.60 * A_t
    y0_sq = 2.0 * dA / k_t
    A = np.ones_like(x, dtype=float)
    dA_x = np.zeros_like(x, dtype=float)
    x_t = L_A + L_n
    m1 = (x > L_A) & (x <= x_t)
    s1 = (x[m1] - L_A) / L_n
    A[m1] = A_t + (1.0 - A_t) * (1.0 - 4.0 * s1**3 + 3.0 * s1**4)
    dA_x[m1] = (1.0 - A_t) * (-12.0 * s1**2 + 12.0 * s1**3) / L_n
    m2 = x > x_t
    y = x[m2] - x_t
    den = y * y + y0_sq
    A[m2] = A_t + dA * (y * y / den)
    dA_x[m2] = dA * (2.0 * y * y0_sq / (den * den))
    R = np.sqrt(A)
    dR = 0.5 * dA_x / R
    return R, dR


def nozzle_R_smooth_3d(x: np.ndarray, M_c: float, L_n: float, s_max: float = 0.28) -> tuple[np.ndarray, np.ndarray]:
    """Smooth C^2 cosine-converging + 15.6-deg cone diverging 3D rocket bell contour (separation-free in 3D)."""
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


def exact_straight_bessel_mc_eig(
    M_c: float, s_mn: float, n_p: float = 0.0, tau_bar: float = TAU_BAR,
    omega_window: tuple[float, float] | None = None, N: int = 96,
) -> complex:
    """Exact inviscid 1D Chebyshev eigen-solver for a straight 3D cylinder (R=1) with choked inlet & Marble-Candel exit."""
    x, Dx, w_cc = cheb_diff(N, L_A)
    n = N + 1
    has_f = abs(n_p) > 0.0
    dim = 4 * n + (5 if has_f else 0)
    Mat = np.zeros((dim, dim), dtype=complex)
    B = np.eye(dim, dtype=complex)
    sl = lambda q: slice(q * n, (q + 1) * n)
    Mat[sl(0), sl(0)] = -M_c * Dx
    Mat[sl(0), sl(1)] = -Dx
    Mat[sl(0), sl(2)] = -1j * s_mn * np.eye(n)
    Mat[sl(1), sl(1)] = -M_c * Dx
    Mat[sl(1), sl(3)] = -Dx
    Mat[sl(2), sl(2)] = -M_c * Dx
    Mat[sl(2), sl(3)] = -1j * s_mn * np.eye(n)
    Mat[sl(3), sl(1)] = -Dx
    Mat[sl(3), sl(2)] = -1j * s_mn * np.eye(n)
    Mat[sl(3), sl(3)] = -M_c * Dx
    if has_f:
        wf = np.exp(-0.5 * ((x - X_F) / SIG_F) ** 2)
        wf /= float(np.sum(w_cc * wf))
        ws = w_cc * wf; inv_ts = 4.0 / tau_bar; ib = 4 * n
        Mat[ib + 0, sl(3)] = OMEGA_HP * ws; Mat[ib + 0, ib + 0] = -OMEGA_HP
        Mat[ib + 1, sl(3)] = inv_ts * ws; Mat[ib + 1, ib + 0] = -inv_ts; Mat[ib + 1, ib + 1] = -inv_ts
        for r in range(1, 4):
            Mat[ib + 1 + r, ib + r] = inv_ts; Mat[ib + 1 + r, ib + 1 + r] = -inv_ts
        Mat[sl(3), ib + 4] = n_p * wf
    for r in (0, n, 2 * n):
        Mat[r, :] = 0.0; B[r, :] = 0.0
    Mat[0, 0] = 1.0; Mat[0, 3 * n] = -1.0
    Mat[n, n] = 1.0; Mat[n, 3 * n] = M_c
    Mat[2 * n, 2 * n] = 1.0
    r_ex = 3 * n + N
    Mat[r_ex, :] = 0.0; B[r_ex, :] = 0.0
    Mat[r_ex, n + N] = 1.0; Mat[r_ex, 3 * n + N] = -0.5 * (GAMMA - 1.0) * M_c
    vals = eig(Mat, B, right=False)
    if omega_window is None:
        w_lo, w_hi = (0.5, 2.6) if s_mn == 0.0 else (0.70 * s_mn, 1.35 * s_mn)
    else:
        w_lo, w_hi = omega_window
    cands = [v for v in vals if np.isfinite(v) and w_lo <= v.imag <= w_hi and -0.50 < v.real < 0.35]
    cands.sort(key=lambda z: z.real, reverse=True)
    return complex(cands[0])


def quasi3d_bessel_bell_eig(
    M_c: float, s_mn: float, L_n: float = 0.50, L_d: float = L_D_STD,
    contour: str = "entry012", n_p: float = 0.0, tau_bar: float = TAU_BAR,
    omega_window: tuple[float, float] | None = None, N: int = 96,
) -> complex:
    """Inviscid quasi-3D WKB Bessel-wavenumber (k_perp(x) = s_mn / R(x)) transonic bell Chebyshev solver."""
    L_tot = L_A + L_n + L_d
    x, Dx, w_cc = cheb_diff(N, L_tot)
    R, dR = nozzle_R_c2_entry012(x, M_c, L_n) if contour == "entry012" else nozzle_R_smooth_3d(x, M_c, L_n)
    A = R * R; dA = 2.0 * R * dR
    A_t = 1.0 / isentropic_area_ratio(M_c); x_t = L_A + L_n
    M = np.zeros_like(x)
    for i, xi in enumerate(x):
        ar = float(A[i] / A_t)
        if ar <= 1.0 + 1e-14:
            M[i] = 1.0
        elif xi < x_t:
            M[i] = brentq(lambda mm: isentropic_area_ratio(mm) - ar, 1e-7, 1.0 - 1e-13)
        else:
            M[i] = brentq(lambda mm: isentropic_area_ratio(mm) - ar, 1.0 + 1e-13, 12.0)
    fac_c = 1.0 + 0.5 * (GAMMA - 1.0) * M_c * M_c
    fac = 1.0 + 0.5 * (GAMMA - 1.0) * M * M
    T = fac_c / fac; rho = T ** (1.0 / (GAMMA - 1.0)); p = P_BAR * (T ** (GAMMA / (GAMMA - 1.0))); ux = M * np.sqrt(T)
    n = N + 1
    drho = Dx @ rho; dux = Dx @ ux; dp = Dx @ p; dAA = dA / A; dRR = dR / R
    has_f = abs(n_p) > 0.0
    dim = 4 * n + (5 if has_f else 0)
    Mat = np.zeros((dim, dim), dtype=complex); B = np.eye(dim, dtype=complex)
    sl = lambda q: slice(q * n, (q + 1) * n)
    Mat[sl(0), sl(0)] = -np.diag(ux) @ Dx - np.diag(dux + dAA * ux)
    Mat[sl(0), sl(1)] = -np.diag(rho) @ Dx - np.diag(drho + dAA * rho)
    Mat[sl(0), sl(2)] = -1j * s_mn * np.diag(rho / R)
    Mat[sl(1), sl(0)] = np.diag(dp / (rho * rho))
    Mat[sl(1), sl(1)] = -np.diag(ux) @ Dx - np.diag(dux)
    Mat[sl(1), sl(3)] = -np.diag(1.0 / rho) @ Dx
    Mat[sl(2), sl(2)] = -np.diag(ux) @ Dx - np.diag(ux * dRR)
    Mat[sl(2), sl(3)] = -1j * s_mn * np.diag(1.0 / (R * rho))
    Mat[sl(3), sl(1)] = -np.diag(GAMMA * p) @ Dx - np.diag(dp + GAMMA * p * dAA)
    Mat[sl(3), sl(2)] = -1j * s_mn * np.diag(GAMMA * p / R)
    Mat[sl(3), sl(3)] = -np.diag(ux) @ Dx - np.diag(GAMMA * (dux + dAA * ux))
    if has_f:
        wf = np.where(x <= L_A, np.exp(-0.5 * ((x - X_F) / SIG_F) ** 2), 0.0)
        wf /= float(np.sum(w_cc * wf))
        ws = w_cc * wf; inv_ts = 4.0 / tau_bar; ib = 4 * n
        Mat[ib + 0, sl(3)] = OMEGA_HP * ws; Mat[ib + 0, ib + 0] = -OMEGA_HP
        Mat[ib + 1, sl(3)] = inv_ts * ws; Mat[ib + 1, ib + 0] = -inv_ts; Mat[ib + 1, ib + 1] = -inv_ts
        for r in range(1, 4):
            Mat[ib + 1 + r, ib + r] = inv_ts; Mat[ib + 1 + r, ib + 1 + r] = -inv_ts
        Mat[sl(3), ib + 4] = n_p * wf
    for r in (0, n, 2 * n):
        Mat[r, :] = 0.0; B[r, :] = 0.0
    Mat[0, 0] = 1.0; Mat[0, 3 * n] = -1.0
    Mat[n, n] = 1.0; Mat[n, 3 * n] = M_c
    Mat[2 * n, 2 * n] = 1.0
    vals = eig(Mat, B, right=False)
    if omega_window is None:
        w_lo, w_hi = (0.4, 2.7) if s_mn == 0.0 else (0.72 * s_mn, 1.28 * s_mn)
    else:
        w_lo, w_hi = omega_window
    cands = [v for v in vals if np.isfinite(v) and w_lo <= v.imag <= w_hi and -0.45 < v.real < 0.35]
    cands.sort(key=lambda z: z.real, reverse=True)
    return complex(cands[0])


# ---------------------------------------------------------------------------
# 3. True 3D Cylindrical Body-Fitted Finite-Volume Compressible Euler Solver
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


def inlet_choked_3d(rho_i, ux_i, p_i, M_c: float, u_port: np.ndarray | None = None):
    c_i = np.sqrt(GAMMA * p_i / rho_i)
    Jm = ux_i - (2.0 / (GAMMA - 1.0)) * c_i
    c_g = np.maximum(c_i.copy(), 0.2)
    a_exp = 2.0 / (GAMMA - 1.0)
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
    """Exact Marble-Candel compact choked exit BC: u_x = M_c * c along outgoing J+."""
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

    def R_of(xv):
        if mode in ("straight", "closed"):
            return np.ones_like(xv)
        if mode == "periodic_bump":
            return np.sqrt(1.0 - 0.35 * np.sin(np.pi * xv / L_tot) ** 2)
        if mode == "bell_c2":
            return nozzle_R_c2_entry012(xv, M_c, L_n)[0]
        return nozzle_R_smooth_3d(xv, M_c, L_n)[0]

    Rf = R_of(xf); Rm = R_of(xc)
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
    return {
        "xc": xc, "xf": xf, "etac": etac, "etaf": etaf, "thc": thc, "dx": dx, "deta": deta, "dth": dth,
        "L_tot": L_tot, "Rm": Rm, "Rf": Rf, "gcl_x": gcl_x, "gcl_r": gcl_r, "dA_in": dA_in,
        "V": V[:, :, None], "r_bar": r_bar[:, :, None], "Sx": Sx[:, :, None], "r_xf": r_xf[:, :, None],
        "Sm": Sm[:, :, None], "nxe": nxe[:, :, None], "nre": nre[:, :, None], "r_ef": r_ef[:, :, None],
        "S_th": S_th[:, :, None], "r_thf": r_thf[:, :, None], "hoop": hoop[:, :, None],
    }


def relax_axi_3d_bell_base(
    nx: int, nr: int, M_c: float = 0.12, L_n: float = 0.60, L_d: float = 0.45,
    mode: str = "bell_smooth", t_relax: float = 160.0,
) -> dict:
    """Relax 2D axisymmetric (nth=1) transonic 3D bell steady base flow."""
    g = build_3d_geom(nx, nr, 1, M_c=M_c, L_n=L_n, L_d=L_d, mode=mode)
    V = g["V"][:, :, 0]; Sx = g["Sx"][:, :, 0]; Sm = g["Sm"][:, :, 0]
    nxe = g["nxe"][:, :, 0]; nre = g["nre"][:, :, 0]; hoop = g["hoop"][:, :, 0]
    xc, Rm, dx, deta = g["xc"], g["Rm"], g["dx"], g["deta"]
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
        return res

    t = 0.0; last_rhs = 1.0
    while t < t_relax - 1e-12:
        r_s = np.maximum(Q[0] / V, 1e-10); u_s = Q[1] / (V * r_s); v_s = Q[2] / (V * r_s)
        pr_s = np.maximum((GAMMA - 1.0) * (Q[4] / V - 0.5 * r_s * (u_s * u_s + v_s * v_s)), 1e-10)
        c_s = np.sqrt(GAMMA * pr_s / r_s)
        dt = min(0.32 / float(np.max((np.abs(u_s) + c_s) / dx + (np.abs(v_s) + c_s) / (deta * Rm[:, None]))), t_relax - t)
        k1 = rhs_axi(Q); Q1 = Q + dt * k1
        k2 = rhs_axi(Q1); Q = 0.5 * (Q + Q1 + dt * k2)
        t += dt; last_rhs = float(np.max(np.abs(k1 / V)))
        if t > 20.0 and last_rhs < 1e-9:
            break

    r_s = Q[0] / V; u_s = Q[1] / (V * r_s); v_s = Q[2] / (V * r_s)
    pr_s = (GAMMA - 1.0) * (Q[4] / V - 0.5 * r_s * (u_s * u_s + v_s * v_s))
    c_s = np.sqrt(GAMMA * pr_s / r_s); M2d = np.sqrt(u_s * u_s + v_s * v_s) / c_s
    it_t = int(np.argmin(np.abs(xc - x_t)))
    div_mask = xc >= x_t
    return {
        "prim_axi": np.stack([r_s, u_s, v_s, np.zeros_like(r_s), pr_s], axis=0),
        "rhs_max": last_rhs,
        "M_throat_center": float(M2d[it_t, 0]), "M_throat_wall": float(M2d[it_t, -1]),
        "M_div_wall_min": float(np.min(M2d[div_mask, -1])),
        "M_exit_center": float(M2d[-1, 0]), "M_exit_wall": float(M2d[-1, -1]),
        "M_in_center": float(M2d[0, 0]), "M_in_wall": float(M2d[0, -1]),
    }


def simulate_3d_cylindrical(
    nx: int = 24, nr: int = 12, nth: int = 48, t_end: float = 30.0,
    M_c: float = 0.12, L_n: float = 0.60, L_d: float = 0.45, mode: str = "straight",
    init_type: str = "spin_1T", A_eq: float = 1e-4,
    n_p: float = 0.0, cav_spec: list[dict] | None = None,
    prim_axi_base: np.ndarray | None = None, cfl: float = 0.32, record_stride: int = 12,
) -> dict:
    """Full 3D cylindrical (x, r, theta) conservative FV compressible Euler solver."""
    g = build_3d_geom(nx, nr, nth, M_c=M_c, L_n=L_n, L_d=L_d, mode=mode)
    V, rb = g["V"], g["r_bar"]
    xc, etac, thc, dx, deta, dth = g["xc"], g["etac"], g["thc"], g["dx"], g["deta"], g["dth"]
    X3, E3, T3 = np.meshgrid(xc, etac, thc, indexing="ij")
    ch_mask = xc <= L_A
    ch_env = np.where(X3 <= L_A, 1.0, 0.0)

    # Exact 3D chamber volume weights normalized over ALL (n_ch, nr, nth) cells (fixes Review #11 M1!)
    V_full = np.broadcast_to(V, (nx, nr, nth))
    V_ch = V_full[ch_mask]
    w_ch = V_ch / float(np.sum(V_ch))
    X_ch, E_ch, T_ch = X3[ch_mask], E3[ch_mask], T3[ch_mask]
    phi_1T = jv(1, S11 * E_ch) / (np.sqrt(0.5 * (1.0 - 1.0 / S11**2)) * abs(jv(1, S11)))
    phi_2T = jv(2, S21 * E_ch) / (np.sqrt(0.5 * (1.0 - 4.0 / S21**2)) * abs(jv(2, S21)))
    phi_1R = jv(0, S02 * E_ch) / abs(jv(0, S02))
    phi_3T = jv(3, S31 * E_ch) / (np.sqrt(0.5 * (1.0 - 9.0 / S31**2)) * abs(jv(3, S31)))
    phi_1L = np.sqrt(2.0) * np.cos(np.pi * X_ch / L_A)
    phi_2L = np.sqrt(2.0) * np.cos(2.0 * np.pi * X_ch / L_A)

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
    elif init_type == "3T":
        j3 = jv(3, S31 * E3) / abs(jv(3, S31))
        dp = A_eq * j3 * np.cos(3.0 * T3) * ch_env
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

    def compute_rhs(Qs, fp_s, cav_states):
        r_s, u_s, v_s, w_s, p_s, _ = from_Q(Qs)
        P = np.stack([r_s, u_s, v_s, w_s, p_s], axis=0)

        u_port = None
        if cavs:
            u_port = np.zeros((nr, nth))
            for (_, qt_c), cf in zip(cav_states, cavs):
                u_port += np.tensordot(qt_c, cf["W"], axes=(0, 0))

        Px = np.empty((5, nx + 4, nr, nth)); Px[:, 2:-2] = P
        p_in_face = p_s[0]
        if mode == "periodic_bump":
            Px[:, 0:2] = P[:, -2:]; Px[:, -2:] = P[:, 0:2]
        elif mode == "closed" or init_type == "quiescent":
            for k in (0, 1):
                Px[:, k] = P[:, 0]; Px[1, k] = -P[1, 0]
                Px[:, -(k + 1)] = P[:, -1]; Px[1, -(k + 1)] = -P[1, -1]
        elif mode == "straight":
            ri, ui, vi, wi, pi = inlet_choked_3d(r_s[0], u_s[0], p_s[0], M_c, u_port=u_port)
            p_in_face = pi
            ro, uo, vo, wo, po = outlet_mc_choked_3d(r_s[-1], u_s[-1], v_s[-1], w_s[-1], p_s[-1], M_c)
            for k in (0, 1):
                Px[0, k] = ri; Px[1, k] = ui; Px[2, k] = vi; Px[3, k] = wi; Px[4, k] = pi
                Px[0, -(k + 1)] = ro; Px[1, -(k + 1)] = uo; Px[2, -(k + 1)] = vo; Px[3, -(k + 1)] = wo; Px[4, -(k + 1)] = po
        else:
            ri, ui, vi, wi, pi = inlet_choked_3d(r_s[0], u_s[0], p_s[0], M_c, u_port=u_port)
            p_in_face = pi
            for k in (0, 1):
                Px[0, k] = ri; Px[1, k] = ui; Px[2, k] = vi; Px[3, k] = wi; Px[4, k] = pi
            Px[:, -2] = P[:, -1]; Px[:, -1] = P[:, -1]

        sx = van_leer(Px[:, 1:-1] - Px[:, :-2], Px[:, 2:] - Px[:, 1:-1])
        PLx = Px[:, 1:-2] + 0.5 * sx[:, :-1]; PRx = Px[:, 2:-1] - 0.5 * sx[:, 1:]
        Fx = hllc_3d(np.maximum(PLx[0], 1e-10), PLx[1], PLx[2], PLx[3], np.maximum(PLx[4], 1e-10),
                     np.maximum(PRx[0], 1e-10), PRx[1], PRx[2], PRx[3], np.maximum(PRx[4], 1e-10))
        FxQ = np.stack([g["Sx"] * Fx[0], g["Sx"] * Fx[1], g["Sx"] * Fx[2], g["Sx"] * g["r_xf"] * Fx[3], g["Sx"] * Fx[4]], axis=0)

        Pe = np.empty((5, nx, nr + 4, nth)); Pe[:, :, 2:-2] = P
        opp = np.roll(P[:, :, 0:2, :], nth // 2, axis=3)
        Pe[:, :, 1] = opp[:, :, 0]; Pe[2:4, :, 1] *= -1.0
        Pe[:, :, 0] = opp[:, :, 1]; Pe[2:4, :, 0] *= -1.0
        Pe[:, :, nr + 2] = P[:, :, nr - 1]; Pe[:, :, nr + 3] = P[:, :, nr - 1]
        se = van_leer(Pe[:, :, 1:-1] - Pe[:, :, :-2], Pe[:, :, 2:] - Pe[:, :, 1:-1])
        PLe = Pe[:, :, 1:-2] + 0.5 * se[:, :, :-1]; PRe = Pe[:, :, 2:-1] - 0.5 * se[:, :, 1:]
        nxe, nre = g["nxe"], g["nre"]
        vnL = nxe * PLe[1] + nre * PLe[2]; vt1L = -nre * PLe[1] + nxe * PLe[2]
        vnR = nxe * PRe[1] + nre * PRe[2]; vt1R = -nre * PRe[1] + nxe * PRe[2]
        vnR[:, -1] = -vnL[:, -1]; vt1R[:, -1] = vt1L[:, -1]
        PRe[0, :, -1] = PLe[0, :, -1]; PRe[3, :, -1] = PLe[3, :, -1]; PRe[4, :, -1] = PLe[4, :, -1]
        Fe_n = hllc_3d(np.maximum(PLe[0], 1e-10), vnL, vt1L, PLe[3], np.maximum(PLe[4], 1e-10),
                       np.maximum(PRe[0], 1e-10), vnR, vt1R, PRe[3], np.maximum(PRe[4], 1e-10))
        Fe_x = nxe * Fe_n[1] - nre * Fe_n[2]; Fe_r = nre * Fe_n[1] + nxe * Fe_n[2]
        Sm = g["Sm"]
        FeQ = np.stack([Sm * Fe_n[0], Sm * Fe_x, Sm * Fe_r, Sm * g["r_ef"] * Fe_n[3], Sm * Fe_n[4]], axis=0)
        FeQ[:, :, 0] = 0.0

        Pt = np.concatenate([P[:, :, :, -2:], P, P[:, :, :, :2]], axis=3)
        st = van_leer(Pt[:, :, :, 1:-1] - Pt[:, :, :, :-2], Pt[:, :, :, 2:] - Pt[:, :, :, 1:-1])
        PLt = Pt[:, :, :, 1:-2] + 0.5 * st[:, :, :, :-1]; PRt = Pt[:, :, :, 2:-1] - 0.5 * st[:, :, :, 1:]
        Gt = hllc_3d(np.maximum(PLt[0], 1e-10), PLt[3], PLt[1], PLt[2], np.maximum(PLt[4], 1e-10),
                     np.maximum(PRt[0], 1e-10), PRt[3], PRt[1], PRt[2], np.maximum(PRt[4], 1e-10))
        GtQ = np.stack([g["S_th"] * Gt[0], g["S_th"] * Gt[2], g["S_th"] * Gt[3], g["S_th"] * g["r_thf"] * Gt[1], g["S_th"] * Gt[4]], axis=0)

        res = -(FxQ[:, 1:] - FxQ[:, :-1]) - (FeQ[:, :, 1:] - FeQ[:, :, :-1]) - (GtQ[:, :, :, 1:] - GtQ[:, :, :, :-1])
        res[2] += (p_s + r_s * w_s * w_s) * g["hoop"]

        dfp = np.zeros_like(fp_s)
        if has_flame:
            p_sense = np.sum(wf_3d * (p_s - p_b), axis=0) * dx
            dfp[0] = OMEGA_HP * (p_sense - fp_s[0])
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
    ts, e_ch, e_1t, e_2t, e_1r, e_3t = [], [], [], [], [], []
    e_1l, e_2l, e_mean = [], [], []
    c1_re, c2_re, c0_re, c3_re, p_pk = [], [], [], [], []
    w1_th = np.exp(-1j * T_ch); w2_th = np.exp(-2j * T_ch); w3_th = np.exp(-3j * T_ch)
    dr_eff = deta * g["Rm"][:, None, None]

    while t < t_end - 1e-12:
        r_c, u_c, v_c, w_c, p_c, c_c = from_Q(Q)
        lam = (np.abs(u_c) + c_c) / dx + (np.abs(v_c) + c_c) / dr_eff + (np.abs(w_c) + c_c) / (rb * dth)
        dt = min(cfl / float(np.max(lam)), t_end - t)

        if step % record_stride == 0:
            dp_c = (p_c - p_b)[ch_mask]; du_c = (u_c - ux_b)[ch_mask]; dv_c = (v_c - ur_b)[ch_mask]; dw_c = (w_c - ut_b)[ch_mask]
            ts.append(t)
            e_ch.append(0.5 * float(np.sum(w_ch * (dp_c * dp_c + du_c * du_c + dv_c * dv_c + dw_c * dw_c))))
            c1 = complex(np.sum(w_ch * dp_c * phi_1T * w1_th))
            c2 = complex(np.sum(w_ch * dp_c * phi_2T * w2_th))
            c0 = float(np.sum(w_ch * dp_c * phi_1R))
            c3 = complex(np.sum(w_ch * dp_c * phi_3T * w3_th))
            c1l = float(np.sum(w_ch * dp_c * phi_1L))
            c2l = float(np.sum(w_ch * dp_c * phi_2L))
            cmn = float(np.sum(w_ch * dp_c))
            e_1t.append(2.0 * abs(c1) ** 2); e_2t.append(2.0 * abs(c2) ** 2)
            e_1r.append(c0 * c0); e_3t.append(2.0 * abs(c3) ** 2)
            e_1l.append(c1l * c1l); e_2l.append(c2l * c2l); e_mean.append(cmn * cmn)
            c1_re.append(c1.real); c2_re.append(c2.real); c0_re.append(c0); c3_re.append(c3.real)
            p_pk.append(float(np.max(np.abs(dp_c)) / P_BAR * 100.0))
            if mode == "periodic_bump":
                tot = np.array([float(np.sum(Q[k])) for k in range(5)])
                max_cons = np.maximum(max_cons, np.abs(tot - init_tot) / np.maximum(np.abs(init_tot), 1.0))

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
        "e_1r": np.array(e_1r), "e_3t": np.array(e_3t), "e_1l": np.array(e_1l), "e_2l": np.array(e_2l),
        "e_mean": np.array(e_mean),
        "c1_re": np.array(c1_re), "c2_re": np.array(c2_re), "c0_re": np.array(c0_re), "c3_re": np.array(c3_re),
        "p_max_pct": np.array(p_pk),
        "quiescent_err": float(np.max(np.abs(Q - Q0))),
        "gcl_x": g["gcl_x"], "gcl_r": g["gcl_r"],
        "periodic_cons_err": {"mass": float(max_cons[0]), "mom_x": float(max_cons[1]),
                              "ang_mom": float(max_cons[3]), "energy": float(max_cons[4])},
    }


def simulate_2d_disk_cascade(nr: int = 24, nth: int = 96, ic_kind: str = "spin_1T", A_eq: float = 0.08, t_end: float = 16.0) -> dict:
    """High-resolution 2D (r, theta) cylindrical cross-section Euler solver isolating 1T -> 1R vs 1T -> 2T."""
    dr = 1.0 / nr; dth = 2.0 * np.pi / nth
    rf = np.linspace(0.0, 1.0, nr + 1); rc = 0.5 * (rf[:-1] + rf[1:]); thc = (np.arange(nth) + 0.5) * dth
    d_r2 = rf[1:] ** 2 - rf[:-1] ** 2
    V = 0.5 * d_r2[:, None] * dth
    rb = ((2.0 / 3.0) * (rf[1:] ** 3 - rf[:-1] ** 3) / d_r2)[:, None]
    S_r = rf[:, None] * dth; S_th = dr; hoop = S_r[1:] - S_r[:-1]; r_thf = (0.5 * d_r2 / dr)[:, None]
    R2, T2 = np.meshgrid(rc, thc, indexing="ij")
    w_area = V / np.pi
    phi_1T = jv(1, S11 * R2) / (np.sqrt(0.5 * (1.0 - 1.0 / S11**2)) * abs(jv(1, S11)))
    phi_2T = jv(2, S21 * R2) / (np.sqrt(0.5 * (1.0 - 4.0 / S21**2)) * abs(jv(2, S21)))
    phi_1R = jv(0, S02 * R2) / abs(jv(0, S02))

    u_r = np.zeros((nr, nth)); u_t = np.zeros((nr, nth))
    j1 = jv(1, S11 * R2) / abs(jv(1, S11)); j1p = jvp(1, S11 * R2) / abs(jv(1, S11))
    if ic_kind == "spin_1T":
        dp = A_eq * j1 * np.cos(T2)
        u_r = A_eq * j1p * np.sin(T2)
        u_t = A_eq * (j1 / (S11 * R2)) * np.cos(T2)
    else:
        dp = A_eq * np.sqrt(2.0) * j1 * np.cos(T2)
    r_s = 1.0 + dp; p_s = P_BAR * (r_s ** GAMMA)
    E0 = p_s / (GAMMA - 1.0) + 0.5 * r_s * (u_r * u_r + u_t * u_t)
    Q = np.stack([V * r_s, V * r_s * u_r, V * rb * r_s * u_t, V * E0], axis=0)

    def rhs(Qs):
        r = np.maximum(Qs[0] / V, 1e-10); vr = Qs[1] / (V * r); vt = Qs[2] / (V * rb * r)
        pr = np.maximum((GAMMA - 1.0) * (Qs[3] / V - 0.5 * r * (vr * vr + vt * vt)), 1e-10)
        P = np.stack([r, np.zeros_like(r), vr, vt, pr], axis=0)
        Pe = np.empty((5, nr + 4, nth)); Pe[:, 2:-2] = P
        opp = np.roll(P[:, 0:2, :], nth // 2, axis=2)
        Pe[:, 1] = opp[:, 0]; Pe[2:4, 1] *= -1.0; Pe[:, 0] = opp[:, 1]; Pe[2:4, 0] *= -1.0
        Pe[:, nr + 2] = P[:, nr - 1]; Pe[:, nr + 3] = P[:, nr - 1]
        se = van_leer(Pe[:, 1:-1] - Pe[:, :-2], Pe[:, 2:] - Pe[:, 1:-1])
        PLe = Pe[:, 1:-2] + 0.5 * se[:, :-1]; PRe = Pe[:, 2:-1] - 0.5 * se[:, 1:]
        PRe[0, -1] = PLe[0, -1]; PRe[2, -1] = -PLe[2, -1]; PRe[3, -1] = PLe[3, -1]; PRe[4, -1] = PLe[4, -1]
        Fr = hllc_3d(np.maximum(PLe[0], 1e-10), PLe[2], PLe[1], PLe[3], np.maximum(PLe[4], 1e-10),
                     np.maximum(PRe[0], 1e-10), PRe[2], PRe[1], PRe[3], np.maximum(PRe[4], 1e-10))
        FrQ = np.stack([S_r * Fr[0], S_r * Fr[1], S_r * rf[:, None] * Fr[3], S_r * Fr[4]], axis=0)
        FrQ[:, 0] = 0.0
        Pt = np.concatenate([P[:, :, -2:], P, P[:, :, :2]], axis=2)
        st = van_leer(Pt[:, :, 1:-1] - Pt[:, :, :-2], Pt[:, :, 2:] - Pt[:, :, 1:-1])
        PLt = Pt[:, :, 1:-2] + 0.5 * st[:, :, :-1]; PRt = Pt[:, :, 2:-1] - 0.5 * st[:, :, 1:]
        Gt = hllc_3d(np.maximum(PLt[0], 1e-10), PLt[3], PLt[1], PLt[2], np.maximum(PLt[4], 1e-10),
                     np.maximum(PRt[0], 1e-10), PRt[3], PRt[1], PRt[2], np.maximum(PRt[4], 1e-10))
        GtQ = np.stack([S_th * Gt[0], S_th * Gt[3], S_th * r_thf * Gt[1], S_th * Gt[4]], axis=0)
        res = -(FrQ[:, 1:] - FrQ[:, :-1]) - (GtQ[:, :, 1:] - GtQ[:, :, :-1])
        res[1] += (pr + r * vt * vt) * hoop
        return res

    t = 0.0; step = 0
    ts, e1t_h, e2t_h, c0_h = [], [], [], []
    w1_t = np.exp(-1j * T2); w2_t = np.exp(-2j * T2)
    while t < t_end - 1e-12:
        r = np.maximum(Q[0] / V, 1e-10); vr = Q[1] / (V * r); vt = Q[2] / (V * rb * r)
        pr = np.maximum((GAMMA - 1.0) * (Q[3] / V - 0.5 * r * (vr * vr + vt * vt)), 1e-10)
        c = np.sqrt(GAMMA * pr / r)
        dt = min(0.32 / float(np.max((np.abs(vr) + c) / dr + (np.abs(vt) + c) / (rb * dth))), t_end - t)
        if step % 6 == 0:
            dp = pr - P_BAR
            c1 = complex(np.sum(w_area * dp * phi_1T * w1_t))
            c2 = complex(np.sum(w_area * dp * phi_2T * w2_t))
            c0 = float(np.sum(w_area * dp * phi_1R))
            ts.append(t); e1t_h.append(2.0 * abs(c1) ** 2); e2t_h.append(2.0 * abs(c2) ** 2); c0_h.append(c0)
        k1 = rhs(Q); Q1 = Q + dt * k1; k2 = rhs(Q1); Q = 0.5 * (Q + Q1 + dt * k2)
        t += dt; step += 1

    ts_a = np.array(ts); e1_a = np.array(e1t_h); e2_a = np.array(e2t_h); c0_a = np.array(c0_h)
    m_late = ts_a >= 0.40 * t_end
    m_vlate = ts_a >= 10.0
    c0_late = c0_a[m_late]
    e1r_ac = float(np.var(c0_late))
    e1r_ac_vlate = float(np.var(c0_a[m_vlate]))
    e1r_dc = float(np.mean(c0_late) ** 2)
    return {
        "E_1T_mean": float(np.mean(e1_a[m_late])),
        "E_2T_mean": float(np.mean(e2_a[m_late])),
        "E_1R_ac": e1r_ac,
        "E_1R_ac_10_16": e1r_ac_vlate,
        "E_1R_dc": e1r_dc,
        "E_2T_over_E1T_pct": float(np.mean(e2_a[m_late]) / max(np.mean(e1_a[m_late]), 1e-30) * 100.0),
        "E_1R_ac_over_E1T_pct": float(e1r_ac / max(np.mean(e1_a[m_late]), 1e-30) * 100.0),
    }


def _sim3d_worker(args):
    lbl, kw = args
    return lbl, simulate_3d_cylindrical(**kw)


def _disk_worker(args):
    lbl, kw = args
    return lbl, simulate_2d_disk_cascade(**kw)


def run_batch(worker_fn, tasks: list[tuple[str, dict]], max_workers: int | None = None) -> dict:
    nw = max_workers or min(6, max(1, (os.cpu_count() or 2) - 2), len(tasks))
    if nw <= 1 or len(tasks) <= 1:
        return dict(worker_fn(t) for t in tasks)
    with ProcessPoolExecutor(max_workers=nw) as ex:
        return dict(ex.map(worker_fn, tasks))


def measure_fft_omega(t: np.ndarray, sig: np.ndarray, t_lo: float = 2.0, f_lo: float = 0.5, f_hi: float = 6.0) -> float:
    m = t >= t_lo
    tu = np.linspace(t[m][0], t[m][-1], 1024)
    su = np.interp(tu, t[m], sig[m])
    su = su - np.mean(su)
    env = np.maximum(np.sqrt(np.convolve(su * su, np.ones(41) / 41.0, mode="same")), 1e-30)
    su_n = (su / env) * np.hanning(len(su))
    nfft = 32768
    freqs = np.fft.rfftfreq(nfft, d=(tu[1] - tu[0])) * 2.0 * np.pi
    spec = np.abs(np.fft.rfft(su_n, n=nfft))
    mf = (freqs >= f_lo) & (freqs <= f_hi)
    return float(freqs[mf][np.argmax(spec[mf])])


def fit_alpha(res: dict, t_lo: float, t_hi: float, key: str = "e_1t") -> float:
    t = res["t"]; m = (t >= t_lo) & (t <= t_hi)
    return float(np.polyfit(t[m], 0.5 * np.log(np.maximum(res[key][m], 1e-30)), 1)[0])


def plot_all_figures(metrics: dict, nl_res_20: dict, nl_res_120: dict) -> None:
    exp2_Ln_rows = metrics["exp2_3d_bell_cutoff"]["Ln_sweep_Mc012_N160"]
    exp2_hopf_3d = metrics["exp2_3d_bell_cutoff"]["Mc_hopf_nv0_Ln050_tau016"]
    A_grid = metrics["exp3_chiral_1T_to_1R_cascade"]["A_grid"]
    exp3_disk_res = metrics["exp3_chiral_1T_to_1R_cascade"]["disk_runs"]
    slope_stand_small = metrics["exp3_chiral_1T_to_1R_cascade"]["stand_1R_ac_slope_smallA"]
    slope_spin_all = metrics["exp3_chiral_1T_to_1R_cascade"]["spin_1R_ac_slope_all"]
    e4 = metrics["exp4_shroud_vs_hub_cavities"]
    analytical_weights = e4["analytical_weights"]
    lin_12 = e4["linear_placement_12x48_matched"]
    lin_20 = e4["linear_placement_20x64_matched"]
    nl_out_120 = e4["reacting_t120_shootout_12x48"]
    nl_out_20 = e4["reacting_t48_1R_shootout_20x64"]

    # Fig 1: Converged N=160 Bessel Bell Cutoff Collapse & Fixed-tau=0.16 Hopf Envelope
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.5), dpi=150)
    Lns = [r["L_n"] for r in exp2_Ln_rows]
    ax1.plot(Lns, [-r["thin2d_1T"] for r in exp2_Ln_rows], "o--", color="#777777", lw=2.2, label="2D Thin-Shell $1T$ ($m=1$, Entry 012)")
    ax1.plot(Lns, [-r["cyl3d_1T"] for r in exp2_Ln_rows], "s-", color="#d62728", lw=2.5, label="3D Cylindrical $1T$ ($s_{11}=1.841$)")
    ax1.plot(Lns, [-r["cyl3d_2T"] for r in exp2_Ln_rows], "^-.", color="#1f77b4", lw=2.1, label="3D Cylindrical $2T$ ($s_{21}=3.054$)")
    ax1.plot(Lns, [-r["cyl3d_1R"] for r in exp2_Ln_rows], "d:", color="#2ca02c", lw=2.1, label="3D Fundamental $1R$ ($s_{02}=3.832$)")
    ax1.set_xlabel("Bell Nozzle Contraction Length $L_n$", fontsize=11)
    ax1.set_ylabel("Inviscid Quasi-3D Cold Bell Damping $-\\alpha$", fontsize=11)
    ax1.set_title("3D Bessel Cutoff Collapse vs. 2D Thin-Shell ($\\bar{M}_c=0.12, N=160$)", fontsize=11.3, fontweight="bold")
    ax1.grid(True, alpha=0.28)
    ax1.legend(fontsize=8.6, loc="upper right", framealpha=0.94)

    Ma = [r["M_c"] for r in exp2_hopf_3d]
    ax2.plot(Ma, [r["n_pH_2D_min12"] for r in exp2_hopf_3d], "o--", color="#777777", lw=2.2, label="2D Thin-Shell min($1T, 2T$) ($\\bar{\\tau}=0.16$)")
    ax2.plot(Ma, [r["n_pH_3D_global"] for r in exp2_hopf_3d], "s-", color="#d62728", lw=2.5, label="3D Cylindrical Global ($1L, 1T, 2T, 1R, 3T$)")
    ax2.plot(Ma, [r["n_pH_1T"] for r in exp2_hopf_3d], "^-.", color="#1f77b4", lw=1.9, label="3D Cylindrical $1T$ ($s_{11}$)")
    ma_1r = [r["M_c"] for r in exp2_hopf_3d if r["n_pH_1R"] is not None]
    y_1r = [r["n_pH_1R"] for r in exp2_hopf_3d if r["n_pH_1R"] is not None]
    ax2.plot(ma_1r, y_1r, "d:", color="#2ca02c", lw=1.9, label="3D Cylindrical $1R$ ($s_{02}$)")
    ax2.set_xlabel("Chamber Mean Mach Number $\\bar{M}_c$", fontsize=11)
    ax2.set_ylabel("Pressure-Coupled Hopf Gain $n_{pH}$ ($n_v=0, \\bar{\\tau}=0.16$)", fontsize=11)
    ax2.set_title("3D vs. 2D Thin-Shell Bare Bell Hopf Envelope ($L_n=0.50$)", fontsize=11.3, fontweight="bold")
    ax2.set_ylim(-0.01, 0.42)
    ax2.grid(True, alpha=0.28)
    ax2.legend(fontsize=8.2, loc="upper left", framealpha=0.94)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp014_3d_bessel_bell_cutoff_collapse.png", dpi=150)
    plt.close(fig)

    # Fig 2: Chiral 1T -> 1R vs 2T Selection Rule (A0 = 0.005..0.10) + Analytical Shroud vs Hub Weights
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.5), dpi=150)
    Aa = np.array(A_grid)
    ax1.loglog(Aa, [exp3_disk_res[f"disk24_stand_A{a:g}"]["E_1R_ac"] for a in A_grid], "s-", color="#d62728", lw=2.5,
               label=f"Standing $1T \\to$ AC $1R$ (small-$A$: $A_0^{{{slope_stand_small:.2f}}}$)")
    ax1.loglog(Aa, [exp3_disk_res[f"disk24_stand_A{a:g}"]["E_2T_mean"] for a in A_grid], "^--", color="#ff7f0e", lw=2.0,
               label="Standing $1T \\to 2T$ (detuned $1.659\\times$)")
    ax1.loglog(Aa, [exp3_disk_res[f"disk24_spin_A{a:g}"]["E_2T_mean"] for a in A_grid], "o-.", color="#1f77b4", lw=2.0,
               label="Spinning $1T \\to 2T$ (detuned $1.659\\times$)")
    ax1.loglog(Aa, [exp3_disk_res[f"disk24_spin_A{a:g}"]["E_1R_ac"] for a in A_grid], "d:", color="#2ca02c", lw=2.2,
               label=f"Spinning $1T \\to$ AC $1R$ ($\\sim 600\\times$ lower, $A_0^{{{slope_spin_all:.2f}}}$)")
    ax1.set_xlabel("Initial $1T$ Amplitude $A_0$", fontsize=11)
    ax1.set_ylabel("Daughter Modal Energy ($24\\times 96$ Disk Euler)", fontsize=11)
    ax1.set_title("3D Chiral $1T \\to 1R$ Selection Rule ($s_{02}/s_{11}=2.081$)", fontsize=11.3, fontweight="bold")
    ax1.grid(True, which="both", alpha=0.26)
    ax1.legend(fontsize=8.2, loc="upper left", framealpha=0.94)

    rr_g = np.linspace(0.0, 1.0, 300)
    ax2.plot(rr_g, bessel_modal_weight(1, S11, rr_g), "-", color="#1f77b4", lw=2.4, label=f"$1T$ ($s_{{11}}$): Shroud $W(1)={analytical_weights['shroud_r1']['1T']:.2f}$, Hub $0$")
    ax2.plot(rr_g, bessel_modal_weight(2, S21, rr_g), "--", color="#ff7f0e", lw=2.2, label=f"$2T$ ($s_{{21}}$): Shroud $W(1)={analytical_weights['shroud_r1']['2T']:.2f}$, Hub $0$")
    ax2.plot(rr_g, bessel_modal_weight(0, S02, rr_g), "-", color="#d62728", lw=2.6, label=f"$1R$ ($s_{{02}}$): Hub $W(0)={analytical_weights['hub_r0']['1R']:.2f}$ ($6.16\\times$ Shroud!)")
    ax2.axvspan(0.0, 0.26, color="#d62728", alpha=0.10, label="Micro-Hub Zone ($r \\leq 0.26$, $0.35\\%$ vol)")
    ax2.axvspan(0.78, 1.00, color="#1f77b4", alpha=0.10, label="Outer Shroud Zone ($r \\geq 0.78$, $2.65\\%$ vol)")
    ax2.set_xlabel("Normalized Faceplate Radius $r/R$", fontsize=11)
    ax2.set_ylabel("Normalized Modal Energy Density $W_{mn}(r)$", fontsize=11)
    ax2.set_title("3D Faceplate Cavity Placement Law: Shroud ($1T/2T$) vs. Hub ($1R$)", fontsize=11.3, fontweight="bold")
    ax2.set_ylim(-0.1, 6.6)
    ax2.grid(True, alpha=0.28)
    ax2.legend(fontsize=8.2, loc="upper right", framealpha=0.94)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp014_3d_chiral_1R_cascade_and_shroud_hub.png", dpi=150)
    plt.close(fig)

    # Fig 3: Left = Matched-Window Cavity Gain Delta-alpha (incl. 24x20x64 1R);
    # Right = Refined 20x20x64 Reacting 1R Shootout (Shroud Fails vs Micro-Hub Quenches) + t=120 1T/2T
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.5), dpi=150)
    labels = ["$1T$ ($0.5\\%$)\n20x64", "$1T$ ($2.0\\%$)\n12x48", "$2T$ ($2.0\\%$)\n12x48", "$1R$ ($2.0\\%$/$0.35\\%$)\n20x64"]
    x_p = np.arange(len(labels)); bw = 0.24
    unif_d = [
        -(lin_20["f20_w05_1T_unif"]["alpha_7_22"] - lin_20["bare20_1T"]["alpha_7_22"]),
        -(lin_12["ep24_1T_uniform"] - lin_12["bare12_1T"]),
        -(lin_12["ep26_2T_uniform"] - lin_12["bare12_2T"]),
        -(lin_20["f20_1R_uniform"]["alpha_7_22"] - lin_20["bare20_1R"]["alpha_7_22"]),
    ]
    shrd_d = [
        -(lin_20["f20_w05_1T_shrd"]["alpha_7_22"] - lin_20["bare20_1T"]["alpha_7_22"]),
        -(lin_12["ep24_1T_shroud"] - lin_12["bare12_1T"]),
        -(lin_12["ep26_2T_shroud"] - lin_12["bare12_2T"]),
        -(lin_20["f20_1R_shroud"]["alpha_7_22"] - lin_20["bare20_1R"]["alpha_7_22"]),
    ]
    hub_d = [
        0.0,
        -(lin_12["ep24_1T_hub"] - lin_12["bare12_1T"]),
        0.0,
        -(lin_20["f20_1R_hub035"]["alpha_7_22"] - lin_20["bare20_1R"]["alpha_7_22"]),
    ]
    ax1.bar(x_p - bw, unif_d, bw, color="#ff7f0e", label="Uniform Faceplate ($\\Delta\\alpha_{1R}\\equiv 0$ exact!)")
    ax1.bar(x_p,      shrd_d, bw, color="#1f77b4", label="Outer Shroud ($r\\in[0.78,1]$)")
    ax1.bar(x_p + bw, hub_d,  bw, color="#d62728", label="Hub ($0.35\\%$ micro-hub on $1R$, $2.27\\times$ shroud!)")
    ax1.set_xticks(x_p); ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.set_ylabel("Matched-Window Cavity Damping Gain $-\\Delta\\alpha$ ($[7,22]$)", fontsize=10.5)
    ax1.set_ylim(0.0, 0.075)
    ax1.set_title("Matched-Window 3D Placement Gain (incl. Refined $24\\times 20\\times 64$ $1R$)", fontsize=10.5, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.28)
    ax1.legend(fontsize=8.0, loc="upper right", framealpha=0.94)

    for cid, rr, st, lbl, col, ls in [
        ("nl20_bare_1R",   nl_res_20["nl20_bare_1R"],   nl_out_20["nl20_bare_1R"],   "Bare 3D 1R (20x20x64)", "#d62728", "-"),
        ("nl20_shroud_1R", nl_res_20["nl20_shroud_1R"], nl_out_20["nl20_shroud_1R"], "Shroud-Only 3.0% 1R (20x20x64, FAILS!)", "#ff7f0e", "-."),
        ("nl20_dualz_1R",  nl_res_20["nl20_dualz_1R"],  nl_out_20["nl20_dualz_1R"],  "Dual-Zone 0.35% Hub 1R (20x20x64, QUENCH!)", "#2ca02c", "-"),
        ("nl_bare_1T",     nl_res_120["nl_bare_1T"],    nl_out_120["nl_bare_1T"],    "Bare 3D spin 1T (t=120)", "#8c564b", "--"),
        ("nl_dualz_1T",    nl_res_120["nl_dualz_1T"],   nl_out_120["nl_dualz_1T"],   "Dual-Zone spin 1T (t=120, k=0 bulk drift)", "#1f77b4", "-"),
        ("nl_dualz_st1T",  nl_res_120["nl_dualz_st1T"], nl_out_120["nl_dualz_st1T"], "Dual-Zone stand 1T (t=120, k=0 bulk drift)", "#9467bd", ":"),
        ("nl_dualz_2T",    nl_res_120["nl_dualz_2T"],   nl_out_120["nl_dualz_2T"],   "Dual-Zone spin 2T (t=120)", "#17becf", "-."),
    ]:
        p_end = st["p_peak_48"] if "20_" in cid else st["p_peak_120"]
        ax2.semilogy(rr["t"], np.maximum(rr["e_ch"], 1e-11), color=col, ls=ls, lw=2.0,
                     label=f"{lbl} (P={p_end:.2f}%)")
    ax2.set_xlabel("Time $t$", fontsize=11)
    ax2.set_ylabel("True-Norm. 3D Wave Energy $E_{\\mathrm{wave}}(t)$", fontsize=11)
    ax2.set_title("Reacting 3D ($n_p=0.30$): Refined 20x20x64 1R Quench & $t=120$ Audit", fontsize=10.4, fontweight="bold")
    ax2.set_ylim(1e-8, 5e-3)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=7.2, loc="lower right", framealpha=0.94)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp014_3d_shroud_hub_pde_shootout.png", dpi=150)
    plt.close(fig)
    print("Saved all 3 hardened figures.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot-only", action="store_true", help="Regenerate all 3 figures from existing exp014_metrics.json (rerunning only the 7 Fig 3 right-panel curves).")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp014_metrics.json"
    if args.replot_only:
        metrics = json.loads(out_path.read_text(encoding="utf-8"))
        cav_shroud_only = [
            {"beta": 0.0210, "wH": S11, "xiH": 0.22, "sigma_o": 0.10, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": True},
            {"beta": 0.0090, "wH": S21, "xiH": 0.24, "sigma_o": 0.08, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": True},
        ]
        cav_dualzone = [
            {"beta": 0.0185, "wH": S11, "xiH": 0.22, "sigma_o": 0.10, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": True},
            {"beta": 0.0080, "wH": S21, "xiH": 0.24, "sigma_o": 0.08, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": True},
            {"beta": 0.0035, "wH": S02, "xiH": 0.14, "sigma_o": 0.04, "n_cav": 1, "r_lo": 0.00, "r_hi": 0.26, "nonlinear": True},
        ]
        t_plot = [
            ("nl20_bare_1R",   dict(nx=20, nr=20, nth=64, t_end=48.0,  M_c=0.12, mode="straight", init_type="1R",       A_eq=0.04, n_p=0.30, record_stride=24)),
            ("nl20_shroud_1R", dict(nx=20, nr=20, nth=64, t_end=48.0,  M_c=0.12, mode="straight", init_type="1R",       A_eq=0.04, n_p=0.30, cav_spec=cav_shroud_only, record_stride=24)),
            ("nl20_dualz_1R",  dict(nx=20, nr=20, nth=64, t_end=48.0,  M_c=0.12, mode="straight", init_type="1R",       A_eq=0.04, n_p=0.30, cav_spec=cav_dualzone, record_stride=24)),
            ("nl_bare_1T",     dict(nx=20, nr=12, nth=48, t_end=120.0, M_c=0.12, mode="straight", init_type="spin_1T",  A_eq=0.04, n_p=0.30, record_stride=24)),
            ("nl_dualz_1T",    dict(nx=20, nr=12, nth=48, t_end=120.0, M_c=0.12, mode="straight", init_type="spin_1T",  A_eq=0.04, n_p=0.30, cav_spec=cav_dualzone, record_stride=24)),
            ("nl_dualz_st1T",  dict(nx=20, nr=12, nth=48, t_end=120.0, M_c=0.12, mode="straight", init_type="stand_1T", A_eq=0.04, n_p=0.30, cav_spec=cav_dualzone, record_stride=24)),
            ("nl_dualz_2T",    dict(nx=20, nr=12, nth=48, t_end=120.0, M_c=0.12, mode="straight", init_type="spin_2T",  A_eq=0.04, n_p=0.30, cav_spec=cav_dualzone, record_stride=24)),
        ]
        res_p = run_batch(_sim3d_worker, t_plot)
        plot_all_figures(metrics, res_p, res_p)
        return

    print("=" * 96)
    print("Entry 014 (Hardened per Peer Review #11): Full 3D Cylindrical Chamber & Transonic Bell")
    print("=" * 96)

    # ------------------------------------------------------------------
    # Experiment 1: 3D GCL, Conservation, Bessel FFTs, Num Damping & Axial Refinement
    # ------------------------------------------------------------------
    print("\n--- Experiment 1: 3D Benchmarks, Per-Mode Numerical Damping & Axial Refinement ---")
    exp1_tasks = [
        ("quiescent_3d_bell",  dict(nx=24, nr=12, nth=32, t_end=4.0, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", init_type="quiescent", A_eq=0.0, record_stride=20)),
        ("cons_3d_curved_bell",dict(nx=20, nr=10, nth=32, t_end=2.5, M_c=0.12, mode="periodic_bump", init_type="periodic_3d_wave", A_eq=0.06, record_stride=15)),
        ("fft_1T_c", dict(nx=6, nr=12, nth=36, t_end=14.0, M_c=0.0, mode="closed", init_type="stand_1T", A_eq=1e-4, record_stride=4)),
        ("fft_1T_f", dict(nx=6, nr=16, nth=48, t_end=14.0, M_c=0.0, mode="closed", init_type="stand_1T", A_eq=1e-4, record_stride=4)),
        ("fft_2T_c", dict(nx=6, nr=12, nth=36, t_end=14.0, M_c=0.0, mode="closed", init_type="stand_2T", A_eq=1e-4, record_stride=4)),
        ("fft_2T_f", dict(nx=6, nr=16, nth=48, t_end=14.0, M_c=0.0, mode="closed", init_type="stand_2T", A_eq=1e-4, record_stride=4)),
        ("fft_1R_c", dict(nx=6, nr=12, nth=36, t_end=14.0, M_c=0.0, mode="closed", init_type="1R", A_eq=1e-4, record_stride=4)),
        ("fft_1R_f", dict(nx=6, nr=16, nth=48, t_end=14.0, M_c=0.0, mode="closed", init_type="1R", A_eq=1e-4, record_stride=4)),
        ("fft_3T_f", dict(nx=6, nr=16, nth=48, t_end=14.0, M_c=0.0, mode="closed", init_type="3T", A_eq=1e-4, record_stride=4)),
        # Per-mode closed-cylinder numerical damping (including matched 14x48 for M_c=0.28 per m3!)
        ("num_12x48_1T", dict(nx=6, nr=12, nth=48, t_end=15.0, M_c=0.0, mode="closed", init_type="spin_1T", A_eq=1e-4, record_stride=6)),
        ("num_12x48_2T", dict(nx=6, nr=12, nth=48, t_end=15.0, M_c=0.0, mode="closed", init_type="spin_2T", A_eq=1e-4, record_stride=6)),
        ("num_12x48_1R", dict(nx=6, nr=12, nth=48, t_end=15.0, M_c=0.0, mode="closed", init_type="1R",      A_eq=1e-4, record_stride=6)),
        ("num_14x48_1T", dict(nx=6, nr=14, nth=48, t_end=15.0, M_c=0.0, mode="closed", init_type="spin_1T", A_eq=1e-4, record_stride=6)),
        ("num_20x64_1T", dict(nx=6, nr=20, nth=64, t_end=15.0, M_c=0.0, mode="closed", init_type="spin_1T", A_eq=1e-4, record_stride=6)),
        ("num_20x64_2T", dict(nx=6, nr=20, nth=64, t_end=15.0, M_c=0.0, mode="closed", init_type="spin_2T", A_eq=1e-4, record_stride=6)),
        ("num_20x64_1R", dict(nx=6, nr=20, nth=64, t_end=15.0, M_c=0.0, mode="closed", init_type="1R",      A_eq=1e-4, record_stride=6)),
        # Straight-cylinder axial refinement (Review #11 C1: nx=24 -> 48 -> 96 moves toward inviscid -0.0424!)
        ("mc_1T_nx24", dict(nx=24, nr=12, nth=48, t_end=32.0, M_c=0.12, mode="straight", init_type="spin_1T", A_eq=1e-4, record_stride=12)),
        ("mc_1T_nx48", dict(nx=48, nr=12, nth=48, t_end=32.0, M_c=0.12, mode="straight", init_type="spin_1T", A_eq=1e-4, record_stride=16)),
        ("mc_1T_nx96", dict(nx=96, nr=12, nth=48, t_end=32.0, M_c=0.12, mode="straight", init_type="spin_1T", A_eq=1e-4, record_stride=24)),
        ("mc_2T_nx24", dict(nx=24, nr=12, nth=48, t_end=32.0, M_c=0.12, mode="straight", init_type="spin_2T", A_eq=1e-4, record_stride=12)),
        ("mc_2T_nx96", dict(nx=96, nr=12, nth=48, t_end=32.0, M_c=0.12, mode="straight", init_type="spin_2T", A_eq=1e-4, record_stride=24)),
        ("mc_1R_nx24", dict(nx=24, nr=12, nth=48, t_end=32.0, M_c=0.12, mode="straight", init_type="1R",      A_eq=1e-4, record_stride=12)),
        ("mc_3d_M28_1T", dict(nx=40, nr=14, nth=48, t_end=22.0, M_c=0.28, mode="straight", init_type="spin_1T", A_eq=1e-4, record_stride=10)),
    ]
    exp1_res = run_batch(_sim3d_worker, exp1_tasks)
    rq = exp1_res["quiescent_3d_bell"]; rc = exp1_res["cons_3d_curved_bell"]
    print(f"  1A Quiescent Bell: gcl_x={rq['gcl_x']:.2e}, gcl_r={rq['gcl_r']:.2e}, drift={rq['quiescent_err']:.2e}")
    print(f"  1B Curved Bell Cons: mass={rc['periodic_cons_err']['mass']:.2e}, ang_mom={rc['periodic_cons_err']['ang_mom']:.2e}, energy={rc['periodic_cons_err']['energy']:.2e}")

    bessel_fft = {}
    for lbl, sig_k, exact_s in [
        ("fft_1T_c", "c1_re", S11), ("fft_1T_f", "c1_re", S11),
        ("fft_2T_c", "c2_re", S21), ("fft_2T_f", "c2_re", S21),
        ("fft_1R_c", "c0_re", S02), ("fft_1R_f", "c0_re", S02),
        ("fft_3T_f", "c3_re", S31),
    ]:
        wm = measure_fft_omega(exp1_res[lbl]["t"], exp1_res[lbl][sig_k], t_lo=1.5)
        err = abs(wm - exact_s) / exact_s * 100.0
        bessel_fft[lbl] = {"measured_omega": wm, "exact_s_mn": exact_s, "rel_err_pct": err}
        print(f"  1C [{lbl:9s}] meas={wm:.4f} vs exact={exact_s:.4f} (err={err:.2f}%)")

    num_audit = {}
    for cid, ek in [("num_12x48_1T", "e_1t"), ("num_12x48_2T", "e_2t"), ("num_12x48_1R", "e_1r"),
                    ("num_14x48_1T", "e_1t"),
                    ("num_20x64_1T", "e_1t"), ("num_20x64_2T", "e_2t"), ("num_20x64_1R", "e_1r")]:
        num_audit[cid] = fit_alpha(exp1_res[cid], 4.0, 14.0, key=ek)
        print(f"  1E [{cid:13s}] alpha_num={num_audit[cid]:+.5f}")

    ex_1T = exact_straight_bessel_mc_eig(0.12, S11, N=96)
    ex_2T = exact_straight_bessel_mc_eig(0.12, S21, N=96)
    ex_1R = exact_straight_bessel_mc_eig(0.12, S02, N=96)
    v28_ex = exact_straight_bessel_mc_eig(0.28, S11, N=96)
    a28 = fit_alpha(exp1_res["mc_3d_M28_1T"], 8.0, 20.0, key="e_1t")
    a28_net = a28 - num_audit["num_14x48_1T"]
    axial_refinement = {
        "exact_inviscid_Mc012": {
            "1T_alpha": ex_1T.real, "1T_omega": ex_1T.imag,
            "2T_alpha": ex_2T.real, "2T_omega": ex_2T.imag,
            "1R_alpha": ex_1R.real, "1R_omega": ex_1R.imag,
        },
        "1T_nx24": {"fit_12_30": fit_alpha(exp1_res["mc_1T_nx24"], 12.0, 30.0, "e_1t"), "fit_16_30": fit_alpha(exp1_res["mc_1T_nx24"], 16.0, 30.0, "e_1t")},
        "1T_nx48": {"fit_12_30": fit_alpha(exp1_res["mc_1T_nx48"], 12.0, 30.0, "e_1t"), "fit_16_30": fit_alpha(exp1_res["mc_1T_nx48"], 16.0, 30.0, "e_1t")},
        "1T_nx96": {"fit_12_30": fit_alpha(exp1_res["mc_1T_nx96"], 12.0, 30.0, "e_1t"), "fit_16_30": fit_alpha(exp1_res["mc_1T_nx96"], 16.0, 30.0, "e_1t")},
        "2T_nx24": {"fit_12_30": fit_alpha(exp1_res["mc_2T_nx24"], 12.0, 30.0, "e_2t"), "fit_16_30": fit_alpha(exp1_res["mc_2T_nx24"], 16.0, 30.0, "e_2t")},
        "2T_nx96": {"fit_12_30": fit_alpha(exp1_res["mc_2T_nx96"], 12.0, 30.0, "e_2t"), "fit_16_30": fit_alpha(exp1_res["mc_2T_nx96"], 16.0, 30.0, "e_2t")},
        "1R_nx24": {"fit_12_30": fit_alpha(exp1_res["mc_1R_nx24"], 12.0, 30.0, "e_1r"), "fit_16_30": fit_alpha(exp1_res["mc_1R_nx24"], 16.0, 30.0, "e_1r")},
        "resolved_M28_1T": {
            "exact_cheb_N96": v28_ex.real, "pde_3d_40x14x48": a28,
            "matched_num_14x48": num_audit["num_14x48_1T"], "pde_net_of_num": a28_net,
            "rel_err_raw_pct": abs(a28 - v28_ex.real) / abs(v28_ex.real) * 100.0,
            "rel_err_net_pct": abs(a28_net - v28_ex.real) / abs(v28_ex.real) * 100.0,
        },
    }
    print(f"  1D Straight 1T axial refinement (exact={ex_1T.real:+.5f}): nx24={axial_refinement['1T_nx24']['fit_12_30']:+.5f}, nx48={axial_refinement['1T_nx48']['fit_12_30']:+.5f}, nx96={axial_refinement['1T_nx96']['fit_12_30']:+.5f} (late [16,30]={axial_refinement['1T_nx96']['fit_16_30']:+.5f})")
    print(f"  1E Resolved M_c=0.28 1T: exact={v28_ex.real:+.5f}, PDE={a28:+.5f}, net(14x48)={a28_net:+.5f} ({axial_refinement['resolved_M28_1T']['rel_err_net_pct']:.2f}%)")

    # ------------------------------------------------------------------
    # Experiment 2: 3D Bessel Bell Cutoff Collapse (N=160, single-branch 1R) & True-3D Bell Refinement
    # ------------------------------------------------------------------
    print("\n--- Experiment 2: 3D Bessel Bell Cutoff (N=160) & True-3D Meridional Bell Refinement ---")
    Ln_grid = [0.05, 0.15, 0.30, 0.40, 0.50, 0.60, 0.70]
    exp2_Ln_rows = []
    for Ln in Ln_grid:
        v2d = quasi3d_bessel_bell_eig(0.12, 1.0, L_n=Ln, contour="entry012", omega_window=(0.65, 1.50), N=160)
        v1T = quasi3d_bessel_bell_eig(0.12, S11, L_n=Ln, contour="entry012", omega_window=(1.40, 2.35), N=160)
        v2T = quasi3d_bessel_bell_eig(0.12, S21, L_n=Ln, contour="entry012", omega_window=(2.40, 3.65), N=160)
        # Track fundamental 1R branch cleanly around s_02 = 3.83 (Review #11 M7: prevents overtone branch-hopping!)
        v1R = quasi3d_bessel_bell_eig(0.12, S02, L_n=Ln, contour="entry012", omega_window=(3.35, 4.12), N=160)
        row = {
            "L_n": Ln,
            "thin2d_1T": v2d.real, "thin2d_1T_w": v2d.imag,
            "cyl3d_1T": v1T.real, "cyl3d_1T_w": v1T.imag,
            "cyl3d_2T": v2T.real, "cyl3d_2T_w": v2T.imag,
            "cyl3d_1R": v1R.real, "cyl3d_1R_w": v1R.imag,
        }
        exp2_Ln_rows.append(row)
        print(f"  2A (N=160) L_n={Ln:.2f}: 2D-1T={v2d.real:+.5f} | 3D-1T={v1T.real:+.5f}({v1T.imag:.2f}) | 3D-2T={v2T.real:+.5f}({v2T.imag:.2f}) | 3D-1R={v1R.real:+.5f}({v1R.imag:.2f})")

    def find_first_hopf(fn) -> float | None:
        pts = [0.0, 0.05, 0.15, 0.30, 0.50, 0.80, 1.20, 2.00]
        v_prev = fn(0.0)
        if v_prev >= 0.0:
            return 0.0
        for p_hi in pts[1:]:
            v_hi = fn(p_hi)
            if v_hi > 0.0:
                p_lo = pts[pts.index(p_hi) - 1]
                return float(brentq(fn, p_lo, p_hi))
            v_prev = v_hi
        return None

    M_grid = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22]
    exp2_hopf_3d = []
    modes_3d = [("1L", 0.0, (0.40, 2.70)), ("1T", S11, (1.40, 2.45)), ("2T", S21, (2.40, 3.70)),
                ("1R", S02, (3.10, 4.50)), ("3T", S31, (3.40, 4.95))]
    for Mb in M_grid:
        npH = {}
        for mname, smn, ww in modes_3d:
            f = lambda nv, s=smn, w=ww: quasi3d_bessel_bell_eig(Mb, s, L_n=0.50, contour="entry012", n_p=nv, omega_window=w, N=96).real
            npH[mname] = find_first_hopf(f)
        finite_modes = {k: v for k, v in npH.items() if v is not None}
        lim = min(finite_modes, key=finite_modes.get)
        npH_2d = {}
        for mn, mm, ww in [("1T", 1.0, (0.60, 1.50)), ("2T", 2.0, (1.40, 2.80))]:
            f = lambda nv, s=mm, w=ww: quasi3d_bessel_bell_eig(Mb, s, L_n=0.50, contour="entry012", n_p=nv, omega_window=w, N=96).real
            npH_2d[mn] = find_first_hopf(f)
        fin_2d = [v for v in npH_2d.values() if v is not None]
        row = {"M_c": Mb, **{f"n_pH_{k}": v for k, v in npH.items()}, "n_pH_3D_global": finite_modes[lim], "limiting_3D": lim,
               "n_pH_2D_1T": npH_2d["1T"], "n_pH_2D_2T": npH_2d["2T"], "n_pH_2D_min12": min(fin_2d) if fin_2d else None}
        exp2_hopf_3d.append(row)
        r1_str = "no_cross<=2" if npH["1R"] is None else f"{npH['1R']:.3f}"
        print(f"  2A Hopf M={Mb:.2f} (tau=0.16, n_v=0): 3D={finite_modes[lim]:.3f}({lim}) [1T:{npH['1T']:.3f}, 2T:{npH['2T']:.3f}, 1R:{r1_str}] | 2D_min={row['n_pH_2D_min12']:.3f}")

    # M4 cross-check: TAU_BAR = 0.40 vs 0.16 at M_c = 0.12, 0.14
    tau040_check = []
    for Mb in (0.12, 0.14):
        f3 = lambda nv: quasi3d_bessel_bell_eig(Mb, S11, L_n=0.50, contour="entry012", n_p=nv, tau_bar=0.40, omega_window=(1.40, 2.45), N=96).real
        f2_1 = lambda nv: quasi3d_bessel_bell_eig(Mb, 1.0, L_n=0.50, contour="entry012", n_p=nv, tau_bar=0.40, omega_window=(0.60, 1.50), N=96).real
        f2_2 = lambda nv: quasi3d_bessel_bell_eig(Mb, 2.0, L_n=0.50, contour="entry012", n_p=nv, tau_bar=0.40, omega_window=(1.40, 2.80), N=96).real
        n3 = find_first_hopf(f3); n21 = find_first_hopf(f2_1); n22 = find_first_hopf(f2_2)
        tau040_check.append({"M_c": Mb, "tau_bar": 0.40, "n_pH_3D_1T": n3, "n_pH_2D_1T": n21, "n_pH_2D_2T": n22,
                             "ratio_2Dmin_over_3D1T": min(n21, n22) / n3})
        print(f"  2A Tau=0.40 check M={Mb:.2f}: 3D-1T={n3:.4f}, 2D-1T={n21:.4f}, 2D-2T={n22:.4f}, 2Dmin/3D1T={min(n21,n22)/n3:.2f}x")

    # 2B: True 3D axisymmetric bell bases (including steep bell_c2 L_n=0.50 per M6 & nr=20 smooth bell per C2!)
    axi_c2_50 = relax_axi_3d_bell_base(40, 12, M_c=0.12, L_n=0.50, L_d=0.25, mode="bell_c2", t_relax=120.0)
    axi_60_40x12 = relax_axi_3d_bell_base(40, 12, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", t_relax=160.0)
    axi_60_56x16 = relax_axi_3d_bell_base(56, 16, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", t_relax=160.0)
    axi_60_80x12 = relax_axi_3d_bell_base(80, 12, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", t_relax=160.0)
    axi_60_56x20 = relax_axi_3d_bell_base(56, 20, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", t_relax=160.0)
    axi_80_44x12 = relax_axi_3d_bell_base(44, 12, M_c=0.12, L_n=0.80, L_d=0.50, mode="bell_smooth", t_relax=160.0)
    print(f"  2B Steep bell_c2 (Ln=0.50, 40x12): M_div_wall_min={axi_c2_50['M_div_wall_min']:.3f}, M_exit_wall={axi_c2_50['M_exit_wall']:.3f}")
    print(f"  2B Smooth bell (Ln=0.60, 56x20): rhs={axi_60_56x20['rhs_max']:.2e} | Throat M_c={axi_60_56x20['M_throat_center']:.3f}, M_w={axi_60_56x20['M_throat_wall']:.3f} | Exit M={axi_60_56x20['M_exit_center']:.3f}-{axi_60_56x20['M_exit_wall']:.3f}")

    exp2b_tasks = [
        ("bell3d_Ln60_40x12_1T", dict(nx=40, nr=12, nth=48, t_end=35.0, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", init_type="spin_1T", A_eq=1e-4, prim_axi_base=axi_60_40x12["prim_axi"], record_stride=12)),
        ("bell3d_Ln60_56x16_1T", dict(nx=56, nr=16, nth=48, t_end=35.0, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", init_type="spin_1T", A_eq=1e-4, prim_axi_base=axi_60_56x16["prim_axi"], record_stride=16)),
        ("bell3d_Ln60_80x12_1T", dict(nx=80, nr=12, nth=48, t_end=35.0, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", init_type="spin_1T", A_eq=1e-4, prim_axi_base=axi_60_80x12["prim_axi"], record_stride=20)),
        ("bell3d_Ln60_56x20_1T", dict(nx=56, nr=20, nth=48, t_end=35.0, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", init_type="spin_1T", A_eq=1e-4, prim_axi_base=axi_60_56x20["prim_axi"], record_stride=20)),
        ("bell3d_Ln60_40x12_1R", dict(nx=40, nr=12, nth=48, t_end=35.0, M_c=0.12, L_n=0.60, L_d=0.45, mode="bell_smooth", init_type="1R",      A_eq=1e-4, prim_axi_base=axi_60_40x12["prim_axi"], record_stride=12)),
        ("bell3d_Ln80_44x12_1T", dict(nx=44, nr=12, nth=48, t_end=35.0, M_c=0.12, L_n=0.80, L_d=0.50, mode="bell_smooth", init_type="spin_1T", A_eq=1e-4, prim_axi_base=axi_80_44x12["prim_axi"], record_stride=12)),
    ]
    exp2b_res = run_batch(_sim3d_worker, exp2b_tasks)
    exp2b_summary = {}
    for cid, Ln_v, Ld_v, s_mn, ek, ww in [
        ("bell3d_Ln60_40x12_1T", 0.60, 0.45, S11, "e_1t", (1.40, 2.35)),
        ("bell3d_Ln60_56x16_1T", 0.60, 0.45, S11, "e_1t", (1.40, 2.35)),
        ("bell3d_Ln60_80x12_1T", 0.60, 0.45, S11, "e_1t", (1.40, 2.35)),
        ("bell3d_Ln60_56x20_1T", 0.60, 0.45, S11, "e_1t", (1.40, 2.35)),
        ("bell3d_Ln60_40x12_1R", 0.60, 0.45, S02, "e_1r", (3.35, 4.15)),
        ("bell3d_Ln80_44x12_1T", 0.80, 0.50, S11, "e_1t", (1.40, 2.35)),
    ]:
        vq = quasi3d_bessel_bell_eig(0.12, s_mn, L_n=Ln_v, L_d=Ld_v, contour="smooth3d", omega_window=ww, N=96)
        ap1 = fit_alpha(exp2b_res[cid], 14.0, 33.0, key=ek)
        ap2 = fit_alpha(exp2b_res[cid], 20.0, 34.0, key=ek)
        exp2b_summary[cid] = {"quasi3d_wkb_alpha": vq.real, "quasi3d_wkb_omega": vq.imag,
                              "true_3d_bell_alpha_14_33": ap1, "true_3d_bell_alpha_20_34": ap2}
        print(f"  2B [{cid:21s}] quasi3D-WKB={vq.real:+.5f} | true-3D [14,33]={ap1:+.5f}, [20,34]={ap2:+.5f}")

    # ------------------------------------------------------------------
    # Experiment 3: 3D Chiral 1T -> 1R vs 1T -> 2T Selection Rule (Small-A0 Quartic & Fixed 3D Energies)
    # ------------------------------------------------------------------
    print("\n--- Experiment 3: 3D Chiral 1T -> 1R vs 1T -> 2T Selection Rule ---")
    A_grid = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
    exp3_disk_tasks = []
    for A0 in A_grid:
        exp3_disk_tasks.append((f"disk24_spin_A{A0:g}",  dict(nr=24, nth=96, ic_kind="spin_1T", A_eq=A0, t_end=16.0)))
        exp3_disk_tasks.append((f"disk24_stand_A{A0:g}", dict(nr=24, nth=96, ic_kind="stand_1T", A_eq=A0, t_end=16.0)))
    exp3_disk_tasks.append(("disk32_spin_A0.08",  dict(nr=32, nth=128, ic_kind="spin_1T", A_eq=0.08, t_end=16.0)))
    exp3_disk_tasks.append(("disk32_stand_A0.08", dict(nr=32, nth=128, ic_kind="stand_1T", A_eq=0.08, t_end=16.0)))
    exp3_disk_res = run_batch(_disk_worker, exp3_disk_tasks)
    for A0 in A_grid:
        sp = exp3_disk_res[f"disk24_spin_A{A0:g}"]; st = exp3_disk_res[f"disk24_stand_A{A0:g}"]
        rat = st["E_1R_ac"] / max(sp["E_1R_ac"], 1e-30)
        rat_l = st["E_1R_ac_10_16"] / max(sp["E_1R_ac_10_16"], 1e-30)
        print(f"  3A A0={A0:5g}: spin(E1R_ac={sp['E_1R_ac']:.2e}) | stand(E2T={st['E_2T_over_E1T_pct']:.3f}%, E1R_ac={st['E_1R_ac']:.2e} [{st['E_1R_ac_over_E1T_pct']:.3f}%]) | ratio [6.4,16]={rat:.0f}x, [10,16]={rat_l:.0f}x")
    sp32 = exp3_disk_res["disk32_spin_A0.08"]; st32 = exp3_disk_res["disk32_stand_A0.08"]
    print(f"  3A Grid 32x128 (A0=0.08): spin E1R_ac={sp32['E_1R_ac']:.2e}, stand E1R_ac={st32['E_1R_ac']:.2e} ({st32['E_1R_ac_over_E1T_pct']:.2f}%), ratio={st32['E_1R_ac']/sp32['E_1R_ac']:.0f}x")

    A_small = [0.005, 0.01, 0.02]
    stand_e1r_all = [exp3_disk_res[f"disk24_stand_A{A0:g}"]["E_1R_ac"] for A0 in A_grid]
    spin_e1r_all = [exp3_disk_res[f"disk24_spin_A{A0:g}"]["E_1R_ac"] for A0 in A_grid]
    stand_e1r_small = [exp3_disk_res[f"disk24_stand_A{A0:g}"]["E_1R_ac"] for A0 in A_small]
    spin_e1r_small = [exp3_disk_res[f"disk24_spin_A{A0:g}"]["E_1R_ac"] for A0 in A_small]
    slope_stand_small = float(np.polyfit(np.log(A_small), np.log(stand_e1r_small), 1)[0])
    slope_spin_small = float(np.polyfit(np.log(A_small), np.log(spin_e1r_small), 1)[0])
    slope_stand_all = float(np.polyfit(np.log(A_grid), np.log(stand_e1r_all), 1)[0])
    slope_spin_all = float(np.polyfit(np.log(A_grid), np.log(spin_e1r_all), 1)[0])
    print(f"  3A Slopes: small-A (<=0.02) stand={slope_stand_small:.3f}, spin={slope_spin_small:.3f} | full range stand={slope_stand_all:.3f}, spin={slope_spin_all:.3f}")

    exp3b_tasks = [
        ("th3d_spin1T_nl",  dict(nx=20, nr=12, nth=48, t_end=26.0, M_c=0.12, mode="straight", init_type="spin_1T", A_eq=0.06, n_p=0.24, record_stride=12)),
        ("th3d_stand1T_nl", dict(nx=20, nr=12, nth=48, t_end=26.0, M_c=0.12, mode="straight", init_type="stand_1T", A_eq=0.06, n_p=0.24, record_stride=12)),
    ]
    exp3b_res = run_batch(_sim3d_worker, exp3b_tasks)
    exp3b_stats = {}
    for cid, _ in exp3b_tasks:
        rr = exp3b_res[cid]; t = rr["t"]; ml = t >= 0.50 * t[-1]
        c0_l = rr["c0_re"][ml]
        exp3b_stats[cid] = {
            "E_1T_late": float(np.mean(rr["e_1t"][ml])),
            "E_2T_late": float(np.mean(rr["e_2t"][ml])),
            "E_1R_ac_late": float(np.var(c0_l)),
            "E_1R_dc_late": float(np.mean(c0_l) ** 2),
            "p_peak_pct": float(np.mean(rr["p_max_pct"][ml])),
        }
        print(f"  3B [{cid:16s}] P={exp3b_stats[cid]['p_peak_pct']:.2f}% | E_1T={exp3b_stats[cid]['E_1T_late']:.3e}, E_2T={exp3b_stats[cid]['E_2T_late']:.3e}, E_1R_ac={exp3b_stats[cid]['E_1R_ac_late']:.3e}")

    # ------------------------------------------------------------------
    # Experiment 4: Matched-Window Shroud vs Hub (12x48 & 20x64), 4-pt EP Sweep & t=120 / 20x20x64 Shootout
    # ------------------------------------------------------------------
    print("\n--- Experiment 4: Matched-Window Shroud vs Hub (12x48 & 20x64) & Long-Horizon Shootout ---")
    analytical_weights = {
        "shroud_r1": {"1T": float(bessel_modal_weight(1, S11, 1.0)), "2T": float(bessel_modal_weight(2, S21, 1.0)),
                      "1R": float(bessel_modal_weight(0, S02, 1.0)), "3T": float(bessel_modal_weight(3, S31, 1.0))},
        "hub_r0":    {"1T": 0.0, "2T": 0.0, "1R": float(bessel_modal_weight(0, S02, 0.0)), "3T": 0.0},
    }

    # 4B: Matched t_end=24, record_stride=10, window [7, 22] for ALL bare & cavity runs (Review #11 M2, M3, m6!)
    lin_tasks_12 = [
        ("bare12_1T",        dict(nx=24, nr=12, nth=48, t_end=24.0, M_c=0.12, mode="straight", init_type="spin_1T", A_eq=1e-4, record_stride=10)),
        ("bare12_2T",        dict(nx=24, nr=12, nth=48, t_end=24.0, M_c=0.12, mode="straight", init_type="spin_2T", A_eq=1e-4, record_stride=10)),
        ("bare12_1R",        dict(nx=24, nr=12, nth=48, t_end=24.0, M_c=0.12, mode="straight", init_type="1R",      A_eq=1e-4, record_stride=10)),
        ("w05_1T_shroud",    dict(nx=24, nr=12, nth=48, t_end=24.0, M_c=0.12, mode="straight", init_type="spin_1T", A_eq=1e-4, cav_spec=[{"beta": 0.005, "wH": S11, "xiH": 0.14, "sigma_o": 0.06, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": False}], record_stride=10)),
        ("w05_1T_uniform",   dict(nx=24, nr=12, nth=48, t_end=24.0, M_c=0.12, mode="straight", init_type="spin_1T", A_eq=1e-4, cav_spec=[{"beta": 0.005, "wH": S11, "xiH": 0.14, "sigma_o": 0.06, "n_cav": 5, "r_lo": 0.00, "r_hi": 1.00, "nonlinear": False}], record_stride=10)),
        # 4-point xi_H sweep on 2.0% shroud 1T to directly locate the 3D EP peak (Review #11 m6!)
        ("ep10_1T_shroud",   dict(nx=24, nr=12, nth=48, t_end=24.0, M_c=0.12, mode="straight", init_type="spin_1T", A_eq=1e-4, cav_spec=[{"beta": 0.020, "wH": S11, "xiH": 0.10, "sigma_o": 0.06, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": False}], record_stride=10)),
        ("ep14_1T_shroud",   dict(nx=24, nr=12, nth=48, t_end=24.0, M_c=0.12, mode="straight", init_type="spin_1T", A_eq=1e-4, cav_spec=[{"beta": 0.020, "wH": S11, "xiH": 0.14, "sigma_o": 0.06, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": False}], record_stride=10)),
        ("ep24_1T_shroud",   dict(nx=24, nr=12, nth=48, t_end=24.0, M_c=0.12, mode="straight", init_type="spin_1T", A_eq=1e-4, cav_spec=[{"beta": 0.020, "wH": S11, "xiH": 0.24, "sigma_o": 0.06, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": False}], record_stride=10)),
        ("ep36_1T_shroud",   dict(nx=24, nr=12, nth=48, t_end=24.0, M_c=0.12, mode="straight", init_type="spin_1T", A_eq=1e-4, cav_spec=[{"beta": 0.020, "wH": S11, "xiH": 0.36, "sigma_o": 0.06, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": False}], record_stride=10)),
        ("ep24_1T_uniform",  dict(nx=24, nr=12, nth=48, t_end=24.0, M_c=0.12, mode="straight", init_type="spin_1T", A_eq=1e-4, cav_spec=[{"beta": 0.020, "wH": S11, "xiH": 0.24, "sigma_o": 0.06, "n_cav": 5, "r_lo": 0.00, "r_hi": 1.00, "nonlinear": False}], record_stride=10)),
        ("ep24_1T_hub",      dict(nx=24, nr=12, nth=48, t_end=24.0, M_c=0.12, mode="straight", init_type="spin_1T", A_eq=1e-4, cav_spec=[{"beta": 0.020, "wH": S11, "xiH": 0.24, "sigma_o": 0.06, "n_cav": 1, "r_lo": 0.00, "r_hi": 0.26, "nonlinear": False}], record_stride=10)),
        ("ep26_2T_shroud",   dict(nx=24, nr=12, nth=48, t_end=24.0, M_c=0.12, mode="straight", init_type="spin_2T", A_eq=1e-4, cav_spec=[{"beta": 0.020, "wH": S21, "xiH": 0.26, "sigma_o": 0.06, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": False}], record_stride=10)),
        ("ep26_2T_uniform",  dict(nx=24, nr=12, nth=48, t_end=24.0, M_c=0.12, mode="straight", init_type="spin_2T", A_eq=1e-4, cav_spec=[{"beta": 0.020, "wH": S21, "xiH": 0.26, "sigma_o": 0.06, "n_cav": 5, "r_lo": 0.00, "r_hi": 1.00, "nonlinear": False}], record_stride=10)),
        ("ep14_1R_shroud",   dict(nx=24, nr=12, nth=48, t_end=24.0, M_c=0.12, mode="straight", init_type="1R",      A_eq=1e-4, cav_spec=[{"beta": 0.020, "wH": S02, "xiH": 0.14, "sigma_o": 0.06, "n_cav": 1, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": False}], record_stride=10)),
        ("ep14_1R_uniform",  dict(nx=24, nr=12, nth=48, t_end=24.0, M_c=0.12, mode="straight", init_type="1R",      A_eq=1e-4, cav_spec=[{"beta": 0.020, "wH": S02, "xiH": 0.14, "sigma_o": 0.06, "n_cav": 1, "r_lo": 0.00, "r_hi": 1.00, "nonlinear": False}], record_stride=10)),
        ("ep14_1R_hub035",   dict(nx=24, nr=12, nth=48, t_end=24.0, M_c=0.12, mode="straight", init_type="1R",      A_eq=1e-4, cav_spec=[{"beta": 0.0035,"wH": S02, "xiH": 0.14, "sigma_o": 0.03, "n_cav": 1, "r_lo": 0.00, "r_hi": 0.26, "nonlinear": False}], record_stride=10)),
        ("ep38_1R_hub20",    dict(nx=24, nr=12, nth=48, t_end=24.0, M_c=0.12, mode="straight", init_type="1R",      A_eq=1e-4, cav_spec=[{"beta": 0.020, "wH": S02, "xiH": 0.38, "sigma_o": 0.14, "n_cav": 1, "r_lo": 0.00, "r_hi": 0.26, "nonlinear": False}], record_stride=10)),
    ]
    lin_res_12 = run_batch(_sim3d_worker, lin_tasks_12)
    lin_12 = {}
    for cid, _ in lin_tasks_12:
        ek = "e_1r" if "_1R" in cid else ("e_2t" if "_2T" in cid else "e_1t")
        lin_12[cid] = fit_alpha(lin_res_12[cid], 7.0, 22.0, key=ek)
        print(f"  4B (12x48) [{cid:16s}] alpha[7,22]={lin_12[cid]:+.5f}")

    # Refined 24x20x64 linear placement audit (Review #11 M2 & M3!)
    lin_tasks_20 = [
        ("bare20_1T",       dict(nx=24, nr=20, nth=64, t_end=24.0, M_c=0.12, mode="straight", init_type="spin_1T", A_eq=1e-4, record_stride=12)),
        ("bare20_1R",       dict(nx=24, nr=20, nth=64, t_end=24.0, M_c=0.12, mode="straight", init_type="1R",      A_eq=1e-4, record_stride=12)),
        ("f20_w05_1T_shrd", dict(nx=24, nr=20, nth=64, t_end=24.0, M_c=0.12, mode="straight", init_type="spin_1T", A_eq=1e-4, cav_spec=[{"beta": 0.005, "wH": S11, "xiH": 0.14, "sigma_o": 0.06, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": False}], record_stride=12)),
        ("f20_w05_1T_unif", dict(nx=24, nr=20, nth=64, t_end=24.0, M_c=0.12, mode="straight", init_type="spin_1T", A_eq=1e-4, cav_spec=[{"beta": 0.005, "wH": S11, "xiH": 0.14, "sigma_o": 0.06, "n_cav": 5, "r_lo": 0.00, "r_hi": 1.00, "nonlinear": False}], record_stride=12)),
        ("f20_1R_shroud",   dict(nx=24, nr=20, nth=64, t_end=24.0, M_c=0.12, mode="straight", init_type="1R",      A_eq=1e-4, cav_spec=[{"beta": 0.020, "wH": S02, "xiH": 0.14, "sigma_o": 0.06, "n_cav": 1, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": False}], record_stride=12)),
        ("f20_1R_uniform",  dict(nx=24, nr=20, nth=64, t_end=24.0, M_c=0.12, mode="straight", init_type="1R",      A_eq=1e-4, cav_spec=[{"beta": 0.020, "wH": S02, "xiH": 0.14, "sigma_o": 0.06, "n_cav": 1, "r_lo": 0.00, "r_hi": 1.00, "nonlinear": False}], record_stride=12)),
        ("f20_1R_hub035",   dict(nx=24, nr=20, nth=64, t_end=24.0, M_c=0.12, mode="straight", init_type="1R",      A_eq=1e-4, cav_spec=[{"beta": 0.0035,"wH": S02, "xiH": 0.14, "sigma_o": 0.03, "n_cav": 1, "r_lo": 0.00, "r_hi": 0.26, "nonlinear": False}], record_stride=12)),
        ("f20_1R_hub20",    dict(nx=24, nr=20, nth=64, t_end=24.0, M_c=0.12, mode="straight", init_type="1R",      A_eq=1e-4, cav_spec=[{"beta": 0.020, "wH": S02, "xiH": 0.38, "sigma_o": 0.14, "n_cav": 1, "r_lo": 0.00, "r_hi": 0.26, "nonlinear": False}], record_stride=12)),
    ]
    lin_res_20 = run_batch(_sim3d_worker, lin_tasks_20)
    lin_20 = {}
    for cid, _ in lin_tasks_20:
        ek = "e_1r" if "_1R" in cid else "e_1t"
        a7 = fit_alpha(lin_res_20[cid], 7.0, 22.0, key=ek)
        a14 = fit_alpha(lin_res_20[cid], 14.0, 24.0, key=ek)
        lin_20[cid] = {"alpha_7_22": a7, "alpha_14_24": a14}
        print(f"  4B (20x64) [{cid:16s}] alpha[7,22]={a7:+.5f}, alpha[14,24]={a14:+.5f}")

    # 4C: Long-horizon t=120 reacting 3D shootout (20x12x48) + refined 20x20x64 (t=48) 1R shootout!
    cav_shroud_only = [
        {"beta": 0.0210, "wH": S11, "xiH": 0.22, "sigma_o": 0.10, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": True},
        {"beta": 0.0090, "wH": S21, "xiH": 0.24, "sigma_o": 0.08, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": True},
    ]
    cav_dualzone = [
        {"beta": 0.0185, "wH": S11, "xiH": 0.22, "sigma_o": 0.10, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": True},
        {"beta": 0.0080, "wH": S21, "xiH": 0.24, "sigma_o": 0.08, "n_cav": 5, "r_lo": 0.78, "r_hi": 1.00, "nonlinear": True},
        {"beta": 0.0035, "wH": S02, "xiH": 0.14, "sigma_o": 0.04, "n_cav": 1, "r_lo": 0.00, "r_hi": 0.26, "nonlinear": True},
    ]
    nl_tasks_120 = [
        ("nl_bare_1T",     dict(nx=20, nr=12, nth=48, t_end=120.0, M_c=0.12, mode="straight", init_type="spin_1T",  A_eq=0.04, n_p=0.30, record_stride=24)),
        ("nl_bare_st1T",   dict(nx=20, nr=12, nth=48, t_end=120.0, M_c=0.12, mode="straight", init_type="stand_1T", A_eq=0.04, n_p=0.30, record_stride=24)),
        ("nl_bare_2T",     dict(nx=20, nr=12, nth=48, t_end=120.0, M_c=0.12, mode="straight", init_type="spin_2T",  A_eq=0.04, n_p=0.30, record_stride=24)),
        ("nl_bare_1R",     dict(nx=20, nr=12, nth=48, t_end=120.0, M_c=0.12, mode="straight", init_type="1R",       A_eq=0.04, n_p=0.30, record_stride=24)),
        ("nl_shroud_1T",   dict(nx=20, nr=12, nth=48, t_end=120.0, M_c=0.12, mode="straight", init_type="spin_1T",  A_eq=0.04, n_p=0.30, cav_spec=cav_shroud_only, record_stride=24)),
        ("nl_shroud_st1T", dict(nx=20, nr=12, nth=48, t_end=120.0, M_c=0.12, mode="straight", init_type="stand_1T", A_eq=0.04, n_p=0.30, cav_spec=cav_shroud_only, record_stride=24)),
        ("nl_shroud_1R",   dict(nx=20, nr=12, nth=48, t_end=120.0, M_c=0.12, mode="straight", init_type="1R",       A_eq=0.04, n_p=0.30, cav_spec=cav_shroud_only, record_stride=24)),
        ("nl_dualz_1T",    dict(nx=20, nr=12, nth=48, t_end=120.0, M_c=0.12, mode="straight", init_type="spin_1T",  A_eq=0.04, n_p=0.30, cav_spec=cav_dualzone, record_stride=24)),
        ("nl_dualz_st1T",  dict(nx=20, nr=12, nth=48, t_end=120.0, M_c=0.12, mode="straight", init_type="stand_1T", A_eq=0.04, n_p=0.30, cav_spec=cav_dualzone, record_stride=24)),
        ("nl_dualz_2T",    dict(nx=20, nr=12, nth=48, t_end=120.0, M_c=0.12, mode="straight", init_type="spin_2T",  A_eq=0.04, n_p=0.30, cav_spec=cav_dualzone, record_stride=24)),
        ("nl_dualz_1R",    dict(nx=20, nr=12, nth=48, t_end=120.0, M_c=0.12, mode="straight", init_type="1R",       A_eq=0.04, n_p=0.30, cav_spec=cav_dualzone, record_stride=24)),
    ]
    nl_res_120 = run_batch(_sim3d_worker, nl_tasks_120)
    nl_out_120 = {}
    for cid, _ in nl_tasks_120:
        rr = nl_res_120[cid]; t = rr["t"]
        m48 = (t >= 38.4) & (t <= 48.0)
        m120 = t >= 110.4
        nl_out_120[cid] = {
            "p_peak_48": float(np.mean(rr["p_max_pct"][m48])),
            "p_peak_120": float(np.mean(rr["p_max_pct"][m120])),
            "e_wave_48": float(np.mean(rr["e_ch"][m48])),
            "e_wave_120": float(np.mean(rr["e_ch"][m120])),
            "slope_28_48": fit_alpha(rr, 28.8, 48.0, "e_ch"),
            "slope_84_120": fit_alpha(rr, 84.0, 120.0, "e_ch"),
            "e_1t_120": float(np.mean(rr["e_1t"][m120])),
            "e_2t_120": float(np.mean(rr["e_2t"][m120])),
            "e_1r_120": float(np.mean(rr["e_1r"][m120])),
            "e_1l_120": float(np.mean(rr["e_1l"][m120])),
            "e_2l_120": float(np.mean(rr["e_2l"][m120])),
            "e_mean_120": float(np.mean(rr["e_mean"][m120])),
            "E_1R_ac_48": float(np.var(rr["c0_re"][m48])),
        }
        s = nl_out_120[cid]
        print(f"  4C(t=120) [{cid:15s}] P48={s['p_peak_48']:5.2f}%->P120={s['p_peak_120']:5.2f}% | slp[84,120]={s['slope_84_120']:+.4f} | 1T={s['e_1t_120']:.1e}, 1L={s['e_1l_120']:.1e}, 2L={s['e_2l_120']:.1e}, mn={s['e_mean_120']:.1e}")

    nl_tasks_20 = [
        ("nl20_bare_1R",   dict(nx=20, nr=20, nth=64, t_end=48.0, M_c=0.12, mode="straight", init_type="1R", A_eq=0.04, n_p=0.30, record_stride=24)),
        ("nl20_shroud_1R", dict(nx=20, nr=20, nth=64, t_end=48.0, M_c=0.12, mode="straight", init_type="1R", A_eq=0.04, n_p=0.30, cav_spec=cav_shroud_only, record_stride=24)),
        ("nl20_dualz_1R",  dict(nx=20, nr=20, nth=64, t_end=48.0, M_c=0.12, mode="straight", init_type="1R", A_eq=0.04, n_p=0.30, cav_spec=cav_dualzone, record_stride=24)),
    ]
    nl_res_20 = run_batch(_sim3d_worker, nl_tasks_20)
    nl_out_20 = {}
    for cid, _ in nl_tasks_20:
        rr = nl_res_20[cid]; t = rr["t"]; ml = t >= 38.4
        nl_out_20[cid] = {
            "p_peak_48": float(np.mean(rr["p_max_pct"][ml])),
            "e_wave_48": float(np.mean(rr["e_ch"][ml])),
            "slope_28_48": fit_alpha(rr, 28.8, 48.0, "e_ch"),
        }
        s = nl_out_20[cid]
        print(f"  4C(20x20x64) [{cid:15s}] P48={s['p_peak_48']:5.2f}% | E={s['e_wave_48']:.2e} | slp={s['slope_28_48']:+.4f}")

    # Save hardened metrics JSON BEFORE plotting
    metrics = {
        "bessel_constants": {"s_11_1T": S11, "s_21_2T": S21, "s_02_1R": S02, "s_31_3T": S31, "s_12_1T1R": S12,
                             "ratio_2T_over_1T": S21 / S11, "ratio_1R_over_1T": S02 / S11},
        "exp1_3d_benchmarks": {
            "quiescent_bell_gcl_x": rq["gcl_x"], "quiescent_bell_gcl_r": rq["gcl_r"],
            "quiescent_bell_err": rq["quiescent_err"],
            "periodic_3d_bell_cons_err": rc["periodic_cons_err"],
            "closed_cyl_bessel_fft": bessel_fft,
            "closed_cyl_num_damping": num_audit,
            "straight_axial_refinement": axial_refinement,
        },
        "exp2_3d_bell_cutoff": {
            "Ln_sweep_Mc012_N160": exp2_Ln_rows,
            "Mc_hopf_nv0_Ln050_tau016": exp2_hopf_3d,
            "tau040_hopf_cross_check": tau040_check,
            "axi_base_c2_Ln50_40x12": {k: v for k, v in axi_c2_50.items() if k != "prim_axi"},
            "axi_base_smooth_Ln60_56x20": {k: v for k, v in axi_60_56x20.items() if k != "prim_axi"},
            "axi_base_smooth_Ln80_44x12": {k: v for k, v in axi_80_44x12.items() if k != "prim_axi"},
            "true_3d_bell_vs_quasi3d": exp2b_summary,
        },
        "exp3_chiral_1T_to_1R_cascade": {
            "A_grid": A_grid,
            "disk_runs": exp3_disk_res,
            "stand_1R_ac_slope_smallA": slope_stand_small,
            "spin_1R_ac_slope_smallA": slope_spin_small,
            "stand_1R_ac_slope_all": slope_stand_all,
            "spin_1R_ac_slope_all": slope_spin_all,
            "reacting_3d_throughflow": exp3b_stats,
        },
        "exp4_shroud_vs_hub_cavities": {
            "analytical_weights": analytical_weights,
            "linear_placement_12x48_matched": lin_12,
            "linear_placement_20x64_matched": lin_20,
            "reacting_t120_shootout_12x48": nl_out_120,
            "reacting_t48_1R_shootout_20x64": nl_out_20,
        },
    }
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved hardened metrics to {out_path}")

    plot_all_figures(metrics, nl_res_20, nl_res_120)


if __name__ == "__main__":
    main()
