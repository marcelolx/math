"""Entry 013 (rocket-combustion): Dual-Harmonic Baffle-Cavity Retuning (wH_1T, wH_2T) &
Coupled 2D Compressible Navier-Stokes Transonic Bell Nozzle — Breaking the Universal
1T -> 2T Cascade Attractor.

Couples Entry 011's full 2D conservative compressible Navier-Stokes discretization (Pr = 0.72,
Stokes' hypothesis, constant-eddy-viscosity closure Re = rho_bar*c_bar*R/mu) with Entry 012's
explicitly resolved C^2 2D thin-shell variable-(A(x), R(x)) transonic converging-diverging
bell nozzle (M_e ~ 1.83 > 1, zero outlet BCs) to answer the central open question left by
Peer Reviews #8 and #9:

    Why did the retuned 2% 80/20 baffle-cavity hybrid still lose 6.9% bombs to a ~5% pure-2T
    upper branch in BOTH 2D NS (011: 4.13%) and resolved 120x300 transonic nozzles (012:
    5.07% annular, 5.54% bell)—and what minimum passive hardware + eddy viscosity breaks it?

Key physical mechanisms tested:
  1. Higher-Harmonic Baffle Downshift Law: Protruding N_b = 5, L_b = 0.45 radial baffles
     downshift 2T by ~18% (from unbaffled ~2.01 down to ~1.65, nearly twice the ~10-12%
     downshift of 1T!). Leaving wH_2T at unbaffled 2.01 (Entries 010-012) detuned the 2T
     sub-ring by ~22%, wasting >60% of its linear damping and exposing the 1T -> 2T cascade.
  2. Coupled Viscous-Transonic Bell Rescue: Turbulent eddy viscosity (Re = 500-1000) boosts
     bell cold 2T damping 6-9x (rescuing the inviscid 2T evanescent-cutoff collapse and
     quenching both inviscid M_c = 0.10 m = 3, 4 whistles), while bare bell 1T remains
     linearly unstable across M_c in [0.10, 0.22] even at Re = 500.
  3. Resolved 120x300 (t = 300) All-Mode Nonlinear Quench Audit: Tests dual-retuned 2.0%
     80/20 hybrids (wH_1T = 0.96, wH_2T = 1.65) in BOTH inviscid and coupled viscous
     (Re = 1000, Re = 500; free-slip and converged t = 150 no-slip wakes) transonic bell
     nozzles against BOTH 6.9% spin_1T AND direct 6.9% spin_2T bombs on resolved 120x300 grids!

Experiments:
  Exp 1: Exact coupled viscous-transonic benchmarks — quiescent well-balancing (~1e-16),
         periodic variable-(A,R) bell viscous mass/angular-momentum/total-energy conservation
         (~1e-16), single-domain compressible NS transonic Chebyshev vs. 96x200 2D NS-Bell
         PDE (Re = 500, 1000; 1T & 2T), and direct FFT measurement of baffled 1T/2T downshift.
  Exp 2: Coupled viscous-transonic bell global Hopf envelopes (m = 0..5, Re in {inf, 2000,
         1000, 500}) quantifying viscous rescue of the bell 2T cutoff collapse.
  Exp 3: Dual-Harmonic Baffle-Cavity Retuning Law — full wH_2T resonance curve (1.35..2.15),
         on-resonance tight (1.5%) vs. wide (5.0%) 2T orifice audit, and 80/20 vs. 60/40 split.
  Exp 4: Resolved 120x300 (t = 300) breaking of the 1T -> 2T cascade attractor — inviscid
         untuned vs. dual-retuned 80/20 (1T & 2T bombs, tight vs. wide neck) and coupled
         viscous (Re = 1000 & Re = 500, slip & t = 150 no-slip wakes) transonic bell nozzles.
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


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

GAMMA = 1.25
PR = 0.72
P_BAR = 1.0 / GAMMA       # 0.80 => c_c = 1.0, rho_c = 1.0, T_c = 1.0
RHO_BAR = 1.0
L_A = 1.5
L_D = 0.25
X_F = 0.25
SIG_F = 0.08
OMEGA_HP = 0.15
TAU_BAR = 0.55
Q_S = 0.036
KAPPA_NL = 4.5


# ---------------------------------------------------------------------------
# 1. C^2 Transonic CD Nozzle Geometry & Exact Isentropic Base Flow
# ---------------------------------------------------------------------------

def isentropic_area_ratio(M: float) -> float:
    g = GAMMA
    return (1.0 / M) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * M * M)) ** ((g + 1.0) / (2.0 * (g - 1.0)))


def nozzle_profile_c2(x: np.ndarray, M_c: float, L_n: float, L_d: float = L_D) -> tuple[np.ndarray, np.ndarray]:
    A_t = 1.0 / isentropic_area_ratio(M_c)
    k_t = 12.0 * (1.0 - A_t) / (L_n * L_n)
    delta_A = 0.60 * A_t
    y0_sq = 2.0 * delta_A / k_t
    A = np.ones_like(x, dtype=float)
    dA = np.zeros_like(x, dtype=float)
    x_t = L_A + L_n
    m1 = (x > L_A) & (x <= x_t)
    s1 = (x[m1] - L_A) / L_n
    A[m1] = A_t + (1.0 - A_t) * (1.0 - 4.0 * s1**3 + 3.0 * s1**4)
    dA[m1] = (1.0 - A_t) * (-12.0 * s1**2 + 12.0 * s1**3) / L_n
    m2 = x > x_t
    y = x[m2] - x_t
    denom = y * y + y0_sq
    A[m2] = A_t + delta_A * (y * y / denom)
    dA[m2] = delta_A * (2.0 * y * y0_sq / (denom * denom))
    return A, dA


def exact_isentropic_base(x: np.ndarray, M_c: float, L_n: float, L_d: float = L_D, geom: str = "bell") -> dict[str, np.ndarray]:
    A, dA = nozzle_profile_c2(x, M_c, L_n, L_d)
    A_t = 1.0 / isentropic_area_ratio(M_c)
    x_t = L_A + L_n
    M = np.zeros_like(x, dtype=float)
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
    T = fac_c / fac
    rho = T ** (1.0 / (GAMMA - 1.0))
    p = P_BAR * (T ** (GAMMA / (GAMMA - 1.0)))
    c = np.sqrt(T)
    ux = M * c
    R = np.ones_like(A) if geom == "annular" else np.sqrt(A)
    dR = np.zeros_like(A) if geom == "annular" else 0.5 * dA / R
    return {"A": A, "dA": dA, "R": R, "dR": dR, "rho": rho, "ux": ux, "p": p, "c": c, "T": T, "M": M}


# ---------------------------------------------------------------------------
# 2. Exact Single-Domain Compressible NS Transonic Chebyshev Eigen-Solver
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


_VISC_BASE_CACHE: dict[tuple, dict[str, np.ndarray]] = {}


def solve_1d_viscous_cheb_base(
    x: np.ndarray, Dx: np.ndarray, M_c: float, L_n: float, Re: float, geom: str = "bell"
) -> dict[str, np.ndarray]:
    """Exact 1D steady compressible Navier-Stokes transonic base flow on Chebyshev nodes."""
    key = (round(float(M_c), 8), round(float(L_n), 8), "inf" if not np.isfinite(Re) else round(float(Re), 6), geom, len(x))
    if key in _VISC_BASE_CACHE:
        return _VISC_BASE_CACHE[key]
    A, dA = nozzle_profile_c2(x, M_c, L_n, L_D)
    base0 = exact_isentropic_base(x, M_c, L_n, L_D, geom=geom)
    if not np.isfinite(Re):
        _VISC_BASE_CACHE[key] = base0
        return base0
    mu = 1.0 / Re
    kap = mu / ((GAMMA - 1.0) * PR)
    g = GAMMA
    n = len(x)
    y = np.concatenate([base0["rho"], base0["ux"], base0["p"]])

    def res(yv: np.ndarray) -> np.ndarray:
        r = yv[:n]; u = yv[n:2 * n]; p = yv[2 * n:]
        T = g * p / r; du = Dx @ u; dT = Dx @ T
        E = p / (g - 1.0) + 0.5 * r * u * u
        txx = (4.0 / 3.0) * mu * du
        f0 = Dx @ (A * r * u)
        f1 = Dx @ (A * (r * u * u + p - txx)) - p * dA
        f2 = Dx @ (A * (u * (E + p) - u * txx - kap * dT))
        # Isentropic choked injector at x = 0: rho*u = M_c, p = P_BAR * rho^gamma
        f0[0] = r[0] * u[0] - M_c
        f2[0] = p[0] - P_BAR * (r[0] ** g)
        # Zero-gradient viscous outflow at supersonic exit x = L_tot
        f1[-1] = du[-1]
        f2[-1] = dT[-1]
        return np.concatenate([f0, f1, f2])

    eps = 1e-7
    for _ in range(14):
        F = res(y)
        if float(np.max(np.abs(F))) < 1e-12:
            break
        J = np.zeros((3 * n, 3 * n))
        for j in range(3 * n):
            yp = y.copy(); yp[j] += eps
            J[:, j] = (res(yp) - F) / eps
        y += np.linalg.solve(J, -F)

    r = y[:n]; u = y[n:2 * n]; p = y[2 * n:]
    T = g * p / r; c = np.sqrt(T)
    R = np.ones_like(A) if geom == "annular" else np.sqrt(A)
    dR = np.zeros_like(A) if geom == "annular" else 0.5 * dA / R
    out = {"A": A, "dA": dA, "R": R, "dR": dR, "rho": r, "ux": u, "p": p, "c": c, "T": T, "M": u / c}
    _VISC_BASE_CACHE[key] = out
    return out


def exact_viscous_transonic_eig(
    M_c: float, m: int, L_n: float = 0.50, geom: str = "bell",
    Re: float = np.inf, n_p: float = 0.0, n_v: float = 0.0,
    omega_window: tuple[float, float] | None = None, N: int = 84,
    check_both_chiral_signs: bool = True,
) -> complex:
    """Exact conservative linearization of the 2D thin-shell compressible Navier-Stokes PDE
    around the exact 1D steady viscous/inviscid transonic base flow on Chebyshev nodes.
    """
    if check_both_chiral_signs and abs(n_v) > 0.0 and m != 0:
        vp = exact_viscous_transonic_eig(M_c, abs(m), L_n, geom, Re, n_p, n_v, omega_window, N, False)
        vn = exact_viscous_transonic_eig(M_c, -abs(m), L_n, geom, Re, n_p, n_v, omega_window, N, False)
        return vp if vp.real >= vn.real else vn

    L_tot = L_A + L_n + L_D
    x, Dx, w_cc = cheb_diff(N, L_tot)
    base = solve_1d_viscous_cheb_base(x, Dx, M_c, L_n, Re, geom=geom)
    A_x, dA, R_x = base["A"], base["dA"], base["R"]
    r0, u0, p0, T0 = base["rho"], base["ux"], base["p"], base["T"]
    g = GAMMA
    n = N + 1
    mu = (1.0 / Re) if np.isfinite(Re) else 0.0
    kap = (mu / ((g - 1.0) * PR)) if np.isfinite(Re) else 0.0
    du0 = Dx @ u0
    txx0 = (4.0 / 3.0) * mu * du0
    ttt0 = -(2.0 / 3.0) * mu * du0
    H0 = (g / (g - 1.0)) * p0 + 0.5 * r0 * u0 * u0

    has_f = (abs(n_p) > 0.0 or abs(n_v) > 0.0)
    dim = 4 * n + (10 if has_f else 0)
    Mat = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    sl = lambda q: slice(q * n, (q + 1) * n)

    # Mass matrix B = dW / d[rho', ux', ut', p']
    B[sl(0), sl(0)] = np.diag(A_x)
    B[sl(1), sl(0)] = np.diag(A_x * u0)
    B[sl(1), sl(1)] = np.diag(A_x * r0)
    B[sl(2), sl(2)] = np.diag(A_x * R_x * r0)
    B[sl(3), sl(0)] = np.diag(0.5 * A_x * u0 * u0)
    B[sl(3), sl(1)] = np.diag(A_x * r0 * u0)
    B[sl(3), sl(3)] = np.diag(A_x / (g - 1.0))
    if has_f:
        B[4 * n:, 4 * n:] = np.eye(10, dtype=complex)

    # Exact conservative operator (inviscid + full compressible viscous & Fourier heat fluxes)
    # Row 0: Mass
    Mat[sl(0), sl(0)] = -Dx @ np.diag(A_x * u0)
    Mat[sl(0), sl(1)] = -Dx @ np.diag(A_x * r0)
    Mat[sl(0), sl(2)] = -1j * m * np.diag(A_x * r0 / R_x)

    # Row 1: Axial momentum
    Mat[sl(1), sl(0)] = -Dx @ np.diag(A_x * u0 * u0)
    Mat[sl(1), sl(1)] = -Dx @ np.diag(2.0 * A_x * r0 * u0) + mu * (Dx @ np.diag((4.0 / 3.0) * A_x) @ Dx - (m * m) * np.diag(A_x / (R_x * R_x)))
    Mat[sl(1), sl(2)] = -1j * m * np.diag(A_x * r0 * u0 / R_x) + 1j * m * mu * (Dx @ np.diag((-2.0 / 3.0) * A_x / R_x) + np.diag(A_x / R_x) @ Dx)
    Mat[sl(1), sl(3)] = -Dx @ np.diag(A_x) + np.diag(dA)

    # Row 2: Angular momentum
    Mat[sl(2), sl(1)] = 1j * m * mu * (Dx @ np.diag(A_x) - np.diag((2.0 / 3.0) * A_x) @ Dx)
    Mat[sl(2), sl(2)] = -Dx @ np.diag(A_x * R_x * r0 * u0) + mu * (Dx @ np.diag(A_x * R_x) @ Dx - (4.0 / 3.0) * (m * m) * np.diag(A_x / R_x))
    Mat[sl(2), sl(3)] = -1j * m * np.diag(A_x)

    # Row 3: Total energy (including exact base-stress perturbations -txx0*u' and ttt0*v')
    Tp = np.diag(g / r0)
    Tr = np.diag(-T0 / r0)
    LapT = kap * (Dx @ np.diag(A_x) @ Dx - (m * m) * np.diag(A_x / (R_x * R_x)))
    Mat[sl(3), sl(0)] = -Dx @ np.diag(0.5 * A_x * (u0 ** 3)) + LapT @ Tr
    Mat[sl(3), sl(1)] = -Dx @ np.diag(A_x * (H0 + r0 * u0 * u0 - txx0)) + mu * (Dx @ np.diag((4.0 / 3.0) * A_x * u0) @ Dx - (m * m) * np.diag(A_x * u0 / (R_x * R_x)))
    Mat[sl(3), sl(2)] = -1j * m * np.diag(A_x * (H0 - ttt0) / R_x) + 1j * m * mu * (Dx @ np.diag((-2.0 / 3.0) * A_x * u0 / R_x) + np.diag(A_x * u0 / R_x) @ Dx)
    Mat[sl(3), sl(3)] = -Dx @ np.diag((g / (g - 1.0)) * A_x * u0) + LapT @ Tp

    if has_f:
        wf = np.where(x <= L_A, np.exp(-0.5 * ((x - X_F) / SIG_F) ** 2), 0.0)
        wf /= float(np.sum(w_cc * wf))
        w_sense = w_cc * wf
        inv_ts = 4.0 / TAU_BAR; ib = 4 * n
        Mat[ib + 0, sl(3)] = OMEGA_HP * w_sense; Mat[ib + 0, ib + 0] = -OMEGA_HP
        Mat[ib + 1, sl(3)] = inv_ts * w_sense; Mat[ib + 1, ib + 0] = -inv_ts; Mat[ib + 1, ib + 1] = -inv_ts
        for r in range(1, 4):
            Mat[ib + 1 + r, ib + r] = inv_ts; Mat[ib + 1 + r, ib + 1 + r] = -inv_ts
        Mat[ib + 5, sl(2)] = OMEGA_HP * w_sense; Mat[ib + 5, ib + 5] = -OMEGA_HP
        Mat[ib + 6, sl(2)] = inv_ts * w_sense; Mat[ib + 6, ib + 5] = -inv_ts; Mat[ib + 6, ib + 6] = -inv_ts
        for r in range(1, 4):
            Mat[ib + 6 + r, ib + 5 + r] = inv_ts; Mat[ib + 6 + r, ib + 6 + r] = -inv_ts
        Mat[sl(3), ib + 4] = A_x * n_p * wf / (g - 1.0)
        Mat[sl(3), ib + 9] = A_x * n_v * wf / (g - 1.0)

    # Inlet BCs at x = 0 linearized about exact viscous/inviscid base (r0[0], u0[0], p0[0]):
    # p' - gamma*P_BAR*r0^(gamma-1)*rho' = 0, u0*rho' + r0*u' = 0, v' = 0
    for r in (0, n, 2 * n):
        Mat[r, :] = 0.0; B[r, :] = 0.0
    Mat[0, 0] = -g * P_BAR * (r0[0] ** (g - 1.0)); Mat[0, 3 * n] = 1.0
    Mat[n, 0] = u0[0]; Mat[n, n] = r0[0]
    Mat[2 * n, 2 * n] = 1.0

    # For finite Re: zero-gradient viscous outflow at supersonic exit x = L_tot (u'_x, u'_th, T')
    if np.isfinite(Re):
        for r_ex in (n + N, 2 * n + N, 3 * n + N):
            Mat[r_ex, :] = 0.0; B[r_ex, :] = 0.0
        Mat[n + N, sl(1)] = Dx[N, :]
        Mat[2 * n + N, sl(2)] = Dx[N, :]
        Mat[3 * n + N, sl(3)] = Dx[N, :] @ Tp
        Mat[3 * n + N, sl(0)] = Dx[N, :] @ Tr

    vals = eig(Mat, B, right=False)
    w_lo, w_hi = omega_window if omega_window is not None else ((0.55, 1.50) if abs(m) == 1 else (0.05, 7.0))
    cands = [v for v in vals if np.isfinite(v) and w_lo <= v.imag <= w_hi and -0.50 < v.real < 0.35]
    cands.sort(key=lambda z: z.real, reverse=True)
    return complex(cands[0])


# ---------------------------------------------------------------------------
# 3. Coupled 2D Conservative Compressible NS + Variable-(A, R) Transonic Solver
# ---------------------------------------------------------------------------

def van_leer_limiter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = a * b
    return np.where(ab > 0.0, 2.0 * ab / np.where(ab > 0.0, a + b, 1.0), 0.0)


def phi_flame(q: np.ndarray) -> np.ndarray:
    x = q / Q_S
    return Q_S * np.tanh(x + KAPPA_NL * (x ** 3))


def prim_from_W(W: np.ndarray, A_c: np.ndarray, R_c: np.ndarray):
    rho = np.maximum(W[0] / A_c, 1e-10)
    ux = W[1] / (A_c * rho)
    ut = W[2] / (A_c * R_c * rho)
    E = W[3] / A_c
    p = np.maximum((GAMMA - 1.0) * (E - 0.5 * rho * (ux * ux + ut * ut)), 1e-10)
    c = np.sqrt(GAMMA * p / rho)
    return rho, ux, ut, p, c


def W_from_prim(rho, ux, ut, p, A_c, R_c) -> np.ndarray:
    E = p / (GAMMA - 1.0) + 0.5 * rho * (ux * ux + ut * ut)
    return np.stack([A_c * rho, A_c * rho * ux, A_c * R_c * rho * ut, A_c * E], axis=0)


def hllc_2d_normal(rhoL, vnL, vtL, pL, rhoR, vnR, vtR, pR) -> np.ndarray:
    cL = np.sqrt(GAMMA * pL / rhoL); cR = np.sqrt(GAMMA * pR / rhoR)
    EL = pL / (GAMMA - 1.0) + 0.5 * rhoL * (vnL * vnL + vtL * vtL)
    ER = pR / (GAMMA - 1.0) + 0.5 * rhoR * (vnR * vnR + vtR * vtR)
    sL = np.minimum(vnL - cL, vnR - cR); sR = np.maximum(vnL + cL, vnR + cR)
    denom = np.where(np.abs(rhoL * (sL - vnL) - rhoR * (sR - vnR)) > 1e-14,
                     rhoL * (sL - vnL) - rhoR * (sR - vnR), 1e-14)
    sS = (pR - pL + rhoL * vnL * (sL - vnL) - rhoR * vnR * (sR - vnR)) / denom
    FL = np.stack([rhoL * vnL, rhoL * vnL * vnL + pL, rhoL * vnL * vtL, vnL * (EL + pL)], axis=0)
    FR = np.stack([rhoR * vnR, rhoR * vnR * vnR + pR, rhoR * vnR * vtR, vnR * (ER + pR)], axis=0)
    fL = rhoL * (sL - vnL) / np.where(np.abs(sL - sS) > 1e-14, sL - sS, 1e-14)
    UsL = np.stack([fL, fL * sS, fL * vtL, fL * (EL / rhoL + (sS - vnL) * (sS + pL / (rhoL * np.where(np.abs(sL - vnL) > 1e-14, sL - vnL, 1e-14))))], axis=0)
    FsL = FL + sL * (UsL - np.stack([rhoL, rhoL * vnL, rhoL * vtL, EL], axis=0))
    fR = rhoR * (sR - vnR) / np.where(np.abs(sR - sS) > 1e-14, sR - sS, 1e-14)
    UsR = np.stack([fR, fR * sS, fR * vtR, fR * (ER / rhoR + (sS - vnR) * (sS + pR / (rhoR * np.where(np.abs(sR - vnR) > 1e-14, sR - vnR, 1e-14))))], axis=0)
    FsR = FR + sR * (UsR - np.stack([rhoR, rhoR * vnR, rhoR * vtR, ER], axis=0))
    return np.where(sL >= 0.0, FL, np.where(sS >= 0.0, FsL, np.where(sR >= 0.0, FsR, FR)))


def inlet_choked_ghost(rho_i, ux_i, p_i, M_c: float, u_port: np.ndarray | None = None):
    if M_c <= 0.0 and u_port is None:
        return rho_i.copy(), -ux_i.copy(), np.zeros_like(ux_i), p_i.copy()
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
    return rho_g, ux_g, np.zeros_like(ux_g), p_g


def build_grid_geometry(nx: int, M_c: float, L_n: float, geom: str = "bell", periodic_bump: bool = False):
    if periodic_bump:
        L_tot = L_A; dx = L_tot / nx
        x = (np.arange(nx) + 0.5) * dx; xf = np.arange(nx + 1) * dx
        A_f = 1.0 - 0.35 * np.sin(np.pi * xf / L_tot) ** 2
        A_c = 0.5 * (A_f[:-1] + A_f[1:])
    else:
        L_tot = L_A + L_n + L_D; dx = L_tot / nx
        x = (np.arange(nx) + 0.5) * dx; xf = np.arange(nx + 1) * dx
        A_f, _ = nozzle_profile_c2(xf, M_c, L_n, L_D)
        A_c = 0.5 * (A_f[:-1] + A_f[1:])
    R_f = np.ones_like(A_f) if geom == "annular" else np.sqrt(A_f)
    R_c = np.ones_like(A_c) if geom == "annular" else np.sqrt(A_c)
    return {"x": x, "xf": xf, "dx": dx, "L_tot": L_tot, "A_c": A_c[:, None], "A_f": A_f[:, None], "R_c": R_c[:, None], "R_f": R_f[:, None]}


def relax_1d_viscous_transonic_base(
    nx: int, M_c: float, L_n: float = 0.50, geom: str = "bell", Re: float = np.inf, t_relax: float = 300.0
) -> tuple[np.ndarray, float]:
    """Relax exact 1D steady viscous/inviscid transonic base flow on grid nx to stationarity."""
    g = build_grid_geometry(nx, M_c, L_n, geom=geom)
    x, dx = g["x"], g["dx"]
    A_c, A_f, R_c, R_f = g["A_c"], g["A_f"], g["R_c"], g["R_f"]
    dA_dx = (A_f[1:] - A_f[:-1]) / dx
    visc = np.isfinite(Re)
    mu = (1.0 / Re) if visc else 0.0
    kap = (mu / ((GAMMA - 1.0) * PR)) if visc else 0.0
    nu_max = mu * max(4.0 / 3.0, GAMMA / PR)

    base = exact_isentropic_base(x, M_c, L_n, L_D, geom=geom)
    W = W_from_prim(base["rho"][:, None], base["ux"][:, None], np.zeros((nx, 1)), base["p"][:, None], A_c, R_c)

    def rhs_1d(Ws):
        rho, ux, ut, p, _ = prim_from_W(Ws, A_c, R_c)
        P = np.stack([rho, ux, ut, p], axis=0)
        P_ext = np.empty((4, nx + 4, 1)); P_ext[:, 2:-2, :] = P
        r_in, u_in, ut_in, p_in = inlet_choked_ghost(rho[0], ux[0], p[0], M_c)
        for k in (0, 1):
            P_ext[0, k] = r_in; P_ext[1, k] = u_in; P_ext[2, k] = ut_in; P_ext[3, k] = p_in
        P_ext[:, -2] = P[:, -1]; P_ext[:, -1] = P[:, -1]
        s = van_leer_limiter(P_ext[:, 1:-1] - P_ext[:, :-2], P_ext[:, 2:] - P_ext[:, 1:-1])
        PL = P_ext[:, 1:-2] + 0.5 * s[:, :-1]; PR_ = P_ext[:, 2:-1] - 0.5 * s[:, 1:]
        Fx = hllc_2d_normal(
            np.maximum(PL[0], 1e-10), PL[1], PL[2], np.maximum(PL[3], 1e-10),
            np.maximum(PR_[0], 1e-10), PR_[1], PR_[2], np.maximum(PR_[3], 1e-10),
        )
        Fx_W = np.stack([A_f * Fx[0], A_f * Fx[1], A_f * R_f * Fx[2], A_f * Fx[3]], axis=0)
        if visc:
            u_g = P_ext[1, 1:-1, :]
            T_g = GAMMA * P_ext[3, 1:-1, :] / np.maximum(P_ext[0, 1:-1, :], 1e-10)
            du_dx = (u_g[1:] - u_g[:-1]) / dx
            dT_dx = (T_g[1:] - T_g[:-1]) / dx
            u_f = 0.5 * (u_g[1:] + u_g[:-1])
            tau_xx = (4.0 / 3.0) * mu * du_dx
            Fx_W[1] -= A_f * tau_xx
            Fx_W[3] -= A_f * (u_f * tau_xx + kap * dT_dx)
        res = -(Fx_W[:, 1:] - Fx_W[:, :-1]) / dx
        res[1] += p * dA_dx
        return res

    t = 0.0; last_rhs = 1.0
    while t < t_relax:
        rho, ux, _, p, c = prim_from_W(W, A_c, R_c)
        dt = 0.30 * dx / float(np.max(np.abs(ux) + c))
        if visc:
            dt = min(dt, 0.20 * dx * dx / max(nu_max, 1e-30))
        dt = min(dt, t_relax - t)
        k1 = rhs_1d(W); W1 = W + dt * k1
        k2 = rhs_1d(W1); W = 0.5 * (W + W1 + dt * k2)
        t += dt; last_rhs = float(np.max(np.abs(k1)))
        if t > 8.0 and last_rhs < 5e-13:
            break
    return W, last_rhs


def simulate_2d_viscous_transonic(
    nx: int = 80,
    nth: int = 200,
    t_end: float = 60.0,
    M_c: float = 0.12,
    L_n: float = 0.50,
    geom: str = "bell",
    Re: float = np.inf,
    n_p: float = 0.0,
    n_v: float = 0.0,
    n_baffles: int = 0,
    L_baffle: float = 0.0,
    baffle_bc: str = "slip",
    cav_spec: list[dict] | None = None,
    init_type: str = "spin_1T",
    A_eq: float = 1e-4,
    periodic_bump: bool = False,
    W_base: np.ndarray | None = None,
    cfl: float = 0.30,
    record_stride: int = 16,
) -> dict:
    """Coupled 2D conservative compressible NS + variable-(A(x), R(x)) transonic CD nozzle solver."""
    g = build_grid_geometry(nx, M_c, L_n, geom=geom, periodic_bump=periodic_bump)
    x, dx, L_tot = g["x"], g["dx"], g["L_tot"]
    A_c, A_f, R_c, R_f = g["A_c"], g["A_f"], g["R_c"], g["R_f"]
    dA_dx = (A_f[1:] - A_f[:-1]) / dx
    dR_dx = (R_f[1:] - R_f[:-1]) / dx
    dth = 2.0 * np.pi / nth
    th = (np.arange(nth) + 0.5) * dth
    X2, TH2 = np.meshgrid(x, th, indexing="ij")

    visc = np.isfinite(Re)
    mu = (1.0 / Re) if visc else 0.0
    kap = (mu / ((GAMMA - 1.0) * PR)) if visc else 0.0
    nu_max = mu * max(4.0 / 3.0, GAMMA / PR)
    min_R = float(np.min(R_c))
    dt_visc_lim = (0.20 / (nu_max * (1.0 / (dx * dx) + 1.0 / ((min_R * dth) ** 2)))) if visc else np.inf

    ch_mask = x <= L_A
    wf = np.where(ch_mask, np.exp(-0.5 * ((x - X_F) / SIG_F) ** 2), 0.0)
    wf /= float(np.sum(wf) * dx)
    wf_col = wf[:, None]

    baffle_mask = np.zeros((nx, nth), dtype=bool)
    if n_baffles > 0 and L_baffle > 0.0:
        nx_b = int(round(L_baffle / dx))
        blade_faces = [int(round(k * nth / n_baffles)) % nth for k in range(n_baffles)]
        for jf in blade_faces:
            baffle_mask[:nx_b, jf] = True
    bm_r = np.roll(baffle_mask, -1, axis=1)

    cavs = []
    if cav_spec:
        Cd = 0.65
        for cf in cav_spec:
            nc = int(cf.get("n_cav", 5))
            phase_off = float(cf.get("phase_offset", 0.5))
            beta_k = float(cf["beta"]); wH = float(cf["wH"]); xiH = float(cf["xiH"])
            sig_o = float(cf.get("sigma_o", 0.045)); nl_drag = bool(cf.get("nonlinear", True))
            V_k = beta_k * (2.0 * np.pi * L_A) / nc
            A_ok = (sig_o * 2.0 * np.pi) / nc
            r_nl = (1.0 / (2.0 * (Cd ** 2) * (A_ok ** 2))) if nl_drag else 0.0
            W_port = np.zeros((nc, nth))
            sig_th = max(1.5 * dth, 0.5 * (2.0 * np.pi * sig_o / nc))
            for k in range(nc):
                th_k = (k + phase_off) * (2.0 * np.pi / nc)
                d_ang = np.angle(np.exp(1j * (th - th_k)))
                wk = np.exp(-0.5 * (d_ang / sig_th) ** 2)
                W_port[k] = wk / float(np.sum(wk) * dth)
            cavs.append({"V_k": V_k, "wH": wH, "xiH": xiH, "r_nl": r_nl, "W": W_port, "Pc": np.zeros(nc), "qt": np.zeros(nc)})

    if periodic_bump:
        W_b = W_from_prim(np.full((nx, nth), RHO_BAR), np.full((nx, nth), M_c), np.zeros((nx, nth)), np.full((nx, nth), P_BAR), A_c, R_c)
    elif init_type == "quiescent":
        W_b = W_from_prim(np.full((nx, nth), RHO_BAR), np.zeros((nx, nth)), np.zeros((nx, nth)), np.full((nx, nth), P_BAR), A_c, R_c)
    elif W_base is not None:
        W_b = np.broadcast_to(W_base, (4, nx, nth)).copy()
    else:
        W1, _ = relax_1d_viscous_transonic_base(nx, M_c, L_n, geom=geom, Re=Re)
        W_b = np.broadcast_to(W1, (4, nx, nth)).copy()

    rho_b, ux_b, ut_b, p_b, _ = prim_from_W(W_b, A_c, R_c)
    rho = rho_b.copy(); ux = ux_b.copy(); ut = ut_b.copy(); p = p_b.copy()

    ch_env = np.where(X2 <= L_A, 1.0, 0.0)
    if init_type in ("quiescent", "steady"):
        pass
    elif init_type == "periodic_2d_wave":
        dp = A_eq * (np.cos(2.0 * np.pi * X2 / L_tot + TH2) + 0.4 * np.sin(2.0 * TH2))
        rho += dp; p += dp; ux += 0.3 * A_eq * np.sin(2.0 * np.pi * X2 / L_tot); ut += 0.5 * A_eq * np.cos(TH2)
    elif init_type == "spin_1T":
        dp = A_eq * np.cos(TH2) * ch_env; rho += dp; p += dp; ut += A_eq * np.cos(TH2) * ch_env
    elif init_type == "spin_2T":
        dp = A_eq * np.cos(2.0 * TH2) * ch_env; rho += dp; p += dp; ut += A_eq * np.cos(2.0 * TH2) * ch_env
    else:
        raise ValueError(f"Unknown init_type={init_type}")

    W = W_from_prim(rho, ux, ut, p, A_c, R_c)
    W0_init = W.copy()
    has_flame = (abs(n_p) > 0.0 or abs(n_v) > 0.0)
    fp = np.zeros((5, nth)); fv = np.zeros((5, nth))
    inv_ts = 4.0 / TAU_BAR
    init_totals = np.array([float(np.sum(W[k]) * dx * dth) for k in range(4)])

    def compute_rhs(Ws, fp_s, fv_s, cav_states):
        rho_s, ux_s, ut_s, p_s, _ = prim_from_W(Ws, A_c, R_c)
        P = np.stack([rho_s, ux_s, ut_s, p_s], axis=0)

        u_port = None
        if cavs:
            u_port = np.zeros(nth)
            for (_, qt_c), cf in zip(cav_states, cavs):
                u_port += qt_c @ cf["W"]

        P_ext = np.empty((4, nx + 4, nth))
        P_ext[:, 2:-2, :] = P
        p_in_face = p_s[0]
        if periodic_bump:
            P_ext[:, 0:2, :] = P[:, -2:, :]
            P_ext[:, -2:, :] = P[:, 0:2, :]
        elif init_type == "quiescent":
            for k in (0, 1):
                P_ext[:, k, :] = P[:, 0, :]; P_ext[1, k, :] = -P[1, 0, :]
                P_ext[:, -(k + 1), :] = P[:, -1, :]; P_ext[1, -(k + 1), :] = -P[1, -1, :]
        else:
            r_in, u_in, ut_in, p_in = inlet_choked_ghost(rho_s[0], ux_s[0], p_s[0], M_c, u_port=u_port)
            p_in_face = p_in
            for k in (0, 1):
                P_ext[0, k] = r_in; P_ext[1, k] = u_in; P_ext[2, k] = ut_in; P_ext[3, k] = p_in
            P_ext[:, -2, :] = P[:, -1, :]; P_ext[:, -1, :] = P[:, -1, :]

        # Inviscid x-fluxes
        sx = van_leer_limiter(P_ext[:, 1:-1, :] - P_ext[:, :-2, :], P_ext[:, 2:, :] - P_ext[:, 1:-1, :])
        PLx = P_ext[:, 1:-2, :] + 0.5 * sx[:, :-1, :]; PRx = P_ext[:, 2:-1, :] - 0.5 * sx[:, 1:, :]
        Fx = hllc_2d_normal(
            np.maximum(PLx[0], 1e-10), PLx[1], PLx[2], np.maximum(PLx[3], 1e-10),
            np.maximum(PRx[0], 1e-10), PRx[1], PRx[2], np.maximum(PRx[3], 1e-10),
        )
        Fx_W = np.stack([A_f * Fx[0], A_f * Fx[1], A_f * R_f * Fx[2], A_f * Fx[3]], axis=0)

        # Inviscid theta-fluxes (split-face reflecting baffles)
        P_th = np.concatenate([P[:, :, -2:], P, P[:, :, :2]], axis=2)
        dL = P_th[:, :, 1:-1] - P_th[:, :, :-2]; dR = P_th[:, :, 2:] - P_th[:, :, 1:-1]
        if n_baffles > 0 and L_baffle > 0.0:
            dL[:, :, 1:-1] = np.where(baffle_mask[None, :, :], 0.0, dL[:, :, 1:-1])
            dR[:, :, 1:-1] = np.where(bm_r[None, :, :], 0.0, dR[:, :, 1:-1])
        st = van_leer_limiter(dL, dR)
        PLt = P_th[:, :, 1:-2] + 0.5 * st[:, :, :-1]; PRt = P_th[:, :, 2:-1] - 0.5 * st[:, :, 1:]
        Gt_n = hllc_2d_normal(
            np.maximum(PLt[0], 1e-10), PLt[2], PLt[1], np.maximum(PLt[3], 1e-10),
            np.maximum(PRt[0], 1e-10), PRt[2], PRt[1], np.maximum(PRt[3], 1e-10),
        )
        AR = A_c / R_c
        Gt_W = np.stack([AR * Gt_n[0], AR * Gt_n[2], A_c * Gt_n[1], AR * Gt_n[3]], axis=0)

        if n_baffles > 0 and L_baffle > 0.0:
            F_wL = hllc_2d_normal(
                np.maximum(PLt[0, :, :-1], 1e-10), PLt[2, :, :-1], PLt[1, :, :-1], np.maximum(PLt[3, :, :-1], 1e-10),
                np.maximum(PLt[0, :, :-1], 1e-10), -PLt[2, :, :-1], PLt[1, :, :-1], np.maximum(PLt[3, :, :-1], 1e-10),
            )
            F_wR = hllc_2d_normal(
                np.maximum(PRt[0, :, :-1], 1e-10), -PRt[2, :, :-1], PRt[1, :, :-1], np.maximum(PRt[3, :, :-1], 1e-10),
                np.maximum(PRt[0, :, :-1], 1e-10), PRt[2, :, :-1], PRt[1, :, :-1], np.maximum(PRt[3, :, :-1], 1e-10),
            )
            GwL = np.stack([AR * F_wL[0], AR * F_wL[2], A_c * F_wL[1], AR * F_wL[3]], axis=0)
            GwR = np.stack([AR * F_wR[0], AR * F_wR[2], A_c * F_wR[1], AR * F_wR[3]], axis=0)
            G_left = np.where(baffle_mask[None, :, :], GwR, Gt_W[:, :, :-1])
            G_right = np.where(bm_r[None, :, :], np.roll(GwL, -1, axis=2), Gt_W[:, :, 1:])
        else:
            G_left = Gt_W[:, :, :-1]
            G_right = Gt_W[:, :, 1:]

        # Conservative face-centered viscous & Fourier heat fluxes (exact mass/ang-mom/energy conservation!)
        if visc:
            ux_g = P_ext[1, 1:-1, :]; ut_g = P_ext[2, 1:-1, :]
            T_g = GAMMA * P_ext[3, 1:-1, :] / np.maximum(P_ext[0, 1:-1, :], 1e-10)
            # x-faces (nx + 1, nth)
            dux_dx_f = (ux_g[1:] - ux_g[:-1]) / dx
            dut_dx_f = (ut_g[1:] - ut_g[:-1]) / dx
            dT_dx_f = (T_g[1:] - T_g[:-1]) / dx
            ux_f = 0.5 * (ux_g[1:] + ux_g[:-1]); ut_f = 0.5 * (ut_g[1:] + ut_g[:-1])
            dux_dth_f = (np.roll(ux_f, -1, axis=1) - np.roll(ux_f, 1, axis=1)) / (2.0 * dth)
            dut_dth_f = (np.roll(ut_f, -1, axis=1) - np.roll(ut_f, 1, axis=1)) / (2.0 * dth)
            tau_xx_f = mu * ((4.0 / 3.0) * dux_dx_f - (2.0 / 3.0) * dut_dth_f / R_f)
            tau_xt_f = mu * (dux_dth_f / R_f + dut_dx_f)
            Fx_W[1] -= A_f * tau_xx_f
            Fx_W[2] -= A_f * R_f * tau_xt_f
            Fx_W[3] -= A_f * (ux_f * tau_xx_f + ut_f * tau_xt_f + kap * dT_dx_f)

            # theta-faces (nx, nth): face j is between cell j-1 (left) and cell j (right)
            T_c = GAMMA * p_s / np.maximum(rho_s, 1e-10)
            ux_L = np.roll(ux_s, 1, axis=1); ut_L = np.roll(ut_s, 1, axis=1); T_L = np.roll(T_c, 1, axis=1)
            dux_dth_t = (ux_s - ux_L) / dth
            dut_dth_t = (ut_s - ut_L) / dth
            dT_dth_t = (T_c - T_L) / dth
            ux_tf = 0.5 * (ux_s + ux_L); ut_tf = 0.5 * (ut_s + ut_L)
            dux_dx_c = (ux_g[2:] - ux_g[:-2]) / (2.0 * dx)
            dut_dx_c = (ut_g[2:] - ut_g[:-2]) / (2.0 * dx)
            dux_dx_tf = 0.5 * (dux_dx_c + np.roll(dux_dx_c, 1, axis=1))
            dut_dx_tf = 0.5 * (dut_dx_c + np.roll(dut_dx_c, 1, axis=1))
            tau_xt_t = mu * (dux_dth_t / R_c + dut_dx_tf)
            tau_tt_t = mu * ((4.0 / 3.0) * dut_dth_t / R_c - (2.0 / 3.0) * dux_dx_tf)

            Gv_L = np.zeros((4, nx, nth))
            Gv_L[1] = AR * tau_xt_t
            Gv_L[2] = A_c * tau_tt_t
            Gv_L[3] = AR * (ux_tf * tau_xt_t + ut_tf * tau_tt_t + kap * dT_dth_t / R_c)
            Gv_R = np.roll(Gv_L, -1, axis=2)

            if n_baffles > 0 and L_baffle > 0.0:
                # Solid adiabatic impermeable baffle blades (q_th = 0, ut = 0 => zero viscous energy flux!)
                if baffle_bc == "noslip":
                    # Right side of blade (cell j, left face = baffle_mask): ux_wall = 0 => dux/dth = +2*ux_s/dth
                    # Left side of blade (cell j, right face = bm_r): ux_wall = 0 => dux/dth = -2*ux_s/dth
                    tau_wR = mu * (2.0 * ux_s / (R_c * dth))
                    tau_wL = mu * (-2.0 * ux_s / (R_c * dth))
                else:
                    tau_wR = np.zeros_like(ux_s); tau_wL = np.zeros_like(ux_s)
                Gv_L[1] = np.where(baffle_mask, AR * tau_wR, Gv_L[1])
                Gv_L[2] = np.where(baffle_mask, 0.0, Gv_L[2])
                Gv_L[3] = np.where(baffle_mask, 0.0, Gv_L[3])
                Gv_R[1] = np.where(bm_r, AR * tau_wL, Gv_R[1])
                Gv_R[2] = np.where(bm_r, 0.0, Gv_R[2])
                Gv_R[3] = np.where(bm_r, 0.0, Gv_R[3])

            G_left = G_left - Gv_L
            G_right = G_right - Gv_R

        rhs_W = -(Fx_W[:, 1:, :] - Fx_W[:, :-1, :]) / dx - (G_right - G_left) / dth
        rhs_W[1] += p_s * dA_dx + (A_c * rho_s * ut_s * ut_s / R_c) * dR_dx

        dfp = np.zeros_like(fp_s); dfv = np.zeros_like(fv_s)
        if has_flame:
            p_sense = np.sum(wf_col * (p_s - p_b), axis=0) * dx
            u_sense = np.sum(wf_col * ut_s, axis=0) * dx
            dfp[0] = OMEGA_HP * (p_sense - fp_s[0])
            dfp[1] = inv_ts * ((p_sense - fp_s[0]) - fp_s[1])
            for r in range(1, 4): dfp[r + 1] = inv_ts * (fp_s[r] - fp_s[r + 1])
            dfv[0] = OMEGA_HP * (u_sense - fv_s[0])
            dfv[1] = inv_ts * ((u_sense - fv_s[0]) - fv_s[1])
            for r in range(1, 4): dfv[r + 1] = inv_ts * (fv_s[r] - fv_s[r + 1])
            q_eff = n_p * phi_flame(fp_s[4]) + n_v * phi_flame(fv_s[4])
            rhs_W[3] += A_c * (wf_col * q_eff[None, :]) / (GAMMA - 1.0)

        dcav = []
        if cavs:
            p_face = p_in_face - p_b[0]
            for cf, (Pc_c, qt_c) in zip(cavs, cav_states):
                wH = cf["wH"]; xiH = cf["xiH"]; r_nl = cf["r_nl"]; Vk = cf["V_k"]
                p_bar_k = (cf["W"] @ p_face) * dth
                dPc = qt_c / Vk
                dqt = (wH * wH) * Vk * (p_bar_k - Pc_c - r_nl * np.abs(qt_c) * qt_c) - 2.0 * xiH * wH * qt_c
                dcav.append((dPc, dqt))

        return rhs_W, dfp, dfv, dcav

    t = 0.0; step = 0
    ts, e_ch, e_1t, e_2t, p_max_h, p1_re, p2_re = [], [], [], [], [], [], []
    w_1T = np.exp(-1j * th); w_2T = np.exp(-2j * th)
    max_cons_err = np.zeros(4); last_rhs_max = 0.0

    while t < t_end - 1e-12:
        rho_c, ux_c, ut_c, p_c, c_c = prim_from_W(W, A_c, R_c)
        lam_x = float(np.max(np.abs(ux_c) + c_c)) / dx
        lam_t = float(np.max((np.abs(ut_c) + c_c) / R_c)) / dth
        dt = min(cfl / (lam_x + lam_t), dt_visc_lim, t_end - t)

        if step % record_stride == 0:
            dp_ch = (p_c - p_b)[ch_mask]; dux_ch = (ux_c - ux_b)[ch_mask]; dut_ch = ut_c[ch_mask]
            ts.append(t)
            e_ch.append(0.5 * float(np.mean(dp_ch * dp_ch + dux_ch * dux_ch + dut_ch * dut_ch)))
            pm1 = np.mean(dp_ch * w_1T[None, :], axis=1)
            pm2 = np.mean(dp_ch * w_2T[None, :], axis=1)
            e_1t.append(float(np.mean(np.abs(pm1) ** 2)))
            e_2t.append(float(np.mean(np.abs(pm2) ** 2)))
            p1_re.append(float(np.real(pm1[-1])))
            p2_re.append(float(np.real(pm2[-1])))
            p_max_h.append(float(np.max(np.abs(dp_ch)) / P_BAR * 100.0))
            if periodic_bump:
                cur_tot = np.array([float(np.sum(W[k]) * dx * dth) for k in range(4)])
                max_cons_err = np.maximum(max_cons_err, np.abs(cur_tot - init_totals) / np.maximum(np.abs(init_totals), 1.0))

        c0 = [(cf["Pc"], cf["qt"]) for cf in cavs]
        k1W, k1p, k1v, k1c = compute_rhs(W, fp, fv, c0)
        last_rhs_max = float(np.max(np.abs(k1W)))
        W1 = W + dt * k1W; fp1 = fp + dt * k1p; fv1 = fv + dt * k1v
        c1 = [(s0 + dt * ds1, v0 + dt * dv1) for (s0, v0), (ds1, dv1) in zip(c0, k1c)]

        k2W, k2p, k2v, k2c = compute_rhs(W1, fp1, fv1, c1)
        W = 0.5 * (W + W1 + dt * k2W)
        fp = 0.5 * (fp + fp1 + dt * k2p)
        fv = 0.5 * (fv + fv1 + dt * k2v)
        for cf, (s0, v0), (s1, v1), (ds2, dv2) in zip(cavs, c0, c1, k2c):
            cf["Pc"] = 0.5 * (s0 + s1 + dt * ds2); cf["qt"] = 0.5 * (v0 + v1 + dt * dv2)
        t += dt; step += 1

    rho_f, ux_f, ut_f, p_f, _ = prim_from_W(W, A_c, R_c)
    dp_ch = (p_f - p_b)[ch_mask]; dux_ch = (ux_f - ux_b)[ch_mask]; dut_ch = ut_f[ch_mask]
    ts.append(t)
    e_ch.append(0.5 * float(np.mean(dp_ch * dp_ch + dux_ch * dux_ch + dut_ch * dut_ch)))
    pm1 = np.mean(dp_ch * w_1T[None, :], axis=1); pm2 = np.mean(dp_ch * w_2T[None, :], axis=1)
    e_1t.append(float(np.mean(np.abs(pm1) ** 2))); e_2t.append(float(np.mean(np.abs(pm2) ** 2)))
    p1_re.append(float(np.real(pm1[-1]))); p2_re.append(float(np.real(pm2[-1])))
    p_max_h.append(float(np.max(np.abs(dp_ch)) / P_BAR * 100.0))

    return {
        "t": np.array(ts), "e_ch": np.array(e_ch), "e_1t": np.array(e_1t), "e_2t": np.array(e_2t),
        "p1_re": np.array(p1_re), "p2_re": np.array(p2_re), "p_max_pct": np.array(p_max_h),
        "quiescent_err": float(np.max(np.abs(W - W0_init))),
        "periodic_cons_err": {"mass": float(max_cons_err[0]), "mom_x": float(max_cons_err[1]),
                              "ang_mom": float(max_cons_err[2]), "energy": float(max_cons_err[3])},
        "final_rhs_max": last_rhs_max,
        "W_final": W,
    }


def _sim_worker(args):
    lbl, kw = args
    return lbl, simulate_2d_viscous_transonic(**kw)


def run_batch(tasks: list[tuple[str, dict]], max_workers: int | None = None) -> dict:
    nw = max_workers or min(6, max(1, (os.cpu_count() or 2) - 2), len(tasks))
    if nw <= 1 or len(tasks) <= 1:
        return dict(_sim_worker(t) for t in tasks)
    with ProcessPoolExecutor(max_workers=nw) as ex:
        return dict(ex.map(_sim_worker, tasks))


def measure_fft_omega(t: np.ndarray, sig: np.ndarray, t_lo: float = 16.0) -> float:
    m = t >= t_lo
    tu = np.linspace(t[m][0], t[m][-1], 1024)
    su = np.interp(tu, t[m], sig[m])
    su = su - np.mean(su)
    env = np.maximum(np.sqrt(np.convolve(su * su, np.ones(41) / 41.0, mode="same")), 1e-30)
    su_n = (su / env) * np.hanning(len(su))
    nfft = 32768
    freqs = np.fft.rfftfreq(nfft, d=(tu[1] - tu[0])) * 2.0 * np.pi
    spec = np.abs(np.fft.rfft(su_n, n=nfft))
    mask_f = (freqs >= 0.4) & (freqs <= 4.0)
    return float(freqs[mask_f][np.argmax(spec[mask_f])])


def fit_alpha(res: dict, t_lo: float = 18.0, t_hi: float = 48.0, key: str = "e_ch") -> float:
    t = res["t"]; m = (t >= t_lo) & (t <= t_hi)
    return float(np.polyfit(t[m], 0.5 * np.log(np.maximum(res[key][m], 1e-30)), 1)[0])


def stats_nl(res: dict, win_lo: float = 18.0, win_hi: float = 55.0) -> dict:
    t = res["t"]; e = res["e_ch"]; pm = res["p_max_pct"]
    t_end = float(t[-1])
    m_late = t >= 0.80 * t_end
    m_mid = (t >= 0.50 * t_end) & (t <= 0.70 * t_end)
    e_late = float(np.mean(e[m_late])); e_mid = float(np.mean(e[m_mid]))
    m_win = (t >= win_lo) & (t <= min(win_hi, t_end))
    a_win = float(np.polyfit(t[m_win], 0.5 * np.log(np.maximum(e[m_win], 1e-30)), 1)[0]) if np.sum(m_win) >= 4 else 0.0
    m_tail = t >= 0.75 * t_end
    a_tail = float(np.polyfit(t[m_tail], 0.5 * np.log(np.maximum(e[m_tail], 1e-30)), 1)[0]) if np.sum(m_tail) >= 4 else 0.0
    e1_l = float(np.mean(res["e_1t"][m_late])); e2_l = float(np.mean(res["e_2t"][m_late]))
    return {
        "t_end": t_end, "p_over_pbar_pct": float(np.mean(pm[m_late])),
        "e_wave_late": e_late, "e_wave_late_over_mid": e_late / max(e_mid, 1e-30),
        "e2_over_e1_late": e2_l / max(e1_l, 1e-30), "alpha_fit_win": a_win, "alpha_tail": a_tail,
    }


def make_hyb(w1: float = 0.96, w2: float = 1.65, frac1: float = 0.80, sig1: float = 0.045, sig2: float = 0.050, beta_tot: float = 0.020):
    return [
        {"beta": beta_tot * frac1, "wH": w1, "xiH": 0.12, "sigma_o": sig1, "n_cav": 5, "phase_offset": 0.5, "nonlinear": True},
        {"beta": beta_tot * (1.0 - frac1), "wH": w2, "xiH": 0.12, "sigma_o": sig2, "n_cav": 5, "phase_offset": 0.5, "nonlinear": True},
    ]


def plot_fig1(metrics: dict) -> None:
    visc_bell_hopf = metrics["exp2_viscous_bell_rescue_hopf"]
    baffle_downshift = metrics["exp1_coupled_benchmarks"]["baffle_downshift_fft"]
    w2_grid = metrics["exp3_dual_retuning_curve"]["w2_grid"]
    exp3_stats = metrics["exp3_dual_retuning_curve"]["runs"]
    w2_bare = baffle_downshift["omega_2T_bare"]
    w2_baf = baffle_downshift["omega_2T_baffled"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.6), dpi=150)

    M_a = [d["M_c"] for d in visc_bell_hopf["inf"]]
    lift_2000 = abs(visc_bell_hopf["2000"][1]["cold_2T"]) / max(abs(visc_bell_hopf["inf"][1]["cold_2T"]), 1e-12)
    for rtag, lbl, col, ls in [
        ("inf",  "Inviscid Bell ($\\mathrm{Re}=\\infty$, $2T$ cutoff collapse)", "#d62728", "-"),
        ("2000", f"Viscous Bell ($\\mathrm{{Re}}=2000$, $2T$ lifted ${lift_2000:.1f}\\times$)", "#ff7f0e", "--"),
        ("1000", "Viscous Bell ($\\mathrm{Re}=1000$, $2T$ rescued)", "#1f77b4", "-."),
        ("500",  "Viscous Bell ($\\mathrm{Re}=500$, $1T$-limited everywhere)", "#2ca02c", "-"),
    ]:
        ax1.plot(M_a, [d["n_pH_global"] for d in visc_bell_hopf[rtag]], "o", ls=ls, color=col, lw=2.3, label=lbl)
    ax1.axhline(0.22, color="#555555", ls=":", lw=1.4, label="Benchmark flame $n_p = 0.22$")
    ax1.set_xlabel("Chamber Mean Mach Number $\\bar{M}_c$", fontsize=11)
    ax1.set_ylabel("Multi-Mode Global Hopf Gain $n_{pH,\\mathrm{global}}$ ($m=0..5$)", fontsize=11)
    ax1.set_title("Viscous Rescue of Bell-Nozzle $2T$ Evanescent Collapse ($L_n=0.50$)", fontsize=12, fontweight="bold")
    ax1.set_ylim(-0.01, 0.45)
    ax1.grid(True, alpha=0.28)
    ax1.legend(fontsize=8.2, loc="upper left", framealpha=0.94)

    lin_damps = [-exp3_stats[f"lin2T_w{w2:.2f}"]["alpha_fit_win"] for w2 in w2_grid]
    nl_peaks = [exp3_stats[f"NL2T_w{w2:.2f}"]["p_over_pbar_pct"] for w2 in w2_grid]
    ax2_r = ax2.twinx()
    l1, = ax2.plot(w2_grid, lin_damps, "o-", color="#1f77b4", lw=2.4, label="Linear $2T$ damping $-\\alpha_{2T}$ ($A_0=10^{-4}$)")
    l2, = ax2_r.plot(w2_grid, nl_peaks, "s--", color="#d62728", lw=2.2, label="Direct $2T$ shock $P_{\\mathrm{peak}}/\\bar{p}$ (%)")
    ax2.axvline(w2_bare, color="#777777", ls=":", lw=1.5, label=f"Unbaffled bare $\\omega_{{2T}} = {w2_bare:.2f}$ (detuned!)")
    ax2.axvline(w2_baf, color="#2ca02c", ls="-.", lw=1.7, label=f"Baffled FFT $\\omega_{{2T}} = {w2_baf:.2f}$ ({baffle_downshift['shift_2T_pct']:+.1f}%)")
    ax2.set_xlabel("$2T$ Cavity Tuning Frequency $\\omega_{H,2T}$ (Fixed $2\\%$ $80/20$ Budget)", fontsize=11)
    ax2.set_ylabel("Linear $2T$ Damping Rate $-\\alpha_{2T}$", fontsize=11, color="#1f77b4")
    ax2_r.set_ylabel("Direct $6.9\\%$ $2T$ Bomb Limit Cycle (%)", fontsize=11, color="#d62728")
    ax2.set_ylim(0.010, 0.044)
    ax2_r.set_ylim(4.05, 5.75)
    ax2.set_title("Higher-Harmonic Baffle-Cavity Retuning Law ($N_b=5, L_b=0.45$)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.28)
    lines = [l1, l2] + ax2.get_lines()[1:]
    ax2.legend(lines, [ln.get_label() for ln in lines], fontsize=8.0, loc="upper right", framealpha=0.94)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp013_viscous_bell_rescue_and_2T_retuning.png", dpi=150)
    plt.close(fig)


def run_exp5_viscous_1t_backreaction() -> tuple[dict, dict]:
    """Experiment 5: Viscous Bell 1T Back-Reaction Mechanism & Resolved 120x300 All-Mode Quench."""
    print("\n--- Experiment 5: Viscous Bell 1T Back-Reaction & Resolved 120x300 3.0% 85/15 Quench ---")
    W80_inf, _ = relax_1d_viscous_transonic_base(80, 0.12, 0.50, "bell", np.inf)
    W80_1000, _ = relax_1d_viscous_transonic_base(80, 0.12, 0.50, "bell", 1000.0)
    W80_500, _ = relax_1d_viscous_transonic_base(80, 0.12, 0.50, "bell", 500.0)
    W80_c_inf, _ = relax_1d_viscous_transonic_base(80, 0.12, 0.05, "annular", np.inf)
    W80_c_1000, _ = relax_1d_viscous_transonic_base(80, 0.12, 0.05, "annular", 1000.0)
    W120_inf, _ = relax_1d_viscous_transonic_base(120, 0.12, 0.50, "bell", np.inf)
    W120_1000, _ = relax_1d_viscous_transonic_base(120, 0.12, 0.50, "bell", 1000.0)

    cf_20 = make_hyb(w1=0.96, w2=1.65, frac1=0.80, sig1=0.045, sig2=0.050, beta_tot=0.020)
    cf_30 = make_hyb(w1=0.96, w2=1.65, frac1=0.85, sig1=0.060, sig2=0.050, beta_tot=0.030)

    lin_tasks = [
        ("cold_baf_bell_inf",  dict(nx=80, nth=200, t_end=50.0, M_c=0.12, L_n=0.50, geom="bell", Re=np.inf, n_baffles=5, L_baffle=0.45, init_type="spin_1T", A_eq=1e-4, W_base=W80_inf, record_stride=8)),
        ("cold_baf_bell_1000", dict(nx=80, nth=200, t_end=50.0, M_c=0.12, L_n=0.50, geom="bell", Re=1000.0, n_baffles=5, L_baffle=0.45, init_type="spin_1T", A_eq=1e-4, W_base=W80_1000, record_stride=8)),
        ("cold_baf_bell_500",  dict(nx=80, nth=200, t_end=50.0, M_c=0.12, L_n=0.50, geom="bell", Re=500.0,  n_baffles=5, L_baffle=0.45, init_type="spin_1T", A_eq=1e-4, W_base=W80_500, record_stride=8)),
        ("react_baf_bell_inf_80",   dict(nx=80, nth=200, t_end=50.0, M_c=0.12, L_n=0.50, geom="bell", Re=np.inf, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, init_type="spin_1T", A_eq=1e-4, W_base=W80_inf, record_stride=8)),
        ("react_baf_bell_1000_80",  dict(nx=80, nth=200, t_end=50.0, M_c=0.12, L_n=0.50, geom="bell", Re=1000.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, init_type="spin_1T", A_eq=1e-4, W_base=W80_1000, record_stride=8)),
        ("react_baf_comp_inf_80",   dict(nx=80, nth=200, t_end=50.0, M_c=0.12, L_n=0.05, geom="annular", Re=np.inf, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, init_type="spin_1T", A_eq=1e-4, W_base=W80_c_inf, record_stride=8)),
        ("react_baf_comp_1000_80",  dict(nx=80, nth=200, t_end=50.0, M_c=0.12, L_n=0.05, geom="annular", Re=1000.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, init_type="spin_1T", A_eq=1e-4, W_base=W80_c_1000, record_stride=8)),
        ("hyb20_bell_inf_80",       dict(nx=80, nth=200, t_end=50.0, M_c=0.12, L_n=0.50, geom="bell", Re=np.inf, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_20, init_type="spin_1T", A_eq=1e-4, W_base=W80_inf, record_stride=8)),
        ("hyb20_bell_1000_80",      dict(nx=80, nth=200, t_end=50.0, M_c=0.12, L_n=0.50, geom="bell", Re=1000.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_20, init_type="spin_1T", A_eq=1e-4, W_base=W80_1000, record_stride=8)),
        ("hyb20_bell_500_80",       dict(nx=80, nth=200, t_end=50.0, M_c=0.12, L_n=0.50, geom="bell", Re=500.0,  n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_20, init_type="spin_1T", A_eq=1e-4, W_base=W80_500, record_stride=8)),
        ("react_baf_bell_inf_120",  dict(nx=120, nth=300, t_end=50.0, M_c=0.12, L_n=0.50, geom="bell", Re=np.inf, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, init_type="spin_1T", A_eq=1e-4, W_base=W120_inf, record_stride=12)),
        ("react_baf_bell_1000_120", dict(nx=120, nth=300, t_end=50.0, M_c=0.12, L_n=0.50, geom="bell", Re=1000.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, init_type="spin_1T", A_eq=1e-4, W_base=W120_1000, record_stride=12)),
    ]
    lin_res = run_batch(lin_tasks)
    lin_out = {}
    for cid, _ in lin_tasks:
        a = fit_alpha(lin_res[cid], 20.0, 48.0, key="e_1t")
        w = measure_fft_omega(lin_res[cid]["t"], lin_res[cid]["p1_re"])
        lin_out[cid] = {"alpha_1T": a, "omega_1T": w}
        print(f"  [{cid:25s}] alpha_1T={a:+.5f}, omega_1T={w:.4f}")

    # Resolved 120x300 verification of 3.0% 85/15 dual-retuned all-mode quench at Re=1000 + 80x200 2.0/2.5/3.0 comparison
    cf_25 = make_hyb(w1=0.96, w2=1.65, frac1=0.85, sig1=0.055, sig2=0.050, beta_tot=0.025)
    nl_tasks = [
        ("R80_Re1000_20_8020_1T",  dict(nx=80, nth=200, t_end=200.0, M_c=0.12, L_n=0.50, geom="bell", Re=1000.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_20, init_type="spin_1T", A_eq=0.055, W_base=W80_1000, record_stride=24)),
        ("R80_Re1000_25_8515_1T",  dict(nx=80, nth=200, t_end=200.0, M_c=0.12, L_n=0.50, geom="bell", Re=1000.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_25, init_type="spin_1T", A_eq=0.055, W_base=W80_1000, record_stride=24)),
        ("R80_Re1000_30_8515_1T",  dict(nx=80, nth=200, t_end=200.0, M_c=0.12, L_n=0.50, geom="bell", Re=1000.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_30, init_type="spin_1T", A_eq=0.055, W_base=W80_1000, record_stride=24)),
        ("R120_Re1000_30_8515_1T", dict(nx=120, nth=300, t_end=240.0, M_c=0.12, L_n=0.50, geom="bell", Re=1000.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_30, init_type="spin_1T", A_eq=0.055, W_base=W120_1000, record_stride=32)),
        ("R120_Re1000_30_8515_2T", dict(nx=120, nth=300, t_end=240.0, M_c=0.12, L_n=0.50, geom="bell", Re=1000.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_30, init_type="spin_2T", A_eq=0.055, W_base=W120_1000, record_stride=32)),
    ]
    nl_res = run_batch(nl_tasks)
    nl_out = {}
    for cid, _ in nl_tasks:
        st = stats_nl(nl_res[cid])
        nl_out[cid] = st
        print(f"  [{cid:23s}] t={st['t_end']:4.0f} | P={st['p_over_pbar_pct']:5.2f}% | E={st['e_wave_late']:.2e} (e2/e1={st['e2_over_e1_late']:.2f}) | tail={st['alpha_tail']:+.4f}")

    # Figure 3: Viscous Bell 1T Back-Reaction & Resolved 120x300 All-Mode Quench
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.5), dpi=150)

    labels = ["Cold Bell\n(Baffled)", "Reacting Bell\n(Baffled, 120x300)", "Reacting Compact\n(Baffled, Ln=0.05)", "Reacting Bell\n(Baffle+2% Hyb)"]
    vals_inf = [
        -lin_out["cold_baf_bell_inf"]["alpha_1T"],
        -lin_out["react_baf_bell_inf_120"]["alpha_1T"],
        -lin_out["react_baf_comp_inf_80"]["alpha_1T"],
        -lin_out["hyb20_bell_inf_80"]["alpha_1T"],
    ]
    vals_1000 = [
        -lin_out["cold_baf_bell_1000"]["alpha_1T"],
        -lin_out["react_baf_bell_1000_120"]["alpha_1T"],
        -lin_out["react_baf_comp_1000_80"]["alpha_1T"],
        -lin_out["hyb20_bell_1000_80"]["alpha_1T"],
    ]
    x_pos = np.arange(len(labels))
    w = 0.35
    ax1.bar(x_pos - w / 2, vals_inf, w, color="#1f77b4", label="Inviscid ($\\mathrm{Re}=\\infty$)")
    ax1.bar(x_pos + w / 2, vals_1000, w, color="#d62728", label="Viscous ($\\mathrm{Re}=1000$)")
    ax1.axhline(0.0, color="#333333", lw=1.2)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=9.2)
    ax1.set_ylabel("Linear $1T$ Damping Margin $-\\alpha_{1T}$ (>0 stable)", fontsize=11)
    ax1.set_title("Viscous Bell $1T$ Back-Reaction vs. Compact Nozzle", fontsize=12, fontweight="bold")
    ax1.set_ylim(-0.015, 0.112)
    ax1.grid(True, axis="y", alpha=0.28)
    ax1.legend(fontsize=9.2, loc="upper left", framealpha=0.94)

    for cid, lbl, col, ls in [
        ("R80_Re1000_20_8020_1T",  "Re=1000, 2.0% 80/20: 1T bomb (80x200)", "#d62728", "--"),
        ("R80_Re1000_25_8515_1T",  "Re=1000, 2.5% 85/15: 1T bomb (80x200)", "#ff7f0e", "-."),
        ("R120_Re1000_30_8515_1T", "Re=1000, 3.0% 85/15: 1T bomb (120x300)", "#1f77b4", "-"),
        ("R120_Re1000_30_8515_2T", "Re=1000, 3.0% 85/15: 2T bomb (120x300)", "#2ca02c", "-"),
    ]:
        rr = nl_res[cid]; st = nl_out[cid]
        ax2.semilogy(rr["t"], np.maximum(rr["e_ch"], 1e-13), color=col, ls=ls, lw=2.3,
                     label=f"{lbl} (P={st['p_over_pbar_pct']:.2f}%)")
    ax2.set_xlabel("Time $t$", fontsize=11)
    ax2.set_ylabel("Chamber Wave Energy $E_{\\mathrm{wave}}(t)$", fontsize=11)
    ax2.set_title("Resolved 120x300 All-Mode Quench at $\\mathrm{Re}=1000$ ($3.0\\%$ $85/15$)", fontsize=12, fontweight="bold")
    ax2.set_ylim(1e-12, 1e-2)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=8.3, loc="lower left", framealpha=0.94)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp013_viscous_1T_backreaction_and_quench.png", dpi=150)
    plt.close(fig)
    print("Saved plot -> exp013_viscous_1T_backreaction_and_quench.png")
    return {"linear_backreaction": lin_out, "resolved_quench_3pct": nl_out}, nl_res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-exp5", action="store_true", help="Run Exp 5 only, merge into existing exp013_metrics.json, and refresh Fig 1 & Fig 3.")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp013_metrics.json"
    if args.only_exp5:
        metrics = json.loads(out_path.read_text(encoding="utf-8"))
        exp5_data, _ = run_exp5_viscous_1t_backreaction()
        metrics["exp5_viscous_1T_backreaction_and_quench"] = exp5_data
        out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        plot_fig1(metrics)
        print(f"Updated metrics -> {out_path} and refreshed Fig 1 & Fig 3.")
        return

    print("=" * 92)
    print("Entry 013: Dual-Harmonic Baffle-Cavity Retuning & Coupled 2D NS Transonic Bell Nozzle")
    print("=" * 92)

    # ------------------------------------------------------------------
    # Experiment 1: Coupled Viscous-Transonic Benchmarks & Higher-Harmonic Baffle Downshift
    # ------------------------------------------------------------------
    print("\n--- Experiment 1: Coupled Viscous-Transonic Benchmarks & Baffle Downshift FFT ---")
    W80_inf, _ = relax_1d_viscous_transonic_base(80, 0.12, 0.50, "bell", np.inf)
    W96_500, rhs_96_500 = relax_1d_viscous_transonic_base(96, 0.12, 0.50, "bell", 500.0)
    W96_1000, rhs_96_1000 = relax_1d_viscous_transonic_base(96, 0.12, 0.50, "bell", 1000.0)
    W120_inf, _ = relax_1d_viscous_transonic_base(120, 0.12, 0.50, "bell", np.inf)
    W120_1000, _ = relax_1d_viscous_transonic_base(120, 0.12, 0.50, "bell", 1000.0)
    W120_500, _ = relax_1d_viscous_transonic_base(120, 0.12, 0.50, "bell", 500.0)

    exp1_tasks = [
        ("quiescent_bell_Re500", dict(nx=48, nth=64, t_end=10.0, M_c=0.12, L_n=0.50, geom="bell", Re=500.0, init_type="quiescent", A_eq=0.0, record_stride=20)),
        ("cons_bell_Re200",      dict(nx=48, nth=96, t_end=8.0, M_c=0.12, L_n=0.50, geom="bell", Re=200.0, init_type="periodic_2d_wave", A_eq=0.10, periodic_bump=True, record_stride=20)),
        # Viscous transonic Chebyshev vs 96x200 PDE verification (1T and 2T, Re = 500 & 1000)
        ("pde_bell_Re500_1T",    dict(nx=96, nth=200, t_end=55.0, M_c=0.12, L_n=0.50, geom="bell", Re=500.0, init_type="spin_1T", A_eq=1e-4, W_base=W96_500, record_stride=12)),
        ("pde_bell_Re1000_1T",   dict(nx=96, nth=200, t_end=55.0, M_c=0.12, L_n=0.50, geom="bell", Re=1000.0, init_type="spin_1T", A_eq=1e-4, W_base=W96_1000, record_stride=12)),
        ("pde_bell_Re500_2T",    dict(nx=96, nth=200, t_end=55.0, M_c=0.12, L_n=0.50, geom="bell", Re=500.0, init_type="spin_2T", A_eq=1e-4, W_base=W96_500, record_stride=12)),
        ("pde_bell_Re1000_2T",   dict(nx=96, nth=200, t_end=55.0, M_c=0.12, L_n=0.50, geom="bell", Re=1000.0, init_type="spin_2T", A_eq=1e-4, W_base=W96_1000, record_stride=12)),
        # Direct FFT measurement of Higher-Harmonic Baffle Downshift (bare vs baffled bell, 1T & 2T, reacting)
        ("fft_bare_1T",          dict(nx=80, nth=200, t_end=55.0, M_c=0.12, L_n=0.50, geom="bell", n_p=0.22, n_v=0.04, init_type="spin_1T", A_eq=1e-4, W_base=W80_inf, record_stride=4)),
        ("fft_baf_1T",           dict(nx=80, nth=200, t_end=55.0, M_c=0.12, L_n=0.50, geom="bell", n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, init_type="spin_1T", A_eq=1e-4, W_base=W80_inf, record_stride=4)),
        ("fft_bare_2T",          dict(nx=80, nth=200, t_end=55.0, M_c=0.12, L_n=0.50, geom="bell", n_p=0.22, n_v=0.04, init_type="spin_2T", A_eq=1e-4, W_base=W80_inf, record_stride=4)),
        ("fft_baf_2T",           dict(nx=80, nth=200, t_end=55.0, M_c=0.12, L_n=0.50, geom="bell", n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, init_type="spin_2T", A_eq=1e-4, W_base=W80_inf, record_stride=4)),
    ]
    exp1_res = run_batch(exp1_tasks)
    ce = exp1_res["cons_bell_Re200"]["periodic_cons_err"]
    print(f"  Exp 1A Quiescent bell NS (Re=500): {exp1_res['quiescent_bell_Re500']['quiescent_err']:.2e}")
    print(f"  Exp 1B Periodic bell NS (Re=200): mass={ce['mass']:.2e}, ang_mom={ce['ang_mom']:.2e}, energy={ce['energy']:.2e}")

    visc_cheb_vs_pde = {}
    for lbl, Re_v, mm, fkey in [("pde_bell_Re500_1T", 500.0, 1, "e_1t"), ("pde_bell_Re1000_1T", 1000.0, 1, "e_1t"),
                                ("pde_bell_Re500_2T", 500.0, 2, "e_2t"), ("pde_bell_Re1000_2T", 1000.0, 2, "e_2t")]:
        ww = (0.55, 1.50) if mm == 1 else (1.40, 3.00)
        v72 = exact_viscous_transonic_eig(0.12, mm, 0.50, "bell", Re=Re_v, omega_window=ww, N=72)
        v96 = exact_viscous_transonic_eig(0.12, mm, 0.50, "bell", Re=Re_v, omega_window=ww, N=96)
        ap = fit_alpha(exp1_res[lbl], 30.0, 52.0, key=fkey)
        rel = abs(ap - v96.real) / abs(v96.real) * 100.0
        visc_cheb_vs_pde[lbl] = {"exact_N72": v72.real, "exact_N96": v96.real, "omega": v96.imag, "pde_96x200": ap, "rel_err_pct": rel}
        print(f"  Exp 1C [{lbl:18s}] exact(N=96)={v96.real:+.5f} (spec={abs(v96-v72):.1e}) | PDE(96x200,{fkey})={ap:+.5f} (rel={rel:.2f}%)")

    w1_bare = measure_fft_omega(exp1_res["fft_bare_1T"]["t"], exp1_res["fft_bare_1T"]["p1_re"])
    w1_baf = measure_fft_omega(exp1_res["fft_baf_1T"]["t"], exp1_res["fft_baf_1T"]["p1_re"])
    w2_bare = measure_fft_omega(exp1_res["fft_bare_2T"]["t"], exp1_res["fft_bare_2T"]["p2_re"])
    w2_baf = measure_fft_omega(exp1_res["fft_baf_2T"]["t"], exp1_res["fft_baf_2T"]["p2_re"])
    baffle_downshift = {
        "omega_1T_bare": w1_bare, "omega_1T_baffled": w1_baf, "shift_1T_pct": (w1_baf - w1_bare) / w1_bare * 100.0,
        "omega_2T_bare": w2_bare, "omega_2T_baffled": w2_baf, "shift_2T_pct": (w2_baf - w2_bare) / w2_bare * 100.0,
    }
    print(f"  Exp 1D Baffle Downshift FFT: 1T {w1_bare:.3f} -> {w1_baf:.3f} ({baffle_downshift['shift_1T_pct']:+.1f}%) | 2T {w2_bare:.3f} -> {w2_baf:.3f} ({baffle_downshift['shift_2T_pct']:+.1f}%)")

    # ------------------------------------------------------------------
    # Experiment 2: Viscous Rescue of the Bell-Nozzle 2T Cutoff Collapse (m = 0..5)
    # ------------------------------------------------------------------
    print("\n--- Experiment 2: Coupled Viscous-Transonic Bell Global Hopf Envelopes (m = 0..5) ---")
    M_grid = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22]
    visc_bell_hopf = {}
    for Re_v in (np.inf, 2000.0, 1000.0, 500.0):
        rtag = "inf" if not np.isfinite(Re_v) else f"{int(Re_v)}"
        rows = []
        for Mb in M_grid:
            npH = {}; colds = {}
            for mm in (0, 1, 2, 3, 4, 5):
                ww = (0.55, 1.50) if mm == 1 else (0.05, 7.0)
                vc = exact_viscous_transonic_eig(Mb, mm, 0.50, "bell", Re=Re_v, omega_window=ww, N=72)
                colds[mm] = vc.real
                if vc.real >= 0.0:
                    npH[mm] = 0.0
                else:
                    f = lambda nv, m_k=mm, w_w=ww: exact_viscous_transonic_eig(Mb, m_k, 0.50, "bell", Re=Re_v, n_p=nv, omega_window=w_w, N=72).real
                    npH[mm] = float(brentq(f, 0.0, 0.85)) if f(0.85) > 0.0 else 0.85
            gm_m = min(npH, key=npH.get)
            r1 = exact_viscous_transonic_eig(Mb, 1, 0.50, "bell", Re=Re_v, n_p=0.22, n_v=0.04, omega_window=(0.55, 1.50), N=72)
            r2 = exact_viscous_transonic_eig(Mb, 2, 0.50, "bell", Re=Re_v, n_p=0.22, n_v=0.04, omega_window=(1.40, 3.00), N=72)
            rows.append({
                "M_c": Mb, "cold_1T": colds[1], "cold_2T": colds[2],
                "react_1T": r1.real, "react_2T": r2.real,
                "n_pH_m0": npH[0], "n_pH_1T": npH[1], "n_pH_2T": npH[2], "n_pH_3T": npH[3],
                "n_pH_global": npH[gm_m], "limiting_m": int(gm_m),
            })
        visc_bell_hopf[rtag] = rows
        print(f"  Bell NS Re={rtag:4s}: " + " | ".join(f"M={r['M_c']:.2f}:{r['n_pH_global']:.3f}(m{r['limiting_m']})" for r in rows))

    # ------------------------------------------------------------------
    # Experiment 3: Dual-Harmonic Baffle-Cavity Retuning Curve (80x200, wH_2T sweep & split)
    # ------------------------------------------------------------------
    print("\n--- Experiment 3: Dual-Harmonic Retuning Curve (wH_2T sweep, orifice & split) ---")
    w2_grid = [1.35, 1.45, 1.55, 1.60, 1.65, 1.70, 1.80, 1.90, 2.01, 2.15]
    exp3_tasks = []
    for w2 in w2_grid:
        cf = make_hyb(w1=0.96, w2=w2, frac1=0.80, sig1=0.045, sig2=0.050)
        exp3_tasks.append((f"lin2T_w{w2:.2f}", dict(nx=80, nth=200, t_end=60.0, M_c=0.12, L_n=0.50, geom="bell", n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf, init_type="spin_2T", A_eq=1e-4, W_base=W80_inf, record_stride=12)))
        exp3_tasks.append((f"NL2T_w{w2:.2f}",  dict(nx=80, nth=200, t_end=150.0, M_c=0.12, L_n=0.50, geom="bell", n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf, init_type="spin_2T", A_eq=0.055, W_base=W80_inf, record_stride=20)))
    # On-resonance (wH_2T = 1.65) tight vs wide orifice AND 60/40 split comparison
    cf_t15_165 = make_hyb(w1=0.96, w2=1.65, frac1=0.80, sig1=0.045, sig2=0.015)
    cf_6040_165 = make_hyb(w1=0.96, w2=1.65, frac1=0.60, sig1=0.045, sig2=0.050)
    exp3_tasks.extend([
        ("NL2T_w1.65_tight15", dict(nx=80, nth=200, t_end=150.0, M_c=0.12, L_n=0.50, geom="bell", n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_t15_165, init_type="spin_2T", A_eq=0.055, W_base=W80_inf, record_stride=20)),
        ("NL1T_6040_w1.65",    dict(nx=80, nth=200, t_end=160.0, M_c=0.12, L_n=0.50, geom="bell", n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_6040_165, init_type="spin_1T", A_eq=0.055, W_base=W80_inf, record_stride=20)),
        ("NL2T_6040_w1.65",    dict(nx=80, nth=200, t_end=160.0, M_c=0.12, L_n=0.50, geom="bell", n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_6040_165, init_type="spin_2T", A_eq=0.055, W_base=W80_inf, record_stride=20)),
    ])
    exp3_res = run_batch(exp3_tasks)
    exp3_stats = {}
    for cid, _ in exp3_tasks:
        st = stats_nl(exp3_res[cid])
        exp3_stats[cid] = st
        print(f"  [{cid:18s}] P={st['p_over_pbar_pct']:5.2f}% | E={st['e_wave_late']:.2e} | a={st['alpha_fit_win']:+.4f} | tail={st['alpha_tail']:+.4f}")

    # ------------------------------------------------------------------
    # Experiment 4: Resolved 120x300 (t = 300) Breaking of the 1T -> 2T Cascade Attractor
    # ------------------------------------------------------------------
    print("\n--- Experiment 4: Resolved 120x300 (t = 300) All-Mode Nonlinear Quench Audit ---")
    # Relax converged t = 150 no-slip viscous wake on 80x200 for Re = 500 bell no-slip verification
    W80_500_1d, _ = relax_1d_viscous_transonic_base(80, 0.12, 0.50, "bell", 500.0)
    wake_80_500 = simulate_2d_viscous_transonic(
        nx=80, nth=200, t_end=150.0, M_c=0.12, L_n=0.50, geom="bell", Re=500.0,
        n_baffles=5, L_baffle=0.45, baffle_bc="noslip", init_type="steady", A_eq=0.0,
        W_base=W80_500_1d, record_stride=80,
    )
    W80_500_noslip = wake_80_500["W_final"]
    print(f"  Converged 80x200 t=150 no-slip bell wake (Re=500): rhs={wake_80_500['final_rhs_max']:.2e}")

    cf_untuned = make_hyb(w1=0.96, w2=2.01, frac1=0.80, sig1=0.045, sig2=0.015)
    cf_dual_w50 = make_hyb(w1=0.96, w2=1.65, frac1=0.80, sig1=0.045, sig2=0.050)
    cf_dual_t15 = make_hyb(w1=0.96, w2=1.65, frac1=0.80, sig1=0.045, sig2=0.015)

    exp4_tasks = [
        # 7 resolved 120x300 (t = 240..300) headline runs covering inviscid, Re=1000, Re=500 across 1T & 2T bombs
        ("R120_inf_untuned_1T",   dict(nx=120, nth=300, t_end=300.0, M_c=0.12, L_n=0.50, geom="bell", Re=np.inf, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_untuned, init_type="spin_1T", A_eq=0.055, W_base=W120_inf, record_stride=32)),
        ("R120_inf_dualW50_1T",   dict(nx=120, nth=300, t_end=300.0, M_c=0.12, L_n=0.50, geom="bell", Re=np.inf, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_dual_w50, init_type="spin_1T", A_eq=0.055, W_base=W120_inf, record_stride=32)),
        ("R120_inf_dualT15_1T",   dict(nx=120, nth=300, t_end=300.0, M_c=0.12, L_n=0.50, geom="bell", Re=np.inf, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_dual_t15, init_type="spin_1T", A_eq=0.055, W_base=W120_inf, record_stride=32)),
        ("R120_inf_dualW50_2T",   dict(nx=120, nth=300, t_end=240.0, M_c=0.12, L_n=0.50, geom="bell", Re=np.inf, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_dual_w50, init_type="spin_2T", A_eq=0.055, W_base=W120_inf, record_stride=32)),
        ("R120_Re1000_dualW50_1T",dict(nx=120, nth=300, t_end=300.0, M_c=0.12, L_n=0.50, geom="bell", Re=1000.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_dual_w50, init_type="spin_1T", A_eq=0.055, W_base=W120_1000, record_stride=32)),
        ("R120_Re1000_dualW50_2T",dict(nx=120, nth=300, t_end=240.0, M_c=0.12, L_n=0.50, geom="bell", Re=1000.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_dual_w50, init_type="spin_2T", A_eq=0.055, W_base=W120_1000, record_stride=32)),
        ("R120_Re500_dualW50_2T", dict(nx=120, nth=300, t_end=240.0, M_c=0.12, L_n=0.50, geom="bell", Re=500.0,  n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_dual_w50, init_type="spin_2T", A_eq=0.055, W_base=W120_500, record_stride=32)),
        # Converged t=150 no-slip wake checks on 80x200 (dual-retuned 1T & 2T vs untuned 2T)
        ("R80_Re500_noslip_1T",   dict(nx=80, nth=200, t_end=240.0, M_c=0.12, L_n=0.50, geom="bell", Re=500.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, baffle_bc="noslip", cav_spec=cf_dual_w50, init_type="spin_1T", A_eq=0.055, W_base=W80_500_noslip, record_stride=24)),
        ("R80_Re500_noslip_2T",   dict(nx=80, nth=200, t_end=240.0, M_c=0.12, L_n=0.50, geom="bell", Re=500.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, baffle_bc="noslip", cav_spec=cf_dual_w50, init_type="spin_2T", A_eq=0.055, W_base=W80_500_noslip, record_stride=24)),
        ("R80_Re500_noslip_unt2T",dict(nx=80, nth=200, t_end=240.0, M_c=0.12, L_n=0.50, geom="bell", Re=500.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, baffle_bc="noslip", cav_spec=cf_untuned, init_type="spin_2T", A_eq=0.055, W_base=W80_500_noslip, record_stride=24)),
    ]
    exp4_res = run_batch(exp4_tasks)
    exp4_stats = {}
    for cid, _ in exp4_tasks:
        st = stats_nl(exp4_res[cid])
        exp4_stats[cid] = st
        print(f"  [{cid:23s}] t={st['t_end']:4.0f} | P={st['p_over_pbar_pct']:5.2f}% | E={st['e_wave_late']:.2e} (l/m={st['e_wave_late_over_mid']:.3f}, e2/e1={st['e2_over_e1_late']:.1f}) | tail={st['alpha_tail']:+.4f}")

    # ------------------------------------------------------------------
    # Experiment 5: Viscous Bell 1T Back-Reaction & Resolved 120x300 3.0% 85/15 Quench
    # ------------------------------------------------------------------
    exp5_data, _ = run_exp5_viscous_1t_backreaction()

    # Save metrics BEFORE plotting
    metrics = {
        "exp1_coupled_benchmarks": {
            "quiescent_bell_Re500_err": exp1_res["quiescent_bell_Re500"]["quiescent_err"],
            "periodic_bell_Re200_cons_err": ce,
            "base_rhs_96": {"Re500": rhs_96_500, "Re1000": rhs_96_1000},
            "visc_cheb_vs_pde_96x200": visc_cheb_vs_pde,
            "baffle_downshift_fft": baffle_downshift,
        },
        "exp2_viscous_bell_rescue_hopf": visc_bell_hopf,
        "exp3_dual_retuning_curve": {
            "w2_grid": w2_grid,
            "runs": exp3_stats,
        },
        "exp4_resolved_120x300_quench": {
            "noslip_wake_80_rhs": wake_80_500["final_rhs_max"],
            "runs": exp4_stats,
        },
        "exp5_viscous_1T_backreaction_and_quench": exp5_data,
    }
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved metrics -> {out_path}")

    # ------------------------------------------------------------------
    # Figure 1: Viscous Rescue of Bell Global Hopf & Dual-Harmonic wH_2T Resonance Curve
    # ------------------------------------------------------------------
    plot_fig1(metrics)

    # ------------------------------------------------------------------
    # Figure 2: Resolved 120x300 (t = 300) Breaking of the 1T -> 2T Cascade Attractor
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.6), dpi=150)

    for cid, lbl, col, ls in [
        ("R120_inf_untuned_1T",    "Inviscid Bell + Untuned 2T (wH2=2.01, 120x300)", "#d62728", "-"),
        ("R120_inf_dualT15_1T",    "Inviscid Bell + Dual-Retuned (wH2=1.65, tight 1.5%)", "#ff7f0e", "--"),
        ("R120_inf_dualW50_1T",    "Inviscid Bell + Dual-Retuned (wH2=1.65, wide 5.0%)", "#1f77b4", "-"),
        ("R120_Re1000_dualW50_1T", "Viscous Bell Re=1000 + Dual-Retuned (120x300)", "#2ca02c", "-."),
        ("R80_Re500_noslip_1T",    "No-Slip Wake Bell Re=500 + Dual-Retuned (80x200)", "#9467bd", ":"),
    ]:
        rr = exp4_res[cid]; st = exp4_stats[cid]
        ax1.semilogy(rr["t"], np.maximum(rr["e_ch"], 1e-12), color=col, ls=ls, lw=2.2,
                     label=f"{lbl} (P={st['p_over_pbar_pct']:.2f}%)")
    ax1.set_xlabel("Time $t$", fontsize=11)
    ax1.set_ylabel("Chamber Wave Energy $E_{\\mathrm{wave}}(t)$", fontsize=11)
    ax1.set_title("6.9% 1T Bomb: Breaking the 1T -> 2T Cascade on 120x300", fontsize=12, fontweight="bold")
    ax1.set_ylim(1e-9, 1e-2)
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=7.9, loc="lower left", framealpha=0.92)

    for cid, lbl, col, ls in [
        ("R120_inf_dualW50_2T",    "Inviscid Bell + Dual-Retuned (120x300)", "#ff7f0e", ":"),
        ("R80_Re500_noslip_unt2T", "No-Slip Bell Re=500 + Untuned 2T (wH2=2.01)", "#d62728", "--"),
        ("R120_Re1000_dualW50_2T", "Viscous Bell Re=1000 + Dual-Retuned (120x300)", "#1f77b4", "-"),
        ("R120_Re500_dualW50_2T",  "Viscous Bell Re=500 + Dual-Retuned (120x300)", "#2ca02c", "-"),
        ("R80_Re500_noslip_2T",    "No-Slip Bell Re=500 + Dual-Retuned (80x200)", "#9467bd", "-."),
    ]:
        rr = exp4_res[cid]; st = exp4_stats[cid]
        ax2.semilogy(rr["t"], np.maximum(rr["e_ch"], 1e-12), color=col, ls=ls, lw=2.2,
                     label=f"{lbl} (P={st['p_over_pbar_pct']:.2f}%)")
    ax2.set_xlabel("Time $t$", fontsize=11)
    ax2.set_ylabel("Chamber Wave Energy $E_{\\mathrm{wave}}(t)$", fontsize=11)
    ax2.set_title("Direct 6.9% 2T Bomb: Coupled Viscous + Dual-Retuned Quench", fontsize=12, fontweight="bold")
    ax2.set_ylim(1e-9, 1e-2)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=8.0, loc="lower left", framealpha=0.92)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp013_120x300_cascade_break_and_2T_quench.png", dpi=150)
    plt.close(fig)
    print("Saved plots -> exp013_viscous_bell_rescue_and_2T_retuning.png, exp013_120x300_cascade_break_and_2T_quench.png")
    print("Done!")


if __name__ == "__main__":
    main()
