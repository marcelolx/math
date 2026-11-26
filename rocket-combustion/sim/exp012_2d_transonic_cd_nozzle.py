"""Entry 012 (rocket-combustion, Hardened per Peer Review #9): 2D Thin-Shell Transonic
Converging-Diverging Nozzle Geometry — Bell Evanescent-Cutoff Collapse vs. Annular Slot
Tuning, Like-for-Like Entropy-Noise Filtering & Resolved 120x300 2T Cascade Audit.

Replaces the compact 0D Marble-Candel outlet boundary condition (Entries 009-011) with an
explicitly resolved, C^2-smooth 2D (x, theta) variable-(A(x), R(x)) thin-shell streamtube
transonic converging-diverging nozzle accelerating steady throughflow from chamber Mach M_c
through the interior sonic throat (M = 1 at x_t = L_a + L_n) to a strictly supersonic exit
(M_e ~ 1.83 > 1 at x = L_a + L_n + L_d), requiring ZERO outlet boundary conditions.

Governing 2D conservative variable-(A(x), R(x)) compressible Euler equations for
W = (A*rho, A*rho*ux, A*rho*R*ut, A*E)^T on [0, L_tot] x [0, 2*pi) (gamma = 1.25):

    d(A*rho)/dt       + d(A*rho*ux)/dx         + d((A/R)*rho*ut)/dtheta       = 0
    d(A*rho*ux)/dt    + d(A*(rho*ux^2 + p))/dx + d((A/R)*rho*ux*ut)/dtheta    = p*dA/dx + (A*rho*ut^2/R)*dR/dx
    d(A*rho*R*ut)/dt  + d(A*rho*ux*R*ut)/dx    + d(A*(rho*ut^2 + p))/dtheta   = 0
    d(A*E)/dt         + d(A*ux*(E + p))/dx     + d((A/R)*ut*(E + p))/dtheta   = A*(w_f/(gamma-1))*Q_flame

Hardening upgrades per independent read-only Peer Review #9 (Claude Opus 5.5):
  - C1 & C2 (Resolved 120x300 t = 300 nonlinear audit): Retracts the coarse-grid 80x200
    quench claims for bare annular (42x) and annular+hybrid (521x); on resolved 120x300 to
    t = 300, bare annular sustains a ~4.0% 2T-dominated limit cycle (exact unstable 2T
    +0.00225), and annular+hybrid regrows after t ~ 77 via the 1T -> 2T cascade into a 5.07%
    pure-2T limit cycle (bell+hybrid: 5.54%; resolved energy ratio 1.19x, NOT 521x).
  - C3 & M3 (Like-for-like Exp 2 entropy-noise filtering): Computes an exact like-for-like
    1D compact Marble-Candel reference (delta_rho = -0.05, zero swirl, nx = 200 & 400),
    extends L_n in {0.15, 0.35, 0.70, 1.10} to t_end = 26.0 on both 1x and 2x grids to
    capture both 1st-pass and true 2nd-pass peaks, and documents bell = annular for m = 0.
  - M1, M4, M5, M6: Replaces short-L_n single-domain Chebyshev with a 2-domain characteristic-
    patched Chebyshev solver (smoothly recovering compact MC at L_n = 0.02: -0.0286 vs -0.0284),
    adds a 3-contour sensitivity table (C^2 quartic, C^1 cubic, C^1 cosine), extends Hopf
    maps to m = 0..5, documents both inviscid bell M_c = 0.10 whistles (m = 3, 4) plus
    eigenvector amplitude ratio & Re = 1e5 viscous quench, and states 2D thin-shell scope.
  - M2 & m1-m5: Fits annular 1T on t in [60, 100] (eliminating T_beat = 32.1 bias, matching
    exact -0.0729 to ~1.4%), documents lin_bell20 convergence (80x200 -> 120x300), reports
    the full m = 1 reacting spectrum (omega ~ 1.85 branch), and cleans up figure labels.
"""

from __future__ import annotations

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
P_BAR = 1.0 / GAMMA       # 0.80 => c_c = 1.0, rho_c = 1.0 in chamber
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
# 1. Smooth Transonic CD Nozzle Contour Family & Exact Isentropic Base Flow
# ---------------------------------------------------------------------------

def isentropic_area_ratio(M: float) -> float:
    g = GAMMA
    return (1.0 / M) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * M * M)) ** ((g + 1.0) / (2.0 * (g - 1.0)))


def nozzle_profile_family(
    x: np.ndarray, M_c: float, L_n: float, L_d: float = L_D, contour: str = "quartic_c2"
) -> tuple[np.ndarray, np.ndarray]:
    """Converging-diverging area profile A(x) and dA/dx for a contour family on [0, L_a + L_n + L_d].

    Contours on [L_a, x_t = L_a + L_n]:
      - 'quartic_c2' (baseline): 1 - 4*s^3 + 3*s^4 (C^2 everywhere, A''(x_t) = 12*(1-A_t)/L_n^2)
      - 'cubic_c1':              1 - 3*s^2 + 2*s^3 (C^1 Hermite, A''(x_t) = 6*(1-A_t)/L_n^2)
      - 'cosine_c1':             0.5*(1 + cos(pi*s)) (C^1 cosine, A''(x_t) = 0.5*pi^2*(1-A_t)/L_n^2)
    Diverging [x_t, x_t + L_d]: curvature-matched rational expansion to strictly supersonic M_e > 1.
    """
    A_t = 1.0 / isentropic_area_ratio(M_c)
    x_t = L_A + L_n
    A = np.ones_like(x, dtype=float)
    dA = np.zeros_like(x, dtype=float)

    m1 = (x > L_A) & (x <= x_t)
    s1 = (x[m1] - L_A) / L_n
    if contour == "quartic_c2":
        A[m1] = A_t + (1.0 - A_t) * (1.0 - 4.0 * s1**3 + 3.0 * s1**4)
        dA[m1] = (1.0 - A_t) * (-12.0 * s1**2 + 12.0 * s1**3) / L_n
        k_t = 12.0 * (1.0 - A_t) / (L_n * L_n)
    elif contour == "cubic_c1":
        A[m1] = A_t + (1.0 - A_t) * (1.0 - 3.0 * s1**2 + 2.0 * s1**3)
        dA[m1] = (1.0 - A_t) * (-6.0 * s1 + 6.0 * s1**2) / L_n
        k_t = 6.0 * (1.0 - A_t) / (L_n * L_n)
    elif contour == "cosine_c1":
        A[m1] = A_t + 0.5 * (1.0 - A_t) * (1.0 + np.cos(np.pi * s1))
        dA[m1] = -0.5 * (1.0 - A_t) * (np.pi / L_n) * np.sin(np.pi * s1)
        k_t = 0.5 * (1.0 - A_t) * ((np.pi / L_n) ** 2)
    else:
        raise ValueError(f"Unknown contour={contour}")

    delta_A = 0.60 * A_t
    y0_sq = 2.0 * delta_A / k_t
    m2 = x > x_t
    y = x[m2] - x_t
    denom = y * y + y0_sq
    A[m2] = A_t + delta_A * (y * y / denom)
    dA[m2] = delta_A * (2.0 * y * y0_sq / (denom * denom))
    return A, dA


def exact_isentropic_base(
    x: np.ndarray, M_c: float, L_n: float, L_d: float = L_D, geom: str = "annular", contour: str = "quartic_c2"
) -> dict[str, np.ndarray]:
    """Exact steady 1D isentropic transonic base flow (rho, ux, p, c, M, A, dA, R, dR)."""
    A, dA = nozzle_profile_family(x, M_c, L_n, L_d, contour=contour)
    A_t = 1.0 / isentropic_area_ratio(M_c)
    x_t = L_A + L_n
    M = np.zeros_like(x, dtype=float)
    for i, xi in enumerate(x):
        ar = float(A[i] / A_t)
        if ar <= 1.0 + 1e-14:
            M[i] = 1.0
        elif xi < x_t:
            M[i] = brentq(lambda m: isentropic_area_ratio(m) - ar, 1e-7, 1.0 - 1e-13)
        else:
            M[i] = brentq(lambda m: isentropic_area_ratio(m) - ar, 1.0 + 1e-13, 12.0)

    fac_c = 1.0 + 0.5 * (GAMMA - 1.0) * M_c * M_c
    fac = 1.0 + 0.5 * (GAMMA - 1.0) * M * M
    T = fac_c / fac
    rho = T ** (1.0 / (GAMMA - 1.0))
    p = P_BAR * (T ** (GAMMA / (GAMMA - 1.0)))
    c = np.sqrt(T)
    ux = M * c

    if geom == "annular":
        R = np.ones_like(A)
        dR = np.zeros_like(A)
    elif geom == "bell":
        R = np.sqrt(A)
        dR = 0.5 * dA / R
    else:
        raise ValueError(f"Unknown geom={geom}")

    return {"A": A, "dA": dA, "R": R, "dR": dR, "rho": rho, "ux": ux, "p": p, "c": c, "M": M}


# ---------------------------------------------------------------------------
# 2. Exact Compact MC & 2-Domain Characteristic-Patched Chebyshev Transonic Solvers
# ---------------------------------------------------------------------------

def cheb_diff(N: int, L: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chebyshev-Gauss-Lobatto nodes x in [0, L], derivative matrix Dx, Clenshaw-Curtis weights."""
    k = np.arange(N + 1)
    z = np.cos(np.pi * k / N)
    c = np.ones(N + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c *= (-1.0) ** k
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


def exact_compact_mc_eig(
    M_c: float, m: int, n_p: float = 0.0, n_v: float = 0.0,
    omega_window: tuple[float, float] | None = None, N: int = 44,
    check_both_chiral_signs: bool = True,
) -> complex:
    """Entry 009 compact Marble-Candel reference eigenvalue on [0, L_a]."""
    if check_both_chiral_signs and abs(n_v) > 0.0 and m != 0:
        vp = exact_compact_mc_eig(M_c, abs(m), n_p, n_v, omega_window, N, False)
        vn = exact_compact_mc_eig(M_c, -abs(m), n_p, n_v, omega_window, N, False)
        return vp if vp.real >= vn.real else vn
    x, Dx, w_cc = cheb_diff(N, L_A)
    n = N + 1
    I = np.eye(n)
    has_f = (abs(n_p) > 0.0 or abs(n_v) > 0.0)
    dim = 4 * n + (10 if has_f else 0)
    A = np.zeros((dim, dim), dtype=complex)
    B = np.eye(dim, dtype=complex)
    sl = lambda q: slice(q * n, (q + 1) * n)
    A[sl(0), sl(0)] = -M_c * Dx; A[sl(0), sl(1)] = -Dx; A[sl(0), sl(2)] = -1j * m * I
    A[sl(1), sl(1)] = -M_c * Dx; A[sl(1), sl(3)] = -Dx
    A[sl(2), sl(2)] = -M_c * Dx; A[sl(2), sl(3)] = -1j * m * I
    A[sl(3), sl(1)] = -Dx; A[sl(3), sl(2)] = -1j * m * I; A[sl(3), sl(3)] = -M_c * Dx
    if has_f:
        wf = np.exp(-0.5 * ((x - X_F) / SIG_F) ** 2)
        wf /= float(np.sum(w_cc * wf))
        w_sense = w_cc * wf
        inv_ts = 4.0 / TAU_BAR; ib = 4 * n
        A[ib + 0, sl(3)] = OMEGA_HP * w_sense; A[ib + 0, ib + 0] = -OMEGA_HP
        A[ib + 1, sl(3)] = inv_ts * w_sense; A[ib + 1, ib + 0] = -inv_ts; A[ib + 1, ib + 1] = -inv_ts
        for r in range(1, 4): A[ib + 1 + r, ib + r] = inv_ts; A[ib + 1 + r, ib + 1 + r] = -inv_ts
        A[ib + 5, sl(2)] = OMEGA_HP * w_sense; A[ib + 5, ib + 5] = -OMEGA_HP
        A[ib + 6, sl(2)] = inv_ts * w_sense; A[ib + 6, ib + 5] = -inv_ts; A[ib + 6, ib + 6] = -inv_ts
        for r in range(1, 4): A[ib + 6 + r, ib + 5 + r] = inv_ts; A[ib + 6 + r, ib + 6 + r] = -inv_ts
        A[sl(3), ib + 4] = n_p * wf; A[sl(3), ib + 9] = n_v * wf
    for r in (0, n, 2 * n): A[r, :] = 0.0; B[r, :] = 0.0
    A[0, 0] = 1.0; A[0, 3 * n] = -1.0
    A[n, n] = 1.0; A[n, 3 * n] = M_c
    A[2 * n, 2 * n] = 1.0
    r_out = 3 * n + N
    A[r_out, :] = 0.0; B[r_out, :] = 0.0
    A[r_out, n + N] = 1.0; A[r_out, 3 * n + N] = -0.5 * M_c * GAMMA; A[r_out, N] = +0.5 * M_c
    vals = eig(A, B, right=False)
    w_lo, w_hi = omega_window if omega_window is not None else (0.05, 7.0)
    cands = [v for v in vals if np.isfinite(v) and w_lo <= v.imag <= w_hi and -0.45 < v.real < 0.35]
    cands.sort(key=lambda z: z.real, reverse=True)
    return complex(cands[0])


def exact_transonic_md_eig(
    M_c: float, m: int, L_n: float, geom: str = "annular",
    contour: str = "quartic_c2", n_p: float = 0.0, n_v: float = 0.0,
    Re: float = np.inf, omega_window: tuple[float, float] | None = None,
    N0: int = 40, N1: int = 56, check_both_chiral_signs: bool = True,
    return_vec: bool = False,
):
    """2-domain characteristic-patched Chebyshev transonic CD nozzle eigen-solver (M6).

    Splits [0, L_a] (uniform chamber, N0) and [L_a, L_a + L_n + L_d_eff] (transonic nozzle, N1)
    with exact 1+3 upwind characteristic patching at x = L_a and ZERO outlet BCs (M_e > 1).
    Resolves short nozzles down to L_n = 0.02 cleanly and supports optional viscous damping
    (nu = 1/Re Laplacian on u'_x, u'_theta for whistle sensitivity check, M4).
    """
    if check_both_chiral_signs and abs(n_v) > 0.0 and m != 0 and not return_vec:
        vp = exact_transonic_md_eig(M_c, abs(m), L_n, geom, contour, n_p, n_v, Re, omega_window, N0, N1, False, False)
        vn = exact_transonic_md_eig(M_c, -abs(m), L_n, geom, contour, n_p, n_v, Re, omega_window, N0, N1, False, False)
        return vp if vp.real >= vn.real else vn

    L_d_eff = min(L_D, max(0.04, 2.0 * L_n))
    x0, D0, w0 = cheb_diff(N0, L_A)
    x1_loc, D1, _ = cheb_diff(N1, L_n + L_d_eff)
    x1 = L_A + x1_loc
    n0, n1 = N0 + 1, N1 + 1
    nt = n0 + n1

    base1 = exact_isentropic_base(x1, M_c, L_n, L_d_eff, geom=geom, contour=contour)
    A1, dA1, R1, dR1 = base1["A"], base1["dA"], base1["R"], base1["dR"]
    rho1, ux1, p1 = base1["rho"], base1["ux"], base1["p"]

    has_f = (abs(n_p) > 0.0 or abs(n_v) > 0.0)
    dim = 4 * nt + (10 if has_f else 0)
    Mat = np.zeros((dim, dim), dtype=complex)
    B = np.eye(dim, dtype=complex)

    def sl0(q: int) -> slice:
        return slice(q * nt, q * nt + n0)

    def sl1(q: int) -> slice:
        return slice(q * nt + n0, (q + 1) * nt)

    I0 = np.eye(n0)
    # Domain 0: straight chamber (A=1, R=1, rho=1, gamma*p=1, ux=M_c)
    Mat[sl0(0), sl0(0)] = -M_c * D0; Mat[sl0(0), sl0(1)] = -D0; Mat[sl0(0), sl0(2)] = -1j * m * I0
    Mat[sl0(1), sl0(1)] = -M_c * D0; Mat[sl0(1), sl0(3)] = -D0
    Mat[sl0(2), sl0(2)] = -M_c * D0; Mat[sl0(2), sl0(3)] = -1j * m * I0
    Mat[sl0(3), sl0(1)] = -D0; Mat[sl0(3), sl0(2)] = -1j * m * I0; Mat[sl0(3), sl0(3)] = -M_c * D0

    # Domain 1: transonic converging-diverging nozzle
    drho1 = D1 @ rho1; dux1 = D1 @ ux1; dp1 = D1 @ p1
    dAA1 = dA1 / A1; dRR1 = dR1 / R1
    Mat[sl1(0), sl1(0)] = -np.diag(ux1) @ D1 - np.diag(dux1 + dAA1 * ux1)
    Mat[sl1(0), sl1(1)] = -np.diag(rho1) @ D1 - np.diag(drho1 + dAA1 * rho1)
    Mat[sl1(0), sl1(2)] = -1j * m * np.diag(rho1 / R1)
    Mat[sl1(1), sl1(0)] = np.diag(dp1 / (rho1 * rho1))
    Mat[sl1(1), sl1(1)] = -np.diag(ux1) @ D1 - np.diag(dux1)
    Mat[sl1(1), sl1(3)] = -np.diag(1.0 / rho1) @ D1
    Mat[sl1(2), sl1(2)] = -np.diag(ux1) @ D1 - np.diag(ux1 * dRR1)
    Mat[sl1(2), sl1(3)] = -1j * m * np.diag(1.0 / (R1 * rho1))
    Mat[sl1(3), sl1(1)] = -np.diag(GAMMA * p1) @ D1 - np.diag(dp1 + GAMMA * p1 * dAA1)
    Mat[sl1(3), sl1(2)] = -1j * m * np.diag(GAMMA * p1 / R1)
    Mat[sl1(3), sl1(3)] = -np.diag(ux1) @ D1 - np.diag(GAMMA * (dux1 + dAA1 * ux1))

    # Optional isotropic momentum viscosity (nu = 1/Re) for whistle viscosity sensitivity (M4)
    if np.isfinite(Re):
        nu = 1.0 / Re
        D0_2 = D0 @ D0; D1_2 = D1 @ D1
        Mat[sl0(1), sl0(1)] += nu * (D0_2 - (m * m) * I0)
        Mat[sl0(2), sl0(2)] += nu * (D0_2 - (m * m) * I0)
        Mat[sl1(1), sl1(1)] += nu * (np.diag(1.0 / rho1) @ D1_2 - np.diag((m * m) / (rho1 * R1 * R1)))
        Mat[sl1(2), sl1(2)] += nu * (np.diag(1.0 / rho1) @ D1_2 - np.diag((m * m) / (rho1 * R1 * R1)))

    if has_f:
        wf0 = np.exp(-0.5 * ((x0 - X_F) / SIG_F) ** 2)
        wf0 /= float(np.sum(w0 * wf0))
        w_sense = w0 * wf0
        inv_ts = 4.0 / TAU_BAR; ib = 4 * nt
        Mat[ib + 0, sl0(3)] = OMEGA_HP * w_sense; Mat[ib + 0, ib + 0] = -OMEGA_HP
        Mat[ib + 1, sl0(3)] = inv_ts * w_sense; Mat[ib + 1, ib + 0] = -inv_ts; Mat[ib + 1, ib + 1] = -inv_ts
        for r in range(1, 4):
            Mat[ib + 1 + r, ib + r] = inv_ts; Mat[ib + 1 + r, ib + 1 + r] = -inv_ts
        Mat[ib + 5, sl0(2)] = OMEGA_HP * w_sense; Mat[ib + 5, ib + 5] = -OMEGA_HP
        Mat[ib + 6, sl0(2)] = inv_ts * w_sense; Mat[ib + 6, ib + 5] = -inv_ts; Mat[ib + 6, ib + 6] = -inv_ts
        for r in range(1, 4):
            Mat[ib + 6 + r, ib + 5 + r] = inv_ts; Mat[ib + 6 + r, ib + 6 + r] = -inv_ts
        Mat[sl0(3), ib + 4] = n_p * wf0
        Mat[sl0(3), ib + 9] = n_v * wf0

    # Inlet BCs at x = 0 (Domain 0 left)
    for r in (0, nt, 2 * nt):
        Mat[r, :] = 0.0; B[r, :] = 0.0
    Mat[0, 0] = 1.0; Mat[0, 3 * nt] = -1.0
    Mat[nt, nt] = 1.0; Mat[nt, 3 * nt] = M_c
    Mat[2 * nt, 2 * nt] = 1.0

    # Interface patching at x = L_a (subsonic: 1 left-running J_- into Dom 0 right, 3 right-running into Dom 1 left)
    rJ = 3 * nt + N0
    Mat[rJ, :] = 0.0; B[rJ, :] = 0.0
    Mat[rJ, nt + N0] = 1.0; Mat[rJ, 3 * nt + N0] = -1.0
    Mat[rJ, nt + n0] = -1.0; Mat[rJ, 3 * nt + n0] = 1.0
    for r in (n0, nt + n0, 2 * nt + n0):
        Mat[r, :] = 0.0; B[r, :] = 0.0
    Mat[n0, n0] = 1.0; Mat[n0, N0] = -1.0
    Mat[nt + n0, nt + n0] = 1.0; Mat[nt + n0, 3 * nt + n0] = 1.0
    Mat[nt + n0, nt + N0] = -1.0; Mat[nt + n0, 3 * nt + N0] = -1.0
    Mat[2 * nt + n0, 2 * nt + n0] = 1.0; Mat[2 * nt + n0, 2 * nt + N0] = -1.0

    if np.isfinite(Re):
        # Smoothness of derivatives across x = L_a for viscous operators + Neumann outflow on u', v'
        # Replace exit rows of u', v' at Domain 1 right (nt - 1) with zero-gradient D1[-1, :]
        for q_v in (1, 2):
            r_ex = q_v * nt + nt - 1
            Mat[r_ex, :] = 0.0; B[r_ex, :] = 0.0
            Mat[r_ex, sl1(q_v)] = D1[-1, :]

    w_lo, w_hi = omega_window if omega_window is not None else (0.05, 7.0)
    if return_vec:
        vals, vecs = eig(Mat, B, right=True)
        idx_c = [i for i, v in enumerate(vals) if np.isfinite(v) and w_lo <= v.imag <= w_hi and -0.45 < v.real < 0.35]
        idx_c.sort(key=lambda i: vals[i].real, reverse=True)
        i0 = idx_c[0]
        v_ut1 = vecs[sl1(2), i0]
        i_t = int(np.argmin(np.abs(x1 - (L_A + L_n))))
        ut_ratio = float(np.abs(v_ut1[i_t]) / max(np.abs(v_ut1[0]), 1e-30))
        return complex(vals[i0]), ut_ratio

    vals = eig(Mat, B, right=False)
    cands = [v for v in vals if np.isfinite(v) and w_lo <= v.imag <= w_hi and -0.45 < v.real < 0.35]
    cands.sort(key=lambda z: z.real, reverse=True)
    return complex(cands[0])


# ---------------------------------------------------------------------------
# 3. 2D Conservative Variable-(A(x), R(x)) van Leer + HLLC + SSP-RK2 Solver
# ---------------------------------------------------------------------------

def van_leer_limiter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smooth C^1 van Leer harmonic TVD slope limiter (eliminates sonic-throat stalling)."""
    ab = a * b
    return np.where(ab > 0.0, 2.0 * ab / np.where(ab > 0.0, a + b, 1.0), 0.0)


def phi_flame(q: np.ndarray) -> np.ndarray:
    x = q / Q_S
    return Q_S * np.tanh(x + KAPPA_NL * (x ** 3))


def prim_from_W(W: np.ndarray, A_c: np.ndarray, R_c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = np.maximum(W[0] / A_c, 1e-10)
    ux = W[1] / (A_c * rho)
    ut = W[2] / (A_c * R_c * rho)
    E = W[3] / A_c
    p = np.maximum((GAMMA - 1.0) * (E - 0.5 * rho * (ux * ux + ut * ut)), 1e-10)
    c = np.sqrt(GAMMA * p / rho)
    return rho, ux, ut, p, c


def W_from_prim(rho: np.ndarray, ux: np.ndarray, ut: np.ndarray, p: np.ndarray, A_c: np.ndarray, R_c: np.ndarray) -> np.ndarray:
    E = p / (GAMMA - 1.0) + 0.5 * rho * (ux * ux + ut * ut)
    return np.stack([A_c * rho, A_c * rho * ux, A_c * R_c * rho * ut, A_c * E], axis=0)


def hllc_2d_normal(
    rhoL: np.ndarray, vnL: np.ndarray, vtL: np.ndarray, pL: np.ndarray,
    rhoR: np.ndarray, vnR: np.ndarray, vtR: np.ndarray, pR: np.ndarray,
) -> np.ndarray:
    cL = np.sqrt(GAMMA * pL / rhoL)
    cR = np.sqrt(GAMMA * pR / rhoR)
    EL = pL / (GAMMA - 1.0) + 0.5 * rhoL * (vnL * vnL + vtL * vtL)
    ER = pR / (GAMMA - 1.0) + 0.5 * rhoR * (vnR * vnR + vtR * vtR)
    sL = np.minimum(vnL - cL, vnR - cR)
    sR = np.maximum(vnL + cL, vnR + cR)
    denom = np.where(
        np.abs(rhoL * (sL - vnL) - rhoR * (sR - vnR)) > 1e-14,
        rhoL * (sL - vnL) - rhoR * (sR - vnR), 1e-14,
    )
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


def inlet_choked_ghost(
    rho_i: np.ndarray, ux_i: np.ndarray, p_i: np.ndarray, M_c: float, u_port: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choked zero-swirl isentropic injector ghost state with optional faceplate cavity port velocity u_port."""
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


def outlet_compact_mc_ghost(rho_i: np.ndarray, ux_i: np.ndarray, ut_i: np.ndarray, p_i: np.ndarray, M_c: float):
    """Exact compact Marble-Candel choked outlet ghost state (ux/c = M_c) for like-for-like Exp 2."""
    c_i = np.sqrt(GAMMA * p_i / rho_i)
    Jp = ux_i + (2.0 / (GAMMA - 1.0)) * c_i
    c_g = Jp / (M_c + 2.0 / (GAMMA - 1.0))
    ux_g = M_c * c_g
    s_fac = p_i / (rho_i ** GAMMA)
    rho_g = ((c_g * c_g) / (GAMMA * s_fac)) ** (1.0 / (GAMMA - 1.0))
    p_g = s_fac * (rho_g ** GAMMA)
    return rho_g, ux_g, ut_i.copy(), p_g


def simulate_1d_compact_mc_entropy(nx: int = 200, M_c: float = 0.12, t_end: float = 20.0) -> dict:
    """Like-for-like 1D compact Marble-Candel entropy-pulse reference (delta_rho = -0.05, zero swirl, C3)."""
    dx = L_A / nx
    x = (np.arange(nx) + 0.5) * dx
    A_c = np.ones((nx, 1)); R_c = np.ones((nx, 1))
    rho = RHO_BAR - 0.05 * np.exp(-0.5 * ((x - 0.55) / 0.08) ** 2)[:, None]
    ux = np.full((nx, 1), M_c)
    ut = np.zeros((nx, 1))
    p = np.full((nx, 1), P_BAR)
    W = W_from_prim(rho, ux, ut, p, A_c, R_c)

    def rhs_mc(Ws):
        r, u, v, pr, _ = prim_from_W(Ws, A_c, R_c)
        P = np.stack([r, u, v, pr], axis=0)
        Pe = np.empty((4, nx + 4, 1)); Pe[:, 2:-2, :] = P
        ri, ui, vi, pi_g = inlet_choked_ghost(r[0], u[0], pr[0], M_c)
        ro, uo, vo, po_g = outlet_compact_mc_ghost(r[-1], u[-1], v[-1], pr[-1], M_c)
        for k in (0, 1):
            Pe[0, k] = ri; Pe[1, k] = ui; Pe[2, k] = vi; Pe[3, k] = pi_g
            Pe[0, -(k + 1)] = ro; Pe[1, -(k + 1)] = uo; Pe[2, -(k + 1)] = vo; Pe[3, -(k + 1)] = po_g
        s = van_leer_limiter(Pe[:, 1:-1] - Pe[:, :-2], Pe[:, 2:] - Pe[:, 1:-1])
        PL = Pe[:, 1:-2] + 0.5 * s[:, :-1]
        PR = Pe[:, 2:-1] - 0.5 * s[:, 1:]
        Fx = hllc_2d_normal(
            np.maximum(PL[0], 1e-10), PL[1], PL[2], np.maximum(PL[3], 1e-10),
            np.maximum(PR[0], 1e-10), PR[1], PR[2], np.maximum(PR[3], 1e-10),
        )
        return -(Fx[:, 1:] - Fx[:, :-1]) / dx

    t = 0.0; ts, pp = [], []
    step = 0
    while t < t_end - 1e-12:
        r, u, _, pr, c = prim_from_W(W, A_c, R_c)
        dt = min(0.30 * dx / float(np.max(np.abs(u) + c)), t_end - t)
        if step % 2 == 0:
            ts.append(t)
            pp.append(float((pr[-1, 0] - P_BAR) / P_BAR * 100.0))
        k1 = rhs_mc(W)
        W1 = W + dt * k1
        k2 = rhs_mc(W1)
        W = 0.5 * (W + W1 + dt * k2)
        t += dt; step += 1
    ts = np.array(ts); pp = np.array(pp)
    m1 = (ts >= 6.0) & (ts <= 9.5)
    m2 = (ts >= 9.5) & (ts <= 13.5)
    return {
        "t": ts, "p_exit_pct": pp,
        "p_first_pass_pct": float(np.min(pp[m1])), "t_first_pass": float(ts[m1][np.argmin(pp[m1])]),
        "p_second_pass_pct": float(np.min(pp[m2])), "t_second_pass": float(ts[m2][np.argmin(pp[m2])]),
    }


def build_grid_geometry(
    nx: int, M_c: float, L_n: float, geom: str = "annular", periodic_bump: bool = False
) -> dict[str, np.ndarray]:
    if periodic_bump:
        L_tot = L_A
        dx = L_tot / nx
        x = (np.arange(nx) + 0.5) * dx
        xf = np.arange(nx + 1) * dx
        A_f = 1.0 - 0.35 * np.sin(np.pi * xf / L_tot) ** 2
        A_c = 0.5 * (A_f[:-1] + A_f[1:])
        R_f = np.ones_like(A_f) if geom == "annular" else np.sqrt(A_f)
        R_c = np.ones_like(A_c) if geom == "annular" else np.sqrt(A_c)
        return {"x": x, "xf": xf, "dx": dx, "L_tot": L_tot, "A_c": A_c[:, None], "A_f": A_f[:, None], "R_c": R_c[:, None], "R_f": R_f[:, None]}

    L_tot = L_A + L_n + L_D
    dx = L_tot / nx
    x = (np.arange(nx) + 0.5) * dx
    xf = np.arange(nx + 1) * dx
    A_f, _ = nozzle_profile_family(xf, M_c, L_n, L_D, contour="quartic_c2")
    A_c = 0.5 * (A_f[:-1] + A_f[1:])
    R_f = np.ones_like(A_f) if geom == "annular" else np.sqrt(A_f)
    R_c = np.ones_like(A_c) if geom == "annular" else np.sqrt(A_c)
    return {"x": x, "xf": xf, "dx": dx, "L_tot": L_tot, "A_c": A_c[:, None], "A_f": A_f[:, None], "R_c": R_c[:, None], "R_f": R_f[:, None]}


def relax_1d_transonic_base(
    nx: int, M_c: float, L_n: float, geom: str = "annular", t_relax: float = 300.0
) -> tuple[np.ndarray, float]:
    """Relax exact 1D steady transonic base flow on grid nx to machine-level stationarity."""
    g = build_grid_geometry(nx, M_c, L_n, geom=geom)
    x, dx = g["x"], g["dx"]
    A_c, A_f, R_c, R_f = g["A_c"], g["A_f"], g["R_c"], g["R_f"]
    dA_dx = (A_f[1:] - A_f[:-1]) / dx

    base = exact_isentropic_base(x, M_c, L_n, L_D, geom=geom)
    W = W_from_prim(base["rho"][:, None], base["ux"][:, None], np.zeros((nx, 1)), base["p"][:, None], A_c, R_c)

    def rhs_1d(Ws: np.ndarray) -> np.ndarray:
        rho, ux, ut, p, _ = prim_from_W(Ws, A_c, R_c)
        P = np.stack([rho, ux, ut, p], axis=0)
        P_ext = np.empty((4, nx + 4, 1))
        P_ext[:, 2:-2, :] = P
        r_in, u_in, ut_in, p_in = inlet_choked_ghost(rho[0], ux[0], p[0], M_c)
        for k in (0, 1):
            P_ext[0, k] = r_in; P_ext[1, k] = u_in; P_ext[2, k] = ut_in; P_ext[3, k] = p_in
        P_ext[:, -2] = P[:, -1]
        P_ext[:, -1] = P[:, -1]
        s = van_leer_limiter(P_ext[:, 1:-1] - P_ext[:, :-2], P_ext[:, 2:] - P_ext[:, 1:-1])
        PL = P_ext[:, 1:-2] + 0.5 * s[:, :-1]
        PR = P_ext[:, 2:-1] - 0.5 * s[:, 1:]
        Fx = hllc_2d_normal(
            np.maximum(PL[0], 1e-10), PL[1], PL[2], np.maximum(PL[3], 1e-10),
            np.maximum(PR[0], 1e-10), PR[1], PR[2], np.maximum(PR[3], 1e-10),
        )
        Fx_W = np.stack([A_f * Fx[0], A_f * Fx[1], A_f * R_f * Fx[2], A_f * Fx[3]], axis=0)
        res = -(Fx_W[:, 1:] - Fx_W[:, :-1]) / dx
        res[1] += p * dA_dx
        return res

    t = 0.0; last_rhs = 1.0
    while t < t_relax:
        rho, ux, _, p, c = prim_from_W(W, A_c, R_c)
        dt = min(0.32 * dx / float(np.max(np.abs(ux) + c)), t_relax - t)
        k1 = rhs_1d(W)
        W1 = W + dt * k1
        k2 = rhs_1d(W1)
        W = 0.5 * (W + W1 + dt * k2)
        t += dt
        last_rhs = float(np.max(np.abs(k1)))
        if t > 8.0 and last_rhs < 5e-13:
            break
    return W, last_rhs


def simulate_2d_transonic(
    nx: int = 64,
    nth: int = 144,
    t_end: float = 45.0,
    M_c: float = 0.12,
    L_n: float = 0.50,
    geom: str = "annular",
    n_p: float = 0.0,
    n_v: float = 0.0,
    n_baffles: int = 0,
    L_baffle: float = 0.0,
    cav_spec: list[dict] | None = None,
    init_type: str = "spin_1T",
    A_eq: float = 1e-4,
    linear_flame: bool = False,
    periodic_bump: bool = False,
    W_base_1d: np.ndarray | None = None,
    cfl: float = 0.30,
    record_stride: int = 12,
) -> dict:
    """2D conservative variable-(A(x), R(x)) compressible Euler solver for a transonic CD nozzle."""
    g = build_grid_geometry(nx, M_c, L_n, geom=geom, periodic_bump=periodic_bump)
    x, dx, L_tot = g["x"], g["dx"], g["L_tot"]
    A_c, A_f, R_c, R_f = g["A_c"], g["A_f"], g["R_c"], g["R_f"]
    dA_dx = (A_f[1:] - A_f[:-1]) / dx
    dR_dx = (R_f[1:] - R_f[:-1]) / dx
    dth = 2.0 * np.pi / nth
    th = (np.arange(nth) + 0.5) * dth
    X2, TH2 = np.meshgrid(x, th, indexing="ij")

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

    cavs = []
    if cav_spec:
        Cd = 0.65
        for cf in cav_spec:
            nc = int(cf.get("n_cav", 5))
            phase_off = float(cf.get("phase_offset", 0.5))
            beta_k = float(cf["beta"])
            wH = float(cf["wH"])
            xiH = float(cf["xiH"])
            sig_o = float(cf.get("sigma_o", 0.045))
            nl_drag = bool(cf.get("nonlinear", True))
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
            cavs.append({
                "V_k": V_k, "wH": wH, "xiH": xiH, "r_nl": r_nl,
                "W": W_port, "Pc": np.zeros(nc), "qt": np.zeros(nc),
            })

    if periodic_bump:
        rho0 = np.full((nx, nth), RHO_BAR)
        ux0 = np.full((nx, nth), M_c)
        ut0 = np.zeros((nx, nth))
        p0 = np.full((nx, nth), P_BAR)
        W_base = W_from_prim(rho0, ux0, ut0, p0, A_c, R_c)
        base_rhs = 0.0
    elif init_type == "quiescent":
        rho0 = np.full((nx, nth), RHO_BAR)
        ux0 = np.zeros((nx, nth))
        ut0 = np.zeros((nx, nth))
        p0 = np.full((nx, nth), P_BAR)
        W_base = W_from_prim(rho0, ux0, ut0, p0, A_c, R_c)
        base_rhs = 0.0
    else:
        if W_base_1d is None:
            W_base_1d, base_rhs = relax_1d_transonic_base(nx, M_c, L_n, geom=geom)
        else:
            base_rhs = 0.0
        W_base = np.broadcast_to(W_base_1d, (4, nx, nth)).copy()

    rho_b, ux_b, ut_b, p_b, _ = prim_from_W(W_base, A_c, R_c)
    rho = rho_b.copy(); ux = ux_b.copy(); ut = ut_b.copy(); p = p_b.copy()

    ch_env = np.where(X2 <= L_A, 1.0, 0.0)
    if init_type in ("quiescent", "steady"):
        pass
    elif init_type == "periodic_2d_wave":
        dp = A_eq * (np.cos(2.0 * np.pi * X2 / L_tot + TH2) + 0.4 * np.sin(2.0 * TH2))
        rho += dp; p += dp; ux += 0.3 * A_eq * np.sin(2.0 * np.pi * X2 / L_tot); ut += 0.5 * A_eq * np.cos(TH2)
    elif init_type == "entropy_spot":
        rho += A_eq * np.exp(-0.5 * ((X2 - 0.55) / 0.08) ** 2)
    elif init_type == "spin_1T":
        dp = A_eq * np.cos(TH2) * ch_env
        rho += dp; p += dp; ut += A_eq * np.cos(TH2) * ch_env
    elif init_type == "spin_2T":
        dp = A_eq * np.cos(2.0 * TH2) * ch_env
        rho += dp; p += dp; ut += A_eq * np.cos(2.0 * TH2) * ch_env
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
            P_ext[:, -2, :] = P[:, -1, :]
            P_ext[:, -1, :] = P[:, -1, :]

        sx = van_leer_limiter(P_ext[:, 1:-1, :] - P_ext[:, :-2, :], P_ext[:, 2:, :] - P_ext[:, 1:-1, :])
        PLx = P_ext[:, 1:-2, :] + 0.5 * sx[:, :-1, :]
        PRx = P_ext[:, 2:-1, :] - 0.5 * sx[:, 1:, :]
        Fx = hllc_2d_normal(
            np.maximum(PLx[0], 1e-10), PLx[1], PLx[2], np.maximum(PLx[3], 1e-10),
            np.maximum(PRx[0], 1e-10), PRx[1], PRx[2], np.maximum(PRx[3], 1e-10),
        )
        Fx_W = np.stack([A_f * Fx[0], A_f * Fx[1], A_f * R_f * Fx[2], A_f * Fx[3]], axis=0)

        P_th = np.concatenate([P[:, :, -2:], P, P[:, :, :2]], axis=2)
        dL = P_th[:, :, 1:-1] - P_th[:, :, :-2]
        dR = P_th[:, :, 2:] - P_th[:, :, 1:-1]
        if n_baffles > 0 and L_baffle > 0.0:
            bm_r = np.roll(baffle_mask, -1, axis=1)
            dL[:, :, 1:-1] = np.where(baffle_mask[None, :, :], 0.0, dL[:, :, 1:-1])
            dR[:, :, 1:-1] = np.where(bm_r[None, :, :], 0.0, dR[:, :, 1:-1])
        st = van_leer_limiter(dL, dR)
        PLt = P_th[:, :, 1:-2] + 0.5 * st[:, :, :-1]
        PRt = P_th[:, :, 2:-1] - 0.5 * st[:, :, 1:]
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
            G_right = np.where(np.roll(baffle_mask, -1, axis=1)[None, :, :], np.roll(GwL, -1, axis=2), Gt_W[:, :, 1:])
            dGt = (G_right - G_left) / dth
        else:
            dGt = (Gt_W[:, :, 1:] - Gt_W[:, :, :-1]) / dth

        rhs_W = -(Fx_W[:, 1:, :] - Fx_W[:, :-1, :]) / dx - dGt
        rhs_W[1] += p_s * dA_dx + (A_c * rho_s * ut_s * ut_s / R_c) * dR_dx

        dfp = np.zeros_like(fp_s); dfv = np.zeros_like(fv_s)
        if has_flame:
            p_sense = np.sum(wf_col * (p_s - p_b), axis=0) * dx
            u_sense = np.sum(wf_col * ut_s, axis=0) * dx
            dfp[0] = OMEGA_HP * (p_sense - fp_s[0])
            dfp[1] = inv_ts * ((p_sense - fp_s[0]) - fp_s[1])
            for r in range(1, 4):
                dfp[r + 1] = inv_ts * (fp_s[r] - fp_s[r + 1])
            dfv[0] = OMEGA_HP * (u_sense - fv_s[0])
            dfv[1] = inv_ts * ((u_sense - fv_s[0]) - fv_s[1])
            for r in range(1, 4):
                dfv[r + 1] = inv_ts * (fv_s[r] - fv_s[r + 1])
            qp_out = fp_s[4] if linear_flame else phi_flame(fp_s[4])
            qv_out = fv_s[4] if linear_flame else phi_flame(fv_s[4])
            q_eff = n_p * qp_out + n_v * qv_out
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
    ts, e_ch, e_1t, e_2t, p_max_h, p_probe = [], [], [], [], [], []
    idx_probe = int(np.argmin(np.abs(x - L_A)))
    w_1T = np.exp(-1j * th); w_2T = np.exp(-2j * th)
    max_cons_err = np.zeros(4)

    while t < t_end - 1e-12:
        rho_c, ux_c, ut_c, p_c, c_c = prim_from_W(W, A_c, R_c)
        lam_x = float(np.max(np.abs(ux_c) + c_c)) / dx
        lam_t = float(np.max((np.abs(ut_c) + c_c) / R_c)) / dth
        dt = min(cfl / (lam_x + lam_t), t_end - t)

        if step % record_stride == 0:
            dp_ch = (p_c - p_b)[ch_mask]
            dux_ch = (ux_c - ux_b)[ch_mask]
            dut_ch = ut_c[ch_mask]
            ts.append(t)
            e_ch.append(0.5 * float(np.mean(dp_ch * dp_ch + dux_ch * dux_ch + dut_ch * dut_ch)))
            e_1t.append(float(np.mean(np.abs(np.mean(dp_ch * w_1T[None, :], axis=1)) ** 2)))
            e_2t.append(float(np.mean(np.abs(np.mean(dp_ch * w_2T[None, :], axis=1)) ** 2)))
            p_max_h.append(float(np.max(np.abs(dp_ch)) / P_BAR * 100.0))
            p_probe.append(float(np.mean(p_c[idx_probe] - p_b[idx_probe]) / P_BAR * 100.0))
            if periodic_bump:
                cur_tot = np.array([float(np.sum(W[k]) * dx * dth) for k in range(4)])
                max_cons_err = np.maximum(max_cons_err, np.abs(cur_tot - init_totals) / np.maximum(np.abs(init_totals), 1.0))

        c0 = [(cf["Pc"], cf["qt"]) for cf in cavs]
        k1W, k1p, k1v, k1c = compute_rhs(W, fp, fv, c0)
        W1 = W + dt * k1W; fp1 = fp + dt * k1p; fv1 = fv + dt * k1v
        c1 = [(s0 + dt * ds1, v0 + dt * dv1) for (s0, v0), (ds1, dv1) in zip(c0, k1c)]

        k2W, k2p, k2v, k2c = compute_rhs(W1, fp1, fv1, c1)
        W = 0.5 * (W + W1 + dt * k2W)
        fp = 0.5 * (fp + fp1 + dt * k2p)
        fv = 0.5 * (fv + fv1 + dt * k2v)
        for cf, (s0, v0), (s1, v1), (ds2, dv2) in zip(cavs, c0, c1, k2c):
            cf["Pc"] = 0.5 * (s0 + s1 + dt * ds2)
            cf["qt"] = 0.5 * (v0 + v1 + dt * dv2)
        t += dt; step += 1

    rho_f, ux_f, ut_f, p_f, _ = prim_from_W(W, A_c, R_c)
    dp_ch = (p_f - p_b)[ch_mask]; dux_ch = (ux_f - ux_b)[ch_mask]; dut_ch = ut_f[ch_mask]
    ts.append(t)
    e_ch.append(0.5 * float(np.mean(dp_ch * dp_ch + dux_ch * dux_ch + dut_ch * dut_ch)))
    e_1t.append(float(np.mean(np.abs(np.mean(dp_ch * w_1T[None, :], axis=1)) ** 2)))
    e_2t.append(float(np.mean(np.abs(np.mean(dp_ch * w_2T[None, :], axis=1)) ** 2)))
    p_max_h.append(float(np.max(np.abs(dp_ch)) / P_BAR * 100.0))
    p_probe.append(float(np.mean(p_f[idx_probe] - p_b[idx_probe]) / P_BAR * 100.0))

    return {
        "t": np.array(ts), "e_ch": np.array(e_ch), "e_1t": np.array(e_1t), "e_2t": np.array(e_2t),
        "p_max_pct": np.array(p_max_h), "p_probe_pct": np.array(p_probe),
        "quiescent_err": float(np.max(np.abs(W - W0_init))),
        "periodic_cons_err": {
            "mass": float(max_cons_err[0]), "mom_x": float(max_cons_err[1]),
            "ang_mom": float(max_cons_err[2]), "energy": float(max_cons_err[3]),
        },
        "base_rhs_max": base_rhs,
    }


def _sim_worker(args):
    lbl, kw = args
    return lbl, simulate_2d_transonic(**kw)


def run_batch(tasks: list[tuple[str, dict]], max_workers: int | None = None) -> dict:
    nw = max_workers or min(6, max(1, (os.cpu_count() or 2) - 2), len(tasks))
    if nw <= 1 or len(tasks) <= 1:
        return dict(_sim_worker(t) for t in tasks)
    with ProcessPoolExecutor(max_workers=nw) as ex:
        return dict(ex.map(_sim_worker, tasks))


def fit_alpha(res: dict, t_lo: float = 20.0, t_hi: float = 52.0, key: str = "e_ch") -> float:
    t = res["t"]; m = (t >= t_lo) & (t <= t_hi)
    return float(np.polyfit(t[m], 0.5 * np.log(np.maximum(res[key][m], 1e-30)), 1)[0])


def stats_nl(res: dict, win_lo: float = 18.0, win_hi: float = 55.0) -> dict:
    t = res["t"]; e = res["e_ch"]; pm = res["p_max_pct"]
    t_end = float(t[-1])
    m_late = t >= 0.80 * t_end
    m_mid = (t >= 0.50 * t_end) & (t <= 0.70 * t_end)
    e_late = float(np.mean(e[m_late]))
    e_mid = float(np.mean(e[m_mid]))
    m_win = (t >= win_lo) & (t <= min(win_hi, t_end))
    a_win = float(np.polyfit(t[m_win], 0.5 * np.log(np.maximum(e[m_win], 1e-30)), 1)[0]) if np.sum(m_win) >= 4 else 0.0
    m_tail = t >= 0.75 * t_end
    a_tail = float(np.polyfit(t[m_tail], 0.5 * np.log(np.maximum(e[m_tail], 1e-30)), 1)[0]) if np.sum(m_tail) >= 4 else 0.0
    e1_l = float(np.mean(res["e_1t"][m_late]))
    e2_l = float(np.mean(res["e_2t"][m_late]))
    return {
        "t_end": t_end,
        "p_over_pbar_pct": float(np.mean(pm[m_late])),
        "e_wave_late": e_late,
        "e_wave_late_over_mid": e_late / max(e_mid, 1e-30),
        "e2_over_e1_late": e2_l / max(e1_l, 1e-30),
        "alpha_fit_win": a_win,
        "alpha_tail": a_tail,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 90)
    print("Entry 012 (Hardened per Peer Review #9): 2D Thin-Shell Transonic CD Nozzle")
    print("=" * 90)

    # ------------------------------------------------------------------
    # Experiment 1: Benchmarks, Multi-Domain Compact Limit (M6) & Clean Exp 1D (M2)
    # ------------------------------------------------------------------
    print("\n--- Experiment 1: Benchmarks, Multi-Domain Compact Limit (M6) & Clean Exp 1D (M2) ---")

    steady_conv = {}
    for nx_s in (60, 100, 160):
        W1, rhs1 = relax_1d_transonic_base(nx_s, 0.12, 0.50, geom="bell")
        g_s = build_grid_geometry(nx_s, 0.12, 0.50, geom="bell")
        rho_s, ux_s, _, p_s, c_s = prim_from_W(W1, g_s["A_c"], g_s["R_c"])
        ex_s = exact_isentropic_base(g_s["x"], 0.12, 0.50, L_D, geom="bell")
        mf_err = float(np.max(np.abs(g_s["A_c"][:, 0] * rho_s[:, 0] * ux_s[:, 0] - 0.12)))
        m_err = float(np.sqrt(np.mean((ux_s[:, 0] / c_s[:, 0] - ex_s["M"]) ** 2)))
        steady_conv[f"nx{nx_s}"] = {
            "rhs_max": rhs1, "mass_flux_max_err": mf_err,
            "mach_l2_err": m_err, "M_exit": float(ux_s[-1, 0] / c_s[-1, 0]),
        }
        print(f"  Exp 1C Steady [nx={nx_s:3d}]: rhs={rhs1:.1e} | mf_err={mf_err:.2e} | M_L2={m_err:.2e} | M_e={ux_s[-1,0]/c_s[-1,0]:.4f}")

    # Multi-domain compact L_n -> 0 limit (M6: cleanly resolves L_n = 0.02!)
    mc_012 = exact_compact_mc_eig(0.12, 1, omega_window=(0.55, 1.50))
    mc_018 = exact_compact_mc_eig(0.18, 1, omega_window=(0.55, 1.50))
    compact_limit = {}
    for ln_c in (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.35, 0.50):
        v12 = exact_transonic_md_eig(0.12, 1, ln_c, geom="annular", omega_window=(0.55, 1.50), N0=44, N1=64)
        v18 = exact_transonic_md_eig(0.18, 1, ln_c, geom="annular", omega_window=(0.55, 1.50), N0=44, N1=64)
        compact_limit[f"Ln_{ln_c:.2f}"] = {
            "ann_M012_alpha": v12.real, "ann_M012_omega": v12.imag,
            "ann_M018_alpha": v18.real, "ann_M018_omega": v18.imag,
        }
    print(f"  Exp 1D MD Compact limit M=0.12: MC={mc_012.real:+.5f} | Ann(Ln=0.02)={compact_limit['Ln_0.02']['ann_M012_alpha']:+.5f} | Ann(Ln=0.05)={compact_limit['Ln_0.05']['ann_M012_alpha']:+.5f}")
    print(f"  Exp 1D MD Compact limit M=0.18: MC={mc_018.real:+.5f} | Ann(Ln=0.02)={compact_limit['Ln_0.02']['ann_M018_alpha']:+.5f}")

    base_cache = {}
    for key_b in [
        (64, 0.12, 0.50, "annular"), (96, 0.12, 0.50, "annular"),
        (64, 0.12, 0.50, "bell"),    (96, 0.12, 0.50, "bell"),
        (64, 0.12, 0.70, "bell"),    (96, 0.12, 0.70, "bell"),
        (96, 0.18, 0.50, "bell"),
        (64, 0.16, 0.50, "annular"), (96, 0.16, 0.50, "annular"),
        (64, 0.16, 0.50, "bell"),    (96, 0.16, 0.50, "bell"),
        (64, 0.12, 0.20, "bell"),    (80, 0.12, 0.20, "bell"), (120, 0.12, 0.20, "bell"),
        (80, 0.12, 0.50, "bell"),    (120, 0.12, 0.50, "bell"),
        (80, 0.12, 0.50, "annular"), (120, 0.12, 0.50, "annular"),
    ]:
        base_cache[key_b] = relax_1d_transonic_base(*key_b)[0]

    exp1_tasks = [
        ("quiescent_ann", dict(nx=48, nth=64, t_end=10.0, M_c=0.12, L_n=0.50, geom="annular", init_type="quiescent", A_eq=0.0, record_stride=20)),
        ("quiescent_bell", dict(nx=48, nth=64, t_end=10.0, M_c=0.12, L_n=0.50, geom="bell", init_type="quiescent", A_eq=0.0, record_stride=20)),
        ("cons_periodic_bell", dict(nx=48, nth=96, t_end=8.0, M_c=0.12, L_n=0.50, geom="bell", init_type="periodic_2d_wave", A_eq=0.10, periodic_bump=True, record_stride=20)),
        # Annular 1T runs to t_end = 105 so t in [60, 100] clears the T_beat = 32.1 beating window (M2!)
        ("pde_ann_M012_Ln050_1T_96", dict(nx=96, nth=200, t_end=105.0, M_c=0.12, L_n=0.50, geom="annular", init_type="spin_1T", A_eq=1e-4, W_base_1d=base_cache[(96, 0.12, 0.50, "annular")], record_stride=16)),
        ("pde_bell_M012_Ln050_1T_64", dict(nx=64, nth=144, t_end=55.0, M_c=0.12, L_n=0.50, geom="bell", init_type="spin_1T", A_eq=1e-4, W_base_1d=base_cache[(64, 0.12, 0.50, "bell")])),
        ("pde_bell_M012_Ln050_1T_96", dict(nx=96, nth=200, t_end=55.0, M_c=0.12, L_n=0.50, geom="bell", init_type="spin_1T", A_eq=1e-4, W_base_1d=base_cache[(96, 0.12, 0.50, "bell")])),
        ("pde_bell_M012_Ln070_1T_64", dict(nx=64, nth=144, t_end=55.0, M_c=0.12, L_n=0.70, geom="bell", init_type="spin_1T", A_eq=1e-4, W_base_1d=base_cache[(64, 0.12, 0.70, "bell")])),
        ("pde_bell_M012_Ln070_1T_96", dict(nx=96, nth=200, t_end=55.0, M_c=0.12, L_n=0.70, geom="bell", init_type="spin_1T", A_eq=1e-4, W_base_1d=base_cache[(96, 0.12, 0.70, "bell")])),
        ("pde_bell_M018_Ln050_1T_96", dict(nx=96, nth=200, t_end=55.0, M_c=0.18, L_n=0.50, geom="bell", init_type="spin_1T", A_eq=1e-4, W_base_1d=base_cache[(96, 0.18, 0.50, "bell")])),
        ("pde_ann_M016_Ln050_2T_64", dict(nx=64, nth=144, t_end=55.0, M_c=0.16, L_n=0.50, geom="annular", init_type="spin_2T", A_eq=1e-4, W_base_1d=base_cache[(64, 0.16, 0.50, "annular")])),
        ("pde_ann_M016_Ln050_2T_96", dict(nx=96, nth=200, t_end=55.0, M_c=0.16, L_n=0.50, geom="annular", init_type="spin_2T", A_eq=1e-4, W_base_1d=base_cache[(96, 0.16, 0.50, "annular")])),
        ("pde_bell_M016_Ln050_2T_64", dict(nx=64, nth=144, t_end=55.0, M_c=0.16, L_n=0.50, geom="bell", init_type="spin_2T", A_eq=1e-4, W_base_1d=base_cache[(64, 0.16, 0.50, "bell")])),
        ("pde_bell_M016_Ln050_2T_96", dict(nx=96, nth=200, t_end=55.0, M_c=0.16, L_n=0.50, geom="bell", init_type="spin_2T", A_eq=1e-4, W_base_1d=base_cache[(96, 0.16, 0.50, "bell")])),
    ]
    exp1_res = run_batch(exp1_tasks)
    ce = exp1_res["cons_periodic_bell"]["periodic_cons_err"]
    print(f"  Exp 1A Quiescent: annular={exp1_res['quiescent_ann']['quiescent_err']:.2e}, bell={exp1_res['quiescent_bell']['quiescent_err']:.2e}")
    print(f"  Exp 1B Periodic bell cons: mass={ce['mass']:.2e}, ang_mom={ce['ang_mom']:.2e}, energy={ce['energy']:.2e}")

    cheb_vs_pde = {}
    for tag, Mb, m_mode, ln_v, gm, lbl64, lbl96, wlo, whi in [
        ("ann_M012_Ln050_1T",  0.12, 1, 0.50, "annular", None,                        "pde_ann_M012_Ln050_1T_96",  60.0, 100.0),
        ("bell_M012_Ln050_1T", 0.12, 1, 0.50, "bell",    "pde_bell_M012_Ln050_1T_64", "pde_bell_M012_Ln050_1T_96", 20.0, 52.0),
        ("bell_M012_Ln070_1T", 0.12, 1, 0.70, "bell",    "pde_bell_M012_Ln070_1T_64", "pde_bell_M012_Ln070_1T_96", 20.0, 52.0),
        ("bell_M018_Ln050_1T", 0.18, 1, 0.50, "bell",    None,                        "pde_bell_M018_Ln050_1T_96", 20.0, 52.0),
        ("ann_M016_Ln050_2T",  0.16, 2, 0.50, "annular", "pde_ann_M016_Ln050_2T_64",  "pde_ann_M016_Ln050_2T_96",  20.0, 52.0),
        ("bell_M016_Ln050_2T", 0.16, 2, 0.50, "bell",    "pde_bell_M016_Ln050_2T_64", "pde_bell_M016_Ln050_2T_96", 20.0, 52.0),
    ]:
        ww = (0.55, 1.50) if m_mode == 1 else (1.50, 3.00)
        v_lo = exact_transonic_md_eig(Mb, m_mode, ln_v, geom=gm, omega_window=ww, N0=40, N1=56)
        v_hi = exact_transonic_md_eig(Mb, m_mode, ln_v, geom=gm, omega_window=ww, N0=52, N1=72)
        a96 = fit_alpha(exp1_res[lbl96], wlo, whi)
        a64 = fit_alpha(exp1_res[lbl64], wlo, whi) if lbl64 else None
        cheb_vs_pde[tag] = {
            "exact_md_lo": v_lo.real, "exact_md_hi": v_hi.real, "omega": v_hi.imag,
            "spectral_diff": abs(v_hi - v_lo), "fit_window": [wlo, whi],
            "pde_64x144": a64, "pde_96x200": a96,
            "num_damp_64": (a64 - v_hi.real) if a64 is not None else None,
            "num_damp_96": a96 - v_hi.real,
        }
        s64 = f"64x144={a64:+.5f} (nd={a64-v_hi.real:+.5f})" if a64 is not None else "64x144=  --   "
        print(f"  Exp 1D [{tag:19s}] exact={v_hi.real:+.5f} (spec={abs(v_hi-v_lo):.1e}) | {s64} | 96x200={a96:+.5f} (nd={a96-v_hi.real:+.5f}) [{int(wlo)}-{int(whi)}]")

    # ------------------------------------------------------------------
    # Experiment 2: Like-for-Like Entropy-Noise Filtering (C3, M3, 1x & 2x grids, t_end = 26)
    # ------------------------------------------------------------------
    print("\n--- Experiment 2: Like-for-Like Finite-Nozzle Entropy Filtering (1x & 2x grids, t_end=26) ---")
    mc_ent_200 = simulate_1d_compact_mc_entropy(nx=200, M_c=0.12, t_end=20.0)
    mc_ent_400 = simulate_1d_compact_mc_entropy(nx=400, M_c=0.12, t_end=20.0)
    p_mc1 = mc_ent_400["p_first_pass_pct"]
    p_mc2 = mc_ent_400["p_second_pass_pct"]
    print(f"  Like-for-like Compact MC (drho=-0.05, 0 swirl): nx=200 1st={mc_ent_200['p_first_pass_pct']:+.4f}% / 2nd={mc_ent_200['p_second_pass_pct']:+.4f}% | nx=400 1st={p_mc1:+.4f}% / 2nd={p_mc2:+.4f}%")

    ent_Lns = (0.15, 0.35, 0.70, 1.10)
    ent_tasks = []
    for ln_e in ent_Lns:
        for gmult, gtag in [(1, "1x"), (2, "2x")]:
            nx_e = int(round(140 * gmult * (L_A + ln_e + L_D) / (L_A + 0.50 + L_D)))
            W1_e, _ = relax_1d_transonic_base(nx_e, 0.12, ln_e, geom="annular", t_relax=300.0)
            ent_tasks.append((
                f"ent_Ln{int(round(ln_e * 100)):03d}_{gtag}",
                dict(nx=nx_e, nth=4, t_end=26.0, M_c=0.12, L_n=ln_e, geom="annular",
                     init_type="entropy_spot", A_eq=-0.05, W_base_1d=W1_e, record_stride=4),
            ))
    ent_res = run_batch(ent_tasks)
    ent_fp_win = {0.15: (6.5, 10.0), 0.35: (7.5, 11.5), 0.70: (9.0, 14.0), 1.10: (10.5, 16.5)}
    ent_sp_win = {0.15: (10.2, 14.5), 0.35: (11.8, 16.0), 0.70: (14.2, 19.0), 1.10: (16.6, 22.0)}
    entropy_filter = {
        "compact_mc_like_for_like": {
            "nx200_1st_pct": mc_ent_200["p_first_pass_pct"], "nx200_2nd_pct": mc_ent_200["p_second_pass_pct"],
            "nx400_1st_pct": p_mc1, "nx400_1st_t": mc_ent_400["t_first_pass"],
            "nx400_2nd_pct": p_mc2, "nx400_2nd_t": mc_ent_400["t_second_pass"],
        },
        "finite_Ln": {},
    }
    for ln_e in ent_Lns:
        rec = {"L_n": ln_e}
        for gtag in ("1x", "2x"):
            rr = ent_res[f"ent_Ln{int(round(ln_e * 100)):03d}_{gtag}"]
            t = rr["t"]; pp = rr["p_probe_pct"]
            w1a, w1b = ent_fp_win[ln_e]; m1 = (t >= w1a) & (t <= w1b)
            w2a, w2b = ent_sp_win[ln_e]; m2 = (t >= w2a) & (t <= w2b)
            p1 = float(np.min(pp[m1])); t1 = float(t[m1][np.argmin(pp[m1])])
            p2 = float(np.min(pp[m2])); t2 = float(t[m2][np.argmin(pp[m2])])
            rec[f"p1_{gtag}_pct"] = p1; rec[f"t1_{gtag}"] = t1
            rec[f"p2_{gtag}_pct"] = p2; rec[f"t2_{gtag}"] = t2
            rec[f"ratio_p1_{gtag}_vs_mc400"] = p1 / p_mc1
            rec[f"ratio_p2_{gtag}_vs_mc400"] = p2 / p_mc2
        entropy_filter["finite_Ln"][f"Ln_{ln_e:.2f}"] = rec
        print(f"  [Ln={ln_e:.2f}] 1st-pass: 1x={rec['p1_1x_pct']:+.4f}% / 2x={rec['p1_2x_pct']:+.4f}% ({rec['ratio_p1_2x_vs_mc400']*100:.1f}% of MC400) | 2nd-pass: 2x={rec['p2_2x_pct']:+.4f}% at t={rec['t2_2x']:.2f} ({rec['ratio_p2_2x_vs_mc400']*100:.1f}%)")

    # ------------------------------------------------------------------
    # Experiment 3: Bell Cutoff vs Annular Tuning (MD), Contour Family (M1) & m<=5 Hopf (M4)
    # ------------------------------------------------------------------
    print("\n--- Experiment 3: MD L_n-Sweeps, Contour Sensitivity (M1) & m=0..5 Hopf Maps (M4) ---")
    Ln_sweep = [0.10, 0.15, 0.20, 0.28, 0.35, 0.42, 0.50, 0.60, 0.70, 0.80]
    ln_curves = {}
    for gm in ("annular", "bell"):
        for Mb in (0.12, 0.18):
            rows = []
            for ln_v in Ln_sweep:
                v1 = exact_transonic_md_eig(Mb, 1, ln_v, geom=gm, omega_window=(0.55, 1.50), N0=40, N1=56)
                v2 = exact_transonic_md_eig(Mb, 2, ln_v, geom=gm, omega_window=(1.50, 3.20), N0=40, N1=56)
                rows.append({"L_n": ln_v, "alpha_1T": v1.real, "omega_1T": v1.imag, "alpha_2T": v2.real, "omega_2T": v2.imag})
            ln_curves[f"{gm}_M{int(round(Mb * 100)):02d}"] = rows

    # Contour-family sensitivity (M1): quartic_c2 vs cubic_c1 vs cosine_c1
    contour_sens = {}
    for ct in ("quartic_c2", "cubic_c1", "cosine_c1"):
        contour_sens[ct] = {
            "ann_M012_Ln050_1T": exact_transonic_md_eig(0.12, 1, 0.50, "annular", ct, omega_window=(0.55, 1.50)).real,
            "bell_M012_Ln050_1T": exact_transonic_md_eig(0.12, 1, 0.50, "bell", ct, omega_window=(0.55, 1.50)).real,
            "bell_M012_Ln070_1T": exact_transonic_md_eig(0.12, 1, 0.70, "bell", ct, omega_window=(0.55, 1.50)).real,
            "bell_M012_Ln080_1T": exact_transonic_md_eig(0.12, 1, 0.80, "bell", ct, omega_window=(0.55, 1.50)).real,
            "bell_M018_Ln050_1T": exact_transonic_md_eig(0.18, 1, 0.50, "bell", ct, omega_window=(0.55, 1.50)).real,
            "bell_M012_Ln050_2T": exact_transonic_md_eig(0.12, 2, 0.50, "bell", ct, omega_window=(1.50, 3.00)).real,
            "bell_M016_Ln050_2T": exact_transonic_md_eig(0.16, 2, 0.50, "bell", ct, omega_window=(1.50, 3.00)).real,
            "bell_M010_Ln050_m3": exact_transonic_md_eig(0.10, 3, 0.50, "bell", ct, omega_window=(2.80, 4.20)).real,
        }
        cs = contour_sens[ct]
        print(f"  Contour [{ct:10s}]: ann1T={cs['ann_M012_Ln050_1T']:+.5f} | bell1T(0.50/0.70/0.80)={cs['bell_M012_Ln050_1T']:+.5f}/{cs['bell_M012_Ln070_1T']:+.5f}/{cs['bell_M012_Ln080_1T']:+.5f} | bell2T(0.12)={cs['bell_M012_Ln050_2T']:+.5f}")

    # Multi-mode global Hopf maps extended to m = 0..5 (M4!)
    M_grid = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22]
    hopf_maps = {}
    for tag, mode_fn in [
        ("compact_mc", lambda Mb, mm, nv: exact_compact_mc_eig(Mb, mm, n_p=nv, omega_window=(0.55, 1.50) if mm == 1 else (0.05, 7.0), N=44)),
        ("annular_Ln050", lambda Mb, mm, nv: exact_transonic_md_eig(Mb, mm, 0.50, "annular", n_p=nv, omega_window=(0.55, 1.50) if mm == 1 else (0.05, 7.0), N0=36, N1=48)),
        ("bell_Ln050", lambda Mb, mm, nv: exact_transonic_md_eig(Mb, mm, 0.50, "bell", n_p=nv, omega_window=(0.55, 1.50) if mm == 1 else (0.05, 7.0), N0=36, N1=48)),
    ]:
        rows = []
        for Mb in M_grid:
            npH = {}; colds = {}
            for mm in (0, 1, 2, 3, 4, 5):
                vc = mode_fn(Mb, mm, 0.0)
                colds[mm] = vc.real
                if vc.real >= 0.0:
                    npH[mm] = 0.0
                else:
                    f = lambda nv, m_k=mm: mode_fn(Mb, m_k, nv).real
                    npH[mm] = float(brentq(f, 0.0, 0.85)) if f(0.85) > 0.0 else 0.85
            gm_m = min(npH, key=npH.get)
            rows.append({
                "M_c": Mb,
                "cold_1T": colds[1], "cold_2T": colds[2], "cold_3T": colds[3], "cold_4T": colds[4], "cold_5T": colds[5],
                "n_pH_m0": npH[0], "n_pH_1T": npH[1], "n_pH_2T": npH[2], "n_pH_3T": npH[3], "n_pH_4T": npH[4], "n_pH_5T": npH[5],
                "n_pH_global": npH[gm_m], "limiting_m": int(gm_m),
            })
        hopf_maps[tag] = rows
        print(f"  Hopf m<=5 [{tag:13s}]: " + " | ".join(f"M={r['M_c']:.2f}:{r['n_pH_global']:.3f}(m{r['limiting_m']})" for r in rows))

    # Inviscid bell M_c = 0.10 whistles (m = 3 & m = 4), eigenvector ratio, and Re = 1e5 quench (M4!)
    w3_val, w3_ratio = exact_transonic_md_eig(0.10, 3, 0.50, "bell", omega_window=(3.0, 4.2), N0=48, N1=68, return_vec=True)
    w4_val, w4_ratio = exact_transonic_md_eig(0.10, 4, 0.50, "bell", omega_window=(4.0, 5.2), N0=48, N1=68, return_vec=True)
    w3_re1e5 = exact_transonic_md_eig(0.10, 3, 0.50, "bell", Re=1e5, omega_window=(3.0, 4.2), N0=44, N1=60)
    w4_re1e5 = exact_transonic_md_eig(0.10, 4, 0.50, "bell", Re=1e5, omega_window=(4.0, 5.2), N0=44, N1=60)
    whistle_audit = {
        "m3_inviscid_alpha": w3_val.real, "m3_omega": w3_val.imag, "m3_ut_throat_over_entrance": w3_ratio, "m3_Re1e5_alpha": w3_re1e5.real,
        "m4_inviscid_alpha": w4_val.real, "m4_omega": w4_val.imag, "m4_ut_throat_over_entrance": w4_ratio, "m4_Re1e5_alpha": w4_re1e5.real,
    }
    print(f"  Bell M=0.10 whistles: m=3 {w3_val.real:+.5f} ({w3_val.imag:.3f}i, |v_t/v_in|={w3_ratio:.2f}, Re1e5={w3_re1e5.real:+.4f}) | m=4 {w4_val.real:+.5f} ({w4_val.imag:.3f}i, Re1e5={w4_re1e5.real:+.4f})")

    # ------------------------------------------------------------------
    # Experiment 4: Resolved 120x300 (t = 300) Nonlinear 2T Cascade Audit (C1, C2, m1-m3)
    # ------------------------------------------------------------------
    print("\n--- Experiment 4: Resolved 120x300 (t=300) Nonlinear Audit (C1, C2) ---")
    cf_bell_hyb = [
        {"beta": 0.016, "wH": 0.96, "xiH": 0.12, "sigma_o": 0.045, "n_cav": 5, "phase_offset": 0.5, "nonlinear": True},
        {"beta": 0.004, "wH": 2.01, "xiH": 0.12, "sigma_o": 0.015, "n_cav": 5, "phase_offset": 0.5, "nonlinear": True},
    ]
    cf_ann_hyb = [
        {"beta": 0.016, "wH": 0.86, "xiH": 0.12, "sigma_o": 0.045, "n_cav": 5, "phase_offset": 0.5, "nonlinear": True},
        {"beta": 0.004, "wH": 2.00, "xiH": 0.12, "sigma_o": 0.015, "n_cav": 5, "phase_offset": 0.5, "nonlinear": True},
    ]

    exact_react_bare = {}
    for lbl_r, gm_r, ln_r in [("bell_Ln050", "bell", 0.50), ("bell_Ln020", "bell", 0.20), ("ann_Ln050", "annular", 0.50)]:
        v1 = exact_transonic_md_eig(0.12, 1, ln_r, gm_r, n_p=0.22, n_v=0.04, omega_window=(0.55, 1.50), N0=44, N1=64)
        v1_full = exact_transonic_md_eig(0.12, 1, ln_r, gm_r, n_p=0.22, n_v=0.04, omega_window=(0.05, 7.0), N0=44, N1=64)
        v2 = exact_transonic_md_eig(0.12, 2, ln_r, gm_r, n_p=0.22, n_v=0.04, omega_window=(1.40, 3.00), N0=44, N1=64)
        exact_react_bare[lbl_r] = {
            "alpha_1T": v1.real, "omega_1T": v1.imag,
            "alpha_m1_all_omega": v1_full.real, "omega_m1_all": v1_full.imag,
            "alpha_2T": v2.real, "omega_2T": v2.imag,
        }
        print(f"  Exact MD reacting [{lbl_r:10s}]: 1T={v1.real:+.5f}({v1.imag:.3f}i) | m=1 all-w={v1_full.real:+.5f}({v1_full.imag:.3f}i) | 2T={v2.real:+.5f}({v2.imag:.3f}i)")

    W64_b50 = base_cache[(64, 0.12, 0.50, "bell")]
    W64_b20 = base_cache[(64, 0.12, 0.20, "bell")]
    W80_b20 = base_cache[(80, 0.12, 0.20, "bell")]
    W120_b20 = base_cache[(120, 0.12, 0.20, "bell")]
    W80_b50 = base_cache[(80, 0.12, 0.50, "bell")]
    W80_a50 = base_cache[(80, 0.12, 0.50, "annular")]
    W120_b50 = base_cache[(120, 0.12, 0.50, "bell")]
    W120_a50 = base_cache[(120, 0.12, 0.50, "annular")]

    exp4_tasks = [
        # Linear checks (including lin_bell20 refinement 80x200 -> 120x300, m1)
        ("lin_bell50_bare_1T",   dict(nx=80, nth=200, t_end=60.0, M_c=0.12, L_n=0.50, geom="bell", n_p=0.22, n_v=0.04, init_type="spin_1T", A_eq=1e-4, W_base_1d=W80_b50)),
        ("lin_bell50_bare_2T",   dict(nx=80, nth=200, t_end=60.0, M_c=0.12, L_n=0.50, geom="bell", n_p=0.22, n_v=0.04, init_type="spin_2T", A_eq=1e-4, W_base_1d=W80_b50)),
        ("lin_bell20_bare_1T_80",dict(nx=80, nth=200, t_end=60.0, M_c=0.12, L_n=0.20, geom="bell", n_p=0.22, n_v=0.04, init_type="spin_1T", A_eq=1e-4, W_base_1d=W80_b20)),
        ("lin_bell20_bare_1T_120",dict(nx=120, nth=300, t_end=60.0, M_c=0.12, L_n=0.20, geom="bell", n_p=0.22, n_v=0.04, init_type="spin_1T", A_eq=1e-4, W_base_1d=W120_b20, record_stride=16)),
        ("lin_ann50_bare_1T",    dict(nx=80, nth=200, t_end=60.0, M_c=0.12, L_n=0.50, geom="annular", n_p=0.22, n_v=0.04, init_type="spin_1T", A_eq=1e-4, W_base_1d=W80_a50)),
        ("lin_bell50_hyb_1T",    dict(nx=80, nth=200, t_end=60.0, M_c=0.12, L_n=0.50, geom="bell", n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_bell_hyb, init_type="spin_1T", A_eq=1e-4, W_base_1d=W80_b50)),
        ("lin_bell50_hyb_2T",    dict(nx=80, nth=200, t_end=60.0, M_c=0.12, L_n=0.50, geom="bell", n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_bell_hyb, init_type="spin_2T", A_eq=1e-4, W_base_1d=W80_b50)),
        ("lin_ann50_hyb_1T",     dict(nx=80, nth=200, t_end=60.0, M_c=0.12, L_n=0.50, geom="annular", n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_ann_hyb, init_type="spin_1T", A_eq=1e-4, W_base_1d=W80_a50)),
        ("lin_ann50_hyb_2T",     dict(nx=80, nth=200, t_end=60.0, M_c=0.12, L_n=0.50, geom="annular", n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_ann_hyb, init_type="spin_2T", A_eq=1e-4, W_base_1d=W80_a50)),
        # Nonlinear shootout: bare bells + 80x200 vs resolved 120x300 (t = 300) for annular & hybrids (C1, C2!)
        ("NL_bell50_bare_1T",    dict(nx=64, nth=144, t_end=160.0, M_c=0.12, L_n=0.50, geom="bell", n_p=0.22, n_v=0.04, init_type="spin_1T", A_eq=0.055, W_base_1d=W64_b50, record_stride=16)),
        ("NL_bell50_bare_2T_80", dict(nx=80, nth=200, t_end=160.0, M_c=0.12, L_n=0.50, geom="bell", n_p=0.22, n_v=0.04, init_type="spin_2T", A_eq=0.055, W_base_1d=W80_b50, record_stride=20)),
        ("NL_bell20_bare_1T",    dict(nx=64, nth=144, t_end=160.0, M_c=0.12, L_n=0.20, geom="bell", n_p=0.22, n_v=0.04, init_type="spin_1T", A_eq=0.055, W_base_1d=W64_b20, record_stride=16)),
        ("NL_ann50_bare_1T_80",  dict(nx=80, nth=200, t_end=300.0, M_c=0.12, L_n=0.50, geom="annular", n_p=0.22, n_v=0.04, init_type="spin_1T", A_eq=0.055, W_base_1d=W80_a50, record_stride=24)),
        ("NL_ann50_bare_1T_120", dict(nx=120, nth=300, t_end=300.0, M_c=0.12, L_n=0.50, geom="annular", n_p=0.22, n_v=0.04, init_type="spin_1T", A_eq=0.055, W_base_1d=W120_a50, record_stride=32)),
        ("NL_bell50_hyb_1T_80",  dict(nx=80, nth=200, t_end=300.0, M_c=0.12, L_n=0.50, geom="bell", n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_bell_hyb, init_type="spin_1T", A_eq=0.055, W_base_1d=W80_b50, record_stride=24)),
        ("NL_bell50_hyb_1T_120", dict(nx=120, nth=300, t_end=300.0, M_c=0.12, L_n=0.50, geom="bell", n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_bell_hyb, init_type="spin_1T", A_eq=0.055, W_base_1d=W120_b50, record_stride=32)),
        ("NL_ann50_hyb_1T_80",   dict(nx=80, nth=200, t_end=300.0, M_c=0.12, L_n=0.50, geom="annular", n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_ann_hyb, init_type="spin_1T", A_eq=0.055, W_base_1d=W80_a50, record_stride=24)),
        ("NL_ann50_hyb_1T_120",  dict(nx=120, nth=300, t_end=300.0, M_c=0.12, L_n=0.50, geom="annular", n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_ann_hyb, init_type="spin_1T", A_eq=0.055, W_base_1d=W120_a50, record_stride=32)),
        ("NL_ann50_hyb_2T_80",   dict(nx=80, nth=200, t_end=240.0, M_c=0.12, L_n=0.50, geom="annular", n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_ann_hyb, init_type="spin_2T", A_eq=0.055, W_base_1d=W80_a50, record_stride=24)),
        ("NL_ann50_hyb_2T_120",  dict(nx=120, nth=300, t_end=240.0, M_c=0.12, L_n=0.50, geom="annular", n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, cav_spec=cf_ann_hyb, init_type="spin_2T", A_eq=0.055, W_base_1d=W120_a50, record_stride=32)),
    ]
    exp4_res = run_batch(exp4_tasks)
    exp4_stats = {}
    for cid, _ in exp4_tasks:
        st = stats_nl(exp4_res[cid])
        exp4_stats[cid] = st
        print(f"  [{cid:22s}] t={st['t_end']:5.0f} | P={st['p_over_pbar_pct']:5.2f}% | E={st['e_wave_late']:.2e} (l/m={st['e_wave_late_over_mid']:.3f}, e2/e1={st['e2_over_e1_late']:.1f}) | a={st['alpha_fit_win']:+.4f} (tail={st['alpha_tail']:+.4f})")

    # ------------------------------------------------------------------
    # Figure 1: MD Bell Evanescent-Cutoff Collapse vs. Annular Slot & m<=5 Hopf Map
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.6), dpi=150)

    L_arr = [d["L_n"] for d in ln_curves["annular_M12"]]
    ax1.plot(L_arr, [-d["alpha_1T"] for d in ln_curves["annular_M12"]], "o-", color="#1f77b4", lw=2.3, label="Annular slot ($R\\equiv 1$), $\\bar{M}_c = 0.12$ ($1T$)")
    ax1.plot(L_arr, [-d["alpha_1T"] for d in ln_curves["bell_M12"]], "s-", color="#d62728", lw=2.3, label="Shrinking bell ($R=\\sqrt{A}$), $\\bar{M}_c = 0.12$ ($1T$)")
    ax1.plot(L_arr, [-d["alpha_1T"] for d in ln_curves["annular_M18"]], "^--", color="#2ca02c", lw=2.0, label="Annular slot ($R\\equiv 1$), $\\bar{M}_c = 0.18$ ($1T$)")
    ax1.plot(L_arr, [-d["alpha_1T"] for d in ln_curves["bell_M18"]], "v--", color="#9467bd", lw=2.0, label="Shrinking bell ($R=\\sqrt{A}$), $\\bar{M}_c = 0.18$ ($1T$)")
    ax1.axhline(-mc_012.real, color="#1f77b4", ls=":", lw=1.4, alpha=0.8, label=f"Compact MC $\\bar{{M}}_c=0.12$ ($-\\alpha={-mc_012.real:.4f}$)")
    ax1.axhline(-mc_018.real, color="#2ca02c", ls=":", lw=1.4, alpha=0.8, label=f"Compact MC $\\bar{{M}}_c=0.18$ ($-\\alpha={-mc_018.real:.4f}$)")
    ax1.scatter([0.50], [-cheb_vs_pde["ann_M012_Ln050_1T"]["pde_96x200"]], s=90, facecolors="none", edgecolors="#0b3c5d", linewidths=2.0, zorder=6, label="2D Transonic PDE (96x200, clean window)")
    ax1.scatter([0.50, 0.70], [-cheb_vs_pde["bell_M012_Ln050_1T"]["pde_96x200"], -cheb_vs_pde["bell_M012_Ln070_1T"]["pde_96x200"]], s=90, facecolors="none", edgecolors="#7f0000", linewidths=2.0, zorder=6)
    ax1.scatter([0.50], [-cheb_vs_pde["bell_M018_Ln050_1T"]["pde_96x200"]], s=90, facecolors="none", edgecolors="#4b0082", linewidths=2.0, zorder=6)
    ax1.set_xlabel("Converging Nozzle Length $L_n / R_c$", fontsize=11)
    ax1.set_ylabel("Cold $1T$ Acoustic Damping Rate $-\\alpha_{1T}$", fontsize=11)
    ax1.set_title("2D Thin-Shell Bell Evanescent Collapse vs. Annular Tuning", fontsize=12, fontweight="bold")
    ax1.set_ylim(-0.005, 0.135)
    ax1.grid(True, alpha=0.28)
    ax1.legend(fontsize=8.1, loc="upper right", framealpha=0.92)

    M_a = [d["M_c"] for d in hopf_maps["compact_mc"]]
    ax2.plot(M_a, [d["n_pH_global"] for d in hopf_maps["compact_mc"]], "k--", lw=2.0, label="Compact Marble-Candel ($L_n \\to 0$, Entry 009)")
    ax2.plot(M_a, [d["n_pH_global"] for d in hopf_maps["annular_Ln050"]], "o-", color="#1f77b4", lw=2.4, label="Annular Slot ($R\\equiv 1, L_n=0.50$): global $m\\leq 5$")
    ax2.plot(M_a, [d["n_pH_1T"] for d in hopf_maps["annular_Ln050"]], "o:", color="#1f77b4", lw=1.4, alpha=0.75, label="Annular Slot ($L_n=0.50$): $1T$-only")
    ax2.plot(M_a, [d["n_pH_global"] for d in hopf_maps["bell_Ln050"]], "s-", color="#d62728", lw=2.4, label="Shrinking Bell ($R=\\sqrt{A}, L_n=0.50$): global $m\\leq 5$")
    ax2.axhline(0.22, color="#666666", ls="-.", lw=1.2, label="Benchmark flame gain $n_p = 0.22$")
    ax2.annotate("Bell $2T$ cutoff collapse\n($n_{pH}=0.060$, $3.8\\times$ below annular)", xy=(0.12, 0.0602), xytext=(0.126, 0.012),
                 fontsize=8.4, color="#b00000", arrowprops=dict(arrowstyle="->", color="#b00000", lw=1.1))
    ax2.annotate("Inviscid $m=3,4$ whistles\n(quenched for $\\mathrm{Re}\\leq 2\\times 10^5$)", xy=(0.10, 0.0), xytext=(0.103, 0.14),
                 fontsize=7.9, color="#7f0000", arrowprops=dict(arrowstyle="->", color="#7f0000", lw=1.0))
    ax2.set_xlabel("Chamber Mean Mach Number $\\bar{M}_c$", fontsize=11)
    ax2.set_ylabel("Multi-Mode Global Hopf Gain $n_{pH,\\mathrm{global}}$ ($m=0..5$)", fontsize=11)
    ax2.set_title("Global All-Mode Hopf Envelope ($L_n = 0.50$ vs. Compact MC)", fontsize=12, fontweight="bold")
    ax2.set_ylim(-0.015, 0.62)
    ax2.grid(True, alpha=0.28)
    ax2.legend(fontsize=8.0, loc="upper left", framealpha=0.92)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp012_bell_cutoff_vs_annular_tuning.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Like-for-Like Entropy Filtering & Resolved 120x300 (t=300) 2T Cascade Audit
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.6), dpi=150)

    ax1.plot(mc_ent_400["t"], mc_ent_400["p_exit_pct"], "k:", lw=1.8, label=f"Like-for-like Compact MC (nx=400, 1st={p_mc1:+.3f}%)")
    for ln_e, col, ls in [(0.15, "#d62728", "-"), (0.35, "#ff7f0e", "--"), (0.70, "#1f77b4", "-."), (1.10, "#2ca02c", "-")]:
        rr = ent_res[f"ent_Ln{int(round(ln_e * 100)):03d}_2x"]
        rec = entropy_filter["finite_Ln"][f"Ln_{ln_e:.2f}"]
        ax1.plot(rr["t"], rr["p_probe_pct"], color=col, ls=ls, lw=2.0,
                 label=f"$L_n = {ln_e:.2f}$ (2x grid: 1st={rec['p1_2x_pct']:+.3f}%, 2nd={rec['p2_2x_pct']:+.3f}%)")
        ax1.scatter([rec["t1_2x"]], [rec["p1_2x_pct"]], color=col, s=40, zorder=5)
    ax1.set_xlabel("Time $t$", fontsize=11)
    ax1.set_ylabel("Nozzle-Entrance Pressure $p'(x=L_a)/\\bar{p}$ (%)", fontsize=11)
    ax1.set_title("Like-for-Like Finite-Nozzle Filtering of Entropy Noise (2x Grid)", fontsize=12, fontweight="bold")
    ax1.set_xlim(5.0, 23.0)
    ax1.grid(True, alpha=0.28)
    ax1.legend(fontsize=8.0, loc="lower right", framealpha=0.92)

    for cid, lbl, col, ls in [
        ("NL_bell50_bare_1T",    "Bare Bell (Ln=0.50, 1T)", "#d62728", "-"),
        ("NL_bell50_bare_2T_80", "Bare Bell (Ln=0.50, 2T cutoff)", "#8c564b", ":"),
        ("NL_ann50_bare_1T_120", "Bare Annular (120x300: 2T cycle!)", "#1f77b4", "-."),
        ("NL_bell50_hyb_1T_120", "Bell + 2% Hybrid (120x300: 2T cascade!)", "#9467bd", "-"),
        ("NL_ann50_hyb_1T_120",  "Annular + 2% Hybrid (120x300: 2T cascade!)", "#2ca02c", "-"),
        ("NL_ann50_hyb_1T_80",   "Annular + 2% Hybrid (80x200: coarse artifact)", "#2ca02c", "--"),
    ]:
        rr = exp4_res[cid]; st = exp4_stats[cid]
        ax2.semilogy(rr["t"], np.maximum(rr["e_ch"], 1e-12), color=col, ls=ls, lw=2.1,
                     label=f"{lbl} (P={st['p_over_pbar_pct']:.2f}%)")
    ax2.set_xlabel("Time $t$", fontsize=11)
    ax2.set_ylabel("Chamber Wave Energy $E_{\\mathrm{wave}}(t)$", fontsize=11)
    ax2.set_title("Resolved 120x300 (t=300) Transonic 1T -> 2T Cascade Audit", fontsize=12, fontweight="bold")
    ax2.set_ylim(1e-7, 1e-2)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=7.9, loc="lower left", framealpha=0.92)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp012_entropy_filter_and_transonic_shocks.png", dpi=150)
    plt.close(fig)

    metrics = {
        "exp1_benchmarks": {
            "quiescent_annular_err": exp1_res["quiescent_ann"]["quiescent_err"],
            "quiescent_bell_err": exp1_res["quiescent_bell"]["quiescent_err"],
            "periodic_bell_cons_err": ce,
            "steady_transonic_conv": steady_conv,
            "compact_mc_ref": {"M012_alpha": mc_012.real, "M012_omega": mc_012.imag, "M018_alpha": mc_018.real, "M018_omega": mc_018.imag},
            "compact_limit_annular_md": compact_limit,
            "cheb_vs_pde_and_num_damp": cheb_vs_pde,
        },
        "exp2_entropy_filtering": entropy_filter,
        "exp3_cutoff_vs_annular": {
            "Ln_curves_md": ln_curves,
            "contour_sensitivity": contour_sens,
            "hopf_maps_Ln050_m0_to_5": hopf_maps,
            "bell_M010_whistle_audit": whistle_audit,
        },
        "exp4_transonic_shock_control": {
            "exact_react_bare_md": exact_react_bare,
            "pde_runs": exp4_stats,
        },
    }
    out_path = RESULTS_DIR / "exp012_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved hardened metrics -> {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
