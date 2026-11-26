"""Entry 011 (rocket-combustion): 2D Conservative Compressible Navier-Stokes Annular
Combustor — Vorticity-Wave Viscous Attenuation Law, Finite-Re Baffle-Tip Convergence,
and the No-Slip Viscous Wake Penalty (Hardened per Peer Review #8).

Upgrades Entry 010's inviscid 2D compressible Euler solver to full 2D conservative
compressible Navier-Stokes (Stokes' hypothesis lambda = -2/3*mu, Fourier heat conduction
kappa = mu / ((gamma - 1) * Pr), Pr = 0.72, effective acoustic Reynolds number
Re = rho_bar*c_bar*R/mu) with BOTH free-slip and no-slip adiabatic solid-wall baffles
(pre-relaxed to converged steady wakes at t = 150, max |dU/dt| ~ 1e-9) and nonlinear
faceplate Helmholtz cavity rings.

Hardening upgrades per independent cross-model Peer Review #8 (Claude):
  - C1: Adds direct nonlinear 64x200 spin_2T runs at M_bar=0.16 (Re=1000, 2000, 5000) and
        computes exact linear crossing Re_cross ~ 5960, showing 2T is linearly stabilized for
        Re <= 5960 but subcritically triggers at Re=2000 (4.38%) & Re=5000 (5.95%) and is
        nonlinearly quenched only for Re <= 1000 (A_0 <= 0.055).
  - C2 & M1: Pre-relaxes ALL no-slip steady wakes to t = 150 (max |dU/dt| ~ 1e-9 vs 1.4e-5
        at t=65), checks floor energy explicitly, and evaluates linear hybrid margins on the
        grid-consistent t in [18, 60] window across 40x120, 60x160, and 80x240.
  - M2 & M3: Adds 100x300 grid convergence for free-slip & no-slip Re=500 baffles (~18.9%
        converged no-slip wake penalty) and reports downstream wake velocity (x > L_b) rather
        than the upstream wall-adjacent cell minimum.
  - M4: Extends Exp 1D Chebyshev-vs-PDE runs to t=42 and fits on t in [20, 40] (0.4-1.4%
        agreement across all four cases, eliminating the higher-mode initial transient).
  - M5: Adds 60x160 nonlinear spin_2T test on the retuned hybrid (exposing a sustained 4.12%
        pure-2T upper branch) and A_0=0.010 vs 0.015 spin_1T bracket showing trigger onset in
        A_0 in (0.010, 0.015].
  - M6: Adds M_bar=0 bare NS run on 80x240 to decompose the converged -0.0260 damping into
        bulk Stokes-Kirchhoff (-0.0017), injector-face u_theta=0 shear (-0.0098), and sharp-tip
        viscous dissipation (-0.0145; blade no-slip minus free-slip < 0.1%).
  - M7 & Minors: Clarifies constant-eddy-viscosity closure scope, exact N=44/60 convergence,
        Re=5000 cold metric in JSON, periodic wall drag disclosure, and zoomed Fig 2 Right y-axis.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig
from scipy.optimize import brentq


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

GAMMA = 1.25
P_BAR = 1.0 / GAMMA       # = 0.80 so c_bar = 1.0
RHO_BAR = 1.0
L_A = 1.5
X_F = 0.25
SIG_F = 0.08
OMEGA_HP = 0.15
TAU_BAR = 0.55
Q_S = 0.036
KAPPA_NL = 4.5
CD_ORIFICE = 0.65
PR = 0.72

W1_COLD = 0.9893
W2_COLD = 1.9799


# ---------------------------------------------------------------------------
# 1. Exact 1D-in-x Chebyshev Linearized Compressible Navier-Stokes Eigen-Solver
# ---------------------------------------------------------------------------

def cheb_diff(N: int, L_a: float = L_A) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = np.arange(N + 1)
    z = np.cos(np.pi * k / N)
    c = np.ones(N + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c *= (-1.0) ** k
    Z = np.tile(z, (N + 1, 1))
    D = (np.outer(c, 1.0 / c)) / (Z.T - Z + np.eye(N + 1))
    D -= np.diag(np.sum(D, axis=1))
    x = 0.5 * L_a * (1.0 - z)
    Dx = -(2.0 / L_a) * D

    th = np.pi * k / N
    w_cc = np.zeros(N + 1)
    for j in range(N + 1):
        s = sum((1.0 if 2 * r == N else 2.0) * np.cos(2 * r * th[j]) / (4 * r * r - 1.0) for r in range(1, N // 2 + 1))
        w_cc[j] = (1.0 if (j == 0 or j == N) else 2.0) / N * (1.0 - s)
    w_cc *= 0.5 * L_a
    return x, Dx, w_cc


def exact_2d_ns_eig(
    M_bar: float,
    m: int,
    Re: float = np.inf,
    n_p: float = 0.0,
    n_v: float = 0.0,
    omega_window: tuple[float, float] | None = None,
    N: int = 44,
    check_both_chiral_signs: bool = True,
) -> complex:
    """Exact least-stable eigenvalue of 2D linearized compressible Navier-Stokes with throughflow M_bar."""
    if check_both_chiral_signs and abs(n_v) > 0.0 and m != 0:
        vp = exact_2d_ns_eig(M_bar, abs(m), Re, n_p, n_v, omega_window, N, False)
        vn = exact_2d_ns_eig(M_bar, -abs(m), Re, n_p, n_v, omega_window, N, False)
        return vp if vp.real >= vn.real else vn

    x, Dx, w_cc = cheb_diff(N, L_A)
    Dxx = Dx @ Dx
    n = N + 1
    I = np.eye(n)
    mu = 0.0 if (not np.isfinite(Re) or Re <= 0.0) else (1.0 / Re)

    wf = np.exp(-0.5 * ((x - X_F) / SIG_F) ** 2)
    wf /= float(np.sum(w_cc * wf))
    w_sense = w_cc * wf

    has_f = (abs(n_p) > 0.0 or abs(n_v) > 0.0)
    dim = 4 * n + (10 if has_f else 0)
    A = np.zeros((dim, dim), dtype=complex)
    B = np.eye(dim, dtype=complex)

    def sl(q: int) -> slice:
        return slice(q * n, (q + 1) * n)

    lap = Dxx - (m * m) * I

    A[sl(0), sl(0)] = -M_bar * Dx; A[sl(0), sl(1)] = -Dx; A[sl(0), sl(2)] = -1j * m * I

    A[sl(1), sl(1)] = -M_bar * Dx + mu * ((4.0 / 3.0) * Dxx - (m * m) * I)
    A[sl(1), sl(2)] = mu * (1.0 / 3.0) * (1j * m) * Dx
    A[sl(1), sl(3)] = -Dx

    A[sl(2), sl(1)] = mu * (1.0 / 3.0) * (1j * m) * Dx
    A[sl(2), sl(2)] = -M_bar * Dx + mu * (Dxx - (4.0 / 3.0) * (m * m) * I)
    A[sl(2), sl(3)] = -1j * m * I

    A[sl(3), sl(0)] = -(mu / PR) * lap
    A[sl(3), sl(1)] = -Dx
    A[sl(3), sl(2)] = -1j * m * I
    A[sl(3), sl(3)] = -M_bar * Dx + (mu * GAMMA / PR) * lap

    if has_f:
        inv_ts = 4.0 / TAU_BAR
        ib = 4 * n
        A[ib + 0, sl(3)] = OMEGA_HP * w_sense; A[ib + 0, ib + 0] = -OMEGA_HP
        A[ib + 1, sl(3)] = inv_ts * w_sense; A[ib + 1, ib + 0] = -inv_ts; A[ib + 1, ib + 1] = -inv_ts
        for r in range(1, 4):
            A[ib + 1 + r, ib + r] = inv_ts; A[ib + 1 + r, ib + 1 + r] = -inv_ts
        A[ib + 5, sl(2)] = OMEGA_HP * w_sense; A[ib + 5, ib + 5] = -OMEGA_HP
        A[ib + 6, sl(2)] = inv_ts * w_sense; A[ib + 6, ib + 5] = -inv_ts; A[ib + 6, ib + 6] = -inv_ts
        for r in range(1, 4):
            A[ib + 6 + r, ib + 5 + r] = inv_ts; A[ib + 6 + r, ib + 6 + r] = -inv_ts
        A[sl(3), ib + 4] = n_p * wf
        A[sl(3), ib + 9] = n_v * wf

    for r_idx in (0, n, 2 * n):
        A[r_idx, :] = 0.0; B[r_idx, :] = 0.0
    A[0, 0] = 1.0; A[0, 3 * n] = -1.0
    A[n, n] = 1.0; A[n, 3 * n] = M_bar
    A[2 * n, 2 * n] = 1.0

    r_out_p = 3 * n + N
    A[r_out_p, :] = 0.0; B[r_out_p, :] = 0.0
    A[r_out_p, n + N] = 1.0
    A[r_out_p, 3 * n + N] = -0.5 * M_bar * GAMMA
    A[r_out_p, 0 * n + N] = +0.5 * M_bar
    if mu > 0.0:
        r_out_ut = 2 * n + N
        A[r_out_ut, :] = 0.0; B[r_out_ut, :] = 0.0
        A[r_out_ut, sl(2)] = Dx[N, :]
        r_out_T = 0 * n + N
        A[r_out_T, :] = 0.0; B[r_out_T, :] = 0.0
        A[r_out_T, sl(3)] = GAMMA * Dx[N, :]
        A[r_out_T, sl(0)] = -Dx[N, :]

    vals = eig(A, B, right=False)
    w_lo, w_hi = omega_window if omega_window is not None else (0.05, 5.5)
    cands = [v for v in vals if np.isfinite(v) and w_lo <= v.imag <= w_hi and -0.60 < v.real < 0.45]
    cands.sort(key=lambda z: z.real, reverse=True)
    return complex(cands[0])


# ---------------------------------------------------------------------------
# 2. 2D Conservative Compressible Navier-Stokes Solver (MUSCL-MC + HLLC + Viscous)
# ---------------------------------------------------------------------------

def mc_limiter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c = 0.5 * (a + b)
    return np.where((a * b) > 0.0, np.sign(c) * np.minimum(np.abs(c), np.minimum(2.0 * np.abs(a), 2.0 * np.abs(b))), 0.0)


def phi_flame(q: np.ndarray) -> np.ndarray:
    x = q / Q_S
    return Q_S * np.tanh(x + KAPPA_NL * (x ** 3))


def prim_2d(U: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = np.maximum(U[0], 1e-10)
    ux = U[1] / rho
    ut = U[2] / rho
    p = np.maximum((GAMMA - 1.0) * (U[3] - 0.5 * rho * (ux * ux + ut * ut)), 1e-10)
    c = np.sqrt(GAMMA * p / rho)
    return rho, ux, ut, p, c


def cons_2d(rho: np.ndarray, ux: np.ndarray, ut: np.ndarray, p: np.ndarray) -> np.ndarray:
    E = p / (GAMMA - 1.0) + 0.5 * rho * (ux * ux + ut * ut)
    return np.stack([rho, rho * ux, rho * ut, E], axis=0)


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
    denom = rhoL * (sL - vnL) - rhoR * (sR - vnR)
    sM = (pR - pL + rhoL * vnL * (sL - vnL) - rhoR * vnR * (sR - vnR)) / np.where(np.abs(denom) < 1e-14, 1e-14, denom)

    FL = np.stack([rhoL * vnL, rhoL * vnL * vnL + pL, rhoL * vnL * vtL, vnL * (EL + pL)], axis=0)
    FR = np.stack([rhoR * vnR, rhoR * vnR * vnR + pR, rhoR * vnR * vtR, vnR * (ER + pR)], axis=0)

    dL = np.where(np.abs(sL - sM) < 1e-14, 1e-14, sL - sM)
    dR = np.where(np.abs(sR - sM) < 1e-14, 1e-14, sR - sM)
    fL = rhoL * (sL - vnL) / dL
    fR = rhoR * (sR - vnR) / dR

    UsL = np.stack([
        fL, fL * sM, fL * vtL,
        fL * (EL / rhoL + (sM - vnL) * (sM + pL / (rhoL * np.where(np.abs(sL - vnL) < 1e-14, 1e-14, sL - vnL)))),
    ], axis=0)
    UsR = np.stack([
        fR, fR * sM, fR * vtR,
        fR * (ER / rhoR + (sM - vnR) * (sM + pR / (rhoR * np.where(np.abs(sR - vnR) < 1e-14, 1e-14, sR - vnR)))),
    ], axis=0)

    UL = np.stack([rhoL, rhoL * vnL, rhoL * vtL, EL], axis=0)
    UR = np.stack([rhoR, rhoR * vnR, rhoR * vtR, ER], axis=0)
    FsL = FL + sL * (UsL - UL)
    FsR = FR + sR * (UsR - UR)
    return np.where(sL >= 0.0, FL, np.where(sM >= 0.0, FsL, np.where(sR >= 0.0, FsR, FR)))


def apply_inlet_bc_port(U_in: np.ndarray, M_bar: float, u_port: np.ndarray) -> np.ndarray:
    _, ux0, _, _, c0 = prim_2d(U_in)
    pow_g = 2.0 / (GAMMA - 1.0)
    J_minus = ux0 - pow_g * c0
    m_flux = M_bar - u_port
    c = np.clip(c0, 0.4, 2.0)
    for _ in range(6):
        ux = J_minus + pow_g * c
        cp = c ** pow_g
        f = cp * ux - m_flux
        df = pow_g * (c ** (pow_g - 1.0)) * ux + pow_g * cp
        c = np.clip(c - f / np.maximum(np.abs(df), 1e-10), 0.35, 2.5)
    ux = J_minus + pow_g * c
    rho = c ** pow_g
    p = P_BAR * (rho ** GAMMA)
    ut = np.zeros_like(rho)
    return cons_2d(rho, ux, ut, p)


def apply_outlet_bc(U_out: np.ndarray, M_bar: float) -> np.ndarray:
    rho0, ux0, ut0, p0, c0 = prim_2d(U_out)
    pow_g = 2.0 / (GAMMA - 1.0)
    J_plus = ux0 + pow_g * c0
    s_out = np.log(np.maximum((p0 / P_BAR) / (rho0 ** GAMMA), 1e-15))
    c = np.maximum(J_plus / (M_bar + pow_g), 0.35)
    ux = M_bar * c
    rho = np.maximum((c * c / np.exp(s_out)) ** (1.0 / (GAMMA - 1.0)), 1e-6)
    p = P_BAR * (rho ** GAMMA) * np.exp(s_out)
    return cons_2d(rho, ux, ut0, p)


def simulate_2d_ns(
    nx: int = 40,
    nth: int = 120,
    t_end: float = 80.0,
    cfl: float = 0.35,
    M_bar: float = 0.12,
    Re: float = 1000.0,
    n_p: float = 0.0,
    n_v: float = 0.0,
    n_baffles: int = 0,
    L_baffle: float = 0.0,
    baffle_bc: str = "slip",
    cav_spec: list[dict] | None = None,
    init_type: str = "spin_1T",
    A_eq: float = 1e-4,
    periodic_x: bool = False,
    U_base: np.ndarray | None = None,
    record_stride: int = 12,
) -> dict:
    """2D (x, theta) conservative compressible Navier-Stokes annular combustor."""
    dx = L_A / nx
    dth = 2.0 * np.pi / nth
    x = (np.arange(nx) + 0.5) * dx
    th = (np.arange(nth) + 0.5) * dth
    X, TH = np.meshgrid(x, th, indexing="ij")

    mu = 0.0 if (not np.isfinite(Re) or Re <= 0.0) else (1.0 / Re)
    kappa = mu / ((GAMMA - 1.0) * PR)
    has_visc = mu > 0.0

    wf = np.exp(-0.5 * ((x - X_F) / SIG_F) ** 2)
    wf /= float(np.sum(wf) * dx)

    baffle_mask = np.zeros((nx, nth), dtype=bool)
    if n_baffles > 0 and L_baffle > 0.0:
        baffle_nx = int(round(L_baffle / dx))
        baffle_faces = [(int(round(k * nth / n_baffles)) - 1) % nth for k in range(n_baffles)]
        for bf in baffle_faces:
            baffle_mask[:baffle_nx, bf] = True
    has_baffle = bool(np.any(baffle_mask))
    baffle_left_adj = baffle_mask.copy()
    baffle_right_adj = np.roll(baffle_mask, 1, axis=1)

    cav_list = []
    for sp in (cav_spec or []):
        beta_f = float(sp["beta"]); wH = float(sp["wH"]); xiH = float(sp["xiH"])
        sig_o = float(sp.get("sigma_o", 0.045)); nc = int(sp.get("n_cav", 5))
        Cd = float(sp.get("Cd", CD_ORIFICE)); nl_on = bool(sp.get("nonlinear", True))
        V_k = beta_f * (2.0 * np.pi * L_A) / nc
        A_ok = (sig_o * 2.0 * np.pi) / nc
        r_nl = (1.0 / (2.0 * (Cd ** 2) * (A_ok ** 2))) if nl_on else 0.0
        W = np.zeros((nc, nth))
        sig_th = max(1.5 * dth, 0.5 * (2.0 * np.pi * sig_o / nc))
        phase_off = float(sp.get("phase_offset", 0.5))
        for k in range(nc):
            th_k = (k + phase_off) * (2.0 * np.pi / nc)
            d_ang = np.angle(np.exp(1j * (th - th_k)))
            wk = np.exp(-0.5 * (d_ang / sig_th) ** 2)
            W[k] = wk / float(np.sum(wk) * dth)
        cav_list.append({"V_k": V_k, "wH": wH, "xiH": xiH, "r_nl": r_nl, "W": W, "Pc": np.zeros(nc), "qt": np.zeros(nc)})

    if U_base is not None:
        rho_b, ux_b, ut_b, p_b, _ = prim_2d(U_base)
    else:
        rho_b = np.ones_like(X); ux_b = np.full_like(X, M_bar); ut_b = np.zeros_like(X); p_b = np.full_like(X, P_BAR)

    dP = np.zeros_like(X); dux = np.zeros_like(X); dut = np.zeros_like(X)
    if init_type == "steady":
        pass
    elif init_type == "spin_1T":
        dP = A_eq * np.cos(TH); dut = A_eq * np.cos(TH)
    elif init_type == "stand_1T":
        dP = np.sqrt(2.0) * A_eq * np.cos(TH)
    elif init_type == "spin_2T":
        dP = A_eq * np.cos(2.0 * TH); dut = A_eq * np.cos(2.0 * TH)
    elif init_type == "cold_1L":
        dP = np.sqrt(2.0) * A_eq * np.cos(np.pi * X / L_A)
    elif init_type == "sk_acoustic_k2":
        dP = np.sqrt(2.0) * A_eq * np.cos(2.0 * TH)
    elif init_type == "shear_diff_k2":
        dux = A_eq * np.cos(2.0 * TH)
    elif init_type == "periodic_2d_wave":
        kx = 2.0 * np.pi / L_A
        dP = A_eq * (0.6 * np.cos(kx * X + TH) + 0.4 * np.cos(2.0 * TH))
        dux = 0.5 * A_eq * np.cos(kx * X + TH)
        dut = 0.5 * A_eq * np.sin(TH)
    else:
        raise ValueError(f"Unknown init_type: {init_type}")

    p0 = p_b + dP
    rho0 = rho_b * np.maximum((p0 / p_b) ** (1.0 / GAMMA), 1e-6)
    U = cons_2d(rho0, ux_b + dux, ut_b + dut, p0)
    U_init = U.copy()

    p_ref = p_b.copy(); ux_ref = ux_b.copy(); ut_ref = ut_b.copy()
    pbf = np.zeros(nth); ubf = np.zeros(nth)
    qp = np.zeros((4, nth)); qu = np.zeros((4, nth))
    has_flame = (abs(n_p) > 0.0 or abs(n_v) > 0.0)
    inv_ts = 4.0 / TAU_BAR

    bl_pad = np.zeros((nx + 2, nth), dtype=bool)
    br_pad = np.zeros((nx + 2, nth), dtype=bool)
    if has_baffle:
        bl_pad[1:-1, :] = baffle_left_adj
        br_pad[1:-1, :] = baffle_right_adj
        if periodic_x:
            bl_pad[0, :] = baffle_left_adj[-1, :]
            bl_pad[-1, :] = baffle_left_adj[0, :]
            br_pad[0, :] = baffle_right_adj[-1, :]
            br_pad[-1, :] = baffle_right_adj[0, :]

    def get_u_port(c_states):
        up = np.zeros(nth)
        for (_, qt), cfg in zip(c_states, cav_list):
            up += qt @ cfg["W"]
        return up

    def rhs(Us, pbs, ubs, qps, qus, c_states):
        u_port = get_u_port(c_states)
        if periodic_x:
            Ugx = np.concatenate([Us[:, -2:, :], Us, Us[:, :2, :]], axis=1)
            U_in = Us[:, 0, :]
        else:
            U_in = apply_inlet_bc_port(Us[:, 0, :], M_bar, u_port)
            U_out = apply_outlet_bc(Us[:, -1, :], M_bar)
            Ugx = np.concatenate([U_in[:, None, :], U_in[:, None, :], Us, U_out[:, None, :], U_out[:, None, :]], axis=1)

        rho_x, ux_x, ut_x, p_x, _ = prim_2d(Ugx)
        Qx = np.stack([rho_x, ux_x, ut_x, p_x], axis=0)
        sx = mc_limiter(Qx[:, 1:-1, :] - Qx[:, :-2, :], Qx[:, 2:, :] - Qx[:, 1:-1, :])
        QLx = Qx[:, 1:-2, :] + 0.5 * sx[:, :-1, :]
        QRx = Qx[:, 2:-1, :] - 0.5 * sx[:, 1:, :]
        Fx = hllc_2d_normal(
            np.maximum(QLx[0], 1e-8), QLx[1], QLx[2], np.maximum(QLx[3], 1e-8),
            np.maximum(QRx[0], 1e-8), QRx[1], QRx[2], np.maximum(QRx[3], 1e-8),
        )

        rho_t, ux_t, ut_t, p_t, _ = prim_2d(Us)
        Qt = np.stack([rho_t, ux_t, ut_t, p_t], axis=0)
        st_lim = mc_limiter(Qt - np.roll(Qt, 1, axis=2), np.roll(Qt, -1, axis=2) - Qt)
        if has_baffle:
            st_lim[:, baffle_left_adj] = 0.0
            st_lim[:, baffle_right_adj] = 0.0
        QLt = Qt + 0.5 * st_lim
        QRt = np.roll(Qt - 0.5 * st_lim, -1, axis=2)

        Ft_n = hllc_2d_normal(
            np.maximum(QLt[0], 1e-8), QLt[2], QLt[1], np.maximum(QLt[3], 1e-8),
            np.maximum(QRt[0], 1e-8), QRt[2], QRt[1], np.maximum(QRt[3], 1e-8),
        )
        Ft = np.empty_like(Ft_n)
        Ft[0] = Ft_n[0]; Ft[1] = Ft_n[2]; Ft[2] = Ft_n[1]; Ft[3] = Ft_n[3]

        if has_baffle:
            Ft_L = Ft.copy(); Ft_R = Ft.copy()
            rL = np.maximum(QLt[0, baffle_mask], 1e-8)
            utL = QLt[2, baffle_mask]; uxL = QLt[1, baffle_mask]; pL = np.maximum(QLt[3, baffle_mask], 1e-8)
            FL_n = hllc_2d_normal(rL, utL, uxL, pL, rL, -utL, uxL, pL)
            Ft_L[0, baffle_mask] = 0.0; Ft_L[1, baffle_mask] = 0.0; Ft_L[2, baffle_mask] = FL_n[1]; Ft_L[3, baffle_mask] = 0.0

            rR = np.maximum(QRt[0, baffle_mask], 1e-8)
            utR = QRt[2, baffle_mask]; uxR = QRt[1, baffle_mask]; pR = np.maximum(QRt[3, baffle_mask], 1e-8)
            FR_n = hllc_2d_normal(rR, -utR, uxR, pR, rR, utR, uxR, pR)
            Ft_R[0, baffle_mask] = 0.0; Ft_R[1, baffle_mask] = 0.0; Ft_R[2, baffle_mask] = FR_n[1]; Ft_R[3, baffle_mask] = 0.0
            dU = -(Fx[:, 1:, :] - Fx[:, :-1, :]) / dx - (Ft_L - np.roll(Ft_R, 1, axis=2)) / dth
        else:
            dU = -(Fx[:, 1:, :] - Fx[:, :-1, :]) / dx - (Ft - np.roll(Ft, 1, axis=2)) / dth

        if has_visc:
            ux_gx = ux_x[1:-1, :]
            ut_gx = ut_x[1:-1, :]
            T_gx = GAMMA * p_x[1:-1, :] / np.maximum(rho_x[1:-1, :], 1e-10)

            dux_dth_c = (np.roll(ux_gx, -1, axis=1) - np.roll(ux_gx, 1, axis=1)) / (2.0 * dth)
            dut_dth_c = (np.roll(ut_gx, -1, axis=1) - np.roll(ut_gx, 1, axis=1)) / (2.0 * dth)
            if has_baffle:
                dut_dth_c[bl_pad] = -2.0 * ut_gx[bl_pad] / dth
                dut_dth_c[br_pad] = +2.0 * ut_gx[br_pad] / dth
                if baffle_bc == "noslip":
                    dux_dth_c[bl_pad] = -2.0 * ux_gx[bl_pad] / dth
                    dux_dth_c[br_pad] = +2.0 * ux_gx[br_pad] / dth
                else:
                    dux_dth_c[bl_pad] = 0.0
                    dux_dth_c[br_pad] = 0.0

            dux_dx_f = (ux_gx[1:, :] - ux_gx[:-1, :]) / dx
            dut_dx_f = (ut_gx[1:, :] - ut_gx[:-1, :]) / dx
            dT_dx_f  = (T_gx[1:, :]  - T_gx[:-1, :])  / dx
            dux_dth_fx = 0.5 * (dux_dth_c[:-1, :] + dux_dth_c[1:, :])
            dut_dth_fx = 0.5 * (dut_dth_c[:-1, :] + dut_dth_c[1:, :])
            ux_fx = 0.5 * (ux_gx[:-1, :] + ux_gx[1:, :])
            ut_fx = 0.5 * (ut_gx[:-1, :] + ut_gx[1:, :])

            txx_fx = mu * ((4.0 / 3.0) * dux_dx_f - (2.0 / 3.0) * dut_dth_fx)
            txt_fx = mu * (dux_dth_fx + dut_dx_f)
            ex_fx  = ux_fx * txx_fx + ut_fx * txt_fx + kappa * dT_dx_f

            dU[1] += (txx_fx[1:, :] - txx_fx[:-1, :]) / dx
            dU[2] += (txt_fx[1:, :] - txt_fx[:-1, :]) / dx
            dU[3] += (ex_fx[1:, :]  - ex_fx[:-1, :])  / dx

            ux_i = ux_gx[1:-1, :]; ut_i = ut_gx[1:-1, :]; T_i = T_gx[1:-1, :]
            dux_dx_c = (ux_gx[2:, :] - ux_gx[:-2, :]) / (2.0 * dx)
            dut_dx_c = (ut_gx[2:, :] - ut_gx[:-2, :]) / (2.0 * dx)

            dux_dth_ft = (np.roll(ux_i, -1, axis=1) - ux_i) / dth
            dut_dth_ft = (np.roll(ut_i, -1, axis=1) - ut_i) / dth
            dT_dth_ft  = (np.roll(T_i,  -1, axis=1) - T_i)  / dth
            dux_dx_ft = 0.5 * (dux_dx_c + np.roll(dux_dx_c, -1, axis=1))
            dut_dx_ft = 0.5 * (dut_dx_c + np.roll(dut_dx_c, -1, axis=1))
            ux_ft = 0.5 * (ux_i + np.roll(ux_i, -1, axis=1))
            ut_ft = 0.5 * (ut_i + np.roll(ut_i, -1, axis=1))

            txt_ft = mu * (dux_dth_ft + dut_dx_ft)
            ttt_ft = mu * ((4.0 / 3.0) * dut_dth_ft - (2.0 / 3.0) * dux_dx_ft)
            et_ft  = ux_ft * txt_ft + ut_ft * ttt_ft + kappa * dT_dth_ft

            if has_baffle:
                txt_L = txt_ft.copy(); txt_R = txt_ft.copy()
                ttt_L = ttt_ft.copy(); ttt_R = ttt_ft.copy()
                et_L  = et_ft.copy();  et_R  = et_ft.copy()

                ux_R_cell = np.roll(ux_i, -1, axis=1)
                ut_R_cell = np.roll(ut_i, -1, axis=1)
                dux_dx_R = np.roll(dux_dx_c, -1, axis=1)

                ttt_L[baffle_mask] = mu * ((4.0 / 3.0) * (-2.0 * ut_i[baffle_mask] / dth) - (2.0 / 3.0) * dux_dx_c[baffle_mask])
                ttt_R[baffle_mask] = mu * ((4.0 / 3.0) * (+2.0 * ut_R_cell[baffle_mask] / dth) - (2.0 / 3.0) * dux_dx_R[baffle_mask])
                if baffle_bc == "noslip":
                    txt_L[baffle_mask] = mu * (-2.0 * ux_i[baffle_mask] / dth)
                    txt_R[baffle_mask] = mu * (+2.0 * ux_R_cell[baffle_mask] / dth)
                else:
                    txt_L[baffle_mask] = 0.0
                    txt_R[baffle_mask] = 0.0
                et_L[baffle_mask] = 0.0
                et_R[baffle_mask] = 0.0

                dU[1] += (txt_L - np.roll(txt_R, 1, axis=1)) / dth
                dU[2] += (ttt_L - np.roll(ttt_R, 1, axis=1)) / dth
                dU[3] += (et_L  - np.roll(et_R,  1, axis=1)) / dth
            else:
                dU[1] += (txt_ft - np.roll(txt_ft, 1, axis=1)) / dth
                dU[2] += (ttt_ft - np.roll(ttt_ft, 1, axis=1)) / dth
                dU[3] += (et_ft  - np.roll(et_ft,  1, axis=1)) / dth

        dcav = []
        if cav_list:
            _, _, _, p_in, _ = prim_2d(U_in)
            P_face = p_in - p_ref[0, :]
            for (Pc, qt), cfg in zip(c_states, cav_list):
                wH = cfg["wH"]; xiH = cfg["xiH"]; r_nl = cfg["r_nl"]; Vk = cfg["V_k"]
                P_bar_k = (cfg["W"] @ P_face) * dth
                dPc = qt / Vk
                dqt = (wH ** 2) * Vk * (P_bar_k - Pc - r_nl * np.abs(qt) * qt) - 2.0 * xiH * wH * qt
                dcav.append((dPc, dqt))

        if has_flame:
            Pf = p_t - p_ref
            uf_rel = ut_t - ut_ref
            pf_s = np.sum(wf[:, None] * Pf, axis=0) * dx
            uf_s = np.sum(wf[:, None] * uf_rel, axis=0) * dx
            dpb = OMEGA_HP * (pf_s - pbs)
            dub = OMEGA_HP * (uf_s - ubs)
            dqp = np.empty_like(qps); dqu = np.empty_like(qus)
            dqp[0] = inv_ts * ((pf_s - pbs) - qps[0])
            dqu[0] = inv_ts * ((uf_s - ubs) - qus[0])
            for r in range(1, 4):
                dqp[r] = inv_ts * (qps[r - 1] - qps[r])
                dqu[r] = inv_ts * (qus[r - 1] - qus[r])
            q_tot = n_p * phi_flame(qps[3]) + n_v * phi_flame(qus[3])
            dU[3] += (1.0 / (GAMMA - 1.0)) * wf[:, None] * q_tot[None, :]
        else:
            dpb = np.zeros_like(pbs); dub = np.zeros_like(ubs)
            dqp = np.zeros_like(qps); dqu = np.zeros_like(qus)

        return dU, dpb, dub, dqp, dqu, dcav

    cav_states = [(c["Pc"].copy(), c["qt"].copy()) for c in cav_list]
    t_list, ew_list, ppk_list, e1_list, e2_list, u2_list = [], [], [], [], [], []
    init_totals = np.sum(U, axis=(1, 2)) * dx * dth
    max_cons_err = np.zeros(4)
    nu_max = mu * max(4.0 / 3.0, GAMMA / PR) if has_visc else 0.0
    dt_visc_cap = (0.20 / (nu_max * (1.0 / (dx * dx) + 1.0 / (dth * dth)))) if has_visc else 1.0

    def record(tv: float):
        _, ux, ut, p, _ = prim_2d(U)
        Pf = p - p_ref
        uxo = ux - ux_ref
        uto = ut - ut_ref
        ew = 0.5 * float(np.mean(Pf * Pf + uxo * uxo + uto * uto))
        phat = np.fft.rfft(np.mean(Pf, axis=0)) / (0.5 * nth)
        if periodic_x:
            cur = np.sum(U, axis=(1, 2)) * dx * dth
            denom = np.maximum(np.abs(init_totals), 1.0)
            np.maximum(max_cons_err, np.abs(cur - init_totals) / denom, out=max_cons_err)
        t_list.append(tv)
        ew_list.append(ew)
        ppk_list.append(float(np.max(np.abs(Pf))))
        e1_list.append(float(np.abs(phat[1]) ** 2))
        e2_list.append(float(np.abs(phat[2]) ** 2))
        u2_list.append(float(np.mean(uxo * uxo)))

    t = 0.0; step = 0; record(0.0)
    last_rhs_max = 0.0
    while t < t_end - 1e-12:
        _, ux_c, ut_c, _, c_c = prim_2d(U)
        dt_adv = cfl / float(np.max((np.abs(ux_c) + c_c) / dx + (np.abs(ut_c) + c_c) / dth))
        dt = min(dt_adv, dt_visc_cap, 0.015, t_end - t)
        k1 = rhs(U, pbf, ubf, qp, qu, cav_states)
        U1 = U + dt * k1[0]
        pb1 = pbf + dt * k1[1]; ub1 = ubf + dt * k1[2]; qp1 = qp + dt * k1[3]; qu1 = qu + dt * k1[4]
        cs1 = [(Pc + dt * dPc, qt + dt * dqt) for (Pc, qt), (dPc, dqt) in zip(cav_states, k1[5])]
        k2 = rhs(U1, pb1, ub1, qp1, qu1, cs1)
        U = 0.5 * (U + U1 + dt * k2[0])
        pbf = 0.5 * (pbf + pb1 + dt * k2[1]); ubf = 0.5 * (ubf + ub1 + dt * k2[2])
        qp = 0.5 * (qp + qp1 + dt * k2[3]); qu = 0.5 * (qu + qu1 + dt * k2[4])
        cav_states = [(0.5 * (Pc + Pc1 + dt * dPc2), 0.5 * (qt + qt1 + dt * dqt2))
                      for (Pc, qt), (Pc1, qt1), (dPc2, dqt2) in zip(cav_states, cs1, k2[5])]
        t += dt; step += 1
        if t >= t_end - 1e-12:
            last_rhs_max = float(np.max(np.abs(k1[0])))
        if step % record_stride == 0 or t >= t_end - 1e-12:
            record(t)

    rho_f, ux_f, ut_f, p_f, _ = prim_2d(U)
    return {
        "x": x, "theta": th, "U_final": U, "ux_final": ux_f, "p_final": p_f,
        "steady_max_err": float(np.max(np.abs(U - U_init))),
        "final_rhs_max": last_rhs_max,
        "periodic_cons_err": {
            "mass": float(max_cons_err[0]), "mom_x": float(max_cons_err[1]),
            "mom_th": float(max_cons_err[2]), "energy": float(max_cons_err[3]),
        },
        "t": np.array(t_list), "e_wave": np.array(ew_list), "p_peak": np.array(ppk_list),
        "e1": np.array(e1_list), "e2": np.array(e2_list), "u2": np.array(u2_list),
    }


def stats_ns(r: dict, win_lo: float = 18.0, win_hi: float = 60.0) -> dict:
    t = r["t"]; n = len(t)
    m1 = int(0.45 * n); m2 = int(0.75 * n)
    em = float(np.mean(r["e_wave"][m1:m2])); el = float(np.mean(r["e_wave"][m2:]))
    pk_pct = float(np.max(r["p_peak"][m2:]) / P_BAR * 100.0)
    afit = float(np.polyfit(t[m1:], 0.5 * np.log(np.maximum(r["e_wave"][m1:], 1e-30)), 1)[0])
    wmsk = (t >= win_lo) & (t <= win_hi) & (r["p_peak"] / P_BAR < 0.005)
    afit_win = float(np.polyfit(t[wmsk], 0.5 * np.log(np.maximum(r["e_wave"][wmsk], 1e-30)), 1)[0]) if int(np.sum(wmsk)) >= 8 else afit
    return {
        "p_over_pbar_pct": pk_pct, "e_wave_late": el,
        "e_wave_late_over_mid": float(el / max(em, 1e-30)),
        "alpha_fit_late": afit, "alpha_fit_win_18_60": afit_win,
        "e1_late": float(np.mean(r["e1"][m2:])), "e2_late": float(np.mean(r["e2"][m2:])),
    }


def _sim_worker(args):
    lbl, kw = args
    return lbl, simulate_2d_ns(**kw)


def _wake_and_pert_worker(args):
    lbl, base_kw, pert_kw = args
    U_base = None; wake_info = {}
    if base_kw is not None:
        rb = simulate_2d_ns(**base_kw)
        U_base = rb["U_final"]
        _, ux_w, _, p_w, _ = prim_2d(U_base)
        xv = rb["x"]
        # Downstream wake velocity metrics (strictly x > L_b = 0.45, Peer Review #8 M3!)
        ds_mask = xv > 0.45
        idx_050 = int(np.argmin(np.abs(xv - 0.50)))
        idx_075 = int(np.argmin(np.abs(xv - 0.75)))
        wake_info = {
            "base_rhs_max": rb["final_rhs_max"],
            "ux_wake_min_downstream": float(np.min(ux_w[ds_mask, :])),
            "ux_wake_min_x050": float(np.min(ux_w[idx_050, :])),
            "ux_wake_min_x075": float(np.min(ux_w[idx_075, :])),
            "ux_core_max": float(np.max(ux_w)),
            "dp_over_pbar_pct": float((np.mean(p_w[0]) - np.mean(p_w[-1])) / P_BAR * 100.0),
        }
    pkw = dict(pert_kw); pkw["U_base"] = U_base
    rp = simulate_2d_ns(**pkw)
    if base_kw is not None:
        rp["wake_info"] = wake_info
        rp["ux_base"] = ux_w
        rp["U_base_saved"] = U_base
    return lbl, rp


def run_batch(tasks, worker=_sim_worker, max_workers: int | None = None) -> dict:
    nw = max_workers or min(6, max(1, (os.cpu_count() or 2) - 2), len(tasks))
    if nw <= 1 or len(tasks) <= 1:
        return dict(worker(t) for t in tasks)
    with ProcessPoolExecutor(max_workers=nw) as ex:
        return dict(ex.map(worker, tasks))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 86)
    print("Entry 011 (Hardened per Peer Review #8): 2D Compressible NS — Vorticity & Baffle Wakes")
    print("=" * 86)

    # ------------------------------------------------------------------
    # Experiment 1: Exact NS Benchmarks & Clean t in [20, 40] Chebyshev-vs-PDE Check (M4)
    # ------------------------------------------------------------------
    print("\n--- Experiment 1: Exact Compressible NS Benchmarks & t in [20, 40] Chebyshev Check ---")
    Re_b = 200.0; mu_b = 1.0 / Re_b
    a_sk_ex = -0.5 * mu_b * (2.0 ** 2) * (4.0 / 3.0 + (GAMMA - 1.0) / PR)
    a_sh_ex = -mu_b * (2.0 ** 2)

    exp1_res = run_batch([
        ("steady_slip", dict(nx=40, nth=120, t_end=15.0, M_bar=0.12, Re=500.0, n_baffles=5, L_baffle=0.45, baffle_bc="slip", init_type="steady", A_eq=0.0, record_stride=20)),
        ("cons_ns_bare", dict(nx=40, nth=120, t_end=8.0, M_bar=0.12, Re=200.0, init_type="periodic_2d_wave", A_eq=0.10, periodic_x=True, record_stride=20)),
        ("cons_ns_noslip5", dict(nx=40, nth=120, t_end=8.0, M_bar=0.12, Re=200.0, n_baffles=5, L_baffle=0.45, baffle_bc="noslip", init_type="periodic_2d_wave", A_eq=0.10, periodic_x=True, record_stride=20)),
        ("sk_k2", dict(nx=16, nth=160, t_end=40.0, M_bar=0.0, Re=Re_b, init_type="sk_acoustic_k2", A_eq=1e-4, periodic_x=True, record_stride=10)),
        ("sh_k2", dict(nx=16, nth=160, t_end=40.0, M_bar=0.0, Re=Re_b, init_type="shear_diff_k2", A_eq=1e-4, periodic_x=True, record_stride=10)),
        ("pde_M012_Re500", dict(nx=60, nth=160, t_end=42.0, M_bar=0.12, Re=500.0, init_type="spin_1T", A_eq=1e-4, record_stride=10)),
        ("pde_M012_Re1000", dict(nx=60, nth=160, t_end=42.0, M_bar=0.12, Re=1000.0, init_type="spin_1T", A_eq=1e-4, record_stride=10)),
        ("pde_M018_Re500", dict(nx=60, nth=160, t_end=42.0, M_bar=0.18, Re=500.0, init_type="spin_1T", A_eq=1e-4, record_stride=10)),
        ("pde_M018_Re1000", dict(nx=60, nth=160, t_end=42.0, M_bar=0.18, Re=1000.0, init_type="spin_1T", A_eq=1e-4, record_stride=10)),
    ])

    ce_bare = exp1_res["cons_ns_bare"]["periodic_cons_err"]
    ce_baf = exp1_res["cons_ns_noslip5"]["periodic_cons_err"]
    print(f"  Exp 1A Steady slip-baffle NS: {exp1_res['steady_slip']['steady_max_err']:.2e}")
    print(f"  Exp 1B Periodic NS (bare):   mass={ce_bare['mass']:.2e}, mx={ce_bare['mom_x']:.2e}, mt={ce_bare['mom_th']:.2e}, E={ce_bare['energy']:.2e}")
    print(f"  Exp 1B Periodic NS (5 baf):  mass={ce_baf['mass']:.2e}, E={ce_baf['energy']:.2e} (wall drag: dmx={ce_baf['mom_x']:.3f}, dmt={ce_baf['mom_th']:.4f})")

    m_sk = exp1_res["sk_k2"]["t"] >= 5.0
    a_sk_pde = float(np.polyfit(exp1_res["sk_k2"]["t"][m_sk], 0.5 * np.log(np.maximum(exp1_res["sk_k2"]["e_wave"][m_sk], 1e-30)), 1)[0])
    m_sh = exp1_res["sh_k2"]["t"] >= 5.0
    a_sh_pde = float(np.polyfit(exp1_res["sh_k2"]["t"][m_sh], 0.5 * np.log(np.maximum(exp1_res["sh_k2"]["u2"][m_sh], 1e-30)), 1)[0])
    print(f"  Exp 1C Stokes-Kirchhoff (k=2, Re=200): exact={a_sk_ex:+.6f} | PDE={a_sk_pde:+.6f} (err={abs(a_sk_pde-a_sk_ex)/abs(a_sk_ex)*100:.2f}%)")
    print(f"  Exp 1C Shear diffusion  (k=2, Re=200): exact={a_sh_ex:+.6f} | PDE={a_sh_pde:+.6f} (err={abs(a_sh_pde-a_sh_ex)/abs(a_sh_ex)*100:.2f}%)")

    cheb_vs_pde = {}
    for lbl, Mb, Re_v in [
        ("pde_M012_Re500", 0.12, 500.0), ("pde_M012_Re1000", 0.12, 1000.0),
        ("pde_M018_Re500", 0.18, 500.0), ("pde_M018_Re1000", 0.18, 1000.0),
    ]:
        v44 = exact_2d_ns_eig(Mb, 1, Re_v, omega_window=(0.65, 1.35), N=44)
        v60 = exact_2d_ns_eig(Mb, 1, Re_v, omega_window=(0.65, 1.35), N=60)
        rc = exp1_res[lbl]; m = (rc["t"] >= 20.0) & (rc["t"] <= 40.0)
        af = float(np.polyfit(rc["t"][m], 0.5 * np.log(np.maximum(rc["e_wave"][m], 1e-30)), 1)[0])
        rel = abs(af - v60.real) / abs(v60.real) * 100.0
        cheb_vs_pde[lbl] = {"exact_N44": v44.real, "exact_N60": v60.real, "pde_60x160_t20_40": af, "rel_err_pct": rel}
        print(f"  Exp 1D [{lbl:16s}] exact(N=60)={v60.real:+.5f} | PDE(t=20-40)={af:+.5f} (rel={rel:.2f}%)")

    # ------------------------------------------------------------------
    # Experiment 2: Vorticity-Wave Viscous Law & Nonlinear 64x200 2T Trap Audit (C1)
    # ------------------------------------------------------------------
    print("\n--- Experiment 2: Vorticity-Wave Viscous Law & Nonlinear 64x200 2T Trap Audit (C1) ---")
    M_grid = np.linspace(0.08, 0.25, 18)
    hopf_envelopes = {}
    for Re_v in (500.0, 1000.0, 2000.0, np.inf):
        rtag = "inf" if not np.isfinite(Re_v) else f"{int(Re_v)}"
        curve = []
        for Mb in M_grid:
            Mb_f = float(Mb)
            vc1 = exact_2d_ns_eig(Mb_f, 1, Re_v, omega_window=(0.65, 1.35))
            npH_m = {}
            for mk in (0, 1, 2, 3):
                ww = (0.65, 1.35) if mk == 1 else (0.05, 5.5)
                fm = lambda npv, m_k=mk, w_w=ww: exact_2d_ns_eig(Mb_f, m_k, Re_v, n_p=npv, omega_window=w_w).real
                npH_m[mk] = float(brentq(fm, 0.01, 0.90)) if fm(0.90) > 0.0 else 0.90
            g_val = min(npH_m.values()); g_m = min(npH_m, key=npH_m.get)
            curve.append({
                "M_bar": Mb_f, "cold_alpha_1T": vc1.real, "cold_omega_1T": vc1.imag,
                "n_pH_m0": npH_m[0], "n_pH_1T": npH_m[1], "n_pH_2T": npH_m[2], "n_pH_3T": npH_m[3],
                "n_pH_global": g_val, "limiting_m": int(g_m),
            })
        hopf_envelopes[rtag] = curve

    # Exact linear crossing Re_cross for chiral 2T at M=0.16 (n_p=0.22, n_v=0.04)
    re_cross_2t = float(brentq(lambda r: exact_2d_ns_eig(0.16, 2, r, n_p=0.22, n_v=0.04).real, 2000.0, 15000.0))
    cold_M012_Re5000 = exact_2d_ns_eig(0.12, 1, 5000.0, omega_window=(0.65, 1.35)).real
    m016_chiral_vs_Re = {}
    for Re_v in (300.0, 500.0, 1000.0, 1500.0, 2000.0, 5000.0, np.inf):
        rtag = "inf" if not np.isfinite(Re_v) else f"{int(Re_v)}"
        v1 = exact_2d_ns_eig(0.16, 1, Re_v, n_p=0.22, n_v=0.04, omega_window=(0.65, 1.35))
        v2 = exact_2d_ns_eig(0.16, 2, Re_v, n_p=0.22, n_v=0.04)
        lv2 = (Re_v * (0.16 ** 3) / ((v2.imag ** 2) * L_A)) if np.isfinite(Re_v) else None
        m016_chiral_vs_Re[rtag] = {"alpha_1T": v1.real, "alpha_2T": v2.real, "L_visc_2T_over_La": lv2}
    print(f"  Exact M=0.16 chiral 2T linear crossing: Re_cross = {re_cross_2t:.0f} (cold M=0.12 Re=5000: {cold_M012_Re5000:+.5f})")

    # Direct nonlinear 64x200 2T trap audit across Re = 1000, 2000, 5000 (Peer Review #8 C1!)
    nl2t_runs = run_batch([
        ("M016_2T_Re1000_A040", dict(nx=64, nth=200, t_end=180.0, M_bar=0.16, Re=1000.0, n_p=0.22, n_v=0.04, init_type="spin_2T", A_eq=0.040, record_stride=18)),
        ("M016_2T_Re2000_A020", dict(nx=64, nth=200, t_end=180.0, M_bar=0.16, Re=2000.0, n_p=0.22, n_v=0.04, init_type="spin_2T", A_eq=0.020, record_stride=18)),
        ("M016_2T_Re2000_A040", dict(nx=64, nth=200, t_end=180.0, M_bar=0.16, Re=2000.0, n_p=0.22, n_v=0.04, init_type="spin_2T", A_eq=0.040, record_stride=18)),
        ("M016_2T_Re5000_A040", dict(nx=64, nth=200, t_end=180.0, M_bar=0.16, Re=5000.0, n_p=0.22, n_v=0.04, init_type="spin_2T", A_eq=0.040, record_stride=18)),
    ])
    nl2t_stats = {}
    for cid, rr in nl2t_runs.items():
        st = stats_ns(rr)
        nl2t_stats[cid] = st
        print(f"  [{cid:20s}] P={st['p_over_pbar_pct']:5.2f}% | E={st['e_wave_late']:.2e} (l/m={st['e_wave_late_over_mid']:.3f})")

    # ------------------------------------------------------------------
    # Experiment 3: Converged t=150 Wake, 100x300 No-Slip Penalty (M2/M3) & M=0 Decomposition (M6)
    # ------------------------------------------------------------------
    print("\n--- Experiment 3: t=150 Steady Wakes, 100x300 Convergence (M2/M3) & M=0 Split (M6) ---")
    exp3_jobs = []
    for Re_v, rtag, grids in [
        (np.inf, "Euler", [(40, 120), (60, 160), (80, 240)]),
        (500.0, "Re500", [(40, 120), (60, 160), (80, 240), (100, 300)]),
    ]:
        for nx_g, nth_g in grids:
            exp3_jobs.append((
                f"slip_{rtag}_{nx_g}x{nth_g}", None,
                dict(nx=nx_g, nth=nth_g, t_end=34.0, M_bar=0.12, Re=Re_v, n_baffles=5, L_baffle=0.45,
                     baffle_bc="slip", init_type="stand_1T", A_eq=1e-4, record_stride=10),
            ))
    for nx_g, nth_g in [(40, 120), (60, 160), (80, 240), (100, 300)]:
        exp3_jobs.append((
            f"noslip_Re500_{nx_g}x{nth_g}",
            dict(nx=nx_g, nth=nth_g, t_end=150.0, M_bar=0.12, Re=500.0, n_baffles=5, L_baffle=0.45,
                 baffle_bc="noslip", init_type="steady", A_eq=0.0, record_stride=80),
            dict(nx=nx_g, nth=nth_g, t_end=34.0, M_bar=0.12, Re=500.0, n_baffles=5, L_baffle=0.45,
                 baffle_bc="noslip", init_type="stand_1T", A_eq=1e-4, record_stride=10),
        ))
    # M_bar = 0 decomposition on 80x240: bare vs slip baffles vs noslip baffles (Peer Review #8 M6!)
    exp3_jobs.extend([
        ("M0_Euler_baf_80x240", None, dict(nx=80, nth=240, t_end=34.0, M_bar=0.0, Re=np.inf, n_baffles=5, L_baffle=0.45, baffle_bc="slip", init_type="stand_1T", A_eq=1e-4, record_stride=12)),
        ("M0_bare_Re500_80x240", None, dict(nx=80, nth=240, t_end=34.0, M_bar=0.0, Re=500.0, n_baffles=0, L_baffle=0.0, init_type="stand_1T", A_eq=1e-4, record_stride=12)),
        ("M0_slip_Re500_80x240", None, dict(nx=80, nth=240, t_end=34.0, M_bar=0.0, Re=500.0, n_baffles=5, L_baffle=0.45, baffle_bc="slip", init_type="stand_1T", A_eq=1e-4, record_stride=12)),
        ("M0_noslip_Re500_80x240", None, dict(nx=80, nth=240, t_end=34.0, M_bar=0.0, Re=500.0, n_baffles=5, L_baffle=0.45, baffle_bc="noslip", init_type="stand_1T", A_eq=1e-4, record_stride=12)),
    ])

    exp3_res = run_batch(exp3_jobs, worker=_wake_and_pert_worker)
    exp3_stats = {}
    for lbl, _, _ in exp3_jobs:
        rr = exp3_res[lbl]; t = rr["t"]; m = (t >= 10.0) & (t <= 30.0)
        af = float(np.polyfit(t[m], 0.5 * np.log(np.maximum(rr["e_wave"][m], 1e-30)), 1)[0])
        entry = {"cold_alpha_1T": af}
        if "wake_info" in rr:
            entry.update(rr["wake_info"])
        exp3_stats[lbl] = entry
        extra = f" | rhs={entry['base_rhs_max']:.1e}, u_ds={entry['ux_wake_min_downstream']:.4f}, dp={entry['dp_over_pbar_pct']:+.3f}%" if "ux_wake_min_downstream" in entry else ""
        print(f"  [{lbl:24s}] cold_alpha={af:+.5f}{extra}")

    wake_snap_80 = exp3_res["noslip_Re500_80x240"]

    # ------------------------------------------------------------------
    # Experiment 4: Clean t=150 Base Hybrid Audit (C2), 1T Trigger Bracket & 2T Branch (M5)
    # ------------------------------------------------------------------
    print("\n--- Experiment 4: Clean t=150 Base Hybrid Margin (C2), 1T Bracket & 2T Branch (M5) ---")
    cf_untuned = [
        {"beta": 0.016, "wH": W1_COLD, "xiH": 0.12, "sigma_o": 0.045, "n_cav": 5, "phase_offset": 0.5, "nonlinear": True},
        {"beta": 0.004, "wH": W2_COLD, "xiH": 0.12, "sigma_o": 0.015, "n_cav": 5, "phase_offset": 0.5, "nonlinear": True},
    ]
    cf_retuned = [
        {"beta": 0.016, "wH": 0.900, "xiH": 0.12, "sigma_o": 0.045, "n_cav": 5, "phase_offset": 0.5, "nonlinear": True},
        {"beta": 0.004, "wH": W2_COLD, "xiH": 0.12, "sigma_o": 0.015, "n_cav": 5, "phase_offset": 0.5, "nonlinear": True},
    ]
    cf_blade_root = [
        {"beta": 0.016, "wH": 0.900, "xiH": 0.12, "sigma_o": 0.045, "n_cav": 5, "phase_offset": 0.0, "nonlinear": True},
        {"beta": 0.004, "wH": W2_COLD, "xiH": 0.12, "sigma_o": 0.015, "n_cav": 5, "phase_offset": 0.0, "nonlinear": True},
    ]

    # Reuse the ALREADY-RELAXED t=150 base wakes from Exp 3 (zero duplicate work!)
    U40_500 = exp3_res["noslip_Re500_40x120"]["U_base_saved"]
    U60_500 = exp3_res["noslip_Re500_60x160"]["U_base_saved"]
    U80_500 = exp3_res["noslip_Re500_80x240"]["U_base_saved"]
    U40_1000 = simulate_2d_ns(nx=40, nth=120, t_end=150.0, M_bar=0.12, Re=1000.0, n_baffles=5, L_baffle=0.45,
                              baffle_bc="noslip", init_type="steady", A_eq=0.0, record_stride=80)["U_final"]

    exp4_specs = [
        # Clean linear (A_0 = 1e-4) with converged t=150 wakes (fit window t in [18, 60], C2!)
        ("lin_bare_Euler",          dict(nx=40, nth=120, t_end=75.0, M_bar=0.12, Re=np.inf, n_p=0.22, n_v=0.04, init_type="spin_1T", A_eq=1e-4)),
        ("lin_bare_Re500",          dict(nx=40, nth=120, t_end=75.0, M_bar=0.12, Re=500.0, n_p=0.22, n_v=0.04, init_type="spin_1T", A_eq=1e-4)),
        ("lin_baf_Euler",           dict(nx=40, nth=120, t_end=75.0, M_bar=0.12, Re=np.inf, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, init_type="spin_1T", A_eq=1e-4)),
        ("lin_baf_slip500",         dict(nx=40, nth=120, t_end=75.0, M_bar=0.12, Re=500.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, baffle_bc="slip", init_type="spin_1T", A_eq=1e-4)),
        ("lin_baf_noslip500_40",    dict(nx=40, nth=120, t_end=75.0, M_bar=0.12, Re=500.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, baffle_bc="noslip", U_base=U40_500, init_type="spin_1T", A_eq=1e-4)),
        ("lin_baf_noslip500_60",    dict(nx=60, nth=160, t_end=75.0, M_bar=0.12, Re=500.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, baffle_bc="noslip", U_base=U60_500, init_type="spin_1T", A_eq=1e-4)),
        ("lin_hyb_untuned_ns500",   dict(nx=40, nth=120, t_end=75.0, M_bar=0.12, Re=500.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, baffle_bc="noslip", cav_spec=cf_untuned, U_base=U40_500, init_type="spin_1T", A_eq=1e-4)),
        ("lin_hyb_retuned_ns500_40",dict(nx=40, nth=120, t_end=75.0, M_bar=0.12, Re=500.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, baffle_bc="noslip", cav_spec=cf_retuned, U_base=U40_500, init_type="spin_1T", A_eq=1e-4)),
        ("lin_hyb_retuned_ns500_60",dict(nx=60, nth=160, t_end=75.0, M_bar=0.12, Re=500.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, baffle_bc="noslip", cav_spec=cf_retuned, U_base=U60_500, init_type="spin_1T", A_eq=1e-4)),
        ("lin_hyb_retuned_ns500_80",dict(nx=80, nth=240, t_end=75.0, M_bar=0.12, Re=500.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, baffle_bc="noslip", cav_spec=cf_retuned, U_base=U80_500, init_type="spin_1T", A_eq=1e-4)),
        ("lin_hyb_floor_ns500_60",  dict(nx=60, nth=160, t_end=75.0, M_bar=0.12, Re=500.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, baffle_bc="noslip", cav_spec=cf_retuned, U_base=U60_500, init_type="steady", A_eq=0.0)),
        ("lin_hyb_root_ns500_40",   dict(nx=40, nth=120, t_end=75.0, M_bar=0.12, Re=500.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, baffle_bc="noslip", cav_spec=cf_blade_root, U_base=U40_500, init_type="spin_1T", A_eq=1e-4)),
        ("lin_hyb_2T_ns500_60",     dict(nx=60, nth=160, t_end=75.0, M_bar=0.12, Re=500.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, baffle_bc="noslip", cav_spec=cf_retuned, U_base=U60_500, init_type="spin_2T", A_eq=1e-4)),
        # Nonlinear shootout, 1T trigger bracket (0.010 vs 0.015) & 2T upper branch (M5!)
        ("NL_bare_Euler",           dict(nx=40, nth=120, t_end=160.0, M_bar=0.12, Re=np.inf, n_p=0.22, n_v=0.04, init_type="spin_1T", A_eq=0.055, record_stride=16)),
        ("NL_bare_Re500",           dict(nx=40, nth=120, t_end=160.0, M_bar=0.12, Re=500.0, n_p=0.22, n_v=0.04, init_type="spin_1T", A_eq=0.055, record_stride=16)),
        ("NL_baf_noslip500",        dict(nx=40, nth=120, t_end=160.0, M_bar=0.12, Re=500.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, baffle_bc="noslip", U_base=U40_500, init_type="spin_1T", A_eq=0.055, record_stride=16)),
        ("NL_hyb_Euler",            dict(nx=40, nth=120, t_end=160.0, M_bar=0.12, Re=np.inf, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, baffle_bc="slip", cav_spec=cf_retuned, init_type="spin_1T", A_eq=0.055, record_stride=16)),
        ("NL_hyb_noslip1000",       dict(nx=40, nth=120, t_end=160.0, M_bar=0.12, Re=1000.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, baffle_bc="noslip", cav_spec=cf_retuned, U_base=U40_1000, init_type="spin_1T", A_eq=0.055, record_stride=16)),
        ("NL_hyb_slip500",          dict(nx=40, nth=120, t_end=160.0, M_bar=0.12, Re=500.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, baffle_bc="slip", cav_spec=cf_retuned, init_type="spin_1T", A_eq=0.055, record_stride=16)),
        ("NL_hyb_noslip500_A010",   dict(nx=40, nth=120, t_end=160.0, M_bar=0.12, Re=500.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, baffle_bc="noslip", cav_spec=cf_retuned, U_base=U40_500, init_type="spin_1T", A_eq=0.010, record_stride=16)),
        ("NL_hyb_noslip500_A015",   dict(nx=40, nth=120, t_end=160.0, M_bar=0.12, Re=500.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, baffle_bc="noslip", cav_spec=cf_retuned, U_base=U40_500, init_type="spin_1T", A_eq=0.015, record_stride=16)),
        ("NL_hyb_noslip500_A055",   dict(nx=40, nth=120, t_end=160.0, M_bar=0.12, Re=500.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, baffle_bc="noslip", cav_spec=cf_retuned, U_base=U40_500, init_type="spin_1T", A_eq=0.055, record_stride=16)),
        ("NL_hyb_noslip500_60",     dict(nx=60, nth=160, t_end=160.0, M_bar=0.12, Re=500.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, baffle_bc="noslip", cav_spec=cf_retuned, U_base=U60_500, init_type="spin_1T", A_eq=0.055, record_stride=18)),
        ("NL_hyb_2T_ns500_60",      dict(nx=60, nth=160, t_end=160.0, M_bar=0.12, Re=500.0, n_p=0.22, n_v=0.04, n_baffles=5, L_baffle=0.45, baffle_bc="noslip", cav_spec=cf_retuned, U_base=U60_500, init_type="spin_2T", A_eq=0.055, record_stride=18)),
    ]
    exp4_runs = run_batch(exp4_specs)
    exp4_stats = {}
    for cid, _ in exp4_specs:
        st = stats_ns(exp4_runs[cid], win_lo=18.0, win_hi=60.0)
        exp4_stats[cid] = st
        print(f"  [{cid:25s}] P={st['p_over_pbar_pct']:5.2f}% | E={st['e_wave_late']:.2e} (l/m={st['e_wave_late_over_mid']:.3f}) | a18_60={st['alpha_fit_win_18_60']:+.4f}")

    # ------------------------------------------------------------------
    # Figure 1: Viscous Strouhal Hopf Envelope & 100x300 Baffle Grid Convergence
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.6), dpi=150)

    M_arr = [d["M_bar"] for d in hopf_envelopes["inf"]]
    for rtag, lbl, col, ls in [
        ("inf",  "Inviscid Euler ($\\mathrm{Re} = \\infty$, $2T$ trap at $0.16$)", "#d62728", "-"),
        ("2000", "Compressible NS ($\\mathrm{Re} = 2000$: linear stab., $2T$ subcrit. $4.4\\%$)", "#ff7f0e", "--"),
        ("1000", "Compressible NS ($\\mathrm{Re} = 1000$: $2T$ trap nonlinearly quenched)", "#1f77b4", "-."),
        ("500",  "Compressible NS ($\\mathrm{Re} = 500$, trough & trap filled)", "#2ca02c", "-"),
    ]:
        ax1.plot(M_arr, [d["n_pH_global"] for d in hopf_envelopes[rtag]], ls, color=col, lw=2.2, label=lbl)
    ax1.axhline(0.22, color="#333333", ls=":", lw=1.3, label="Test flame $n_p = 0.22$")
    ax1.axvline(0.12, color="gray", ls=":", lw=1.0, alpha=0.6)
    ax1.axvline(0.16, color="gray", ls=":", lw=1.0, alpha=0.6)
    ax1.set_xlabel("Chamber axial Mach number $\\bar{M}$ ($L_a = 1.5$)")
    ax1.set_ylabel("Exact Chebyshev NS global all-mode Hopf $n_{pH,\\mathrm{global}}$")
    ax1.set_ylim(0.0, 0.62)
    ax1.set_title("Vorticity-Wave Viscous Attenuation ($L_{\\mathrm{visc}}/L_a \\approx \\mathrm{Re}\\,\\bar{M}^3/\\omega^2 L_a$)", fontweight="bold")
    ax1.grid(True, alpha=0.28)
    ax1.legend(loc="upper left", fontsize=7.2, framealpha=0.93)

    g_eul = [40, 60, 80]
    g_ns = [40, 60, 80, 100]
    gmap = {40: "40x120", 60: "60x160", 80: "80x240", 100: "100x300"}
    ax2.plot(g_eul, [-exp3_stats[f"slip_Euler_{gmap[g]}"]["cold_alpha_1T"] for g in g_eul],
             "s--", color="#d62728", lw=2.2, ms=7.0, label="Inviscid Euler ($\\bar{M}=0.12$, drifting)")
    ax2.plot(g_ns, [-exp3_stats[f"slip_Re500_{gmap[g]}"]["cold_alpha_1T"] for g in g_ns],
             "o-", color="#2ca02c", lw=2.4, ms=7.2, label="Free-slip NS ($\\mathrm{Re}=500$: $0.1\\%$ converged)")
    ax2.plot(g_ns, [-exp3_stats[f"noslip_Re500_{gmap[g]}"]["cold_alpha_1T"] for g in g_ns],
             "D-", color="#9467bd", lw=2.3, ms=7.0, label="No-slip wake NS ($\\mathrm{Re}=500$: $\\approx -19\\%$ penalty)")
    ax2.axhline(-exp3_stats["M0_noslip_Re500_80x240"]["cold_alpha_1T"], color="#17becf", ls="-.", lw=1.6,
                label="$\\bar{M}=0$ NS ($-0.026$: face shear + tip diss.)")
    ax2.axhline(-exp3_stats["M0_bare_Re500_80x240"]["cold_alpha_1T"], color="#17becf", ls=":", lw=1.4,
                label="$\\bar{M}=0$ bare NS ($-0.0115$: face shear + SK)")
    ax2.set_xticks(g_ns, ["$40\\times 120$", "$60\\times 160$", "$80\\times 240$", "$100\\times 300$"])
    ax2.set_xlabel("Grid resolution ($n_x \\times n_\\theta$)")
    ax2.set_ylabel("Cold $1T$ baffle damping $-\\alpha_{1T}$ ($N_b=5, L_b=0.45$)")
    ax2.set_ylim(0.0, 0.145)
    ax2.set_title("Converged Finite-$\\mathrm{Re}$ Baffle Damping & No-Slip Penalty", fontweight="bold")
    ax2.grid(True, alpha=0.28)
    ax2.legend(loc="center right", fontsize=7.2, framealpha=0.93)

    fig.suptitle("Rocket Combustion Entry 011 (Part 1, Hardened): Viscous Vorticity Attenuation & 100x300 Convergence", fontsize=11.2, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp011_viscous_strouhal_and_convergence.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Steady No-Slip Wake Field (80x240, t=150) & Zoomed Rotating-Shock Capping + Bracket
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.6), dpi=150)

    ux_wk = wake_snap_80["ux_base"]
    u_ds_80 = exp3_stats["noslip_Re500_80x240"]["ux_wake_min_downstream"]
    im = ax1.pcolormesh(wake_snap_80["theta"], wake_snap_80["x"], ux_wk, shading="auto", cmap="viridis", vmin=0.02, vmax=0.145)
    for k in range(5):
        th_b = k * 2.0 * np.pi / 5.0
        ax1.plot([th_b, th_b], [0.0, 0.45], color="#ffffff", lw=2.5)
    ax1.axhline(X_F, color="#ff7f0e", ls="--", lw=1.2, label="Flame sheet $x_f = 0.25$")
    ax1.axhline(0.45, color="#ffffff", ls=":", lw=1.1, label="Baffle tips $L_b = 0.45$")
    ax1.set_xlabel("Azimuth $\\theta$ (rad)")
    ax1.set_ylabel("Axial coordinate $x$")
    ax1.set_title("Steady No-Slip Viscous Wakes $u_x(x, \\theta)$ ($\\mathrm{Re}=500, 80\\times 240, t=150$)", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=7.8, framealpha=0.9)
    cb = fig.colorbar(im, ax=ax1, pad=0.02)
    cb.set_label(f"Axial velocity $u_x$ ($\\bar{{M}}=0.12$; downstream wake $u_{{\\min}}={u_ds_80:.3f}$)")

    for cid, lbl, col, ls in [
        ("NL_bare_Euler",          "Bare Euler ($\\mathrm{Re}=\\infty$: $8.9\\%$ shock)", "#d62728", "-"),
        ("NL_bare_Re500",          "Bare NS ($\\mathrm{Re}=500$: $5.7\\%$ shock)", "#ff7f0e", "--"),
        ("NL_baf_noslip500",       "No-slip baffles alone ($\\mathrm{Re}=500$: $6.5\\%$)", "#9467bd", "-."),
        ("NL_hyb_Euler",           "Retuned $2\\%$ hybrid — Euler ($5.6\\%$)", "#7f7f7f", ":"),
        ("NL_hyb_noslip500_A055",  "Retuned hybrid — No-slip NS ($4.3\\%, A_0=0.055$)", "#2ca02c", "-"),
        ("NL_hyb_noslip500_A015",  "Retuned hybrid — No-slip NS ($A_0=0.015$: triggers!)", "#0b5e20", "--"),
        ("NL_hyb_noslip500_A010",  "Retuned hybrid — No-slip NS ($A_0=0.010$: decays!)", "#17becf", "-."),
        ("NL_hyb_2T_ns500_60",     "Retuned hybrid — $2T$ upper branch ($60\\times 160$: $4.1\\%$)", "#1f77b4", ":"),
    ]:
        rr = exp4_runs[cid]
        ax2.semilogy(rr["t"], np.maximum(rr["e_wave"], 1e-16), ls, color=col, lw=2.0, label=lbl)
    ax2.set_xlabel("Time $t$")
    ax2.set_ylabel("2D Perturbation Energy $E_{\\mathrm{wave}}(t)$ (log scale)")
    ax2.set_ylim(1e-7, 6e-3)
    ax2.set_title("Viscous Rotating-Shock Capping, $1T$ Trigger Bracket & $2T$ Branch", fontweight="bold")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="lower left", fontsize=7.0, framealpha=0.93)

    fig.suptitle("Rocket Combustion Entry 011 (Part 2, Hardened): Converged No-Slip Wakes & Subcritical Branches", fontsize=11.2, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp011_noslip_wake_and_shock_quench.png")
    plt.close(fig)

    metrics = {
        "exp1_verification": {
            "steady_slip_baffle_err": exp1_res["steady_slip"]["steady_max_err"],
            "periodic_cons_ns_bare": ce_bare,
            "periodic_cons_ns_noslip5": ce_baf,
            "stokes_kirchhoff_k2_Re200": {"exact": a_sk_ex, "pde": a_sk_pde, "rel_err_pct": abs(a_sk_pde - a_sk_ex) / abs(a_sk_ex) * 100.0},
            "shear_diffusion_k2_Re200": {"exact": a_sh_ex, "pde": a_sh_pde, "rel_err_pct": abs(a_sh_pde - a_sh_ex) / abs(a_sh_ex) * 100.0},
            "chebyshev_ns_vs_pde_60x160_t20_40": cheb_vs_pde,
        },
        "exp2_vorticity_viscous_law": {
            "hopf_envelopes": hopf_envelopes,
            "re_cross_M016_chiral_2T": re_cross_2t,
            "cold_M012_Re5000": cold_M012_Re5000,
            "m016_chiral_vs_Re": m016_chiral_vs_Re,
            "m016_nonlinear_2T_64x200": nl2t_stats,
        },
        "exp3_baffle_convergence_and_wake": exp3_stats,
        "exp4_reacting_and_shock_shootout": exp4_stats,
    }
    out_json = RESULTS_DIR / "exp011_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics -> {out_json}")
    print("Saved plots   -> exp011_viscous_strouhal_and_convergence.png, exp011_noslip_wake_and_shock_quench.png")


if __name__ == "__main__":
    main()
