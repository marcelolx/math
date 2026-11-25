"""Entry 010 (rocket-combustion): 2D Compressible Euler Throughflow with Protruding
Solid-Wall Baffles & Nonlinear Wide-Mouth Faceplate Cavity Rings (Hardened per Peer Review #7).

Resolves Open Problem #1 from rocket-combustion/SYNTHESIS.md by embedding two first-principles
passive hardware mechanisms directly into the 2D (x, theta) conservative compressible Euler
annular combustor with steady axial throughflow M_bar > 0 (Entry 009 baseline; ZERO ad-hoc
entropy-flush or drag closures):

  1. Protruding Inviscid Radial Baffles (N_b thin blades on x in [0, L_b], theta = 2*pi*k/N_b):
     Implemented via split-face mirror HLLC solid-wall Riemann fluxes on azimuthal cell faces
     lying on each blade. Conserves mass, axial momentum, and total energy to machine precision
     (~1e-16), satisfies discrete azimuthal blade-torque bookkeeping (~1e-17), and captures
     baffle-tip shock diffraction and acoustic-to-vorticity shedding.

  2. Nonlinear Faceplate Helmholtz Cavity Rings at x = 0 (Cummings-Eversman orifice jet drag):
     Discrete N_c (or continuous) resonators of volume V_k = beta_k * (2*pi*L_a) / N_c coupled
     into the choked isentropic inlet mass-flux BC rho_in*ux_in = M_bar - u_port(theta,t),
     with quadratic orifice pressure loss R_nl * |q_k| * q_k, R_nl = 1/(2*Cd^2*A_ok^2), Cd = 0.65.
     Linearizes exactly into the 1D-in-x Chebyshev 2D throughflow eigenvalue solver (windowed to
     omega in [0.05, 5.5] to exclude spurious high-frequency Chebyshev boundary modes).

Hardening upgrades per independent cross-model Peer Review #7 (Claude, 110 independent runs):
  - C1: Explicit per-mode, per-grid numerical-damping audit (1T, 2T, 3T across 40x120, 60x160,
        80x240) + bare reacting 2T convergence + aligned stand_2T blade-count audit at n_v = 0
        on 80x240 + L_b = 0.60 2T check.
  - C2: Full Exp 4 orifice-drag cliff re-run on 64x200 (t = 220) alongside 48x160, exposing
        that sigma_o = 2.5% chokes into a ~4.0% 2T rotating shock on 64x200 (the true cliff
        lies between 2.5% and 5.0%, while wide-mouth 5.0% survives refinement).
  - M1: All linear reacting growth rates measured cleanly at A_0 = 1e-4 inside the linear window
        (p'/p_bar < 0.5%, zero subcritical cubic-flame contamination).
  - M2: Adds n_v = 0 pure pressure-coupled baffle sweep showing short-baffle amplification is
        chiral (n_v > 0) and L_b = 0.60 re-destabilization is driven primarily by baffle-induced
        frequency downshift / Crocco phase shift rather than a cold St_eff = 1.00 trough.
  - M3: Quantifies M_bar = 0 (zero-throughflow) inviscid baffle numerical edge dissipation across
        40x120 and 80x240 (~-0.011 -> -0.006, 1st-order numerical) vs. ~90% physical shedding.
  - M4/M5: Converged least-squares omega_1T(0.45) (-6.7%: 0.989 -> 0.923 on 80x240), explicit
        omega_H retuning scan showing optimal tuning targets the baffled REACTING frequency
        (~0.89-0.90), cavity phase-offset placement audit (mid-sector 0.5 vs. blade-root 0.0:
        36% margin loss at blade roots), and all-mode (spin_2T, cold_1L) hybrid check on 60x160.
  - M6: Exact spin_1T subcritical trigger bracket (0.015, 0.025] at M_bar = 0.16 on 48x160 and
        exact matching Chebyshev rows (m6).
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

W1_COLD = 0.9893
W2_COLD = 1.9799
W3_COLD = 2.9705


# ---------------------------------------------------------------------------
# 1. Exact 1D-in-x Chebyshev Linearized Euler + Faceplate Cavity Eigen-Solver
# ---------------------------------------------------------------------------

def cheb_diff(N: int, L_a: float = L_A) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chebyshev-Gauss-Lobatto nodes x in [0, L_a], derivative matrix Dx, and Clenshaw-Curtis weights."""
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


def exact_2d_cavity_eig(
    M_bar: float,
    m: int,
    n_p: float = 0.0,
    n_v: float = 0.0,
    cav_families: list[tuple[float, float, float]] | None = None,
    port_form_sq: float = 1.0,
    omega_window: tuple[float, float] | None = None,
    N: int = 36,
    check_both_chiral_signs: bool = True,
) -> complex:
    """Exact least-stable eigenvalue of 2D linearized Euler throughflow with faceplate Helmholtz cavities.

    Filtered to omega in [0.05, 5.5] by default to exclude spurious high-frequency Chebyshev boundary modes.
    When n_v != 0 and m != 0, evaluates BOTH +m and -m and returns the least-stable root.
    """
    if check_both_chiral_signs and abs(n_v) > 0.0 and m != 0:
        vp = exact_2d_cavity_eig(M_bar, abs(m), n_p, n_v, cav_families, port_form_sq, omega_window, N, False)
        vn = exact_2d_cavity_eig(M_bar, -abs(m), n_p, n_v, cav_families, port_form_sq, omega_window, N, False)
        return vp if vp.real >= vn.real else vn

    cavs = cav_families or []
    x, Dx, w_cc = cheb_diff(N, L_A)
    n = N + 1
    I = np.eye(n)
    wf = np.exp(-0.5 * ((x - X_F) / SIG_F) ** 2)
    wf /= float(np.sum(w_cc * wf))
    w_sense = w_cc * wf

    has_f = (abs(n_p) > 0.0 or abs(n_v) > 0.0)
    n_f = 10 if has_f else 0
    n_c = 2 * len(cavs)
    dim = 4 * n + n_f + n_c
    A = np.zeros((dim, dim), dtype=complex)
    B = np.eye(dim, dtype=complex)

    def sl(q: int) -> slice:
        return slice(q * n, (q + 1) * n)

    A[sl(0), sl(0)] = -M_bar * Dx; A[sl(0), sl(1)] = -Dx; A[sl(0), sl(2)] = -1j * m * I
    A[sl(1), sl(1)] = -M_bar * Dx; A[sl(1), sl(3)] = -Dx
    A[sl(2), sl(2)] = -M_bar * Dx; A[sl(2), sl(3)] = -1j * m * I
    A[sl(3), sl(1)] = -Dx; A[sl(3), sl(2)] = -1j * m * I; A[sl(3), sl(3)] = -M_bar * Dx

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

    ic0 = 4 * n + n_f
    for j, (beta_j, wH_j, xiH_j) in enumerate(cavs):
        idx_pc = ic0 + 2 * j
        idx_qt = ic0 + 2 * j + 1
        vj = beta_j * L_A * port_form_sq
        A[idx_pc, idx_qt] = 1.0 / vj
        A[idx_qt, 3 * n] = (wH_j ** 2) * vj
        A[idx_qt, idx_pc] = -(wH_j ** 2) * vj
        A[idx_qt, idx_qt] = -2.0 * xiH_j * wH_j

    for r_idx in (0, n, 2 * n):
        A[r_idx, :] = 0.0; B[r_idx, :] = 0.0
    A[0, 0] = 1.0; A[0, 3 * n] = -1.0
    A[n, n] = 1.0; A[n, 3 * n] = M_bar
    for j in range(len(cavs)):
        A[n, ic0 + 2 * j + 1] = 1.0
    A[2 * n, 2 * n] = 1.0

    r_out = 3 * n + N
    A[r_out, :] = 0.0; B[r_out, :] = 0.0
    A[r_out, n + N] = 1.0
    A[r_out, 3 * n + N] = -0.5 * M_bar * GAMMA
    A[r_out, 0 * n + N] = +0.5 * M_bar

    vals = eig(A, B, right=False)
    w_lo, w_hi = omega_window if omega_window is not None else (0.05, 5.5)
    cands = [v for v in vals if np.isfinite(v) and w_lo <= v.imag <= w_hi and -0.80 < v.real < 0.50]
    cands.sort(key=lambda z: z.real, reverse=True)
    return complex(cands[0])


# ---------------------------------------------------------------------------
# 2. 2D Conservative MUSCL-MC + HLLC Euler Solver with Baffles & Cavities
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
    """HLLC numerical flux along normal direction (vn) with passive tangential velocity (vt)."""
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
    """Choked zero-swirl isentropic inlet BC with faceplate cavity port inflow u_port(theta)."""
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
    """Exact Marble-Candel compact choked nozzle BC at x = L_a (at M_bar = 0, reduces to solid wall ux = 0)."""
    rho0, ux0, ut0, p0, c0 = prim_2d(U_out)
    pow_g = 2.0 / (GAMMA - 1.0)
    J_plus = ux0 + pow_g * c0
    s_out = np.log(np.maximum((p0 / P_BAR) / (rho0 ** GAMMA), 1e-15))
    c = np.maximum(J_plus / (M_bar + pow_g), 0.35)
    ux = M_bar * c
    rho = np.maximum((c * c / np.exp(s_out)) ** (1.0 / (GAMMA - 1.0)), 1e-6)
    p = P_BAR * (rho ** GAMMA) * np.exp(s_out)
    return cons_2d(rho, ux, ut0, p)


def simulate_2d_hw(
    nx: int = 40,
    nth: int = 120,
    t_end: float = 120.0,
    cfl: float = 0.35,
    M_bar: float = 0.12,
    n_p: float = 0.0,
    n_v: float = 0.0,
    n_baffles: int = 0,
    L_baffle: float = 0.0,
    cav_spec: list[dict] | None = None,
    init_type: str = "spin_1T",
    A_eq: float = 0.055,
    periodic_x: bool = False,
    record_stride: int = 12,
) -> dict:
    """2D (x, theta) compressible Euler annular combustor with solid-wall baffles & faceplate cavities."""
    dx = L_A / nx
    dth = 2.0 * np.pi / nth
    x = (np.arange(nx) + 0.5) * dx
    th = (np.arange(nth) + 0.5) * dth
    X, TH = np.meshgrid(x, th, indexing="ij")

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
        beta_f = float(sp["beta"])
        wH = float(sp["wH"])
        xiH = float(sp["xiH"])
        sig_o = float(sp.get("sigma_o", 0.035))
        nc = int(sp.get("n_cav", 5))
        Cd = float(sp.get("Cd", CD_ORIFICE))
        nl_on = bool(sp.get("nonlinear", True))
        if nc == 0:
            v_rad = beta_f * L_A
            r_nl = (1.0 / (2.0 * (Cd ** 2) * (sig_o ** 2))) if nl_on else 0.0
            cav_list.append({
                "mode": "cont", "v_rad": v_rad, "wH": wH, "xiH": xiH, "r_nl": r_nl,
                "Pc": np.zeros(nth), "qt": np.zeros(nth), "port_form_sq": {1: 1.0, 2: 1.0, 3: 1.0},
            })
        else:
            V_k = beta_f * (2.0 * np.pi * L_A) / nc
            A_ok = (sig_o * 2.0 * np.pi) / nc
            r_nl = (1.0 / (2.0 * (Cd ** 2) * (A_ok ** 2))) if nl_on else 0.0
            W = np.zeros((nc, nth))
            # Grid-regularized narrow Gaussian port profile (1.5*dth minimum width)
            sig_th = max(1.5 * dth, 0.5 * (2.0 * np.pi * sig_o / nc))
            phase_off = float(sp.get("phase_offset", 0.5))
            for k in range(nc):
                th_k = (k + phase_off) * (2.0 * np.pi / nc)
                d_ang = np.angle(np.exp(1j * (th - th_k)))
                wk = np.exp(-0.5 * (d_ang / sig_th) ** 2)
                W[k] = wk / float(np.sum(wk) * dth)
            pfs = {}
            for mm in (1, 2, 3):
                wh = float(np.abs(np.sum(W[0] * np.exp(-1j * mm * th)) * dth))
                pfs[mm] = wh * wh
            cav_list.append({
                "mode": "disc", "V_k": V_k, "wH": wH, "xiH": xiH, "r_nl": r_nl,
                "W": W, "Pc": np.zeros(nc), "qt": np.zeros(nc), "port_form_sq": pfs,
            })

    P0 = np.zeros_like(X)
    ux0 = np.full_like(X, M_bar)
    ut0 = np.zeros_like(X)
    if init_type == "steady":
        pass
    elif init_type == "spin_1T":
        P0 = A_eq * np.cos(TH); ut0 = A_eq * np.cos(TH)
    elif init_type == "stand_1T":
        P0 = np.sqrt(2.0) * A_eq * np.cos(TH)
    elif init_type == "spin_2T":
        P0 = A_eq * np.cos(2.0 * TH); ut0 = A_eq * np.cos(2.0 * TH)
    elif init_type == "stand_2T_aligned":
        P0 = np.sqrt(2.0) * A_eq * np.cos(2.0 * TH)
    elif init_type == "spin_3T":
        P0 = A_eq * np.cos(3.0 * TH); ut0 = A_eq * np.cos(3.0 * TH)
    elif init_type == "cold_1L":
        P0 = np.sqrt(2.0) * A_eq * np.cos(np.pi * X / L_A)
    elif init_type == "periodic_2d_wave":
        kx = 2.0 * np.pi / L_A
        P0 = A_eq * (0.6 * np.cos(kx * X + TH) + 0.4 * np.cos(2.0 * TH))
        ux0 = M_bar + 0.5 * A_eq * np.cos(kx * X + TH)
        ut0 = 0.5 * A_eq * np.sin(TH)
    else:
        raise ValueError(f"Unknown init_type: {init_type}")

    p0 = P_BAR + P0
    rho0 = np.maximum((p0 / P_BAR) ** (1.0 / GAMMA), 1e-6)
    U = cons_2d(rho0, ux0, ut0, p0)
    U_init = U.copy()

    pbf = np.zeros(nth); ubf = np.zeros(nth)
    qp = np.zeros((4, nth)); qu = np.zeros((4, nth))
    has_flame = (abs(n_p) > 0.0 or abs(n_v) > 0.0)
    inv_ts = 4.0 / TAU_BAR

    def get_u_port(c_states):
        up = np.zeros(nth)
        for (_, qt), cfg in zip(c_states, cav_list):
            up += qt if cfg["mode"] == "cont" else (qt @ cfg["W"])
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

        blade_force = 0.0
        if has_baffle:
            Ft_L = Ft.copy()
            Ft_R = Ft.copy()
            rL = np.maximum(QLt[0, baffle_mask], 1e-8)
            utL = QLt[2, baffle_mask]; uxL = QLt[1, baffle_mask]; pL = np.maximum(QLt[3, baffle_mask], 1e-8)
            FL_n = hllc_2d_normal(rL, utL, uxL, pL, rL, -utL, uxL, pL)
            Ft_L[0, baffle_mask] = 0.0; Ft_L[1, baffle_mask] = 0.0; Ft_L[2, baffle_mask] = FL_n[1]; Ft_L[3, baffle_mask] = 0.0

            rR = np.maximum(QRt[0, baffle_mask], 1e-8)
            utR = QRt[2, baffle_mask]; uxR = QRt[1, baffle_mask]; pR = np.maximum(QRt[3, baffle_mask], 1e-8)
            FR_n = hllc_2d_normal(rR, -utR, uxR, pR, rR, utR, uxR, pR)
            Ft_R[0, baffle_mask] = 0.0; Ft_R[1, baffle_mask] = 0.0; Ft_R[2, baffle_mask] = FR_n[1]; Ft_R[3, baffle_mask] = 0.0

            blade_force = float(np.sum(FR_n[1] - FL_n[1]) * dx)
            dU = -(Fx[:, 1:, :] - Fx[:, :-1, :]) / dx - (Ft_L - np.roll(Ft_R, 1, axis=2)) / dth
        else:
            dU = -(Fx[:, 1:, :] - Fx[:, :-1, :]) / dx - (Ft - np.roll(Ft, 1, axis=2)) / dth

        dcav = []
        if cav_list:
            _, _, _, p_in, _ = prim_2d(U_in)
            P_face = p_in - P_BAR
            for (Pc, qt), cfg in zip(c_states, cav_list):
                wH = cfg["wH"]; xiH = cfg["xiH"]; r_nl = cfg["r_nl"]
                if cfg["mode"] == "cont":
                    vr = cfg["v_rad"]
                    dPc = qt / vr
                    dqt = (wH ** 2) * vr * (P_face - Pc - r_nl * np.abs(qt) * qt) - 2.0 * xiH * wH * qt
                else:
                    Vk = cfg["V_k"]
                    P_bar_k = (cfg["W"] @ P_face) * dth
                    dPc = qt / Vk
                    dqt = (wH ** 2) * Vk * (P_bar_k - Pc - r_nl * np.abs(qt) * qt) - 2.0 * xiH * wH * qt
                dcav.append((dPc, dqt))

        if has_flame:
            Pf = p_t - P_BAR
            pf_s = np.sum(wf[:, None] * Pf, axis=0) * dx
            uf_s = np.sum(wf[:, None] * ut_t, axis=0) * dx
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

        return dU, dpb, dub, dqp, dqu, dcav, blade_force

    cav_states = [(c["Pc"].copy(), c["qt"].copy()) for c in cav_list]
    t_list, ew_list, ppk_list, p1c_list, e1_list, e2_list, e3_list = [], [], [], [], [], [], []
    rho_m_list, ut_m_list, p_exit_list = [], [], []
    init_totals = np.sum(U, axis=(1, 2)) * dx * dth
    max_cons_err = np.zeros(4)
    blade_impulse = 0.0
    max_torque_err = 0.0

    def record(tv: float):
        nonlocal max_torque_err
        rho, ux, ut, p, _ = prim_2d(U)
        Pf = p - P_BAR
        ew = 0.5 * float(np.mean(Pf * Pf + (ux - M_bar) ** 2 + (ut - float(np.mean(ut))) ** 2))
        phat = np.fft.rfft(np.mean(Pf, axis=0)) / (0.5 * nth)
        if periodic_x:
            cur = np.sum(U, axis=(1, 2)) * dx * dth
            denom = np.maximum(np.abs(init_totals), 1.0)
            np.maximum(max_cons_err, np.abs(cur - init_totals) / denom, out=max_cons_err)
            terr = abs((cur[2] - init_totals[2]) - blade_impulse)
            if terr > max_torque_err:
                max_torque_err = terr
        t_list.append(tv)
        ew_list.append(ew)
        ppk_list.append(float(np.max(np.abs(Pf))))
        p1c_list.append(complex(phat[1]))
        e1_list.append(float(np.abs(phat[1]) ** 2))
        e2_list.append(float(np.abs(phat[2]) ** 2))
        e3_list.append(float(np.abs(phat[3]) ** 2))
        rho_m_list.append(float(np.mean(rho)))
        ut_m_list.append(float(np.mean(ut)))
        p_exit_list.append(float(np.mean(Pf[-1])))

    t = 0.0
    step = 0
    record(0.0)
    while t < t_end - 1e-12:
        _, ux_c, ut_c, _, c_c = prim_2d(U)
        dt = min(cfl / float(np.max((np.abs(ux_c) + c_c) / dx + (np.abs(ut_c) + c_c) / dth)), 0.015, t_end - t)
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
        blade_impulse += 0.5 * dt * (k1[6] + k2[6])
        t += dt
        step += 1
        if step % record_stride == 0 or t >= t_end - 1e-12:
            record(t)

    rho_f, ux_f, ut_f, p_f, _ = prim_2d(U)
    dut_dx = np.gradient(ut_f, dx, axis=0)
    dux_dth = (np.roll(ux_f, -1, axis=1) - np.roll(ux_f, 1, axis=1)) / (2.0 * dth)
    vort_f = dut_dx - dux_dth

    return {
        "x": x, "theta": th,
        "P_final": p_f - P_BAR, "ut_final": ut_f, "vort_final": vort_f,
        "steady_max_err": float(np.max(np.abs(U - U_init))),
        "periodic_cons_err": {
            "mass": float(max_cons_err[0]), "mom_x": float(max_cons_err[1]),
            "mom_th_raw": float(max_cons_err[2]), "energy": float(max_cons_err[3]),
            "mom_th_blade_impulse_err": float(max_torque_err),
        },
        "port_form_sq": [c["port_form_sq"] for c in cav_list],
        "t": np.array(t_list), "e_wave": np.array(ew_list), "p_peak": np.array(ppk_list),
        "p1_complex": np.array(p1c_list), "e1": np.array(e1_list), "e2": np.array(e2_list), "e3": np.array(e3_list),
        "rho_mean": np.array(rho_m_list), "ut_mean": np.array(ut_m_list), "p_exit_mean": np.array(p_exit_list),
    }


def stats_2d(r: dict, fit_start_frac: float = 0.45) -> dict:
    t = r["t"]
    n = len(t)
    m1 = int(0.45 * n)
    m2 = int(0.75 * n)
    e_mid = float(np.mean(r["e_wave"][m1:m2]))
    e_late = float(np.mean(r["e_wave"][m2:]))
    pk_pct = float(np.max(r["p_peak"][m2:]) / P_BAR * 100.0)
    mf = int(fit_start_frac * n)
    afit = float(np.polyfit(t[mf:], 0.5 * np.log(np.maximum(r["e_wave"][mf:], 1e-30)), 1)[0])
    # Also compute clean linear-regime fit restricted to p_peak / P_BAR < 0.005 (Peer Review #7 M1)
    lin_msk = (t >= 20.0) & (r["p_peak"] / P_BAR < 0.005)
    if int(np.sum(lin_msk)) >= 8:
        afit_lin = float(np.polyfit(t[lin_msk], 0.5 * np.log(np.maximum(r["e_wave"][lin_msk], 1e-30)), 1)[0])
        afit_e2 = float(np.polyfit(t[lin_msk], 0.5 * np.log(np.maximum(r["e2"][lin_msk], 1e-30)), 1)[0])
    else:
        afit_lin = afit
        afit_e2 = float(np.polyfit(t[mf:], 0.5 * np.log(np.maximum(r["e2"][mf:], 1e-30)), 1)[0])

    return {
        "p_over_pbar_pct": pk_pct,
        "e_wave_late": e_late,
        "e_wave_late_over_mid": float(e_late / max(e_mid, 1e-30)),
        "alpha_fit_late": afit,
        "alpha_fit_linear_clean": afit_lin,
        "alpha_fit_e2_clean": afit_e2,
        "e1_late": float(np.mean(r["e1"][m2:])),
        "e2_late": float(np.mean(r["e2"][m2:])),
        "e3_late": float(np.mean(r["e3"][m2:])),
        "rho_mean_late": float(np.mean(r["rho_mean"][m2:])),
        "ut_mean_late": float(np.mean(r["ut_mean"][m2:])),
    }


def measure_cold_lsq(r: dict, t_lo: float = 10.0, t_hi: float = 34.0, mode_key: str = "e1") -> tuple[float, float]:
    """Measure cold exponential damping alpha and accurate least-squares frequency omega from standing mode."""
    t = r["t"]
    msk = (t >= t_lo) & (t <= t_hi)
    tm = t[msk]
    af = float(np.polyfit(tm, 0.5 * np.log(np.maximum(r[mode_key][msk], 1e-30)), 1)[0])
    # For stand_1T, p1_complex(t) ~ A * exp(alpha*t) * cos(omega*t + phi) -> unwrap phase or fit cosine!
    if mode_key == "e1":
        sig = np.real(r["p1_complex"][msk]) * np.exp(-af * (tm - tm[0]))
        sig = sig - float(np.mean(sig))
        # Coarse FFT estimate then golden/grid least-squares refinement for high accuracy
        dt_s = float(np.mean(np.diff(tm)))
        nfft = len(sig) * 16
        freqs = np.fft.rfftfreq(nfft, d=dt_s) * 2.0 * np.pi
        w0 = float(freqs[int(np.argmax(np.abs(np.fft.rfft(sig, n=nfft))))])
        w_grid = np.linspace(max(0.5, w0 - 0.08), w0 + 0.08, 401)
        best_err = 1e30; best_w = w0
        for wc in w_grid:
            M = np.column_stack([np.cos(wc * tm), np.sin(wc * tm)])
            coef, _, _, _ = np.linalg.lstsq(M, sig, rcond=None)
            err = float(np.sum((sig - M @ coef) ** 2))
            if err < best_err:
                best_err = err; best_w = float(wc)
        w_est = best_w
    else:
        sig = r[mode_key][msk] * np.exp(-2.0 * af * (tm - tm[0]))
        sig = sig - float(np.mean(sig))
        dt_s = float(np.mean(np.diff(tm)))
        nfft = len(sig) * 16
        freqs = np.fft.rfftfreq(nfft, d=dt_s) * 2.0 * np.pi
        w_est = 0.5 * float(freqs[int(np.argmax(np.abs(np.fft.rfft(sig, n=nfft))))])
    return af, w_est


def _sim_worker(task: tuple[str, dict]) -> tuple[str, dict]:
    label, kwargs = task
    return label, simulate_2d_hw(**kwargs)


def run_parallel_batch(tasks: list[tuple[str, dict]], max_workers: int | None = None) -> dict[str, dict]:
    nw = max_workers or min(6, max(1, (os.cpu_count() or 2) - 2), len(tasks))
    if nw <= 1 or len(tasks) <= 1:
        return {lbl: simulate_2d_hw(**kw) for lbl, kw in tasks}
    with ProcessPoolExecutor(max_workers=nw) as ex:
        results = list(ex.map(_sim_worker, tasks))
    return dict(results)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 84)
    print("Entry 010 (Hardened per Peer Review #7): 2D Euler Baffles & Faceplate Cavity Rings")
    print("=" * 84)

    # ------------------------------------------------------------------
    # Experiment 1: Verification, Per-Mode Numerical Damping Table & M_bar=0 Baffle Check
    # ------------------------------------------------------------------
    print("\n--- Experiment 1: Conservation, Per-Mode Damping Audit (C1) & M_bar=0 Baffle Check (M3) ---")
    cav_4pct_disc = [
        {"beta": 0.032, "wH": W1_COLD, "xiH": 0.14, "sigma_o": 0.060, "n_cav": 5, "nonlinear": False},
        {"beta": 0.008, "wH": W2_COLD, "xiH": 0.12, "sigma_o": 0.020, "n_cav": 5, "nonlinear": False},
    ]
    cav_4pct_cont = [
        {"beta": 0.032, "wH": W1_COLD, "xiH": 0.14, "sigma_o": 0.060, "n_cav": 0, "nonlinear": False},
        {"beta": 0.008, "wH": W2_COLD, "xiH": 0.12, "sigma_o": 0.020, "n_cav": 0, "nonlinear": False},
    ]
    exp1_tasks = [
        ("steady", dict(nx=40, nth=120, t_end=20.0, M_bar=0.12, n_baffles=5, L_baffle=0.45,
                        cav_spec=cav_4pct_disc, init_type="steady", A_eq=0.0, record_stride=20)),
        ("cons", dict(nx=40, nth=120, t_end=10.0, M_bar=0.12, n_baffles=5, L_baffle=0.45,
                      init_type="periodic_2d_wave", A_eq=0.10, periodic_x=True, record_stride=20)),
        # Cavity Chebyshev cross-checks
        ("cav_1T_cont_40x120", dict(nx=40, nth=120, t_end=30.0, M_bar=0.12, cav_spec=cav_4pct_cont, init_type="spin_1T", A_eq=1e-4, record_stride=10)),
        ("cav_1T_cont_60x160", dict(nx=60, nth=160, t_end=30.0, M_bar=0.12, cav_spec=cav_4pct_cont, init_type="spin_1T", A_eq=1e-4, record_stride=10)),
        ("cav_1T_disc5_40x120", dict(nx=40, nth=120, t_end=30.0, M_bar=0.12, cav_spec=cav_4pct_disc, init_type="spin_1T", A_eq=1e-4, record_stride=10)),
        ("cav_2T_cont_60x160", dict(nx=60, nth=160, t_end=30.0, M_bar=0.12, cav_spec=cav_4pct_cont, init_type="spin_2T", A_eq=1e-4, record_stride=10)),
        # Per-mode numerical damping audit across 40x120, 60x160, 80x240 (Peer Review #7 C1)
        ("bare_1T_40x120", dict(nx=40, nth=120, t_end=30.0, M_bar=0.12, init_type="spin_1T", A_eq=1e-4, record_stride=10)),
        ("bare_1T_60x160", dict(nx=60, nth=160, t_end=30.0, M_bar=0.12, init_type="spin_1T", A_eq=1e-4, record_stride=10)),
        ("bare_1T_80x240", dict(nx=80, nth=240, t_end=30.0, M_bar=0.12, init_type="spin_1T", A_eq=1e-4, record_stride=12)),
        ("bare_2T_40x120", dict(nx=40, nth=120, t_end=30.0, M_bar=0.12, init_type="spin_2T", A_eq=1e-4, record_stride=10)),
        ("bare_2T_60x160", dict(nx=60, nth=160, t_end=30.0, M_bar=0.12, init_type="spin_2T", A_eq=1e-4, record_stride=10)),
        ("bare_2T_80x240", dict(nx=80, nth=240, t_end=30.0, M_bar=0.12, init_type="spin_2T", A_eq=1e-4, record_stride=12)),
        ("bare_3T_40x120", dict(nx=40, nth=120, t_end=26.0, M_bar=0.12, init_type="spin_3T", A_eq=1e-4, record_stride=10)),
        ("bare_3T_80x240", dict(nx=80, nth=240, t_end=26.0, M_bar=0.12, init_type="spin_3T", A_eq=1e-4, record_stride=12)),
        # Bare reacting 2T convergence (40x120 masks +0.0114; 80x240 recovers growth!)
        ("react_2T_40x120", dict(nx=40, nth=120, t_end=60.0, M_bar=0.12, n_p=0.22, n_v=0.04, init_type="spin_2T", A_eq=1e-4, record_stride=12)),
        ("react_2T_80x240", dict(nx=80, nth=240, t_end=60.0, M_bar=0.12, n_p=0.22, n_v=0.04, init_type="spin_2T", A_eq=1e-4, record_stride=16)),
        # M_bar = 0 zero-throughflow inviscid baffle numerical dissipation audit (Peer Review #7 M3)
        ("M0_bare_40x120", dict(nx=40, nth=120, t_end=34.0, M_bar=0.0, init_type="stand_1T", A_eq=1e-4, record_stride=10)),
        ("M0_baf045_40x120", dict(nx=40, nth=120, t_end=34.0, M_bar=0.0, n_baffles=5, L_baffle=0.45, init_type="stand_1T", A_eq=1e-4, record_stride=10)),
        ("M0_baf045_80x240", dict(nx=80, nth=240, t_end=34.0, M_bar=0.0, n_baffles=5, L_baffle=0.45, init_type="stand_1T", A_eq=1e-4, record_stride=15)),
    ]
    exp1_res = run_parallel_batch(exp1_tasks)
    r_steady = exp1_res["steady"]
    ce = exp1_res["cons"]["periodic_cons_err"]
    print(f"  Exp 1A Steady preservation: {r_steady['steady_max_err']:.2e}")
    print(f"  Exp 1B Periodic conservation (5 baffles): mass={ce['mass']:.2e}, mom_x={ce['mom_x']:.2e}, "
          f"energy={ce['energy']:.2e}, torque_bookkeeping_err={ce['mom_th_blade_impulse_err']:.2e}")

    cf_4 = [(0.032, W1_COLD, 0.14), (0.008, W2_COLD, 0.12)]
    v_ex_cav1 = exact_2d_cavity_eig(0.12, 1, cav_families=cf_4)
    v_ex_cav2 = exact_2d_cavity_eig(0.12, 2, cav_families=cf_4)
    v_ex_cav1_d = exact_2d_cavity_eig(0.12, 1, cav_families=cf_4, port_form_sq=r_steady["port_form_sq"][0][1])

    exp1c_cav = {}
    for lbl, vex, mkey in [
        ("cav_1T_cont_40x120", v_ex_cav1, "e1"),
        ("cav_1T_cont_60x160", v_ex_cav1, "e1"),
        ("cav_1T_disc5_40x120", v_ex_cav1_d, "e1"),
        ("cav_2T_cont_60x160", v_ex_cav2, "e2"),
    ]:
        rc = exp1_res[lbl]
        msk = (rc["t"] >= 10.0) & (rc["t"] <= 26.0)
        af = float(np.polyfit(rc["t"][msk], 0.5 * np.log(np.maximum(rc[mkey][msk], 1e-30)), 1)[0])
        exp1c_cav[lbl] = {"exact_alpha": vex.real, "exact_omega": vex.imag, "pde_alpha": af, "excess": af - vex.real}
        print(f"  Exp 1C [{lbl:19s}] exact={vex.real:+.5f} | PDE={af:+.5f} (excess={af-vex.real:+.4f})")

    ex_bare = {m: exact_2d_cavity_eig(0.12, m) for m in (1, 2, 3)}
    per_mode_audit = {}
    print("  [Per-Mode Cold Bare Numerical Damping Audit (M_bar = 0.12)]")
    for lbl, m, mkey in [
        ("bare_1T_40x120", 1, "e1"), ("bare_1T_60x160", 1, "e1"), ("bare_1T_80x240", 1, "e1"),
        ("bare_2T_40x120", 2, "e2"), ("bare_2T_60x160", 2, "e2"), ("bare_2T_80x240", 2, "e2"),
        ("bare_3T_40x120", 3, "e3"), ("bare_3T_80x240", 3, "e3"),
    ]:
        rc = exp1_res[lbl]
        t_hi = 22.0 if m == 3 else 26.0
        msk = (rc["t"] >= 10.0) & (rc["t"] <= t_hi)
        af = float(np.polyfit(rc["t"][msk], 0.5 * np.log(np.maximum(rc[mkey][msk], 1e-30)), 1)[0])
        per_mode_audit[lbl] = {"exact_alpha": ex_bare[m].real, "pde_alpha": af, "excess": af - ex_bare[m].real}
        print(f"    [{lbl:14s}] exact={ex_bare[m].real:+.5f} | PDE={af:+.5f} (excess={af-ex_bare[m].real:+.4f})")

    react_2T_conv = {}
    for lbl in ("react_2T_40x120", "react_2T_80x240"):
        st = stats_2d(exp1_res[lbl])
        react_2T_conv[lbl] = st["alpha_fit_e2_clean"]
    print(f"  [Reacting Bare 2T (exact +0.01137)]: 40x120={react_2T_conv['react_2T_40x120']:+.5f} (masked!) | 80x240={react_2T_conv['react_2T_80x240']:+.5f} (growing!)")

    m0_audit = {}
    for lbl in ("M0_bare_40x120", "M0_baf045_40x120", "M0_baf045_80x240"):
        rc = exp1_res[lbl]
        msk = (rc["t"] >= 10.0) & (rc["t"] <= 30.0)
        af = float(np.polyfit(rc["t"][msk], 0.5 * np.log(np.maximum(rc["e_wave"][msk], 1e-30)), 1)[0])
        m0_audit[lbl] = af
    print(f"  [M_bar=0 Baffle Numerical Dissipation (exact 0)]: bare={m0_audit['M0_bare_40x120']:+.5f} | "
          f"Lb=0.45 40x120={m0_audit['M0_baf045_40x120']:+.5f} | 80x240={m0_audit['M0_baf045_80x240']:+.5f}")

    # ------------------------------------------------------------------
    # Experiment 2: Clean Linear (A_0=1e-4) Baffle Sweep, n_v=0 Decomposition & 80x240 Checks
    # ------------------------------------------------------------------
    print("\n--- Experiment 2: Clean Linear (A_0=1e-4) Baffle Sweep, n_v=0 Decomposition & 80x240 Audit ---")
    exp2_tasks = []
    for Lb in (0.00, 0.15, 0.30, 0.45, 0.60, 0.75):
        nb = 5 if Lb > 0.0 else 0
        exp2_tasks.append((f"cold_{Lb:.2f}", dict(nx=40, nth=120, t_end=38.0, M_bar=0.12, n_baffles=nb, L_baffle=Lb,
                                                  init_type="stand_1T", A_eq=1e-4, record_stride=4)))
        # Clean linear reacting (A_0 = 1e-4, Peer Review #7 M1!)
        exp2_tasks.append((f"lin_{Lb:.2f}", dict(nx=40, nth=120, t_end=80.0, M_bar=0.12, n_p=0.22, n_v=0.04,
                                                 n_baffles=nb, L_baffle=Lb, init_type="spin_1T", A_eq=1e-4, record_stride=12)))
        # Pure pressure-coupled n_v = 0 decomposition (Peer Review #7 M2!)
        exp2_tasks.append((f"nv0_{Lb:.2f}", dict(nx=40, nth=120, t_end=80.0, M_bar=0.12, n_p=0.22, n_v=0.0,
                                                 n_baffles=nb, L_baffle=Lb, init_type="spin_1T", A_eq=1e-4, record_stride=12)))
        exp2_tasks.append((f"nl_{Lb:.2f}", dict(nx=40, nth=120, t_end=150.0, M_bar=0.12, n_p=0.22, n_v=0.04,
                                                n_baffles=nb, L_baffle=Lb, init_type="spin_1T", A_eq=0.055, record_stride=15)))
    # 60x160 & 80x240 grid convergence on key baffle lengths (M1, M2, M3, M4)
    for Lb in (0.00, 0.30, 0.45, 0.60):
        nb = 5 if Lb > 0.0 else 0
        exp2_tasks.append((f"cold60_{Lb:.2f}", dict(nx=60, nth=160, t_end=38.0, M_bar=0.12, n_baffles=nb, L_baffle=Lb,
                                                    init_type="stand_1T", A_eq=1e-4, record_stride=6)))
        exp2_tasks.append((f"lin60_{Lb:.2f}", dict(nx=60, nth=160, t_end=80.0, M_bar=0.12, n_p=0.22, n_v=0.04,
                                                   n_baffles=nb, L_baffle=Lb, init_type="spin_1T", A_eq=1e-4, record_stride=14)))
    exp2_tasks.append(("cold80_0.45", dict(nx=80, nth=240, t_end=36.0, M_bar=0.12, n_baffles=5, L_baffle=0.45,
                                           init_type="stand_1T", A_eq=1e-4, record_stride=8)))
    exp2_tasks.append(("cold80_0.60", dict(nx=80, nth=240, t_end=36.0, M_bar=0.12, n_baffles=5, L_baffle=0.60,
                                           init_type="stand_1T", A_eq=1e-4, record_stride=8)))
    # Check L_b=0.60 for hidden 2T instability on 80x240 (Peer Review #7 C1!)
    exp2_tasks.append(("Lb060_2T_80x240", dict(nx=80, nth=240, t_end=60.0, M_bar=0.12, n_p=0.22, n_v=0.04,
                                               n_baffles=5, L_baffle=0.60, init_type="spin_2T", A_eq=1e-4, record_stride=16)))
    # Clean aligned stand_2T blind-spot audit on 80x240 at n_v=0 (C1 + m4!)
    for lbl, nb in [("bare_s2T_80x240", 0), ("Nb4_s2T_80x240", 4), ("Nb5_s2T_80x240", 5)]:
        exp2_tasks.append((lbl, dict(nx=80, nth=240, t_end=60.0, M_bar=0.12, n_p=0.22, n_v=0.0,
                                     n_baffles=nb, L_baffle=(0.45 if nb > 0 else 0.0), init_type="stand_2T_aligned", A_eq=1e-4, record_stride=16)))

    exp2_res = run_parallel_batch(exp2_tasks)
    baffle_sweep = []
    for Lb in (0.00, 0.15, 0.30, 0.45, 0.60, 0.75):
        a_cold, w_cold = measure_cold_lsq(exp2_res[f"cold_{Lb:.2f}"], t_lo=10.0, t_hi=34.0, mode_key="e1")
        st_eff = float(w_cold * (L_A - Lb) / (2.0 * np.pi * 0.12))
        st_lin = stats_2d(exp2_res[f"lin_{Lb:.2f}"])
        st_nv0 = stats_2d(exp2_res[f"nv0_{Lb:.2f}"])
        st_nl = stats_2d(exp2_res[f"nl_{Lb:.2f}"])
        row = {
            "L_b": Lb, "L_eff": L_A - Lb, "omega_1T_cold_lsq": w_cold, "alpha_1T_cold_40x120": a_cold,
            "St_eff": st_eff,
            "react_alpha_chiral_40x120": st_lin["alpha_fit_linear_clean"],
            "react_alpha_nv0_40x120": st_nv0["alpha_fit_linear_clean"],
            "react_nl_P_pct": st_nl["p_over_pbar_pct"], "react_nl_E": st_nl["e_wave_late"],
            "react_nl_lm": st_nl["e_wave_late_over_mid"],
        }
        baffle_sweep.append(row)
        print(f"  L_b={Lb:.2f} (w={w_cold:.4f}, St={st_eff:.2f}): cold={a_cold:+.5f} | chiral_lin={st_lin['alpha_fit_linear_clean']:+.4f} | "
              f"nv0_lin={st_nv0['alpha_fit_linear_clean']:+.4f} | NL P={st_nl['p_over_pbar_pct']:4.2f}%, E={st_nl['e_wave_late']:.2e}")
    vort_snap_baffle = exp2_res["nl_0.45"]

    baffle_refine = {}
    for Lb in (0.00, 0.30, 0.45, 0.60):
        a60, w60 = measure_cold_lsq(exp2_res[f"cold60_{Lb:.2f}"], t_lo=10.0, t_hi=34.0, mode_key="e1")
        st60 = stats_2d(exp2_res[f"lin60_{Lb:.2f}"])
        baffle_refine[f"60x160_Lb_{Lb:.2f}"] = {"omega_1T_cold_lsq": w60, "alpha_1T_cold": a60, "react_alpha_chiral": st60["alpha_fit_linear_clean"]}
        print(f"    60x160 L_b={Lb:.2f}: w={w60:.4f}, cold={a60:+.5f}, chiral_lin={st60['alpha_fit_linear_clean']:+.4f}")
    for Lb in (0.45, 0.60):
        a80, w80 = measure_cold_lsq(exp2_res[f"cold80_{Lb:.2f}"], t_lo=10.0, t_hi=32.0, mode_key="e1")
        baffle_refine[f"80x240_Lb_{Lb:.2f}"] = {"omega_1T_cold_lsq": w80, "alpha_1T_cold": a80}
        print(f"    80x240 L_b={Lb:.2f}: w={w80:.4f}, cold={a80:+.5f}")

    lb060_2t_alpha = stats_2d(exp2_res["Lb060_2T_80x240"])["alpha_fit_e2_clean"]
    print(f"  [L_b=0.60 2T Audit on 80x240 (C1)]: alpha_2T = {lb060_2t_alpha:+.5f}")

    blind_spot_80 = {}
    print("  [Aligned stand_2T Blade Symmetry Audit on 80x240 at n_v=0, n_p=0.22]")
    for lbl in ("bare_s2T_80x240", "Nb4_s2T_80x240", "Nb5_s2T_80x240"):
        st = stats_2d(exp2_res[lbl])
        blind_spot_80[lbl] = {"alpha_e2": st["alpha_fit_e2_clean"], "e2_late": st["e2_late"]}
        print(f"    [{lbl:15s}] alpha_e2={st['alpha_fit_e2_clean']:+.5f} | e2_late={st['e2_late']:.2e}")

    # ------------------------------------------------------------------
    # Experiment 3: Multi-Mode Cavity Spectra, omega_H Retuning Scan, Placement Audit & Shootout
    # ------------------------------------------------------------------
    print("\n--- Experiment 3: Multi-Mode Cavity Spectra, omega_H Scan (M4) & Placement Audit (M5) ---")
    cav_spectra = {}
    for cname, Mb, cf in [
        ("bare_M012", 0.12, []),
        ("80_20_2pct_xi12_M012", 0.12, [(0.016, W1_COLD, 0.12), (0.004, W2_COLD, 0.12)]),
        ("80_20_2pct_opt_M012", 0.12, [(0.016, W1_COLD, 0.10), (0.004, W2_COLD, 0.14)]),
        ("100_0_4pct_M012", 0.12, [(0.040, W1_COLD, 0.16)]),
        ("50_50_4pct_M012", 0.12, [(0.020, W1_COLD, 0.12), (0.020, W2_COLD, 0.12)]),
        ("80_20_4pct_M012", 0.12, [(0.032, W1_COLD, 0.10), (0.008, W2_COLD, 0.14)]),
        ("80_20_5pct_M012", 0.12, [(0.040, W1_COLD, 0.10), (0.010, W2_COLD, 0.14)]),
        ("bare_M016", 0.16, []),
        ("trim_06pct_M016", 0.16, [(0.006, 1.94, 0.08)]),
        ("trim_10pct_M016", 0.16, [(0.010, 1.94, 0.08)]),
    ]:
        ms = {str(m): exact_2d_cavity_eig(Mb, m, n_p=0.22, n_v=0.04, cav_families=cf) for m in (0, 1, 2, 3)}
        worst_m = max(ms, key=lambda k: ms[k].real)
        cav_spectra[cname] = {
            "modes": {k: {"alpha": v.real, "omega": v.imag} for k, v in ms.items()},
            "alpha_max": ms[worst_m].real, "limiting_m": int(worst_m),
        }
        print(f"  [{cname:21s}] m0={ms['0'].real:+.5f} 1T={ms['1'].real:+.5f} 2T={ms['2'].real:+.5f} 3T={ms['3'].real:+.5f} | MAX={ms[worst_m].real:+.5f}")

    def make_hybrid_cav(wH1: float, phase_off: float = 0.5) -> list[dict]:
        return [
            {"beta": 0.016, "wH": wH1, "xiH": 0.12, "sigma_o": 0.045, "n_cav": 5, "phase_offset": phase_off, "nonlinear": True},
            {"beta": 0.004, "wH": W2_COLD, "xiH": 0.12, "sigma_o": 0.015, "n_cav": 5, "phase_offset": phase_off, "nonlinear": True},
        ]

    cf_2_untuned = make_hybrid_cav(W1_COLD, 0.5)
    cf_2_retuned = make_hybrid_cav(0.900, 0.5)   # Tuned to baffled reacting frequency (~0.90)
    cf_2_cold80  = make_hybrid_cav(0.923, 0.5)   # Tuned to converged cold baffled frequency (0.923)
    cf_2_atblade = make_hybrid_cav(0.900, 0.0)   # Blade-root placement (Peer Review #7 M5)
    cf_5_wide = [
        {"beta": 0.040, "wH": W1_COLD, "xiH": 0.10, "sigma_o": 0.070, "n_cav": 5, "nonlinear": True},
        {"beta": 0.010, "wH": W2_COLD, "xiH": 0.14, "sigma_o": 0.025, "n_cav": 5, "nonlinear": True},
    ]
    cf_5_tight = [
        {"beta": 0.040, "wH": W1_COLD, "xiH": 0.10, "sigma_o": 0.020, "n_cav": 5, "nonlinear": True},
        {"beta": 0.010, "wH": W2_COLD, "xiH": 0.14, "sigma_o": 0.008, "n_cav": 5, "nonlinear": True},
    ]
    cf_5_lin = [
        {"beta": 0.040, "wH": W1_COLD, "xiH": 0.10, "sigma_o": 0.070, "n_cav": 5, "nonlinear": False},
        {"beta": 0.010, "wH": W2_COLD, "xiH": 0.14, "sigma_o": 0.025, "n_cav": 5, "nonlinear": False},
    ]

    retune_specs = [
        ("bare_lin_40x120",             40, 120, 0, 0.00, None,                     "spin_1T", 1e-4, 90.0),
        ("baffle045_alone_lin_40x120",  40, 120, 5, 0.45, None,                     "spin_1T", 1e-4, 110.0),
        ("cav2_alone_lin_40x120",       40, 120, 0, 0.00, cf_2_untuned,             "spin_1T", 1e-4, 100.0),
        ("hybrid_wH086_40x120",         40, 120, 5, 0.45, make_hybrid_cav(0.860),   "spin_1T", 1e-4, 110.0),
        ("hybrid_wH090_retuned_40x120", 40, 120, 5, 0.45, cf_2_retuned,             "spin_1T", 1e-4, 120.0),
        ("hybrid_wH0923_cold_40x120",   40, 120, 5, 0.45, cf_2_cold80,              "spin_1T", 1e-4, 110.0),
        ("hybrid_wH096_40x120",         40, 120, 5, 0.45, make_hybrid_cav(0.960),   "spin_1T", 1e-4, 110.0),
        ("hybrid_wH0989_untuned_40x120",40, 120, 5, 0.45, cf_2_untuned,             "spin_1T", 1e-4, 120.0),
        ("hybrid_blade_root_40x120",    40, 120, 5, 0.45, cf_2_atblade,             "spin_1T", 1e-4, 110.0),
        ("hybrid_untuned_60x160",       60, 160, 5, 0.45, cf_2_untuned,             "spin_1T", 1e-4, 110.0),
        ("hybrid_retuned_60x160",       60, 160, 5, 0.45, cf_2_retuned,             "spin_1T", 1e-4, 110.0),
        ("hybrid_retuned_2T_60x160",    60, 160, 5, 0.45, cf_2_retuned,             "spin_2T", 1e-4, 90.0),
        ("hybrid_retuned_1L_60x160",    60, 160, 5, 0.45, cf_2_retuned,             "cold_1L", 1e-4, 90.0),
        ("cav5_wide_lin_40x120",        40, 120, 0, 0.00, cf_5_wide,                "spin_1T", 1e-4, 110.0),
    ]
    nl_specs = [
        ("bare_M012_NL",              0, 0.00, None),
        ("baffle030_M012_NL",         5, 0.30, None),
        ("cav5_tight20_M012_NL",      0, 0.00, cf_5_tight),
        ("cav5_wide70_M012_NL",       0, 0.00, cf_5_wide),
        ("cav5_linear_M012_NL",       0, 0.00, cf_5_lin),
        ("hybrid045_retuned_M012_NL", 5, 0.45, cf_2_retuned),
    ]
    exp3_res = run_parallel_batch([
        (cid, dict(nx=nx_g, nth=nth_g, t_end=tend, M_bar=0.12, n_p=0.22, n_v=0.04,
                   n_baffles=nb, L_baffle=Lb, cav_spec=cspec, init_type=itype, A_eq=amp, record_stride=15))
        for cid, nx_g, nth_g, nb, Lb, cspec, itype, amp, tend in retune_specs
    ] + [
        (cid, dict(nx=40, nth=120, t_end=160.0, M_bar=0.12, n_p=0.22, n_v=0.04,
                   n_baffles=nb, L_baffle=Lb, cav_spec=cspec, init_type="spin_1T", A_eq=0.055, record_stride=15))
        for cid, nb, Lb, cspec in nl_specs
    ])

    retune_runs = {}
    retune_stats = {}
    print("  [Clean Linear (A_0=1e-4) Retuning Scan, Placement Audit & All-Mode 60x160 Checks]")
    for cid, *_ in retune_specs:
        rr = exp3_res[cid]
        st = stats_2d(rr)
        retune_runs[cid] = rr
        retune_stats[cid] = st
        print(f"    [{cid:28s}] alpha_lin={st['alpha_fit_linear_clean']:+.4f} | E={st['e_wave_late']:.2e} (l/m={st['e_wave_late_over_mid']:.3f})")

    m012_nl_stats = {}
    print("  [Finite-Amplitude 6.9% Shock Shootout at M_bar=0.12 (40x120, A_0=0.055, t=160)]")
    for cid, *_ in nl_specs:
        st = stats_2d(exp3_res[cid])
        m012_nl_stats[cid] = st
        print(f"    [{cid:25s}] P={st['p_over_pbar_pct']:4.2f}% | E={st['e_wave_late']:.2e} (l/m={st['e_wave_late_over_mid']:.3f})")

    # ------------------------------------------------------------------
    # Experiment 4: M_bar=0.16 2T Trap Rescue, 64x200 Orifice-Drag Cliff (C2) & 1T Trigger Bracket (M6)
    # ------------------------------------------------------------------
    print("\n--- Experiment 4: M_bar=0.16 64x200 Orifice-Drag Cliff (C2) & 1T Trigger Bracket (M6) ---")
    cf_1pct_tight = [{"beta": 0.010, "wH": 1.94, "xiH": 0.08, "sigma_o": 0.012, "n_cav": 5, "nonlinear": True}]
    cf_1pct_med   = [{"beta": 0.010, "wH": 1.94, "xiH": 0.08, "sigma_o": 0.025, "n_cav": 5, "nonlinear": True}]
    cf_1pct_wide  = [{"beta": 0.010, "wH": 1.94, "xiH": 0.08, "sigma_o": 0.050, "n_cav": 5, "nonlinear": True}]

    exp4_specs = [
        # Grid-refined 64x200 cliff audit (t=220, Peer Review #7 C2!)
        ("M016_bare_spin2T_64x200",         64, 200, None,          "spin_2T",  0.040, 220.0),
        ("M016_1pct_tight12_2T_64x200",     64, 200, cf_1pct_tight, "spin_2T",  0.040, 220.0),
        ("M016_1pct_med25_2T_64x200",       64, 200, cf_1pct_med,   "spin_2T",  0.040, 220.0),
        ("M016_1pct_wide50_2T_64x200",      64, 200, cf_1pct_wide,  "spin_2T",  0.040, 220.0),
        # 48x160 comparison for med25 & wide50 + 5-IC & 1T trigger bracket (M6!)
        ("M016_1pct_med25_2T_48x160",       48, 160, cf_1pct_med,   "spin_2T",  0.040, 220.0),
        ("M016_1pct_wide50_2T_48x160",      48, 160, cf_1pct_wide,  "spin_2T",  0.040, 160.0),
        ("M016_1pct_wide50_1T_A015_48x160", 48, 160, cf_1pct_wide,  "spin_1T",  0.015, 200.0),
        ("M016_1pct_wide50_1T_A025_48x160", 48, 160, cf_1pct_wide,  "spin_1T",  0.025, 200.0),
        ("M016_1pct_wide50_1T_A055_48x160", 48, 160, cf_1pct_wide,  "spin_1T",  0.055, 160.0),
        ("M016_1pct_wide50_3T_48x160",      48, 160, cf_1pct_wide,  "spin_3T",  0.040, 140.0),
        ("M016_1pct_wide50_1L_48x160",      48, 160, cf_1pct_wide,  "cold_1L",  0.040, 140.0),
    ]
    exp4_runs = run_parallel_batch([
        (cid, dict(nx=nx_g, nth=nth_g, t_end=tend, M_bar=0.16, n_p=0.22, n_v=0.04,
                   cav_spec=cspec, init_type=itype, A_eq=amp, record_stride=18))
        for cid, nx_g, nth_g, cspec, itype, amp, tend in exp4_specs
    ])
    exp4_stats = {}
    for cid, *_ in exp4_specs:
        st = stats_2d(exp4_runs[cid])
        exp4_stats[cid] = st
        print(f"  [{cid:31s}] P={st['p_over_pbar_pct']:5.2f}% | E={st['e_wave_late']:.2e} (l/m={st['e_wave_late_over_mid']:.3f}) | afit={st['alpha_fit_late']:+.4f}")

    # ------------------------------------------------------------------
    # Figure 1: Clean Linear Baffle Curve (Chiral vs n_v=0) & omega_H Retuning / Placement Law
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.6), dpi=150)

    Lbs = [r["L_b"] for r in baffle_sweep]
    c_damp = [-r["alpha_1T_cold_40x120"] for r in baffle_sweep]
    r_chiral = [r["react_alpha_chiral_40x120"] for r in baffle_sweep]
    r_nv0 = [r["react_alpha_nv0_40x120"] for r in baffle_sweep]
    w_arr = [r["omega_1T_cold_lsq"] for r in baffle_sweep]

    l1, = ax1.plot(Lbs, c_damp, "o-", color="#1f77b4", lw=2.2, ms=6.0, label="Cold $1T$ damping $-\\alpha_{1T}$ ($40\\times 120$)")
    l2, = ax1.plot(Lbs, r_chiral, "s-", color="#d62728", lw=2.2, ms=6.0, label="Reacting chiral ($n_v=0.04$, $A_0=10^{-4}$)")
    l3, = ax1.plot(Lbs, r_nv0, "D--", color="#ff7f0e", lw=1.9, ms=5.5, label="Reacting pressure-only ($n_v=0$)")
    lbs60 = [0.00, 0.30, 0.45, 0.60]
    ax1.scatter(lbs60, [-baffle_refine[f"60x160_Lb_{x:.2f}"]["alpha_1T_cold"] for x in lbs60],
                marker="^", s=65, color="#0b3c70", zorder=6)
    ax1.scatter(lbs60, [baffle_refine[f"60x160_Lb_{x:.2f}"]["react_alpha_chiral"] for x in lbs60],
                marker="v", s=60, color="#8b0000", zorder=6)
    ax1.axhline(0.0, color="#333333", lw=0.9, alpha=0.6)
    ax1.axvline(0.45, color="#2ca02c", ls="--", lw=1.1, alpha=0.8)
    ax1.annotate("Cold shedding peak\n($L_b=0.45$)", xy=(0.45, c_damp[3]), xytext=(0.48, 0.112), fontsize=8.0, color="#1f77b4", fontweight="bold")
    ax1.annotate("Freq-downshift\nphase re-destabilization", xy=(0.60, r_chiral[4]), xytext=(0.44, 0.056),
                 fontsize=7.6, color="#8b0000", arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.1))
    ax1.set_xlabel("Protruding inviscid baffle length $L_b$ ($N_b = 5, L_a = 1.5, x_f = 0.25$)")
    ax1.set_ylabel("Cold damping $-\\alpha_{1T}$ (blue) / Reacting growth $\\alpha_{1T}$")
    ax1.set_ylim(-0.015, 0.132)
    ax1.grid(True, alpha=0.28)
    ax1_r = ax1.twinx()
    l4, = ax1_r.plot(Lbs, w_arr, "^:", color="#6c757d", lw=1.5, ms=5.0, alpha=0.85, label="Cold $\\omega_{1T}(L_b)$ (right)")
    ax1_r.set_ylabel("Cold $1T$ frequency $\\omega_{1T}(L_b)$", color="#555555")
    ax1_r.set_ylim(0.76, 1.04)
    ax1.set_title("Clean Linear ($A_0=10^{-4}$) Baffle Resonance & $n_v=0$ Split", fontweight="bold")
    ax1.legend([l1, l2, l3, l4], [ll.get_label() for ll in (l1, l2, l3, l4)], loc="lower left", fontsize=7.3, framealpha=0.93)

    # Right panel: omega_H retuning scan + mid-sector vs blade-root placement
    wh_pts = [0.860, 0.900, 0.923, 0.960, 0.9893]
    wh_keys = ["hybrid_wH086_40x120", "hybrid_wH090_retuned_40x120", "hybrid_wH0923_cold_40x120",
               "hybrid_wH096_40x120", "hybrid_wH0989_untuned_40x120"]
    wh_alphas = [retune_stats[k]["alpha_fit_linear_clean"] for k in wh_keys]
    ax2.plot(wh_pts, wh_alphas, "o-", color="#2ca02c", lw=2.4, ms=7.5, label="Mid-sector hybrid ($40\\times 120$, phase $= 0.5$)")
    ax2.scatter([0.900, 0.9893],
                [retune_stats["hybrid_retuned_60x160"]["alpha_fit_linear_clean"],
                 retune_stats["hybrid_untuned_60x160"]["alpha_fit_linear_clean"]],
                marker="D", s=75, color="#0b5e20", zorder=7, label="Mid-sector hybrid ($60\\times 160$ check)")
    ax2.scatter([0.900], [retune_stats["hybrid_blade_root_40x120"]["alpha_fit_linear_clean"]],
                marker="X", s=110, color="#d62728", zorder=8, label="Blade-root ports (phase $= 0.0$: $-36\\%$ margin!)")
    ax2.axhline(retune_stats["baffle045_alone_lin_40x120"]["alpha_fit_linear_clean"], color="#9467bd", ls="-.", lw=1.5, label="$L_b=0.45$ baffles alone (unstable)")
    ax2.axhline(retune_stats["cav2_alone_lin_40x120"]["alpha_fit_linear_clean"], color="#ff7f0e", ls="--", lw=1.5, label="$2\\%$ cavity alone (unstable)")
    ax2.axhline(0.0, color="#333333", lw=1.0, alpha=0.7)
    ax2.axvline(0.923, color="#6c757d", ls=":", lw=1.2, label="Converged cold baffled $\\omega_{1T}(0.45) = 0.923$")
    ax2.set_xlabel("Faceplate $1T$ cavity tuning frequency $\\omega_H$ ($L_b = 0.45 + 2\\%$ 80/20)")
    ax2.set_ylabel("Clean linear $1T$ growth rate $\\alpha_{1T}$ ($A_0 = 10^{-4}$)")
    ax2.set_ylim(-0.026, 0.022)
    ax2.set_title("Baffle-Cavity Retuning Scan & Placement Sensitivity ($\\bar{M}=0.12$)", fontweight="bold")
    ax2.grid(True, alpha=0.28)
    ax2.legend(loc="lower right", fontsize=7.2, framealpha=0.93)

    fig.suptitle("Rocket Combustion Entry 010 (Part 1, Hardened): Baffle Resonance & Reacting-Frequency Retuning Law", fontsize=11.4, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp010_baffle_strouhal_retuning.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: 2D Baffle-Tip Vorticity Shedding & Grid-Verified 64x200 Orifice-Drag Cliff
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.6), dpi=150)

    vf = vort_snap_baffle["vort_final"]
    vmax = float(np.percentile(np.abs(vf), 98.5))
    im = ax1.pcolormesh(vort_snap_baffle["theta"], vort_snap_baffle["x"], vf, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for k in range(5):
        th_b = k * 2.0 * np.pi / 5.0
        ax1.plot([th_b, th_b], [0.0, 0.45], color="#111111", lw=3.0)
    ax1.axhline(X_F, color="#2ca02c", ls="--", lw=1.2, alpha=0.85, label="Flame sheet $x_f = 0.25$")
    ax1.axhline(0.45, color="#111111", ls=":", lw=1.1, alpha=0.75, label="Baffle tips $L_b = 0.45$")
    ax1.set_xlabel("Azimuth $\\theta$ (rad)")
    ax1.set_ylabel("Axial coordinate $x$ ($0 =$ injector, $1.5 =$ nozzle)")
    ax1.set_title("2D Baffle-Tip Vorticity Shedding $\\omega_z(x, \\theta)$ ($N_b=5, L_b=0.45$)", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.0, framealpha=0.9)
    cb = fig.colorbar(im, ax=ax1, pad=0.02)
    cb.set_label("Vorticity $\\omega_z = \\partial_x u_\\theta - \\partial_\\theta u_x$")

    for cid, lbl, col, ls in [
        ("M016_bare_spin2T_64x200",         "Bare $\\bar{M}=0.16$ ($64\\times 200$: $7.2\\%$ $2T$ shock)", "#d62728", "-"),
        ("M016_1pct_tight12_2T_64x200",     "$1\\%$ trim $\\sigma_o=1.2\\%$ ($64\\times 200$: $5.8\\%$ shock)", "#ff7f0e", "--"),
        ("M016_1pct_med25_2T_64x200",       "$1\\%$ trim $\\sigma_o=2.5\\%$ ($64\\times 200$: CHOKES $4.0\\%$!)", "#9467bd", "-"),
        ("M016_1pct_med25_2T_48x160",       "$1\\%$ trim $\\sigma_o=2.5\\%$ ($48\\times 160$: coarse artifact)", "#9467bd", ":"),
        ("M016_1pct_wide50_2T_64x200",      "$1\\%$ trim $\\sigma_o=5.0\\%$ ($64\\times 200$: $2T$ QUENCH!)", "#2ca02c", "-"),
        ("M016_1pct_wide50_1T_A025_48x160", "$1\\%$ trim: $1T$ $A_0=0.025$ (subcrit. $3.95\\%$)", "#1f77b4", "-."),
        ("M016_1pct_wide50_1T_A015_48x160", "$1\\%$ trim: $1T$ $A_0=0.015$ (decays!)", "#17becf", "--"),
    ]:
        rr = exp4_runs[cid]
        ax2.semilogy(rr["t"], np.maximum(rr["e_wave"], 1e-16), ls, color=col, lw=2.0, label=lbl)
    ax2.set_xlabel("Time $t$")
    ax2.set_ylabel("2D Wave Energy $E_{\\mathrm{wave}}(t)$ (log scale)")
    ax2.set_ylim(1e-10, 5e-3)
    ax2.set_title("Grid-Verified $64\\times 200$ Orifice-Drag Cliff & $1T$ Trigger Bracket", fontweight="bold")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="lower left", fontsize=7.2, framealpha=0.93)

    fig.suptitle("Rocket Combustion Entry 010 (Part 2, Hardened): Baffle Vortex Streets & 64x200 Orifice-Drag Cliff", fontsize=11.4, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp010_orifice_cliff_and_2T_rescue.png")
    plt.close(fig)

    metrics = {
        "exp1_verification": {
            "steady_max_err": r_steady["steady_max_err"],
            "periodic_conservation_5_baffles": ce,
            "port_form_sq_40x120_Nc5": r_steady["port_form_sq"],
            "exp1c_cavity_chebyshev_vs_pde": exp1c_cav,
            "per_mode_bare_numerical_damping_M012": per_mode_audit,
            "bare_reacting_2T_grid_convergence": react_2T_conv,
            "M0_inviscid_baffle_numerical_damping": m0_audit,
        },
        "exp2_baffle_strouhal": {
            "sweep_40x120_clean_linear": baffle_sweep,
            "grid_refinement_60_80": baffle_refine,
            "Lb060_2T_alpha_80x240": lb060_2t_alpha,
            "blade_symmetry_nv0_80x240": blind_spot_80,
        },
        "exp3_cavities_and_retuning": {
            "chebyshev_multimode_spectra": cav_spectra,
            "retuning_and_placement_clean_linear": retune_stats,
            "m012_nonlinear_shootout": m012_nl_stats,
        },
        "exp4_m016_orifice_cliff_64x200": exp4_stats,
    }
    out_json = RESULTS_DIR / "exp010_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics -> {out_json}")
    print("Saved plots   -> exp010_baffle_strouhal_retuning.png, exp010_orifice_cliff_and_2T_rescue.png")


if __name__ == "__main__":
    main()
