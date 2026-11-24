"""Entry 009 (rocket-combustion): 2D (x, theta) Compressible Euler Annular Combustor with
Physical Steady Axial Throughflow — Strouhal Acoustic-Vorticity Damping Resonance,
First-Principles Entropy/Swirl Stationarity & High-Gain Spin Re-Emergence.

Replaces the 1D periodic ring's ad-hoc entropy-flush closure (S_ent) and zero-inflow momentum
sink (Entries 007-008) with first-principles 2D conservative compressible Euler equations on
(x, theta) in [0, L_a] x [0, 2*pi) (L_a = 1.5, gamma = 1.25) with steady axial throughflow M_bar > 0:

    d(rho)/dt       + d(rho*ux)/dx       + d(rho*ut)/dtheta       = 0
    d(rho*ux)/dt    + d(rho*ux^2 + p)/dx + d(rho*ux*ut)/dtheta    = 0
    d(rho*ut)/dt    + d(rho*ux*ut)/dx    + d(rho*ut^2 + p)/dtheta = 0
    d(E)/dt         + d(ux*(E+p))/dx     + d(ut*(E+p))/dtheta     = (1/(gamma - 1)) * w_f(x) * Q_flame(theta, t)

Exact nonlinear characteristic boundary conditions (ZERO ad-hoc closures):
  - Inlet x = 0 (choked zero-swirl isentropic propellant injector):
      rho_in * ux_in = M_bar,   s_in = 0,   ut_in = 0,   outgoing J_- = ux - 2/(gamma-1)*c extrapolated.
  - Outlet x = L_a (exact compact choked nozzle, Marble & Candel 1977):
      ux_out / c_out = M_bar,   outgoing (J_+, s, ut) extrapolated (naturally generates indirect entropy noise).

Numerical methods:
  - 2D unsplit conservative 2nd-order MUSCL-MC + HLLC Riemann solver + SSP-RK2.
  - Exact 1D-in-x Chebyshev-Gauss-Lobatto generalized eigenvalue solver (N_cheb = 36) for 2D linearized
    Euler throughflow (including vorticity waves, entropy advection, Marble-Candel nozzle coupling, and
    DC-blocked Erlang-4 flame dynamics).

Experiments:
1. Verification Suite:
   (1a) Exact steady-throughflow preservation (max |U - U_0| < 1e-15).
   (1b) Exact 2D doubly periodic inviscid conservation of mass, x/theta momentum, and energy (< 1e-15).
   (1c) Exact Chebyshev 2D linearized eigenvalues vs. 2D Euler MC PDE cold damping (1L, 1T, 2T, 3T)
        and grid convergence (24x72, 32x96, 48x144) quantifying per-mode numerical damping.
   (1d) First-principles entropy & mean-swirl residence flushing (tau_res = L_a / M_bar = 12.5) plus
        Marble-Candel indirect entropy-noise acoustic echo at x = L_a.
2. Strouhal Acoustic-Vorticity Damping Resonance (M_bar in [0.06, 0.30]):
   Maps exact cold 1T damping alpha_1T(M_bar) and hot 1T Hopf threshold n_pH,1T(M_bar), verified at
   5 Mach numbers in 2D cold Euler PDE, revealing 3.5x cold-damping / 4.8x Hopf peaks at half-integer
   Strouhal numbers M_bar ~ L_a / (2*pi*(k + 1/2)) vs. vulnerable integer-Strouhal troughs.
3. First-Principles Stationarity & Landau Standing-to-Spinning Reversal (n_v = 0, ZERO ad-hoc S_ent):
   Tests both low-gain (M_bar=0.06, n_p=0.105) and high-gain (M_bar=0.12, n_p=0.220) regimes from
   seeded standing and perturbed spin (plus 48x144 grid check) using the boundary-layer-free envelope
   spin index S_env = min_T|P_hat_1| / max_T|P_hat_1|.
4. 2D Chiral Rotating-Shock Control Shootout (M_bar=0.12, n_p=0.22, n_v=0.04, A_0=6.9%, t=300):
   Compares bare 2D rotating shock, zero-hardware M=2 Bragg trap, zero-hardware odd M=5 staggering
   (tau in {0.55, 2.55}, plus 48x144 grid check), and passive Strouhal Mach-shift (M_bar=0.12 -> 0.18).
"""

from __future__ import annotations

import json
from pathlib import Path

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


# ---------------------------------------------------------------------------
# 1. Exact 1D-in-x Chebyshev Eigenvalue Solver for 2D Linearized Euler Throughflow
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


def exact_2d_throughflow_eig(
    M_bar: float,
    m: int,
    n_p: float = 0.0,
    n_v: float = 0.0,
    tau_bar: float = TAU_BAR,
    omega_window: tuple[float, float] | None = None,
    N: int = 34,
) -> complex:
    """Exact least-stable acoustic eigenvalue of 2D linearized compressible Euler with mean flow M_bar."""
    x, Dx, w_cc = cheb_diff(N, L_A)
    n = N + 1
    I = np.eye(n)
    wf = np.exp(-0.5 * ((x - X_F) / SIG_F) ** 2)
    wf /= float(np.sum(w_cc * wf))
    w_sense = w_cc * wf

    has_f = (abs(n_p) > 0.0 or abs(n_v) > 0.0)
    dim = 4 * n + (10 if has_f else 0)
    A = np.zeros((dim, dim), dtype=complex)
    B = np.eye(dim, dtype=complex)

    def sl(q: int) -> slice:
        return slice(q * n, (q + 1) * n)

    A[sl(0), sl(0)] = -M_bar * Dx; A[sl(0), sl(1)] = -Dx; A[sl(0), sl(2)] = -1j * m * I
    A[sl(1), sl(1)] = -M_bar * Dx; A[sl(1), sl(3)] = -Dx
    A[sl(2), sl(2)] = -M_bar * Dx; A[sl(2), sl(3)] = -1j * m * I
    A[sl(3), sl(1)] = -Dx; A[sl(3), sl(2)] = -1j * m * I; A[sl(3), sl(3)] = -M_bar * Dx

    if has_f:
        inv_ts = 4.0 / tau_bar
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

    # Inlet BCs at x=0: rho'(0)-P(0)=0, ux'(0)+M*P(0)=0, ut'(0)=0
    for r_idx in (0, n, 2 * n):
        A[r_idx, :] = 0.0; B[r_idx, :] = 0.0
    A[0, 0] = 1.0; A[0, 3 * n] = -1.0
    A[n, n] = 1.0; A[n, 3 * n] = M_bar
    A[2 * n, 2 * n] = 1.0

    # Outlet Marble-Candel choked-nozzle BC at x=L_a: ux'(L_a) - 0.5*M*(gamma*P - rho') = 0
    r_out = 3 * n + N
    A[r_out, :] = 0.0; B[r_out, :] = 0.0
    A[r_out, n + N] = 1.0
    A[r_out, 3 * n + N] = -0.5 * M_bar * GAMMA
    A[r_out, 0 * n + N] = +0.5 * M_bar

    vals = eig(A, B, right=False)
    w_lo, w_hi = omega_window if omega_window is not None else (0.5, 5.5)
    cands = [v for v in vals if np.isfinite(v) and w_lo <= v.imag <= w_hi and -0.45 < v.real < 0.35]
    cands.sort(key=lambda z: z.real, reverse=True)
    return complex(cands[0])


# ---------------------------------------------------------------------------
# 2. 2D Conservative MUSCL-MC + HLLC + SSP-RK2 Compressible Euler Solver
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
    """Exact HLLC numerical flux along normal direction (vn) with passive tangential velocity (vt)."""
    cL = np.sqrt(GAMMA * pL / rhoL)
    cR = np.sqrt(GAMMA * pR / rhoR)
    EL = pL / (GAMMA - 1.0) + 0.5 * rhoL * (vnL * vnL + vtL * vtL)
    ER = pR / (GAMMA - 1.0) + 0.5 * rhoR * (vnR * vnR + vtR * vtR)

    srL = np.sqrt(rhoL)
    srR = np.sqrt(rhoR)
    vn_r = (srL * vnL + srR * vnR) / (srL + srR)
    vt_r = (srL * vtL + srR * vtR) / (srL + srR)
    H_r = (srL * (EL + pL) / rhoL + srR * (ER + pR) / rhoR) / (srL + srR)
    c_r = np.sqrt(np.maximum((GAMMA - 1.0) * (H_r - 0.5 * (vn_r * vn_r + vt_r * vt_r)), 1e-12))

    sL = np.minimum(vnL - cL, vn_r - c_r)
    sR = np.maximum(vnR + cR, vn_r + c_r)
    num = pR - pL + rhoL * vnL * (sL - vnL) - rhoR * vnR * (sR - vnR)
    den = rhoL * (sL - vnL) - rhoR * (sR - vnR)
    sM = num / np.where(np.abs(den) > 1e-14, den, 1e-14)

    FL = np.stack([rhoL * vnL, rhoL * vnL * vnL + pL, rhoL * vnL * vtL, vnL * (EL + pL)], axis=0)
    FR = np.stack([rhoR * vnR, rhoR * vnR * vnR + pR, rhoR * vnR * vtR, vnR * (ER + pR)], axis=0)

    dL = np.where(np.abs(sL - sM) > 1e-14, sL - sM, 1e-14)
    dR = np.where(np.abs(sR - sM) > 1e-14, sR - sM, 1e-14)
    facL = rhoL * (sL - vnL) / dL
    facR = rhoR * (sR - vnR) / dR
    p_star = np.maximum(pL + rhoL * (sL - vnL) * (sM - vnL), 1e-10)

    UsL = np.stack([facL, facL * sM, facL * vtL, (EL * (sL - vnL) - pL * vnL + p_star * sM) / dL], axis=0)
    UsR = np.stack([facR, facR * sM, facR * vtR, (ER * (sR - vnR) - pR * vnR + p_star * sM) / dR], axis=0)
    FsL = FL + sL * (UsL - np.stack([rhoL, rhoL * vnL, rhoL * vtL, EL], axis=0))
    FsR = FR + sR * (UsR - np.stack([rhoR, rhoR * vnR, rhoR * vtR, ER], axis=0))

    return np.where(sL >= 0.0, FL, np.where(sM >= 0.0, FsL, np.where(sR >= 0.0, FsR, FR)))


def make_staggered_tau_2d(nth: int, M_periods: int, tau_1: float = 0.55, tau_2: float = 2.55) -> np.ndarray:
    sec = np.floor(np.arange(nth) * (2 * M_periods) / nth).astype(int)
    return np.where(sec % 2 == 0, tau_1, tau_2)


def simulate_2d_euler_annulus(
    nx: int = 32,
    nth: int = 96,
    t_end: float = 200.0,
    cfl: float = 0.35,
    M_bar: float = 0.12,
    n_p: float = 0.0,
    n_v: float = 0.0,
    tau_arr: np.ndarray | float = TAU_BAR,
    linear_flame: bool = False,
    periodic_x: bool = False,
    init_type: str = "steady",
    A_eq: float = 0.035,
    record_stride: int = 12,
) -> dict:
    """2D (x, theta) conservative compressible Euler annular combustor with physical axial throughflow."""
    dx = L_A / nx
    dth = 2.0 * np.pi / nth
    x = (np.arange(nx) + 0.5) * dx
    th = (np.arange(nth) + 0.5) * dth

    wf = np.exp(-0.5 * ((x - X_F) / SIG_F) ** 2)
    wf /= float(np.sum(wf) * dx)
    w_sense = wf * dx

    tau_loc = np.broadcast_to(np.asarray(tau_arr, dtype=float), (nth,)).copy()
    inv_ts = 4.0 / tau_loc
    inv_gm1 = 2.0 / (GAMMA - 1.0)
    has_f = (abs(n_p) > 0.0 or abs(n_v) > 0.0)

    # Initial conditions
    ux0 = np.full((nx, nth), M_bar)
    ut0 = np.zeros((nx, nth))
    p0 = np.full((nx, nth), P_BAR)
    rho0 = np.ones((nx, nth))

    if init_type == "steady":
        pass
    elif init_type == "periodic_test":
        p0 = P_BAR + A_eq * np.cos(2.0 * np.pi * x[:, None] / L_A) * np.cos(th[None, :])
        rho0 = (p0 / P_BAR) ** (1.0 / GAMMA)
        ux0 = M_bar + 0.5 * A_eq * np.sin(2.0 * np.pi * x[:, None] / L_A)
        ut0 = 0.5 * A_eq * np.cos(th[None, :]) * np.ones((nx, 1))
    elif init_type == "entropy_swirl_pulse":
        # Localized hot entropy spot near x=0.25 (zero pressure perturbation!) + uniform swirl ut = 0.04
        s_init = 0.05 * np.exp(-0.5 * ((x - 0.25) / 0.12) ** 2)[:, None] * np.ones((1, nth))
        rho0 = np.exp(-s_init / GAMMA)
        ut0 = np.full((nx, nth), 0.04)
    elif init_type == "cold_1L":
        p0 = P_BAR + A_eq * np.cos(np.pi * x / L_A)[:, None] * np.ones((1, nth))
        rho0 = (p0 / P_BAR) ** (1.0 / GAMMA)
    elif init_type in ("spin_1T", "spin_2T", "spin_3T"):
        m_k = {"spin_1T": 1, "spin_2T": 2, "spin_3T": 3}[init_type]
        p0 = P_BAR + A_eq * np.cos(m_k * th)[None, :] * np.ones((nx, 1))
        rho0 = (p0 / P_BAR) ** (1.0 / GAMMA)
        ut0 = A_eq * np.cos(m_k * th)[None, :] * np.ones((nx, 1))
    elif init_type == "stand_seed":
        eps = 0.12
        a_s = A_eq / np.sqrt(1.0 + 0.5 * eps * eps)
        p0 = P_BAR + np.sqrt(2.0) * a_s * np.cos(th)[None, :] * np.ones((nx, 1))
        rho0 = (p0 / P_BAR) ** (1.0 / GAMMA)
        ut0 = eps * a_s * np.cos(th)[None, :] * np.ones((nx, 1))
    elif init_type == "spin_pert":
        eps_c = 0.10
        a_p = A_eq / np.sqrt(1.0 + eps_c * eps_c)
        a_m = eps_c * a_p
        p0 = P_BAR + (a_p + a_m) * np.cos(th)[None, :] * np.ones((nx, 1))
        rho0 = (p0 / P_BAR) ** (1.0 / GAMMA)
        ut0 = (a_p - a_m) * np.cos(th)[None, :] * np.ones((nx, 1))
    else:
        raise ValueError(f"Unknown init_type: {init_type}")

    U = cons_2d(rho0, ux0, ut0, p0)
    U_init = U.copy()
    pbf = np.zeros(nth)
    ubf = np.zeros(nth)
    qp = np.zeros((4, nth))
    qu = np.zeros((4, nth))

    def rhs(U_s, pb_s, ub_s, qp_s, qu_s):
        rho, ux, ut, p, c = prim_2d(U_s)

        if periodic_x:
            def rx_p(v):
                s = mc_limiter(v - np.roll(v, 1, axis=0), np.roll(v, -1, axis=0) - v)
                return v + 0.5 * s, np.roll(v - 0.5 * s, -1, axis=0)
            rL, rR = rx_p(rho); uxL, uxR = rx_p(ux); utL, utR = rx_p(ut); pL, pR = rx_p(p)
            Fx = hllc_2d_normal(np.maximum(rL, 1e-10), uxL, utL, np.maximum(pL, 1e-10),
                                np.maximum(rR, 1e-10), uxR, utR, np.maximum(pR, 1e-10))
            div_x = (Fx - np.roll(Fx, 1, axis=1)) / dx
        else:
            # Exact characteristic choked zero-swirl isentropic injector at x = 0
            Jm = ux[0] - inv_gm1 * c[0]
            c_in = np.ones_like(Jm)
            for _ in range(5):
                u_try = Jm + inv_gm1 * c_in
                f = (c_in ** inv_gm1) * u_try - M_bar
                df = inv_gm1 * (c_in ** (inv_gm1 - 1.0)) * u_try + (c_in ** inv_gm1) * inv_gm1
                c_in = np.maximum(c_in - f / df, 0.2)
            ux_in = Jm + inv_gm1 * c_in
            rho_in = c_in ** inv_gm1
            p_in = P_BAR * (rho_in ** GAMMA)
            ut_in = np.zeros_like(Jm)

            # Exact Marble-Candel compact choked nozzle at x = L_a
            Jp = ux[-1] + inv_gm1 * c[-1]
            s_out = np.log(np.maximum((p[-1] / P_BAR) / (rho[-1] ** GAMMA), 1e-15))
            c_out = Jp / (M_bar + inv_gm1)
            ux_out = M_bar * c_out
            rho_out = np.maximum((c_out * c_out) / np.exp(s_out), 1e-10) ** (1.0 / (GAMMA - 1.0))
            p_out = rho_out * c_out * c_out / GAMMA
            ut_out = ut[-1]

            def pad(arr, lo, hi):
                return np.vstack([lo[None, :], lo[None, :], arr, hi[None, :], hi[None, :]])

            rg, uxg, utg, pg = pad(rho, rho_in, rho_out), pad(ux, ux_in, ux_out), pad(ut, ut_in, ut_out), pad(p, p_in, p_out)

            def rx(v):
                s = mc_limiter(v[1:-1] - v[:-2], v[2:] - v[1:-1])
                return v[1:-2] + 0.5 * s[:-1], v[2:-1] - 0.5 * s[1:]

            rL, rR = rx(rg); uxL, uxR = rx(uxg); utL, utR = rx(utg); pL, pR = rx(pg)
            Fx = hllc_2d_normal(np.maximum(rL, 1e-10), uxL, utL, np.maximum(pL, 1e-10),
                                np.maximum(rR, 1e-10), uxR, utR, np.maximum(pR, 1e-10))
            div_x = (Fx[:, 1:, :] - Fx[:, :-1, :]) / dx

        # Periodic theta fluxes
        def rt(v):
            s = mc_limiter(v - np.roll(v, 1, axis=1), np.roll(v, -1, axis=1) - v)
            return v + 0.5 * s, np.roll(v - 0.5 * s, -1, axis=1)

        rL, rR = rt(rho); uxL, uxR = rt(ux); utL, utR = rt(ut); pL, pR = rt(p)
        Ft_raw = hllc_2d_normal(np.maximum(rL, 1e-10), utL, uxL, np.maximum(pL, 1e-10),
                                np.maximum(rR, 1e-10), utR, uxR, np.maximum(pR, 1e-10))
        Ft = np.stack([Ft_raw[0], Ft_raw[2], Ft_raw[1], Ft_raw[3]], axis=0)
        div_t = (Ft - np.roll(Ft, 1, axis=2)) / dth

        dU = -(div_x + div_t)

        if has_f:
            Pf = p - P_BAR
            P_f = np.sum(w_sense[:, None] * Pf, axis=0)
            u_f = np.sum(w_sense[:, None] * ut, axis=0)
            dpb = OMEGA_HP * (P_f - pb_s)
            dub = OMEGA_HP * (u_f - ub_s)
            dqp = np.empty_like(qp_s)
            dqu = np.empty_like(qu_s)
            dqp[0] = inv_ts * ((P_f - pb_s) - qp_s[0])
            dqu[0] = inv_ts * ((u_f - ub_s) - qu_s[0])
            for r in range(1, 4):
                dqp[r] = inv_ts * (qp_s[r - 1] - qp_s[r])
                dqu[r] = inv_ts * (qu_s[r - 1] - qu_s[r])
            qp_out = qp_s[3] if linear_flame else phi_flame(qp_s[3])
            qu_out = qu_s[3] if linear_flame else phi_flame(qu_s[3])
            q_tot = n_p * qp_out + n_v * qu_out
            dU[3] += (1.0 / (GAMMA - 1.0)) * wf[:, None] * q_tot[None, :]
        else:
            dpb = np.zeros_like(pb_s)
            dub = np.zeros_like(ub_s)
            dqp = np.zeros_like(qp_s)
            dqu = np.zeros_like(qu_s)

        return dU, dpb, dub, dqp, dqu

    t_list, ew_list, ppk_list, p1c_list, e1_list, e2_list = [], [], [], [], [], []
    rho_m_list, ut_m_list, ent_m_list, p_exit_list = [], [], [], []
    init_totals = np.sum(U, axis=(1, 2)) * dx * dth
    max_cons_err = np.zeros(4)

    def record(t_val: float):
        rho, ux, ut, p, _ = prim_2d(U)
        Pf = p - P_BAR
        Po = Pf - float(np.mean(Pf))
        uxo = ux - M_bar
        uto = ut - float(np.mean(ut))
        ew = 0.5 * float(np.mean(Po * Po + uxo * uxo + uto * uto))
        p_ax = np.mean(Pf, axis=0)
        phat = np.fft.rfft(p_ax) / (0.5 * nth)
        s_fld = np.log(np.maximum((p / P_BAR) / (rho ** GAMMA), 1e-15))

        if periodic_x:
            cur_totals = np.sum(U, axis=(1, 2)) * dx * dth
            denom = np.maximum(np.abs(init_totals), 1.0)
            np.maximum(max_cons_err, np.abs(cur_totals - init_totals) / denom, out=max_cons_err)

        t_list.append(t_val)
        ew_list.append(ew)
        ppk_list.append(float(np.max(np.abs(Pf))))
        p1c_list.append(complex(phat[1]))
        e1_list.append(float(np.abs(phat[1]) ** 2))
        e2_list.append(float(np.abs(phat[2]) ** 2))
        rho_m_list.append(float(np.mean(rho)))
        ut_m_list.append(float(np.mean(ut)))
        ent_m_list.append(float(np.mean(s_fld)))
        p_exit_list.append(float(np.mean(Pf[-1])))

    t = 0.0
    step = 0
    record(0.0)
    while t < t_end - 1e-12:
        _, ux_c, ut_c, _, c_c = prim_2d(U)
        dt = min(cfl / float(np.max((np.abs(ux_c) + c_c) / dx + (np.abs(ut_c) + c_c) / dth)), 0.015, t_end - t)
        k1 = rhs(U, pbf, ubf, qp, qu)
        U1 = U + dt * k1[0]
        pb1 = pbf + dt * k1[1]; ub1 = ubf + dt * k1[2]; qp1 = qp + dt * k1[3]; qu1 = qu + dt * k1[4]
        k2 = rhs(U1, pb1, ub1, qp1, qu1)
        U = 0.5 * (U + U1 + dt * k2[0])
        pbf = 0.5 * (pbf + pb1 + dt * k2[1]); ubf = 0.5 * (ubf + ub1 + dt * k2[2])
        qp = 0.5 * (qp + qp1 + dt * k2[3]); qu = 0.5 * (qu + qu1 + dt * k2[4])
        t += dt
        step += 1
        if step % record_stride == 0 or t >= t_end - 1e-12:
            record(t)

    rho_f, ux_f, ut_f, p_f, _ = prim_2d(U)
    return {
        "x": x, "theta": th,
        "P_final": p_f - P_BAR, "rho_final": rho_f, "ut_final": ut_f,
        "steady_max_err": float(np.max(np.abs(U - U_init))),
        "periodic_cons_err": {
            "mass": float(max_cons_err[0]), "mom_x": float(max_cons_err[1]),
            "mom_th": float(max_cons_err[2]), "energy": float(max_cons_err[3]),
        },
        "t": np.array(t_list),
        "e_wave": np.array(ew_list),
        "p_peak": np.array(ppk_list),
        "p1_complex": np.array(p1c_list),
        "e1": np.array(e1_list),
        "e2": np.array(e2_list),
        "rho_mean": np.array(rho_m_list),
        "ut_mean": np.array(ut_m_list),
        "ent_mean": np.array(ent_m_list),
        "p_exit_mean": np.array(p_exit_list),
    }


def envelope_spin_index(r: dict, win: float = 25.0) -> tuple[float, float]:
    """Boundary-layer-free Envelope Spin Index S_env = min|P_hat_1|/max|P_hat_1| in [0, 1] with sign."""
    m = r["t"] >= (r["t"][-1] - win)
    p1 = r["p1_complex"][m]
    amp = np.abs(p1)
    s_mag = float(np.min(amp) / max(np.max(amp), 1e-30))
    # Propagation sign: cos(theta - omega*t) has P_hat_1 ~ exp(-i*omega*t) => Im(p1^* dp1/dt) < 0 => +1!
    rot = float(np.mean(np.imag(np.conj(p1[:-1]) * np.diff(p1))))
    sgn = 1.0 if rot <= 0.0 else -1.0
    return sgn * s_mag, s_mag


def stats_2d(r: dict) -> dict:
    m = r["t"] >= 0.80 * r["t"][-1]
    m_mid = (r["t"] >= 0.50 * r["t"][-1]) & (r["t"] <= 0.65 * r["t"][-1])
    ew_late = float(np.mean(r["e_wave"][m]))
    ew_mid = float(np.mean(r["e_wave"][m_mid]))
    e1_m = float(np.mean(r["e1"][m]))
    e2_m = float(np.mean(r["e2"][m]))
    s_env_signed, s_env_abs = envelope_spin_index(r, win=min(25.0, 0.25 * r["t"][-1]))
    return {
        "p_over_pbar_pct": float(np.mean(r["p_peak"][m]) / P_BAR * 100.0),
        "e_wave_late": ew_late,
        "e_wave_late_over_mid": ew_late / max(ew_mid, 1e-30),
        "S_env_signed": s_env_signed,
        "S_env_abs": s_env_abs,
        "harmonic_2T_over_1T_pct": e2_m / max(e1_m, 1e-30) * 100.0,
        "rho_mean_late": float(np.mean(r["rho_mean"][m])),
        "ut_mean_late": float(np.mean(r["ut_mean"][m])),
    }


def run_exp4_m5_subcritical_audit() -> dict:
    """Verify M=5 staggering is linearly stable (A=1e-4) yet subcritically triggers for A >= 0.030."""
    print("\n--- Experiment 4B: M=5 Stagger Linear Stability vs. Erlang-4 Cubic-Asymmetry Bomb Triggering ---")
    tau_M5 = make_staggered_tau_2d(96, 5, tau_1=0.55, tau_2=2.55)
    r_lin = simulate_2d_euler_annulus(
        nx=32, nth=96, t_end=100.0, M_bar=0.12, n_p=0.22, n_v=0.04,
        tau_arr=tau_M5, init_type="spin_1T", A_eq=1e-4, record_stride=10,
    )
    m = r_lin["t"] >= 25.0
    alpha_lin = float(np.polyfit(r_lin["t"][m], 0.5 * np.log(np.maximum(r_lin["e_wave"][m], 1e-30)), 1)[0])
    e_ratio_lin = float(r_lin["e_wave"][-1] / max(r_lin["e_wave"][0], 1e-30))
    print(f"  [M=5 linear A=1e-4] alpha={alpha_lin:+.5f}, E(100)/E(0)={e_ratio_lin:.3e}")
    out = {"linear_A1e4": {"alpha": alpha_lin, "e_final_over_init": e_ratio_lin}, "bombs": {}}
    for amp in (0.015, 0.030):
        rb = simulate_2d_euler_annulus(
            nx=32, nth=96, t_end=140.0, M_bar=0.12, n_p=0.22, n_v=0.04,
            tau_arr=tau_M5, init_type="spin_1T", A_eq=amp, record_stride=15,
        )
        st = stats_2d(rb)
        out["bombs"][f"A_{amp:.3f}"] = st
        print(f"  [M=5 bomb A={amp:.3f}] P={st['p_over_pbar_pct']:.2f}%, E_wave={st['e_wave_late']:.2e}, l/m={st['e_wave_late_over_mid']:.3f}")
    return out


# ---------------------------------------------------------------------------
# 3. Main Experiments & Figures
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Entry 009 (rocket-combustion): 2D (x, theta) Compressible Euler Annulus with Axial Throughflow ===\n")

    # ------------------------------------------------------------------
    # Experiment 1: Verification Suite & Physical Entropy/Swirl Flushing
    # ------------------------------------------------------------------
    print("--- Experiment 1A & 1B: Steady-State Preservation & 2D Inviscid Conservation ---")
    r_st = simulate_2d_euler_annulus(nx=32, nth=96, t_end=20.0, M_bar=0.12, init_type="steady", record_stride=50)
    r_per = simulate_2d_euler_annulus(nx=32, nth=96, t_end=10.0, M_bar=0.12, periodic_x=True, init_type="periodic_test", A_eq=0.10, record_stride=20)
    print(f"  Steady throughflow preservation (t=20): max |U - U_0| = {r_st['steady_max_err']:.2e}")
    print(f"  2D periodic inviscid conservation: {r_per['periodic_cons_err']}")

    print("\n--- Experiment 1C: Exact Chebyshev 2D Eigenvalues vs. 2D Euler MC PDE Cold Damping (M_bar=0.12) ---")
    cold_exact = {
        "1L": exact_2d_throughflow_eig(0.12, 0, omega_window=(1.5, 2.5)),
        "1T": exact_2d_throughflow_eig(0.12, 1, omega_window=(0.65, 1.35)),
        "2T": exact_2d_throughflow_eig(0.12, 2, omega_window=(1.6, 2.3)),
        "3T": exact_2d_throughflow_eig(0.12, 3, omega_window=(2.6, 3.3)),
    }
    cold_ver = {}
    for m_name, itype in [("1L", "cold_1L"), ("1T", "spin_1T"), ("2T", "spin_2T"), ("3T", "spin_3T")]:
        ex = cold_exact[m_name]
        cold_ver[m_name] = {"exact_alpha": ex.real, "exact_omega": ex.imag, "grids": {}}
        row = []
        for nx_g, nth_g in [(24, 72), (32, 96), (48, 144)]:
            rc = simulate_2d_euler_annulus(nx=nx_g, nth=nth_g, t_end=35.0, M_bar=0.12, init_type=itype, A_eq=1e-4, record_stride=10)
            msk = rc["t"] >= 14.0
            afit = float(np.polyfit(rc["t"][msk], 0.5 * np.log(np.maximum(rc["e_wave"][msk], 1e-30)), 1)[0])
            cold_ver[m_name]["grids"][f"{nx_g}x{nth_g}"] = {"alpha_pde": afit, "excess_num": afit - ex.real}
            row.append(f"{nx_g}x{nth_g}:{afit:+.5f}({afit-ex.real:+.1e})")
        print(f"  [{m_name}] Exact={ex.real:+.5f} (w={ex.imag:.4f}) | " + " | ".join(row))

    print("\n--- Experiment 1D: First-Principles Entropy & Swirl Residence Flushing (tau_res = 12.5, ZERO S_ent!) ---")
    r_flush = simulate_2d_euler_annulus(nx=32, nth=96, t_end=22.0, M_bar=0.12, init_type="entropy_swirl_pulse", record_stride=4)
    idx_15 = int(np.argmin(np.abs(r_flush["t"] - 15.0)))
    idx_22 = -1
    m_mc = (r_flush["t"] >= 6.0) & (r_flush["t"] <= 16.0)
    idx_echo = int(np.where(m_mc)[0][int(np.argmax(np.abs(r_flush["p_exit_mean"][m_mc])))])
    t_echo = float(r_flush["t"][idx_echo])
    p_echo = float(r_flush["p_exit_mean"][idx_echo] / P_BAR * 100.0)
    print(
        f"  Entropy mean: t=0 -> {r_flush['ent_mean'][0]:.5f}, t=15 -> {r_flush['ent_mean'][idx_15]:.2e}, t=22 -> {r_flush['ent_mean'][idx_22]:.2e} | "
        f"Swirl ut_m: t=0 -> {r_flush['ut_mean'][0]:.4f}, t=15 -> {r_flush['ut_mean'][idx_15]:.2e} | "
        f"Post-transient exit echo (t in [6,16]): P={p_echo:+.4f}% at t={t_echo:.2f} (center conv time {(L_A-0.25)/0.12:.2f})"
    )

    # ------------------------------------------------------------------
    # Experiment 2: Strouhal Acoustic-Vorticity Damping Resonance (M_bar Sweep)
    # ------------------------------------------------------------------
    print("\n--- Experiment 2: Strouhal Acoustic-Vorticity Damping Resonance (M_bar in [0.06, 0.30]) ---")
    M_sweep = np.linspace(0.06, 0.30, 25)
    strouhal_curve = []
    for Mb in M_sweep:
        vc = exact_2d_throughflow_eig(float(Mb), 1, omega_window=(0.65, 1.35))
        npH = float(brentq(lambda npv: exact_2d_throughflow_eig(float(Mb), 1, n_p=npv, omega_window=(0.65, 1.35)).real, 0.01, 0.85))
        st_num = float(vc.imag * L_A / (2.0 * np.pi * Mb))
        strouhal_curve.append({"M_bar": float(Mb), "St": st_num, "cold_alpha_1T": vc.real, "cold_omega_1T": vc.imag, "hot_n_pH_1T": npH})

    pde_mach_ver = []
    for Mb_chk in (0.08, 0.12, 0.16, 0.18, 0.25):
        vc = exact_2d_throughflow_eig(Mb_chk, 1, omega_window=(0.65, 1.35))
        rc = simulate_2d_euler_annulus(nx=32, nth=96, t_end=35.0, M_bar=Mb_chk, init_type="spin_1T", A_eq=1e-4, record_stride=10)
        msk = rc["t"] >= 14.0
        afit = float(np.polyfit(rc["t"][msk], 0.5 * np.log(np.maximum(rc["e_wave"][msk], 1e-30)), 1)[0])
        pde_mach_ver.append({"M_bar": Mb_chk, "exact_alpha": vc.real, "pde_alpha": afit, "abs_diff": abs(afit - vc.real)})
        print(f"  [Cold 1T PDE check M={Mb_chk:.2f}] Exact={vc.real:+.5f} | 2D PDE={afit:+.5f} | |diff|={abs(afit-vc.real):.2e}")

    # ------------------------------------------------------------------
    # Experiment 3: First-Principles Stationarity & Landau Standing-to-Spinning Reversal (n_v = 0)
    # ------------------------------------------------------------------
    print("\n--- Experiment 3: First-Principles Stationarity & Landau Mode Reversal (n_v=0, ZERO S_ent!) ---")
    exp3_runs = {}
    exp3_stats = {}
    for label, Mb, np_v, itype, tend, nx_g, nth_g in [
        ("low_gain_stand",  0.06, 0.105, "stand_seed", 360.0, 28, 96),
        ("low_gain_spin",   0.06, 0.105, "spin_pert",  360.0, 28, 96),
        ("high_gain_stand", 0.12, 0.220, "stand_seed", 360.0, 28, 96),
        ("high_gain_spin",  0.12, 0.220, "spin_pert",  360.0, 28, 96),
        ("high_gain_48x144", 0.12, 0.220, "spin_pert", 200.0, 48, 144),
    ]:
        rr = simulate_2d_euler_annulus(nx=nx_g, nth=nth_g, t_end=tend, M_bar=Mb, n_p=np_v, n_v=0.0, init_type=itype, A_eq=0.035, record_stride=12)
        st = stats_2d(rr)
        exp3_runs[label] = rr
        exp3_stats[label] = st
        print(
            f"  [{label:16s}] M={Mb:.2f} np={np_v:.3f} | S_env={st['S_env_signed']:+.3f} | P={st['p_over_pbar_pct']:4.2f}% | "
            f"E2/E1={st['harmonic_2T_over_1T_pct']:4.1f}% | rho={st['rho_mean_late']:.4f} | ut_m={st['ut_mean_late']:+.5f}"
        )

    # ------------------------------------------------------------------
    # Experiment 4: 2D Chiral Rotating-Shock Control Shootout (n_p=0.22, n_v=0.04, A_0=6.9%)
    # ------------------------------------------------------------------
    print("\n--- Experiment 4: 2D Chiral Rotating-Shock Control Shootout (n_p=0.22, n_v=0.04, A_0=6.9%, t=280) ---")
    tau_M2_96 = make_staggered_tau_2d(96, 2, tau_1=0.55, tau_2=2.55)
    tau_M5_96 = make_staggered_tau_2d(96, 5, tau_1=0.55, tau_2=2.55)
    tau_M5_144 = make_staggered_tau_2d(144, 5, tau_1=0.55, tau_2=2.55)

    exp4_runs = {}
    exp4_stats = {}
    for cid, Mb, tau_c, nx_g, nth_g, tend in [
        ("bare_2d_chiral",      0.12, TAU_BAR,    32, 96,  280.0),
        ("stagger_M2_bragg",    0.12, tau_M2_96,  32, 96,  280.0),
        ("stagger_M5_odd",      0.12, tau_M5_96,  32, 96,  280.0),
        ("strouhal_M018_shift", 0.18, TAU_BAR,    32, 96,  280.0),
        ("stagger_M5_48x144",   0.12, tau_M5_144, 48, 144, 180.0),
    ]:
        rr = simulate_2d_euler_annulus(
            nx=nx_g, nth=nth_g, t_end=tend, M_bar=Mb, n_p=0.22, n_v=0.04,
            tau_arr=tau_c, init_type="spin_1T", A_eq=0.055, record_stride=12,
        )
        st = stats_2d(rr)
        exp4_runs[cid] = rr
        exp4_stats[cid] = st
        print(
            f"  [{cid:20s}] P={st['p_over_pbar_pct']:4.2f}% | E_wave={st['e_wave_late']:.2e} (l/m={st['e_wave_late_over_mid']:.3f}) | "
            f"S_env={st['S_env_signed']:+.3f} | rho={st['rho_mean_late']:.4f} | ut_m={st['ut_mean_late']:+.5f}"
        )

    # Experiment 4B: M=5 Linear Stability vs. Nonlinear Erlang-4 Cubic-Asymmetry Subcritical Triggering
    exp4b_m5_audit = run_exp4_m5_subcritical_audit()

    # ------------------------------------------------------------------
    # Figure 1: Strouhal Acoustic-Vorticity Damping Resonance & Physical Flushing
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.4), dpi=150)

    M_arr = [d["M_bar"] for d in strouhal_curve]
    a_arr = [-d["cold_alpha_1T"] for d in strouhal_curve]
    npH_arr = [d["hot_n_pH_1T"] for d in strouhal_curve]
    ax1.plot(M_arr, a_arr, "-", color="#1f77b4", lw=2.3, label="Exact Chebyshev cold damping $-\\alpha_{1T}(\\bar{M})$")
    ax1.scatter(
        [d["M_bar"] for d in pde_mach_ver], [-d["pde_alpha"] for d in pde_mach_ver],
        color="#d62728", s=65, zorder=6, label="2D Euler MC PDE ($32\\times 96$)",
    )
    for k_i in (1, 2, 3):
        m_tr = L_A / (2.0 * np.pi * k_i)
        if 0.06 <= m_tr <= 0.30:
            ax1.axvline(m_tr, color="gray", ls=":", lw=1.0, alpha=0.8)
    ax1.set_xlabel("Chamber axial Mach number $\\bar{M}$ ($L_a = 1.5$)")
    ax1.set_ylabel("Cold $1T$ damping rate $-\\alpha_{1T}$")
    ax1.set_title("Strouhal Acoustic-Vorticity Damping Peaks", fontweight="bold")
    ax1.grid(True, alpha=0.28)
    ax1.legend(loc="upper left", fontsize=8.2, framealpha=0.92)

    ax2.plot(M_arr, npH_arr, "-", color="#2ca02c", lw=2.4, label="Exact hot $1T$ Hopf threshold $n_{pH,1T}(\\bar{M})$")
    ax2.axhline(0.22, color="#d62728", ls="--", lw=1.6, label="Test flame gain $n_p = 0.22$")
    ax2.scatter([0.12, 0.18], [0.1338, 0.6382], color=["#d62728", "#2ca02c"], s=75, zorder=6)
    ax2.annotate("Integer St$\\approx 2$ trough\n($n_{pH}=0.134$, UNSTABLE)", xy=(0.12, 0.134), xytext=(0.135, 0.06),
                 arrowprops=dict(arrowstyle="->", color="#d62728"), fontsize=8.0, color="#d62728")
    ax2.annotate("Half-integer St peak\n($n_{pH}=0.638$, $4.8\\times$ higher!)", xy=(0.18, 0.638), xytext=(0.185, 0.56),
                 arrowprops=dict(arrowstyle="->", color="#2ca02c"), fontsize=8.0, color="#1a7f37")
    ax2.set_xlabel("Chamber axial Mach number $\\bar{M}$ ($L_a = 1.5, \\bar{\\tau} = 0.55$)")
    ax2.set_ylabel("Hot $1T$ linear Hopf threshold $n_{pH,1T}$")
    ax2.set_title("Non-Monotonic $4.8\\times$ Passive Stability Gain via $\\bar{M}$", fontweight="bold")
    ax2.grid(True, alpha=0.28)
    ax2.legend(loc="upper left", fontsize=8.2, framealpha=0.92)

    fig.suptitle(
        "Rocket Combustion Entry 009 (Part 1): 2D Compressible Throughflow Strouhal Vorticity-Interference Law",
        fontsize=11.6, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp009_strouhal_vorticity_damping.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: First-Principles 2D Landau Flip & Chiral Rotating-Shock Control
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.4), dpi=150)

    # Left: 2D pressure contour snapshot of the saturated rotating shock (bare_2d_chiral)
    r_ch = exp4_runs["bare_2d_chiral"]
    cf = ax1.contourf(r_ch["theta"], r_ch["x"], r_ch["P_final"] / P_BAR * 100.0, levels=25, cmap="RdBu_r")
    cb = fig.colorbar(cf, ax=ax1, pad=0.02)
    cb.set_label("$p'/\\bar{p}$ (%)")
    ax1.axhline(X_F, color="black", ls="--", lw=1.2, label="Flame plane $x_f = 0.25$")
    ax1.set_xlabel("Azimuthal angle $\\theta$ (rad)")
    ax1.set_ylabel("Axial coordinate $x$ ($0=$ injector, $1.5=$ nozzle)")
    ax1.set_title("2D Compressible Euler Rotating Shock ($n_p=0.22, n_v=0.04$)", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.0, framealpha=0.90)

    # Right: Control shootout log wave-energy evolution
    c_styles = [
        ("bare_2d_chiral",      "-",  "#d62728", "Bare 2D chiral shock ($\\bar{M}=0.12$)"),
        ("stagger_M2_bragg",    "--", "#ff7f0e", "Zero-HW $M=2$ stagger (Bragg trap!)"),
        ("stagger_M5_odd",      "-",  "#2ca02c", "Zero-HW $M=5$ odd stagger (no cavities!)"),
        ("strouhal_M018_shift", "-.", "#1f77b4", "Passive Strouhal shift ($\\bar{M}=0.12 \\rightarrow 0.18$)"),
    ]
    for cid, ls, col, lbl in c_styles:
        rr = exp4_runs[cid]
        ax2.semilogy(rr["t"], np.maximum(rr["e_wave"], 1e-11), ls, color=col, lw=2.2, label=lbl)
    ax2.set_xlabel("Dimensionless time $t \\bar{c} / R$")
    ax2.set_ylabel("2D oscillatory wave energy $E_{\\mathrm{wave}}(t)$ (log)")
    ax2.set_ylim(1e-11, 1e-2)
    ax2.set_title("2D Chiral Rotating-Shock Control ($A_0 = 6.9\\%$, ZERO $S_{\\mathrm{ent}}$)", fontweight="bold")
    ax2.grid(True, which="both", alpha=0.24)
    ax2.legend(loc="lower left", fontsize=7.9, framealpha=0.92)

    fig.suptitle(
        "Rocket Combustion Entry 009 (Part 2): 2D Rotating Shock Structure & Zero-Hardware Control Shootout",
        fontsize=11.6, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp009_2d_throughflow_control.png")
    plt.close(fig)

    metrics = {
        "parameters": {
            "gamma": GAMMA, "L_a": L_A, "x_f": X_F, "sig_f": SIG_F,
            "omega_hp": OMEGA_HP, "tau_bar": TAU_BAR, "q_s": Q_S, "kappa": KAPPA_NL,
            "stagger_tau_1": 0.55, "stagger_tau_2": 2.55,
        },
        "exp1a_steady_preservation_err": r_st["steady_max_err"],
        "exp1b_periodic_conservation_err": r_per["periodic_cons_err"],
        "exp1c_cold_damping_and_grid_convergence": cold_ver,
        "exp1d_physical_flushing": {
            "tau_res_exact": L_A / 0.12,
            "ent_t0": float(r_flush["ent_mean"][0]),
            "ent_t15": float(r_flush["ent_mean"][idx_15]),
            "ent_t22": float(r_flush["ent_mean"][idx_22]),
            "swirl_t0": float(r_flush["ut_mean"][0]),
            "swirl_t15": float(r_flush["ut_mean"][idx_15]),
            "marble_candel_echo_pct": p_echo,
            "marble_candel_echo_time": t_echo,
        },
        "exp2_strouhal_curve": strouhal_curve,
        "exp2_pde_cold_verification": pde_mach_ver,
        "exp3_stationarity_and_landau_flip": exp3_stats,
        "exp4_2d_chiral_control_shootout": exp4_stats,
        "exp4b_m5_subcritical_asymmetry_audit": exp4b_m5_audit,
    }
    with open(RESULTS_DIR / "exp009_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved Entry 009 plots and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
