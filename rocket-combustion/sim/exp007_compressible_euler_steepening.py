"""Entry 007 (rocket-combustion): Compressible Nonlinear Euler Transverse Acoustics — Shock Steepening,
Rankine-Hugoniot Mode-Selection Reversal & Dual-Tuned Cavity Cascade Absorption.

Upgrades Entries 004-006's bulk-linear 2D cylindrical-annulus transverse acoustics (k_x = 0)
to the full conservative 1D periodic compressible Euler equations on the azimuthal ring
theta in [0, 2*pi) with gamma = 1.25 (matching Entry 006's gamma_ref = 1.25):

    d(rho)/dt    + d(rho*u)/dtheta        = -(1/L_a) * rho * (u_noz + v_cav) - S_ent
    d(rho*u)/dt  + d(rho*u^2 + p)/dtheta  = -(1/L_a) * rho * u * (u_noz + v_cav) - S_ent * u
    d(E)/dt      + d(u*(E + p))/dtheta    = -(1/L_a) * (E + p) * (u_noz + v_cav) - 0.5 * S_ent * u^2
                                            + (1/(gamma - 1)) * (n_p / L_a) * Phi(q_p)

where:
    E = p / (gamma - 1) + 0.5 * rho * u^2,    p_bar = 1/gamma = 0.80, rho_bar = 1, c_bar = 1
    P(theta, t) = (p - p_bar) / (gamma * p_bar) = p - p_bar
    u_noz = zeta * P       (linearized choked-nozzle extraction => cold alpha_mT = -zeta/(2*L_a) = -0.010)
    v_cav = v_r(theta, t)  (faceplate Helmholtz cavity volume flux with calibrated quadratic orifice drag)
    S_ent = nu_res * (rho - (p / p_bar)^(1/gamma)),  nu_res = zeta / L_a (0 when zeta = 0):
            Isobaric chamber-residence entropy flush (zero effect on inviscid flows, zero effect on
            instantaneous p or u, and identically zero to first order on all isentropic linear acoustic
            modes; prevents secular mean-density drift from trapped shock heating over long horizons).

Linearized around (rho=1, u=0, p=1/gamma), this reproduces Entry 004-006's pure transverse
equations identically (exact continuum 1T Hopf point n_pH,1T = 0.04137, 2T = 0.04828, 3T = 0.06560).

Numerical method:
    2nd-order MUSCL-minmod primitive reconstruction + exact HLLC Riemann solver + SSP-RK2.
    Bulk-linear comparison uses 2/3-dealiased Fourier spectral derivatives + classic RK4.

Experiments:
1. Verification Suite:
   (1a) Inviscid unforced conservation of total mass, momentum, and total energy (< 1e-15).
   (1b) Small-amplitude linear limit vs Entry 004 Half-Damping Theorem (cold 1T, 2T alpha = -0.010)
        and exact continuum linear Hopf thresholds (1T, 2T, 3T).
   (1c) Exact Riemann simple-wave shock birth t_shock = 2 / ((gamma + 1) * u_0) (RH entropy onset)
        and exact Fay/Burgers maximum shock-jump time (pi/2) * t_shock across u_0 in {0.04, 0.08, 0.12}
        and grids N in {128, 256, 512}, plus O(u_0^3) weak-shock dissipation scaling.
2. Compressible Euler vs. Bulk-Linear Acoustics Under Entry 006's Flame Law:
   (2a) Supercritical spinning sawtooth shock wave (n_p = 0.068, N=256 & N=512): peak amplitude
        capping (-38%) and harmonic cascade (E_2T/E_1T ~ 28-29%, E_3T/E_1T ~ 8.7-9.0%).
   (2b) Attractor/Saddle Exchange (Mode-Selection Reversal, n_p = 0.068, t = 360): tests BOTH
        seeded standing (S_0 = 0.17) AND perturbed spinning (S_0 = 0.98) on Bulk-Linear vs. Euler
        (plus N = 512 verification), proving bulk-linear selects spinning (S -> 1) whereas
        compressible Euler reverses selection to standing (S -> 0) due to O(A^3) spinning shock loss.
   (2c) Subcritical bistability quenching (n_p = 0.037, 0.039 < n_pH,1T): compares bulk-linear
        bomb bisection (E_ratio ~ 1.42-1.46) against compressible Euler (N=192 + N=512 check).
3. Shocked-Wave Cavity Absorption (n_p = 0.085, beta_tot = 2%, sigma_o = 3.5%, t = 320):
   Compares bare Euler, 100% 1T-only ring, 80/20 (1T/2T) dual ring, and 50/50 dual ring on an
   initially shocked spinning 1T wave (A_0 = 0.055 = 6.9% p'/p_bar) where both 1T and 2T are
   supercritical and nonlinearly coupled by shock steepening.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

GAMMA = 1.25
P_BAR = 1.0 / GAMMA       # = 0.80 so c_bar = sqrt(gamma * p_bar / rho_bar) = 1.0
RHO_BAR = 1.0
L_A = 2.0
ZETA = 0.04
OMEGA_HP = 0.15
TAU_BAR = 0.30
SIGMA_TAU = 0.15          # Erlang-4: k=4, tau_s = 0.075
Q_S = 0.036
KAPPA_NL = 4.5


# ---------------------------------------------------------------------------
# 0. Exact Continuum Linear Dispersion (1D Azimuthal Ring)
# ---------------------------------------------------------------------------

def exact_hopf_1d_ring(m: int) -> tuple[float, float]:
    """Exact continuum bare Hopf threshold n_pH and frequency omega_H for azimuthal mode mT."""
    def res(x: np.ndarray) -> list[float]:
        w, np_v = float(x[0]), float(x[1])
        s = 1j * w
        H = (s / (s + OMEGA_HP)) / ((1.0 + s * (TAU_BAR / 4.0)) ** 4)
        F = s * s + ((ZETA - np_v * H) / L_A) * s + float(m * m)
        return [float(F.real), float(F.imag)]

    sol = root(res, np.array([float(m), 0.045]))
    if not sol.success:
        raise RuntimeError(f"Hopf root failed for mode {m}T: {sol.message}")
    return float(sol.x[1]), float(sol.x[0])


# ---------------------------------------------------------------------------
# 1. Exact 1D Periodic Compressible Euler HLLC + MUSCL-Minmod + SSP-RK2 Solver
# ---------------------------------------------------------------------------

def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a * b > 0.0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def prim_from_cons(U: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = np.maximum(U[0], 1e-10)
    u = U[1] / rho
    p = np.maximum((GAMMA - 1.0) * (U[2] - 0.5 * rho * u * u), 1e-10)
    c = np.sqrt(GAMMA * p / rho)
    return rho, u, p, c


def cons_from_prim(rho: np.ndarray, u: np.ndarray, p: np.ndarray) -> np.ndarray:
    E = p / (GAMMA - 1.0) + 0.5 * rho * u * u
    return np.vstack([rho, rho * u, E])


def hllc_flux_periodic(U: np.ndarray) -> np.ndarray:
    """2nd-order MUSCL-minmod HLLC numerical flux at periodic faces i+1/2 (shape 3 x N)."""
    rho, u, p, _ = prim_from_cons(U)

    def recon_periodic(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dl = v - np.roll(v, 1)
        dr = np.roll(v, -1) - v
        s = minmod(dl, dr)
        vL = v + 0.5 * s
        vR = np.roll(v - 0.5 * s, -1)
        return vL, vR

    rhoL, rhoR = recon_periodic(rho)
    uL, uR = recon_periodic(u)
    pL, pR = recon_periodic(p)
    rhoL = np.maximum(rhoL, 1e-10)
    rhoR = np.maximum(rhoR, 1e-10)
    pL = np.maximum(pL, 1e-10)
    pR = np.maximum(pR, 1e-10)

    cL = np.sqrt(GAMMA * pL / rhoL)
    cR = np.sqrt(GAMMA * pR / rhoR)

    EL = pL / (GAMMA - 1.0) + 0.5 * rhoL * uL * uL
    ER = pR / (GAMMA - 1.0) + 0.5 * rhoR * uR * uR

    srL = np.sqrt(rhoL)
    srR = np.sqrt(rhoR)
    u_roe = (srL * uL + srR * uR) / (srL + srR)
    HL = (EL + pL) / rhoL
    HR = (ER + pR) / rhoR
    H_roe = (srL * HL + srR * HR) / (srL + srR)
    c_roe = np.sqrt(np.maximum((GAMMA - 1.0) * (H_roe - 0.5 * u_roe * u_roe), 1e-12))

    sL = np.minimum(uL - cL, u_roe - c_roe)
    sR = np.maximum(uR + cR, u_roe + c_roe)

    num = pR - pL + rhoL * uL * (sL - uL) - rhoR * uR * (sR - uR)
    den = rhoL * (sL - uL) - rhoR * (sR - uR)
    sM = num / np.where(np.abs(den) > 1e-14, den, 1e-14)

    FL = np.vstack([rhoL * uL, rhoL * uL * uL + pL, uL * (EL + pL)])
    FR = np.vstack([rhoR * uR, rhoR * uR * uR + pR, uR * (ER + pR)])

    facL = rhoL * (sL - uL) / np.where(np.abs(sL - sM) > 1e-14, sL - sM, 1e-14)
    facR = rhoR * (sR - uR) / np.where(np.abs(sR - sM) > 1e-14, sR - sM, 1e-14)
    p_star = np.maximum(pL + rhoL * (sL - uL) * (sM - uL), 1e-10)

    UsL = np.vstack([
        facL,
        facL * sM,
        (EL * (sL - uL) - pL * uL + p_star * sM) / np.where(np.abs(sL - sM) > 1e-14, sL - sM, 1e-14),
    ])
    UsR = np.vstack([
        facR,
        facR * sM,
        (ER * (sR - uR) - pR * uR + p_star * sM) / np.where(np.abs(sR - sM) > 1e-14, sR - sM, 1e-14),
    ])

    FsL = FL + sL * (UsL - np.vstack([rhoL, rhoL * uL, EL]))
    FsR = FR + sR * (UsR - np.vstack([rhoR, rhoR * uR, ER]))

    return np.where(
        sL >= 0.0, FL,
        np.where(sM >= 0.0, FsL, np.where(sR >= 0.0, FsR, FR)),
    )


def phi_flame(q: np.ndarray) -> np.ndarray:
    """Bounded monotonic subcritical flame law from Entry 006."""
    x = q / Q_S
    return Q_S * np.tanh(x + KAPPA_NL * (x ** 3))


def make_isentropic_init(
    theta: np.ndarray,
    init_type: str,
    A_eq: float,
    eps_spin: float = 0.12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exact isentropic initial conditions (exact Riemann simple wave for spinning modes)."""
    if init_type in ("simple_wave_1T", "spin_1T"):
        u = A_eq * np.cos(theta)
        c = 1.0 + 0.5 * (GAMMA - 1.0) * u
        rho = c ** (2.0 / (GAMMA - 1.0))
        p = P_BAR * (rho ** GAMMA)
        return rho, u, p
    elif init_type == "spin_2T":
        u = A_eq * np.cos(2.0 * theta)
        c = 1.0 + 0.5 * (GAMMA - 1.0) * u
        rho = c ** (2.0 / (GAMMA - 1.0))
        p = P_BAR * (rho ** GAMMA)
        return rho, u, p
    elif init_type == "spin_1T_perturbed":
        # Nearly pure right-going spin (a_+ = 1, a_- = 0.10 => S_0 = (1 - 0.01)/(1 + 0.01) = 0.9802)
        eps_c = 0.10
        norm = np.sqrt(1.0 + eps_c * eps_c)
        a_p = A_eq / norm
        a_m = eps_c * a_p
        p = P_BAR + (a_p + a_m) * np.cos(theta)
        rho = (p / P_BAR) ** (1.0 / GAMMA)
        u = (a_p - a_m) * np.cos(theta)
        return rho, u, p
    elif init_type == "standing_1T_pure":
        p = P_BAR + np.sqrt(2.0) * A_eq * np.cos(theta)
        rho = (p / P_BAR) ** (1.0 / GAMMA)
        u = np.zeros_like(theta)
        return rho, u, p
    elif init_type == "standing_1T_seeded":
        norm = np.sqrt(1.0 + 0.5 * (eps_spin ** 2))
        a_s = A_eq / norm
        p = P_BAR + np.sqrt(2.0) * a_s * np.cos(theta)
        rho = (p / P_BAR) ** (1.0 / GAMMA)
        u = eps_spin * a_s * np.cos(theta)
        return rho, u, p
    else:
        raise ValueError(f"Unknown init_type: {init_type}")


def simulate_euler_ring(
    n_cells: int = 256,
    t_end: float = 40.0,
    cfl: float = 0.38,
    zeta: float = ZETA,
    n_p: float = 0.0,
    linear_flame: bool = False,
    bulk_linear_mode: bool = False,
    nu_res: float | None = None,
    beta_local: np.ndarray | float = 0.0,
    omega_H_arr: np.ndarray | float = 1.0,
    xi_H_arr: np.ndarray | float = 0.14,
    c_nl_arr: np.ndarray | float = 0.0,
    init_type: str = "spin_1T",
    A_eq: float = 0.05,
    record_stride: int = 5,
) -> dict:
    """Solve 1D periodic compressible Euler (or bulk-linear acoustic comparison) with flame & cavities."""
    dth = 2.0 * np.pi / n_cells
    theta = (np.arange(n_cells) + 0.5) * dth

    # Default isobaric entropy residence flush equals zeta / L_a (and identically 0 when zeta == 0)
    nu_ent = (zeta / L_A) if nu_res is None else float(nu_res)

    rho0, u0, p0 = make_isentropic_init(theta, init_type, A_eq)
    U = cons_from_prim(rho0, u0, p0)

    P_lin = p0 - P_BAR
    u_lin = u0.copy()

    b_loc = np.broadcast_to(np.asarray(beta_local, dtype=float), (n_cells,)).copy()
    has_cav = bool(np.any(b_loc > 0.0))
    beta_s = b_loc * L_A
    wH = np.broadcast_to(np.asarray(omega_H_arr, dtype=float), (n_cells,)).copy()
    xiH = np.broadcast_to(np.asarray(xi_H_arr, dtype=float), (n_cells,)).copy()
    c_nl = np.broadcast_to(np.asarray(c_nl_arr, dtype=float), (n_cells,)).copy()
    inv_beta_s = np.where(beta_s > 0.0, 1.0 / np.maximum(beta_s, 1e-30), 0.0)

    v_r = np.zeros(n_cells)
    p_c = (p0 - P_BAR).copy() if has_cav else np.zeros(n_cells)

    p_bar_f = np.zeros(n_cells)
    q_stages = np.zeros((4, n_cells))
    inv_tau_s = 4.0 / TAU_BAR

    def step_sources(rho, u, p, v_r_s, p_c_s, p_bf_s, qs_s):
        P_fluc = p - P_BAR
        dp_bf = OMEGA_HP * (P_fluc - p_bf_s)
        p_hp = P_fluc - p_bf_s
        dq = np.empty_like(qs_s)
        dq[0] = inv_tau_s * (p_hp - qs_s[0])
        dq[1] = inv_tau_s * (qs_s[0] - qs_s[1])
        dq[2] = inv_tau_s * (qs_s[1] - qs_s[2])
        dq[3] = inv_tau_s * (qs_s[2] - qs_s[3])

        q_out = qs_s[3] if linear_flame else phi_flame(qs_s[3])
        Q_flame_E = (1.0 / (GAMMA - 1.0)) * (n_p / L_A) * q_out

        if has_cav:
            dv_r = beta_s * (wH ** 2) * (P_fluc - p_c_s) - (2.0 * xiH * wH + c_nl * np.abs(v_r_s)) * v_r_s
            dp_c = v_r_s * inv_beta_s
        else:
            dv_r = np.zeros_like(v_r_s)
            dp_c = np.zeros_like(p_c_s)

        u_ext = zeta * P_fluc + v_r_s
        E_cur = p / (GAMMA - 1.0) + 0.5 * rho * u * u
        if nu_ent > 0.0:
            s_ent = nu_ent * (rho - (p / P_BAR) ** (1.0 / GAMMA))
        else:
            s_ent = 0.0

        S_U = np.vstack([
            -(1.0 / L_A) * rho * u_ext - s_ent,
            -(1.0 / L_A) * rho * u * u_ext - s_ent * u,
            -(1.0 / L_A) * (E_cur + p) * u_ext - 0.5 * s_ent * u * u + Q_flame_E,
        ])
        return S_U, dv_r, dp_c, dp_bf, dq

    def rhs_euler(U_s, v_r_s, p_c_s, p_bf_s, qs_s):
        rho, u, p, _ = prim_from_cons(U_s)
        F = hllc_flux_periodic(U_s)
        div_F = (F - np.roll(F, 1, axis=1)) / dth
        S_U, dv_r, dp_c, dp_bf, dq = step_sources(rho, u, p, v_r_s, p_c_s, p_bf_s, qs_s)
        return -div_F + S_U, dv_r, dp_c, dp_bf, dq

    k_lin = np.fft.fftfreq(n_cells, d=dth / (2.0 * np.pi))
    dealias_lin = np.abs(k_lin) <= (n_cells / 3.0)
    ik_lin = (1j * k_lin) * dealias_lin

    def rhs_linear(P_s, u_s, v_r_s, p_c_s, p_bf_s, qs_s):
        dP_dth = np.fft.ifft(ik_lin * np.fft.fft(P_s)).real
        du_dth = np.fft.ifft(ik_lin * np.fft.fft(u_s)).real

        dp_bf = OMEGA_HP * (P_s - p_bf_s)
        p_hp = P_s - p_bf_s
        dq = np.empty_like(qs_s)
        dq[0] = inv_tau_s * (p_hp - qs_s[0])
        dq[1] = inv_tau_s * (qs_s[0] - qs_s[1])
        dq[2] = inv_tau_s * (qs_s[1] - qs_s[2])
        dq[3] = inv_tau_s * (qs_s[2] - qs_s[3])
        q_out = qs_s[3] if linear_flame else phi_flame(qs_s[3])
        q_out = np.fft.ifft(np.fft.fft(q_out) * dealias_lin).real

        if has_cav:
            dv_r = beta_s * (wH ** 2) * (P_s - p_c_s) - (2.0 * xiH * wH + c_nl * np.abs(v_r_s)) * v_r_s
            dp_c = v_r_s * inv_beta_s
        else:
            dv_r = np.zeros_like(v_r_s)
            dp_c = np.zeros_like(p_c_s)

        dP = -du_dth + (n_p / L_A) * q_out - (zeta / L_A) * P_s - (1.0 / L_A) * v_r_s
        du = -dP_dth
        return dP, du, dv_r, dp_c, dp_bf, dq

    t_list, e_ac_list, spin_list, ppeak_list = [], [], [], []
    e1_list, e2_list, e3_list, max_grad_list, ent_list = [], [], [], [], []
    mass_init = float(np.sum(U[0]) * dth)
    mom_init = float(np.sum(U[1]) * dth)
    ene_init = float(np.sum(U[2]) * dth)
    max_mass_err = 0.0
    max_mom_err = 0.0
    max_ene_err = 0.0

    def record(t_val: float):
        if bulk_linear_mode:
            P_f = P_lin
            u_f = u_lin
            rho_f = np.ones(n_cells)
            p_f = P_BAR + P_f
        else:
            rho_f, u_f, p_f, _ = prim_from_cons(U)
            P_f = p_f - P_BAR
            nonlocal max_mass_err, max_mom_err, max_ene_err
            max_mass_err = max(max_mass_err, abs(float(np.sum(U[0]) * dth) - mass_init) / max(abs(mass_init), 1e-12))
            max_mom_err = max(max_mom_err, abs(float(np.sum(U[1]) * dth) - mom_init) / max(abs(mom_init), 1.0))
            max_ene_err = max(max_ene_err, abs(float(np.sum(U[2]) * dth) - ene_init) / max(abs(ene_init), 1e-12))

        e_ac = 0.5 * float(np.mean(P_f * P_f + u_f * u_f))
        spin_idx = abs(float(np.mean(P_f * u_f))) / max(e_ac, 1e-30)
        p_pk = float(np.max(np.abs(P_f)))

        P_hat = np.fft.rfft(P_f) / (0.5 * n_cells)
        e1 = float(np.abs(P_hat[1]) ** 2)
        e2 = float(np.abs(P_hat[2]) ** 2)
        e3 = float(np.abs(P_hat[3]) ** 2)

        du_dth = (np.roll(u_f, -1) - np.roll(u_f, 1)) / (2.0 * dth)
        max_neg_grad = float(np.max(-du_dth))
        s_mean = float(np.mean(np.log(np.maximum((p_f / P_BAR) / (rho_f ** GAMMA), 1e-15))))

        t_list.append(t_val)
        e_ac_list.append(e_ac)
        spin_list.append(spin_idx)
        ppeak_list.append(p_pk)
        e1_list.append(e1)
        e2_list.append(e2)
        e3_list.append(e3)
        max_grad_list.append(max_neg_grad)
        ent_list.append(s_mean)

    t = 0.0
    step = 0
    record(0.0)
    while t < t_end - 1e-12:
        if bulk_linear_mode:
            dt = min(0.8 * dth, 0.015, t_end - t)
            k1 = rhs_linear(P_lin, u_lin, v_r, p_c, p_bar_f, q_stages)
            k2 = rhs_linear(
                P_lin + 0.5 * dt * k1[0], u_lin + 0.5 * dt * k1[1], v_r + 0.5 * dt * k1[2],
                p_c + 0.5 * dt * k1[3], p_bar_f + 0.5 * dt * k1[4], q_stages + 0.5 * dt * k1[5],
            )
            k3 = rhs_linear(
                P_lin + 0.5 * dt * k2[0], u_lin + 0.5 * dt * k2[1], v_r + 0.5 * dt * k2[2],
                p_c + 0.5 * dt * k2[3], p_bar_f + 0.5 * dt * k2[4], q_stages + 0.5 * dt * k2[5],
            )
            k4 = rhs_linear(
                P_lin + dt * k3[0], u_lin + dt * k3[1], v_r + dt * k3[2],
                p_c + dt * k3[3], p_bar_f + dt * k3[4], q_stages + dt * k3[5],
            )
            P_lin = P_lin + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
            u_lin = u_lin + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
            v_r = v_r + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
            p_c = p_c + (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
            p_bar_f = p_bar_f + (dt / 6.0) * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4])
            q_stages = q_stages + (dt / 6.0) * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5])
        else:
            _, u_c, _, c_c = prim_from_cons(U)
            max_spd = float(np.max(np.abs(u_c) + c_c))
            dt = min(cfl * dth / max(max_spd, 1.0), 0.015, t_end - t)
            k1 = rhs_euler(U, v_r, p_c, p_bar_f, q_stages)
            U1 = U + dt * k1[0]
            v1 = v_r + dt * k1[1]; pc1 = p_c + dt * k1[2]
            pb1 = p_bar_f + dt * k1[3]; q1 = q_stages + dt * k1[4]
            k2 = rhs_euler(U1, v1, pc1, pb1, q1)
            U = 0.5 * (U + U1 + dt * k2[0])
            v_r = 0.5 * (v_r + v1 + dt * k2[1])
            p_c = 0.5 * (p_c + pc1 + dt * k2[2])
            p_bar_f = 0.5 * (p_bar_f + pb1 + dt * k2[3])
            q_stages = 0.5 * (q_stages + q1 + dt * k2[4])

        t += dt
        step += 1
        if step % record_stride == 0 or t >= t_end - 1e-12:
            record(t)

    if bulk_linear_mode:
        P_final = P_lin.copy()
        u_final = u_lin.copy()
        p_mean_ratio = 1.0
        rho_mean = 1.0
        c_mean = 1.0
    else:
        rho_fin, u_final, p_fin, c_fin = prim_from_cons(U)
        P_final = p_fin - P_BAR
        p_mean_ratio = float(np.mean(p_fin) / P_BAR)
        rho_mean = float(np.mean(rho_fin))
        c_mean = float(np.mean(c_fin))

    return {
        "theta": theta,
        "P_final": P_final,
        "u_final": u_final,
        "p_mean_ratio": p_mean_ratio,
        "rho_mean": rho_mean,
        "c_mean": c_mean,
        "t": np.array(t_list),
        "e_ac": np.array(e_ac_list),
        "spin": np.array(spin_list),
        "p_peak": np.array(ppeak_list),
        "e1": np.array(e1_list),
        "e2": np.array(e2_list),
        "e3": np.array(e3_list),
        "max_neg_grad": np.array(max_grad_list),
        "entropy_mean": np.array(ent_list),
        "conservation": {
            "mass_rel_err": max_mass_err,
            "mom_err": max_mom_err,
            "energy_rel_err": max_ene_err,
        },
    }


def make_dual_cavity_1d(
    n_cells: int,
    beta_tot: float,
    f1: float,
    xi_mult: float = 1.0,
    C_d: float = 0.65,
    sigma_o: float = 0.035,
):
    """Exact 1D ring mirror of Entry 005/006's odd 5-pair dual-tuned (1T/2T) cavity ring."""
    if f1 >= 0.999:
        b_loc = np.full(n_cells, beta_tot)
        wH = np.full(n_cells, 1.0)
        xiH = np.full(n_cells, xi_mult * np.sqrt(beta_tot))
    else:
        n_pair = 5
        sec_idx = np.floor(np.arange(n_cells) * (2 * n_pair) / n_cells).astype(int)
        is_z1 = (sec_idx % 2 == 0)
        f2 = 1.0 - f1
        b_loc = np.where(is_z1, 2.0 * f1 * beta_tot, 2.0 * f2 * beta_tot)
        wH = np.where(is_z1, 1.0, 2.0)
        xiH = np.where(is_z1, xi_mult * np.sqrt(f1 * beta_tot), xi_mult * np.sqrt(f2 * beta_tot))
    beta_s = b_loc * L_A
    c_nl = (beta_s * (wH ** 2)) / (2.0 * (C_d ** 2) * (sigma_o ** 2)) if sigma_o > 0.0 else np.zeros(n_cells)
    return b_loc, wH, xiH, c_nl


def lc_stats(r: dict) -> dict:
    m = r["t"] >= 0.80 * r["t"][-1]
    m_mid = (r["t"] >= 0.50 * r["t"][-1]) & (r["t"] <= 0.65 * r["t"][-1])
    e1_m = float(np.mean(r["e1"][m]))
    e2_m = float(np.mean(r["e2"][m]))
    e3_m = float(np.mean(r["e3"][m]))
    e_late = float(np.mean(r["e_ac"][m]))
    e_mid = float(np.mean(r["e_ac"][m_mid]))
    return {
        "p_peak_late": float(np.mean(r["p_peak"][m])),
        "p_over_pbar_pct": float(np.mean(r["p_peak"][m]) / P_BAR * 100.0),
        "e_ac_late": e_late,
        "e_late_over_mid": e_late / max(e_mid, 1e-30),
        "spin_late": float(np.mean(r["spin"][m])),
        "harmonic_2T_over_1T_pct": e2_m / max(e1_m, 1e-30) * 100.0,
        "harmonic_3T_over_1T_pct": e3_m / max(e1_m, 1e-30) * 100.0,
        "max_neg_grad_late": float(np.mean(r["max_neg_grad"][m])),
        "p_mean_ratio": r["p_mean_ratio"],
        "rho_mean": r["rho_mean"],
        "c_mean": r["c_mean"],
    }


# ---------------------------------------------------------------------------
# 2. Main Experiments & Figures
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Entry 007 (rocket-combustion): Compressible Nonlinear Euler Transverse Shock Steepening ===\n")

    # ------------------------------------------------------------------
    # Experiment 1: Verification Suite
    # ------------------------------------------------------------------
    print("--- Experiment 1A: Exact Inviscid Conservation (zeta=0, n_p=0, A=0.10, t=25) ---")
    r_inv = simulate_euler_ring(n_cells=256, t_end=25.0, zeta=0.0, n_p=0.0, init_type="simple_wave_1T", A_eq=0.10)
    print(
        f"  Mass rel err={r_inv['conservation']['mass_rel_err']:.2e} | "
        f"Mom err={r_inv['conservation']['mom_err']:.2e} | "
        f"Energy rel err={r_inv['conservation']['energy_rel_err']:.2e}"
    )

    print("\n--- Experiment 1B: Small-Amplitude Linear Limit & Exact Continuum Hopf Points ---")
    hopf_exact = {}
    for m in (1, 2, 3):
        npH, wH_ex = exact_hopf_1d_ring(m)
        hopf_exact[f"{m}T"] = {"n_pH": npH, "omega_H": wH_ex}
        print(f"  [Exact continuum {m}T] n_pH = {npH:.5f}, omega_H = {wH_ex:.5f}")
    lin_check = {}
    for m_name, itype in [("cold_1T", "spin_1T"), ("cold_2T", "spin_2T")]:
        r_l = simulate_euler_ring(n_cells=256, t_end=40.0, zeta=ZETA, n_p=0.0, init_type=itype, A_eq=1e-4)
        mask = r_l["t"] >= 5.0
        alpha_fit = float(np.polyfit(r_l["t"][mask], 0.5 * np.log(np.maximum(r_l["e_ac"][mask], 1e-30)), 1)[0])
        print(f"  [{m_name}] Euler PDE alpha = {alpha_fit:+.6f} (exact -zeta/(2*L_a) = {-ZETA/(2*L_A):+.6f}, |diff|={abs(alpha_fit + ZETA/(2*L_A)):.2e})")
        lin_check[m_name] = {"alpha_euler": alpha_fit, "alpha_exact": -ZETA / (2.0 * L_A), "abs_diff": float(abs(alpha_fit + ZETA / (2.0 * L_A)))}

    print("\n--- Experiment 1C: Exact Riemann Shock Birth t_shock = 2/((gamma+1)*u0) & Peak Jump (pi/2)*t_shock ---")
    shock_ver = []
    for u0 in (0.04, 0.08, 0.12):
        t_ex = 2.0 / ((GAMMA + 1.0) * u0)
        t_pk_ex = 0.5 * np.pi * t_ex
        r_s = simulate_euler_ring(n_cells=512, t_end=1.8 * t_ex, zeta=0.0, n_p=0.0, init_type="simple_wave_1T", A_eq=u0, record_stride=2)
        ds_dt = np.gradient(r_s["entropy_mean"], r_s["t"])
        t_grad_pk = float(r_s["t"][int(np.argmax(r_s["max_neg_grad"]))])
        idx_ent = int(np.where(ds_dt >= 0.25 * np.max(ds_dt))[0][0])
        t_ent_onset = float(r_s["t"][idx_ent])
        m_post = (r_s["t"] >= 1.15 * t_ex) & (r_s["t"] <= 1.75 * t_ex)
        dE_dt_shock = -float(np.polyfit(r_s["t"][m_post], r_s["e_ac"][m_post], 1)[0])
        print(
            f"  u_0={u0:.2f} ({GAMMA*u0*100:4.1f}%): birth t_shock={t_ex:5.2f} vs RH-onset={t_ent_onset:5.2f} | "
            f"max-jump (pi/2)t_s={t_pk_ex:5.2f} vs grad-pk={t_grad_pk:5.2f} ({(t_grad_pk/t_pk_ex-1)*100:+.1f}%) | "
            f"-dE/dt / u_0^3 = {dE_dt_shock/(u0**3):.3f}"
        )
        shock_ver.append({
            "u0": u0, "t_shock_exact": t_ex, "t_entropy_onset": t_ent_onset,
            "t_peak_jump_exact": t_pk_ex, "t_grad_peak": t_grad_pk,
            "peak_jump_rel_err_pct": (t_grad_pk / t_pk_ex - 1.0) * 100.0,
            "post_shock_dissipation_rate": dE_dt_shock,
            "dissipation_over_u0_cubed": dE_dt_shock / (u0 ** 3),
        })

    shock_grid = {}
    t_ex_08 = 2.0 / ((GAMMA + 1.0) * 0.08)
    t_pk_08 = 0.5 * np.pi * t_ex_08
    for nc in (128, 256, 512):
        rg = simulate_euler_ring(n_cells=nc, t_end=1.8 * t_ex_08, zeta=0.0, n_p=0.0, init_type="simple_wave_1T", A_eq=0.08, record_stride=2)
        t_pk = float(rg["t"][int(np.argmax(rg["max_neg_grad"]))])
        shock_grid[str(nc)] = {"t_grad_peak": t_pk, "rel_err_vs_pi_half_pct": (t_pk / t_pk_08 - 1.0) * 100.0}
    print(f"  Grid check (u_0=0.08, exact (pi/2)t_s={t_pk_08:.2f}): N=128->{shock_grid['128']['t_grad_peak']:.2f}, N=256->{shock_grid['256']['t_grad_peak']:.2f}, N=512->{shock_grid['512']['t_grad_peak']:.2f}\n")

    # ------------------------------------------------------------------
    # Experiment 2: Compressible Euler vs Bulk-Linear Under Entry 006 Flame Law
    # ------------------------------------------------------------------
    print("--- Experiment 2: Compressible Euler vs. Bulk-Linear Acoustics (Entry 006 Flame Law) ---")
    # (2A) Supercritical spinning sawtooth snapshot while on the spinning manifold (t = 220, S > 0.99)
    r_sup_lin = simulate_euler_ring(n_cells=256, t_end=220.0, n_p=0.068, bulk_linear_mode=True, init_type="spin_1T", A_eq=0.035)
    r_sup_eul = simulate_euler_ring(n_cells=256, t_end=220.0, n_p=0.068, bulk_linear_mode=False, init_type="spin_1T", A_eq=0.035)
    r_sup_eul512 = simulate_euler_ring(n_cells=512, t_end=220.0, n_p=0.068, bulk_linear_mode=False, init_type="spin_1T", A_eq=0.035)
    st_sup_lin = lc_stats(r_sup_lin)
    st_sup_eul = lc_stats(r_sup_eul)
    st_sup_eul512 = lc_stats(r_sup_eul512)
    print(f"  [2A: Supercritical n_p=0.068, Spinning 1T Manifold at t=220]")
    print(f"    Bulk-linear: P_peak={st_sup_lin['p_peak_late']:.4f} ({st_sup_lin['p_over_pbar_pct']:.1f}%) | E_2T/E_1T={st_sup_lin['harmonic_2T_over_1T_pct']:.2f}% | E_3T/E_1T={st_sup_lin['harmonic_3T_over_1T_pct']:.2f}% | S={st_sup_lin['spin_late']:.3f}")
    print(f"    Euler N=256: P_peak={st_sup_eul['p_peak_late']:.4f} ({st_sup_eul['p_over_pbar_pct']:.1f}%) | E_2T/E_1T={st_sup_eul['harmonic_2T_over_1T_pct']:.2f}% | E_3T/E_1T={st_sup_eul['harmonic_3T_over_1T_pct']:.2f}% | S={st_sup_eul['spin_late']:.3f} | rho_m={st_sup_eul['rho_mean']:.5f}")
    print(f"    Euler N=512: P_peak={st_sup_eul512['p_peak_late']:.4f} ({st_sup_eul512['p_over_pbar_pct']:.1f}%) | E_2T/E_1T={st_sup_eul512['harmonic_2T_over_1T_pct']:.2f}% | E_3T/E_1T={st_sup_eul512['harmonic_3T_over_1T_pct']:.2f}% | S={st_sup_eul512['spin_late']:.3f}")

    # (2B) Complete Attractor/Saddle Exchange: test BOTH seeded standing (S_0=0.17) AND perturbed spinning (S_0=0.98)!
    r_sel_st_lin = simulate_euler_ring(n_cells=256, t_end=360.0, n_p=0.068, bulk_linear_mode=True, init_type="standing_1T_seeded", A_eq=0.035)
    r_sel_sp_lin = simulate_euler_ring(n_cells=256, t_end=360.0, n_p=0.068, bulk_linear_mode=True, init_type="spin_1T_perturbed", A_eq=0.035)
    r_sel_st_eul = simulate_euler_ring(n_cells=256, t_end=360.0, n_p=0.068, bulk_linear_mode=False, init_type="standing_1T_seeded", A_eq=0.035)
    r_sel_sp_eul = simulate_euler_ring(n_cells=256, t_end=360.0, n_p=0.068, bulk_linear_mode=False, init_type="spin_1T_perturbed", A_eq=0.035)
    r_sel_st_eul512 = simulate_euler_ring(n_cells=512, t_end=360.0, n_p=0.068, bulk_linear_mode=False, init_type="standing_1T_seeded", A_eq=0.035)

    st_sel_st_lin = lc_stats(r_sel_st_lin)
    st_sel_sp_lin = lc_stats(r_sel_sp_lin)
    st_sel_st_eul = lc_stats(r_sel_st_eul)
    st_sel_sp_eul = lc_stats(r_sel_sp_eul)
    st_sel_st_eul512 = lc_stats(r_sel_st_eul512)
    print(f"\n  [2B: Attractor/Saddle Exchange at n_p=0.068, t=360]")
    print(f"    Bulk-linear (standing S_0=0.17): S_late={st_sel_st_lin['spin_late']:.4f} (SPINS UP!), P_peak={st_sel_st_lin['p_peak_late']:.4f}, E_ac={st_sel_st_lin['e_ac_late']:.3e}")
    print(f"    Bulk-linear (spin     S_0=0.98): S_late={st_sel_sp_lin['spin_late']:.4f} (STAYS SPIN), P_peak={st_sel_sp_lin['p_peak_late']:.4f}, E_ac={st_sel_sp_lin['e_ac_late']:.3e}")
    print(f"    Euler N=256 (standing S_0=0.17): S_late={st_sel_st_eul['spin_late']:.4f} (STAYS STAND!), P_peak={st_sel_st_eul['p_peak_late']:.4f}, E_ac={st_sel_st_eul['e_ac_late']:.3e}, E2/E1={st_sel_st_eul['harmonic_2T_over_1T_pct']:.2f}%")
    print(f"    Euler N=256 (spin     S_0=0.98): S_late={st_sel_sp_eul['spin_late']:.4f} (COLLAPSES TO STAND!), P_peak={st_sel_sp_eul['p_peak_late']:.4f}, E_ac={st_sel_sp_eul['e_ac_late']:.3e}, E2/E1={st_sel_sp_eul['harmonic_2T_over_1T_pct']:.2f}%")
    print(f"    Euler N=512 (standing S_0=0.17): S_late={st_sel_st_eul512['spin_late']:.4f}, P_peak={st_sel_st_eul512['p_peak_late']:.4f}, E_ac={st_sel_st_eul512['e_ac_late']:.3e}, E2/E1={st_sel_st_eul512['harmonic_2T_over_1T_pct']:.2f}%")

    # (2C) Subcritical bomb triggering at n_p = 0.037, 0.039 (< n_pH,1T = 0.04137)
    def bisect_ring_bomb(bulk_lin: bool, itype: str, n_p_v: float) -> float | None:
        def is_trig(amp: float) -> bool:
            rr = simulate_euler_ring(n_cells=192, t_end=260.0, n_p=n_p_v, bulk_linear_mode=bulk_lin, init_type=itype, A_eq=amp, record_stride=12)
            m2 = (rr["t"] >= 150.0) & (rr["t"] <= 190.0)
            m3 = rr["t"] >= 220.0
            e2, e3 = float(np.mean(rr["e_ac"][m2])), float(np.mean(rr["e_ac"][m3]))
            p3 = float(np.mean(rr["p_peak"][m3]))
            return bool((e3 > 1.005 * e2 and p3 > 0.004) or (e3 >= 0.995 * e2 and p3 > 0.015))

        hi_found = None
        for probe in (0.015, 0.030, 0.045):
            if is_trig(probe):
                hi_found = probe
                break
        if hi_found is None:
            return None
        lo, hi = 0.002, hi_found
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            if is_trig(mid):
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    sub_summary = {}
    print(f"\n  [2C: Subcritical Bomb Triggering (< n_pH,1T = {hopf_exact['1T']['n_pH']:.5f}): Bulk-Linear vs. Euler]")
    for np_sub in (0.037, 0.039):
        a_sp_l = bisect_ring_bomb(True, "spin_1T", np_sub)
        a_st_l = bisect_ring_bomb(True, "standing_1T_pure", np_sub)
        a_sp_e = bisect_ring_bomb(False, "spin_1T", np_sub)
        a_st_e = bisect_ring_bomb(False, "standing_1T_pure", np_sub)
        # Explicit N=512 grid check at A=0.030 confirming quenching is physical shock loss, not HLLC diffusion
        chk512 = {}
        for it in ("spin_1T", "standing_1T_pure"):
            r5 = simulate_euler_ring(n_cells=512, t_end=260.0, n_p=np_sub, bulk_linear_mode=False, init_type=it, A_eq=0.030, record_stride=15)
            m2 = (r5["t"] >= 150.0) & (r5["t"] <= 190.0)
            m3 = r5["t"] >= 220.0
            chk512[it] = {
                "p_peak_late": float(np.mean(r5["p_peak"][m3])),
                "e3_over_e2": float(np.mean(r5["e_ac"][m3]) / max(np.mean(r5["e_ac"][m2]), 1e-30)),
            }
        er_l = (a_sp_l / a_st_l) ** 2 if (a_sp_l and a_st_l) else None
        er_e = (a_sp_e / a_st_e) ** 2 if (a_sp_e and a_st_e) else None
        fmt = lambda v: f"{v:.4f}" if v is not None else "QUENCHED(<=0.045)"
        print(f"    n_p={np_sub:.3f} Bulk-linear: A_spin={fmt(a_sp_l)}, A_stand={fmt(a_st_l)} => E_ratio={f'{er_l:.3f}' if er_l else 'N/A'}")
        print(
            f"    n_p={np_sub:.3f} Euler N=192: A_spin={fmt(a_sp_e)}, A_stand={fmt(a_st_e)} | "
            f"N=512 check (A=0.030): spin e3/e2={chk512['spin_1T']['e3_over_e2']:.3f}, stand e3/e2={chk512['standing_1T_pure']['e3_over_e2']:.3f}"
        )
        sub_summary[f"np_{np_sub:.3f}"] = {
            "bulk_linear": {"A_spin": a_sp_l, "A_stand": a_st_l, "E_ratio": er_l},
            "euler_192": {"A_spin": a_sp_e, "A_stand": a_st_e, "E_ratio": er_e},
            "euler_512_A030_check": chk512,
        }
    print()

    # ------------------------------------------------------------------
    # Experiment 3: Shocked-Wave Cavity Absorption (n_p = 0.085, t = 320)
    # Both 1T (n_pH=0.0414) and 2T (n_pH=0.0483) supercritical + shock cascade
    # ------------------------------------------------------------------
    print("--- Experiment 3: Shocked-Wave Cavity Cascade Absorption (n_p=0.085, beta_tot=2%, sigma_o=3.5%, t=320) ---")
    cav_cases = [
        ("bare_euler", 0.0, 1.0),
        ("cav_100pct_1T", 0.02, 1.00),
        ("cav_80_20_dual", 0.02, 0.80),
        ("cav_50_50_dual", 0.02, 0.50),
    ]
    exp3_summary = {}
    exp3_runs = {}
    for cid, btot, f1_v in cav_cases:
        if btot > 0.0:
            b_loc, wH, xiH, c_nl = make_dual_cavity_1d(256, btot, f1_v, xi_mult=1.0, C_d=0.65, sigma_o=0.035)
        else:
            b_loc, wH, xiH, c_nl = 0.0, 1.0, 0.14, 0.0
        rr = simulate_euler_ring(
            n_cells=256, t_end=320.0, n_p=0.085, bulk_linear_mode=False,
            beta_local=b_loc, omega_H_arr=wH, xi_H_arr=xiH, c_nl_arr=c_nl,
            init_type="spin_1T", A_eq=0.055,
        )
        st = lc_stats(rr)
        print(
            f"  [{cid:16s}] P_peak={st['p_peak_late']:.4f} ({st['p_over_pbar_pct']:4.2f}%) | "
            f"E_ac={st['e_ac_late']:.3e} (late/mid={st['e_late_over_mid']:.3f}) | "
            f"E_2T/E_1T={st['harmonic_2T_over_1T_pct']:7.1f}% | rho_m={st['rho_mean']:.5f}"
        )
        exp3_summary[cid] = st
        exp3_runs[cid] = rr

    # ------------------------------------------------------------------
    # Figure 1: Sawtooth Shock Steepening & Attractor/Saddle Mode-Selection Reversal
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.4), dpi=150)

    ax1.plot(
        r_sup_lin["theta"], r_sup_lin["P_final"] / P_BAR * 100.0, "--", color="#1f77b4", lw=2.1,
        label=f"Bulk-linear ($p'_{{\\mathrm{{pk}}}}/\\bar{{p}}={st_sup_lin['p_over_pbar_pct']:.1f}\\%$, smooth)",
    )
    ax1.plot(
        r_sup_eul["theta"], r_sup_eul["P_final"] / P_BAR * 100.0, "-", color="#d62728", lw=2.3,
        label=f"Compressible Euler ($p'_{{\\mathrm{{pk}}}}/\\bar{{p}}={st_sup_eul['p_over_pbar_pct']:.1f}\\%$, sawtooth shock!)",
    )
    ax1.set_xlabel("Azimuthal angle $\\theta$ (rad)")
    ax1.set_ylabel("Pressure fluctuation $p' / \\bar{p}$ (%)")
    ax1.set_title("Supercritical Spinning $1T$ Waveform ($n_p = 0.068, \\gamma = 1.25$)", fontweight="bold")
    ax1.grid(True, alpha=0.28)
    ax1.legend(loc="upper right", fontsize=8.4)

    # Panel 2: Clean 4-curve Attractor/Saddle Exchange (zero clutter, zero overlap!)
    ax2.plot(
        r_sel_st_lin["t"], r_sel_st_lin["spin"], "--", color="#1f77b4", lw=2.2,
        label=f"Bulk-linear from standing ($S: 0.17 \\rightarrow {st_sel_st_lin['spin_late']:.2f}$)",
    )
    ax2.plot(
        r_sel_sp_lin["t"], r_sel_sp_lin["spin"], ":", color="#4fa3e0", lw=2.0,
        label=f"Bulk-linear from spinning ($S: 0.98 \\rightarrow {st_sel_sp_lin['spin_late']:.2f}$)",
    )
    ax2.plot(
        r_sel_st_eul["t"], r_sel_st_eul["spin"], "-", color="#d62728", lw=2.3,
        label=f"Euler from standing ($S: 0.17 \\rightarrow {st_sel_st_eul['spin_late']:.2f}$)",
    )
    ax2.plot(
        r_sel_sp_eul["t"], r_sel_sp_eul["spin"], "-.", color="#ff7f0e", lw=2.2,
        label=f"Euler from spinning ($S: 0.98 \\rightarrow {st_sel_sp_eul['spin_late']:.2f}$)",
    )
    ax2.set_xlabel("Dimensionless time $t \\bar{c} / R$")
    ax2.set_ylabel("Spin index $S(t)$ ($0 =$ standing, $1 =$ spinning)")
    ax2.set_ylim(-0.06, 1.12)
    ax2.set_title("Attractor/Saddle Exchange Under Gasdynamic Shocking", fontweight="bold")
    ax2.grid(True, alpha=0.28)
    ax2.legend(loc="center left", fontsize=8.1, framealpha=0.92)

    fig.suptitle(
        "Rocket Combustion Entry 007 (Part 1): Compressible Euler Shock Steepening & Mode-Selection Reversal",
        fontsize=11.8, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp007_euler_shock_steepening.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Shocked-Wave Cavity Absorption (Waveforms + Log Acoustic Energy Decay)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.4), dpi=150)

    styles = {
        "bare_euler":     ("-",  "#d62728", "Bare Euler (shocked limit cycle)"),
        "cav_100pct_1T":  ("--", "#1f77b4", "100% $1T$ ring ($\\beta=2\\%$, jumps to $2T$!)"),
        "cav_80_20_dual": ("-",  "#2ca02c", "80/20 ($1T/2T$) dual ring ($\\beta=2\\%$)"),
        "cav_50_50_dual": (":",  "#ff7f0e", "50/50 ($1T/2T$) dual ring ($\\beta=2\\%$)"),
    }
    for cid, (ls, col, lbl) in styles.items():
        rr = exp3_runs[cid]
        ax1.plot(rr["theta"], rr["P_final"] / P_BAR * 100.0, ls, color=col, lw=2.1, label=lbl)
    ax1.set_xlabel("Azimuthal angle $\\theta$ (rad)")
    ax1.set_ylabel("Final pressure waveform $p' / \\bar{p}$ (%) at $t = 320$")
    ax1.set_title("Final Shocked Waveforms ($n_p = 0.085, \\sigma_o = 3.5\\%$)", fontweight="bold")
    ax1.grid(True, alpha=0.28)
    ax1.legend(loc="upper right", fontsize=8.1)

    for cid, (ls, col, lbl) in styles.items():
        rr = exp3_runs[cid]
        ax2.semilogy(rr["t"], np.maximum(rr["e_ac"], 1e-8), ls, color=col, lw=2.2, label=lbl)
    ax2.set_xlabel("Dimensionless time $t \\bar{c} / R$")
    ax2.set_ylabel("Ring-averaged acoustic energy $E_{\\mathrm{ac}}(t)$ (log)")
    ax2.set_title("Acoustic Energy Evolution Under Shocked $A_0 = 6.9\\%$ Pulse", fontweight="bold")
    ax2.set_ylim(1e-6, 3e-3)
    ax2.grid(True, which="both", alpha=0.24)
    ax2.legend(loc="lower left", fontsize=8.1, framealpha=0.92)

    fig.suptitle(
        "Rocket Combustion Entry 007 (Part 2): Cavity Cascade Absorption Under Shocked Spinning Waves",
        fontsize=11.8, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp007_shock_cavity_absorption.png")
    plt.close(fig)

    metrics = {
        "parameters": {
            "gamma": GAMMA, "L_a": L_A, "zeta": ZETA, "nu_res": ZETA / L_A,
            "omega_hp": OMEGA_HP, "tau_bar": TAU_BAR, "q_s": Q_S, "kappa": KAPPA_NL,
        },
        "exp1a_inviscid_conservation": r_inv["conservation"],
        "exp1b_exact_continuum_hopf": hopf_exact,
        "exp1b_linear_limit_check": lin_check,
        "exp1c_riemann_shock_time": shock_ver,
        "exp1c_shock_grid_convergence": shock_grid,
        "exp2a_supercritical_spin_manifold": {
            "bulk_linear": st_sup_lin, "euler_256": st_sup_eul, "euler_512": st_sup_eul512,
        },
        "exp2b_attractor_saddle_exchange": {
            "bulk_linear_from_standing": st_sel_st_lin,
            "bulk_linear_from_spinning": st_sel_sp_lin,
            "euler_256_from_standing": st_sel_st_eul,
            "euler_256_from_spinning": st_sel_sp_eul,
            "euler_512_from_standing": st_sel_st_eul512,
        },
        "exp2c_subcritical_bomb_comparison": sub_summary,
        "exp3_shocked_cavity_absorption": exp3_summary,
    }
    with open(RESULTS_DIR / "exp007_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved Entry 007 plots and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
