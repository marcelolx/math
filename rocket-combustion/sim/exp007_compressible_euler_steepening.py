"""Entry 007 (rocket-combustion): Compressible Nonlinear Euler Transverse Acoustics — Shock Steepening,
Rankine-Hugoniot Mode-Selection Reversal & Cavity Cascade Absorption (Hardened per Peer Review #5).

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
    S_ent = nu_res * (rho - (p / p_bar)^(1/gamma)),  nu_res = zeta / L_a (default; 0 when zeta = 0):
            Isobaric chamber-residence entropy-flush closure. Leaves instantaneous p and u unchanged
            and vanishes to O(eps^2) on cold isentropic acoustic waves, but acts at O(eps) on
            flame-generated entropy (since heat addition occurs at fixed mass) and is a CONTROLLING
            modeling closure for long-horizon limit-cycle stationarity (sensitivity in nu_res tested).

Numerical methods:
    2nd-order MUSCL (monotonized-central MC or minmod limiter) + exact HLLC Riemann solver + SSP-RK2.
    Bulk-linear comparison uses 2/3-dealiased Fourier spectral derivatives + classic RK4.
    Exact linearized stability of azimuthally patterned cavity/stagger rings is computed via a
    Fourier-collocation eigenvalue solver (N_spec = 96).

Experiments:
1. Verification Suite:
   (1a) Inviscid unforced conservation of total mass, momentum, and total energy (< 1e-15).
   (1b) Small-amplitude linear consistency check (-zeta/(2*L_a)), exact continuum Hopf points
        (1T, 2T, 3T), and per-mode numerical damping table (minmod N=256 vs MC N=256).
   (1c) Exact Riemann simple-wave shock birth t_shock = 2 / ((gamma + 1) * u_0), Fay-Burgers
        peak-jump time (pi/2) * t_shock, and least-squares O(u_0^3) Rankine-Hugoniot dissipation
        constant (0.2209 over tau in [1.15, 1.75]) across u_0 in {0.04, 0.08, 0.12} & grids.
2. Compressible Euler vs. Bulk-Linear Acoustics Under Entry 006's Flame Law:
   (2a) Saturated bulk-linear spinning cycle (t = 600) vs. Euler standing attractor (N=256, N=512)
        and transient spinning sawtooth saddle (t = 220): attractor-vs-attractor capping (~ -29%).
   (2b) Attractor/saddle exchange (seeded standing S_0=0.17 & perturbed spinning S_0=0.98, t=400)
        plus explicit nu_res in {0.0, 0.01, 0.02, 0.05} closure sensitivity check (t = 600).
   (2c) Subcritical bistability quenching (n_p = 0.037, 0.039 < n_pH,1T): bulk-linear bisection vs.
        Euler N=512 MC multi-amplitude sweep A in {0.008, 0.015, 0.030, 0.045} (spin & standing).
3. Shocked-Wave Cavity Absorption (n_p = 0.085, beta_tot = 2%, sigma_o = 3.5%, t = 1600):
   Exact linear eigen-spectrum + low-dissipation MC N=256 long-horizon runs comparing bare Euler,
   100% 1T ring, 80/20 dual ring (exposing linearly unstable 3T alpha=+0.00191 that minmod N=256
   falsely damped!), and 50/50 dual ring (both spinning and equal-energy standing ICs).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, root


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
# 0. Exact Continuum Linear Dispersion & Fourier-Collocation Eigen-Spectrum
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


def exact_fay_burgers_lsq_constant(tau_lo: float = 1.15, tau_hi: float = 1.75) -> tuple[float, float]:
    """Exact Fay-Burgers weak-shock dissipation LSQ slope and mean instantaneous rate over [tau_lo, tau_hi]."""
    taus = np.linspace(tau_lo, tau_hi, 2000)
    phis = np.array([brentq(lambda p: p - tau * np.sin(p), 1e-10, np.pi - 1e-10) for tau in taus])
    rates = ((GAMMA + 1.0) / (3.0 * np.pi)) * (np.sin(phis) ** 3)
    dE_dtau = -rates / (0.5 * (GAMMA + 1.0))
    E_rel = np.cumsum(np.concatenate([[0.0], 0.5 * (dE_dtau[:-1] + dE_dtau[1:]) * np.diff(taus)]))
    lsq_slope = float(-np.polyfit(taus, E_rel, 1)[0] * 0.5 * (GAMMA + 1.0))
    return lsq_slope, float(np.mean(rates))


def linear_ring_eigenspectrum(
    n_p: float,
    n_v: float = 0.0,
    tau_arr: np.ndarray | float = TAU_BAR,
    beta_local: np.ndarray | float = 0.0,
    omega_H_arr: np.ndarray | float = 1.0,
    xi_H_arr: np.ndarray | float = 0.14,
    n_spec: int = 96,
) -> dict:
    """Exact linearized eigen-spectrum of the 1D ring with azimuthally patterned cavities & tau(theta).

    Uses spectral Fourier collocation on N_spec periodic nodes (resolving modes up to |m| < N_spec/2)
    for state vector [P, u, p_bf, qp0..3, (u_bf, qu0..3 if n_v!=0), (v_r, p_c if cavities)].
    """
    N = n_spec
    # Exact Fourier differentiation matrix on N equispaced periodic nodes
    j = np.arange(N)
    k = np.fft.fftfreq(N, d=1.0 / N)
    # D @ f = ifft(i*k*fft(f))
    F_mat = np.fft.fft(np.eye(N), axis=0)
    Finv = np.fft.ifft(np.eye(N), axis=0)
    D = (Finv @ (1j * k[:, None] * F_mat)).real

    tau_loc = np.broadcast_to(np.asarray(tau_arr, dtype=float), (N,)).copy()
    inv_ts = 4.0 / tau_loc

    b_loc = np.broadcast_to(np.asarray(beta_local, dtype=float), (N,)).copy()
    has_cav = bool(np.any(b_loc > 0.0))
    beta_s = b_loc * L_A
    wH = np.broadcast_to(np.asarray(omega_H_arr, dtype=float), (N,)).copy()
    xiH = np.broadcast_to(np.asarray(xi_H_arr, dtype=float), (N,)).copy()
    inv_beta_s = np.where(beta_s > 0.0, 1.0 / np.maximum(beta_s, 1e-30), 0.0)
    use_v = abs(n_v) > 0.0

    # Block layout: 0:P, 1:u, 2:p_bf, 3..6:qp0..3, [u_bf, qu0..3], [v_r, p_c]
    n_blocks = 7 + (5 if use_v else 0) + (2 if has_cav else 0)
    A = np.zeros((n_blocks * N, n_blocks * N), dtype=float)
    I_N = np.eye(N)

    def blk(r: int, c: int):
        return slice(r * N, (r + 1) * N), slice(c * N, (c + 1) * N)

    idx_P, idx_u, idx_pbf = 0, 1, 2
    idx_qp = [3, 4, 5, 6]
    nxt = 7
    if use_v:
        idx_ubf = nxt
        idx_qu = [nxt + 1, nxt + 2, nxt + 3, nxt + 4]
        nxt += 5
    if has_cav:
        idx_vr, idx_pc = nxt, nxt + 1

    # dP/dt = -D u - (zeta/L_a) P + (n_p/L_a) qp3 [+ (n_v/L_a) qu3] [- (1/L_a) v_r]
    A[blk(idx_P, idx_u)] = -D
    A[blk(idx_P, idx_P)] = -(ZETA / L_A) * I_N
    A[blk(idx_P, idx_qp[3])] = (n_p / L_A) * I_N
    if use_v:
        A[blk(idx_P, idx_qu[3])] = (n_v / L_A) * I_N
    if has_cav:
        A[blk(idx_P, idx_vr)] = -(1.0 / L_A) * I_N

    # du/dt = -D P
    A[blk(idx_u, idx_P)] = -D

    # Pressure HPF + Erlang-4
    A[blk(idx_pbf, idx_P)] = OMEGA_HP * I_N
    A[blk(idx_pbf, idx_pbf)] = -OMEGA_HP * I_N
    D_ts = np.diag(inv_ts)
    A[blk(idx_qp[0], idx_P)] = D_ts
    A[blk(idx_qp[0], idx_pbf)] = -D_ts
    A[blk(idx_qp[0], idx_qp[0])] = -D_ts
    for s_i in range(1, 4):
        A[blk(idx_qp[s_i], idx_qp[s_i - 1])] = D_ts
        A[blk(idx_qp[s_i], idx_qp[s_i])] = -D_ts

    # Velocity HPF + Erlang-4
    if use_v:
        A[blk(idx_ubf, idx_u)] = OMEGA_HP * I_N
        A[blk(idx_ubf, idx_ubf)] = -OMEGA_HP * I_N
        A[blk(idx_qu[0], idx_u)] = D_ts
        A[blk(idx_qu[0], idx_ubf)] = -D_ts
        A[blk(idx_qu[0], idx_qu[0])] = -D_ts
        for s_i in range(1, 4):
            A[blk(idx_qu[s_i], idx_qu[s_i - 1])] = D_ts
            A[blk(idx_qu[s_i], idx_qu[s_i])] = -D_ts

    # Cavity ODEs
    if has_cav:
        A[blk(idx_vr, idx_P)] = np.diag(beta_s * (wH ** 2))
        A[blk(idx_vr, idx_pc)] = -np.diag(beta_s * (wH ** 2))
        A[blk(idx_vr, idx_vr)] = -np.diag(2.0 * xiH * wH)
        A[blk(idx_pc, idx_vr)] = np.diag(inv_beta_s)

    eigvals, eigvecs = np.linalg.eig(A)

    # Filter physical oscillatory acoustic modes (0.4 <= omega <= 10.0) and identify dominant mT
    modes = []
    for idx_e, lam in enumerate(eigvals):
        if lam.imag >= 0.4 and lam.imag <= 10.0:
            vec_P = eigvecs[0:N, idx_e]
            vec_u = eigvecs[N:2 * N, idx_e]
            # Dominant azimuthal harmonic m from |fft(P)|^2 (combining +m and -m)
            phat = np.abs(np.fft.fft(vec_P)) ** 2
            mode_pow = [float(phat[m] + phat[N - m]) for m in range(1, 12)]
            dom_m = int(np.argmax(mode_pow) + 1)
            # Signed spin direction of the eigenmode (+1 co-rotating, -1 counter-rotating, 0 standing)
            num_s = float(np.real(np.vdot(vec_P, vec_u)))
            den_s = 0.5 * float(np.real(np.vdot(vec_P, vec_P) + np.vdot(vec_u, vec_u)))
            s_mode = num_s / max(den_s, 1e-30)
            modes.append({
                "alpha": float(lam.real),
                "omega": float(lam.imag),
                "dom_m": dom_m,
                "spin_mode": s_mode,
            })

    modes.sort(key=lambda d: d["alpha"], reverse=True)
    by_mode = {}
    for m_id in (1, 2, 3, 4):
        cand = [x for x in modes if x["dom_m"] == m_id]
        if cand:
            by_mode[f"{m_id}T"] = cand[0]

    return {
        "max_alpha": modes[0]["alpha"] if modes else None,
        "least_stable": modes[0] if modes else None,
        "top_modes": modes[:6],
        "by_harmonic": by_mode,
    }


# ---------------------------------------------------------------------------
# 1. Exact 1D Periodic Compressible Euler HLLC + MUSCL (MC / minmod) + SSP-RK2
# ---------------------------------------------------------------------------

def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a * b > 0.0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def mc_limiter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Monotonized-Central (MC) TVD slope limiter: 2nd-order accurate with ~14x less 3T damping than minmod."""
    c = 0.5 * (a + b)
    same = (a * b) > 0.0
    mag = np.minimum(np.abs(c), np.minimum(2.0 * np.abs(a), 2.0 * np.abs(b)))
    return np.where(same, np.sign(c) * mag, 0.0)


def prim_from_cons(U: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = np.maximum(U[0], 1e-10)
    u = U[1] / rho
    p = np.maximum((GAMMA - 1.0) * (U[2] - 0.5 * rho * u * u), 1e-10)
    c = np.sqrt(GAMMA * p / rho)
    return rho, u, p, c


def cons_from_prim(rho: np.ndarray, u: np.ndarray, p: np.ndarray) -> np.ndarray:
    E = p / (GAMMA - 1.0) + 0.5 * rho * u * u
    return np.vstack([rho, rho * u, E])


def hllc_flux_periodic(U: np.ndarray, limiter: str = "mc") -> np.ndarray:
    """2nd-order MUSCL (MC or minmod) HLLC numerical flux at periodic faces i+1/2 (shape 3 x N)."""
    rho, u, p, _ = prim_from_cons(U)
    lim_fn = mc_limiter if limiter == "mc" else minmod

    def recon_periodic(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dl = v - np.roll(v, 1)
        dr = np.roll(v, -1) - v
        s = lim_fn(dl, dr)
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
    elif init_type == "spin_3T":
        u = A_eq * np.cos(3.0 * theta)
        c = 1.0 + 0.5 * (GAMMA - 1.0) * u
        rho = c ** (2.0 / (GAMMA - 1.0))
        p = P_BAR * (rho ** GAMMA)
        return rho, u, p
    elif init_type == "spin_1T_perturbed":
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
    limiter: str = "mc",
    nu_res: float | None = None,
    beta_local: np.ndarray | float = 0.0,
    omega_H_arr: np.ndarray | float = 1.0,
    xi_H_arr: np.ndarray | float = 0.14,
    c_nl_arr: np.ndarray | float = 0.0,
    init_type: str = "spin_1T",
    A_eq: float = 0.05,
    record_stride: int = 8,
) -> dict:
    """Solve 1D periodic compressible Euler (or bulk-linear acoustic comparison) with flame & cavities."""
    dth = 2.0 * np.pi / n_cells
    theta = (np.arange(n_cells) + 0.5) * dth
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
        s_ent = nu_ent * (rho - (p / P_BAR) ** (1.0 / GAMMA)) if nu_ent > 0.0 else 0.0

        S_U = np.vstack([
            -(1.0 / L_A) * rho * u_ext - s_ent,
            -(1.0 / L_A) * rho * u * u_ext - s_ent * u,
            -(1.0 / L_A) * (E_cur + p) * u_ext - 0.5 * s_ent * u * u + Q_flame_E,
        ])
        return S_U, dv_r, dp_c, dp_bf, dq

    def rhs_euler(U_s, v_r_s, p_c_s, p_bf_s, qs_s):
        rho, u, p, _ = prim_from_cons(U_s)
        F = hllc_flux_periodic(U_s, limiter=limiter)
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

    t_list, e_ac_list, e_wave_list, spin_list, spin_signed_list, ppeak_list = [], [], [], [], [], []
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

        P_osc = P_f - float(np.mean(P_f))
        u_osc = u_f - float(np.mean(u_f))
        e_ac = 0.5 * float(np.mean(P_f * P_f + u_f * u_f))
        e_wave = 0.5 * float(np.mean(P_osc * P_osc + u_osc * u_osc))
        s_signed = float(np.mean(P_osc * u_osc)) / max(e_wave, 1e-30)
        p_pk = float(np.max(np.abs(P_f)))

        P_hat = np.fft.rfft(P_osc) / (0.5 * n_cells)
        e1 = float(np.abs(P_hat[1]) ** 2)
        e2 = float(np.abs(P_hat[2]) ** 2)
        e3 = float(np.abs(P_hat[3]) ** 2)

        du_dth = (np.roll(u_f, -1) - np.roll(u_f, 1)) / (2.0 * dth)
        max_neg_grad = float(np.max(-du_dth))
        s_mean = float(np.mean(np.log(np.maximum((p_f / P_BAR) / (rho_f ** GAMMA), 1e-15))))

        t_list.append(t_val)
        e_ac_list.append(e_ac)
        e_wave_list.append(e_wave)
        spin_list.append(abs(s_signed))
        spin_signed_list.append(s_signed)
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
        u_mean = float(np.mean(u_lin))
    else:
        rho_fin, u_final, p_fin, c_fin = prim_from_cons(U)
        P_final = p_fin - P_BAR
        p_mean_ratio = float(np.mean(p_fin) / P_BAR)
        rho_mean = float(np.mean(rho_fin))
        c_mean = float(np.mean(c_fin))
        u_mean = float(np.mean(u_final))

    return {
        "theta": theta,
        "P_final": P_final,
        "u_final": u_final,
        "p_mean_ratio": p_mean_ratio,
        "rho_mean": rho_mean,
        "c_mean": c_mean,
        "u_mean": u_mean,
        "t": np.array(t_list),
        "e_ac": np.array(e_ac_list),
        "e_wave": np.array(e_wave_list),
        "spin": np.array(spin_list),
        "spin_signed": np.array(spin_signed_list),
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
    ew_late = float(np.mean(r["e_wave"][m]))
    ew_mid = float(np.mean(r["e_wave"][m_mid]))
    return {
        "p_peak_late": float(np.mean(r["p_peak"][m])),
        "p_over_pbar_pct": float(np.mean(r["p_peak"][m]) / P_BAR * 100.0),
        "e_wave_late": ew_late,
        "e_wave_late_over_mid": ew_late / max(ew_mid, 1e-30),
        "spin_late": float(np.mean(r["spin"][m])),
        "spin_signed_late": float(np.mean(r["spin_signed"][m])),
        "harmonic_2T_over_1T_pct": e2_m / max(e1_m, 1e-30) * 100.0,
        "harmonic_3T_over_1T_pct": e3_m / max(e1_m, 1e-30) * 100.0,
        "e1_late": e1_m,
        "e2_late": e2_m,
        "e3_late": e3_m,
        "max_neg_grad_late": float(np.mean(r["max_neg_grad"][m])),
        "p_mean_ratio": r["p_mean_ratio"],
        "rho_mean": r["rho_mean"],
        "c_mean": r["c_mean"],
        "u_mean": r["u_mean"],
    }


# ---------------------------------------------------------------------------
# 2. Main Experiments & Figures
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Entry 007 (Hardened per Peer Review #5): Compressible Nonlinear Euler Transverse Acoustics ===\n")

    # ------------------------------------------------------------------
    # Experiment 1: Verification Suite + Per-Mode Numerical Damping Table
    # ------------------------------------------------------------------
    print("--- Experiment 1A: Exact Inviscid Conservation (zeta=0, n_p=0, A=0.10, t=25, MC N=256) ---")
    r_inv = simulate_euler_ring(n_cells=256, t_end=25.0, zeta=0.0, n_p=0.0, limiter="mc", init_type="simple_wave_1T", A_eq=0.10)
    print(
        f"  Mass rel err={r_inv['conservation']['mass_rel_err']:.2e} | "
        f"Mom err={r_inv['conservation']['mom_err']:.2e} | "
        f"Energy rel err={r_inv['conservation']['energy_rel_err']:.2e}"
    )

    print("\n--- Experiment 1B: Continuum Hopf Points & Per-Mode Numerical Damping (minmod vs MC) ---")
    hopf_exact = {}
    for m in (1, 2, 3):
        npH, wH_ex = exact_hopf_1d_ring(m)
        hopf_exact[f"{m}T"] = {"n_pH": npH, "omega_H": wH_ex}
        print(f"  [Exact continuum {m}T] n_pH = {npH:.5f}, omega_H = {wH_ex:.5f}")

    num_diss = {}
    exact_alpha = -ZETA / (2.0 * L_A)
    for lim in ("minmod", "mc"):
        num_diss[lim] = {}
        row = []
        for m_name, itype in [("1T", "spin_1T"), ("2T", "spin_2T"), ("3T", "spin_3T")]:
            r_l = simulate_euler_ring(n_cells=256, t_end=40.0, zeta=ZETA, n_p=0.0, limiter=lim, init_type=itype, A_eq=1e-4, record_stride=4)
            mask = r_l["t"] >= 5.0
            a_fit = float(np.polyfit(r_l["t"][mask], 0.5 * np.log(np.maximum(r_l["e_wave"][mask], 1e-30)), 1)[0])
            a_num = a_fit - exact_alpha
            num_diss[lim][m_name] = {"alpha_pde": a_fit, "alpha_num": a_num}
            row.append(f"{m_name}:{a_num:+.2e}")
        print(f"  [Cold excess numerical damping N=256, {lim:6s}] " + " | ".join(row))

    lsq_ex, mean_ex = exact_fay_burgers_lsq_constant()
    print(f"\n--- Experiment 1C: Exact Riemann Shock Birth & Fay-Burgers O(u_0^3) Law (LSQ exact={lsq_ex:.5f}, mean={mean_ex:.5f}) ---")
    shock_ver = []
    for u0 in (0.04, 0.08, 0.12):
        t_ex = 2.0 / ((GAMMA + 1.0) * u0)
        t_pk_ex = 0.5 * np.pi * t_ex
        r_s = simulate_euler_ring(n_cells=512, t_end=1.8 * t_ex, zeta=0.0, n_p=0.0, limiter="mc", init_type="simple_wave_1T", A_eq=u0, record_stride=2)
        ds_dt = np.gradient(r_s["entropy_mean"], r_s["t"])
        t_grad_pk = float(r_s["t"][int(np.argmax(r_s["max_neg_grad"]))])
        idx_ent = int(np.where(ds_dt >= 0.25 * np.max(ds_dt))[0][0])
        t_ent_onset = float(r_s["t"][idx_ent])
        m_post = (r_s["t"] >= 1.15 * t_ex) & (r_s["t"] <= 1.75 * t_ex)
        dE_dt_shock = -float(np.polyfit(r_s["t"][m_post], r_s["e_wave"][m_post], 1)[0])
        print(
            f"  u_0={u0:.2f} ({GAMMA*u0*100:4.1f}%): birth={t_ex:5.2f} vs RH={t_ent_onset:5.2f} | "
            f"peak-jump={t_pk_ex:5.2f} vs grad-pk={t_grad_pk:5.2f} ({(t_grad_pk/t_pk_ex-1)*100:+.1f}%) | "
            f"-dE/dt/u0^3={dE_dt_shock/(u0**3):.4f}"
        )
        shock_ver.append({
            "u0": u0, "t_shock_exact": t_ex, "t_entropy_onset": t_ent_onset,
            "t_peak_jump_exact": t_pk_ex, "t_grad_peak": t_grad_pk,
            "peak_jump_rel_err_pct": (t_grad_pk / t_pk_ex - 1.0) * 100.0,
            "post_shock_dissipation_rate": dE_dt_shock,
            "dissipation_over_u0_cubed": dE_dt_shock / (u0 ** 3),
        })

    # ------------------------------------------------------------------
    # Experiment 2: Attractor-vs-Attractor Capping, Mode Reversal & nu_res Sensitivity
    # ------------------------------------------------------------------
    print("\n--- Experiment 2A: Saturated Bulk-Linear Attractor (t=600) vs. Euler Standing Attractor & Spin Saddle ---")
    r_bl_sat = simulate_euler_ring(n_cells=256, t_end=600.0, n_p=0.068, bulk_linear_mode=True, init_type="spin_1T", A_eq=0.035, record_stride=12)
    r_eul_spin_tr = simulate_euler_ring(n_cells=256, t_end=220.0, n_p=0.068, bulk_linear_mode=False, limiter="mc", init_type="spin_1T", A_eq=0.035, record_stride=8)
    r_eul_att_256 = simulate_euler_ring(n_cells=256, t_end=450.0, n_p=0.068, bulk_linear_mode=False, limiter="mc", init_type="standing_1T_seeded", A_eq=0.035, record_stride=10)
    r_eul_att_512 = simulate_euler_ring(n_cells=512, t_end=450.0, n_p=0.068, bulk_linear_mode=False, limiter="mc", init_type="standing_1T_seeded", A_eq=0.035, record_stride=18)

    st_bl_sat = lc_stats(r_bl_sat)
    st_eul_spin_tr = lc_stats(r_eul_spin_tr)
    st_eul_att_256 = lc_stats(r_eul_att_256)
    st_eul_att_512 = lc_stats(r_eul_att_512)
    cap_att_pct = (st_eul_att_512["p_over_pbar_pct"] / st_bl_sat["p_over_pbar_pct"] - 1.0) * 100.0
    cap_tr_pct = (st_eul_spin_tr["p_over_pbar_pct"] / st_bl_sat["p_over_pbar_pct"] - 1.0) * 100.0
    print(f"  Bulk-linear saturated spin (t=600): P={st_bl_sat['p_over_pbar_pct']:.2f}% (late/mid={st_bl_sat['e_wave_late_over_mid']:.3f}) | E2/E1={st_bl_sat['harmonic_2T_over_1T_pct']:.2f}%")
    print(f"  Euler MC spinning saddle   (t=220): P={st_eul_spin_tr['p_over_pbar_pct']:.2f}% ({cap_tr_pct:+.1f}% vs BL) | E2/E1={st_eul_spin_tr['harmonic_2T_over_1T_pct']:.1f}% | S={st_eul_spin_tr['spin_signed_late']:+.3f}")
    print(f"  Euler MC standing att N256 (t=450): P={st_eul_att_256['p_over_pbar_pct']:.2f}% (late/mid={st_eul_att_256['e_wave_late_over_mid']:.3f}) | E2/E1={st_eul_att_256['harmonic_2T_over_1T_pct']:.1f}% | S={st_eul_att_256['spin_signed_late']:+.4f}")
    print(f"  Euler MC standing att N512 (t=450): P={st_eul_att_512['p_over_pbar_pct']:.2f}% ({cap_att_pct:+.1f}% vs BL, late/mid={st_eul_att_512['e_wave_late_over_mid']:.3f}) | E2/E1={st_eul_att_512['harmonic_2T_over_1T_pct']:.1f}% | S={st_eul_att_512['spin_signed_late']:+.4f}")

    print("\n--- Experiment 2B: Attractor/Saddle Exchange & nu_res Closure Sensitivity (n_p=0.068, MC N=256) ---")
    r_sel_st_bl = simulate_euler_ring(n_cells=256, t_end=450.0, n_p=0.068, bulk_linear_mode=True, init_type="standing_1T_seeded", A_eq=0.035, record_stride=10)
    r_sel_sp_bl = simulate_euler_ring(n_cells=256, t_end=450.0, n_p=0.068, bulk_linear_mode=True, init_type="spin_1T_perturbed", A_eq=0.035, record_stride=10)
    r_sel_sp_eul = simulate_euler_ring(n_cells=256, t_end=450.0, n_p=0.068, bulk_linear_mode=False, limiter="mc", init_type="spin_1T_perturbed", A_eq=0.035, record_stride=10)

    nu_sens = {}
    for nu_v in (0.0, 0.01, 0.02, 0.05):
        r_nu_st = simulate_euler_ring(n_cells=256, t_end=600.0, n_p=0.068, limiter="mc", nu_res=nu_v, init_type="standing_1T_seeded", A_eq=0.035, record_stride=14)
        r_nu_sp = simulate_euler_ring(n_cells=256, t_end=600.0, n_p=0.068, limiter="mc", nu_res=nu_v, init_type="spin_1T_perturbed", A_eq=0.035, record_stride=14)
        s_st = lc_stats(r_nu_st)
        s_sp = lc_stats(r_nu_sp)
        nu_sens[f"nu_{nu_v:.2f}"] = {"from_standing_seeded": s_st, "from_spin_perturbed": s_sp}
        print(
            f"  nu_res={nu_v:.2f} (t=600) | stand: S={s_st['spin_signed_late']:+.4f}, P={s_st['p_over_pbar_pct']:.2f}%, rho={s_st['rho_mean']:.4f}, l/m={s_st['e_wave_late_over_mid']:.3f} | "
            f"spin: S={s_sp['spin_signed_late']:+.4f}, P={s_sp['p_over_pbar_pct']:.2f}%, rho={s_sp['rho_mean']:.4f}"
        )

    # ------------------------------------------------------------------
    # Experiment 2C: Subcritical Quenching (N=512 MC multi-amplitude sweep closing probe gap)
    # ------------------------------------------------------------------
    print("\n--- Experiment 2C: Subcritical Bomb Quenching (Bulk-Linear vs. Euler N=512 MC, A in {0.008..0.045}) ---")
    def bisect_bl(itype: str, np_v: float) -> float:
        def is_trig(amp: float) -> bool:
            rr = simulate_euler_ring(n_cells=192, t_end=260.0, n_p=np_v, bulk_linear_mode=True, init_type=itype, A_eq=amp, record_stride=12)
            m2 = (rr["t"] >= 150.0) & (rr["t"] <= 190.0)
            m3 = rr["t"] >= 220.0
            e2, e3 = float(np.mean(rr["e_wave"][m2])), float(np.mean(rr["e_wave"][m3]))
            p3 = float(np.mean(rr["p_peak"][m3]))
            return bool((e3 > 1.005 * e2 and p3 > 0.004) or (e3 >= 0.995 * e2 and p3 > 0.015))
        lo, hi = 0.002, 0.015
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            if is_trig(mid):
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    sub_summary = {}
    for np_sub in (0.037, 0.039):
        a_sp_l = bisect_bl("spin_1T", np_sub)
        a_st_l = bisect_bl("standing_1T_pure", np_sub)
        er_l = (a_sp_l / a_st_l) ** 2
        euler_grid = {}
        for it in ("spin_1T", "standing_1T_pure"):
            euler_grid[it] = {}
            for amp in (0.008, 0.015, 0.030, 0.045):
                r5 = simulate_euler_ring(n_cells=512, t_end=300.0, n_p=np_sub, limiter="mc", init_type=it, A_eq=amp, record_stride=20)
                m2 = (r5["t"] >= 180.0) & (r5["t"] <= 220.0)
                m3 = r5["t"] >= 260.0
                e3_e2 = float(np.mean(r5["e_wave"][m3]) / max(np.mean(r5["e_wave"][m2]), 1e-30))
                euler_grid[it][f"A_{amp:.3f}"] = {
                    "e3_over_e2": e3_e2,
                    "p_peak_late": float(np.mean(r5["p_peak"][m3])),
                    "triggered": bool(e3_e2 > 1.005),
                }
        sp_ratios = [euler_grid["spin_1T"][f"A_{a:.3f}"]["e3_over_e2"] for a in (0.008, 0.015, 0.030, 0.045)]
        st_ratios = [euler_grid["standing_1T_pure"][f"A_{a:.3f}"]["e3_over_e2"] for a in (0.008, 0.015, 0.030, 0.045)]
        print(f"  n_p={np_sub:.3f} Bulk-linear: A_spin={a_sp_l:.4f}, A_stand={a_st_l:.4f} (E_ratio={er_l:.3f})")
        print(f"  n_p={np_sub:.3f} Euler N=512 MC (A=0.008..0.045): ALL DECAY! spin e3/e2={[round(x,3) for x in sp_ratios]}, stand e3/e2={[round(x,3) for x in st_ratios]}")
        sub_summary[f"np_{np_sub:.3f}"] = {
            "bulk_linear": {"A_spin": a_sp_l, "A_stand": a_st_l, "E_ratio": er_l},
            "euler_512_mc_sweep": euler_grid,
        }

    # ------------------------------------------------------------------
    # Experiment 3: Exact Eigen-Spectrum & Low-Dissipation MC Cavity Runs (t=1600)
    # ------------------------------------------------------------------
    print("\n--- Experiment 3: Exact Linear Eigen-Spectrum & Low-Dissipation MC Cavity Absorption (n_p=0.085, t=1600) ---")
    cav_configs = [
        ("bare_euler",         0.0,  1.00, "mc",     "spin_1T",          1600.0),
        ("cav_100pct_1T",      0.02, 1.00, "mc",     "spin_1T",          1600.0),
        ("cav_80_20_mc",       0.02, 0.80, "mc",     "spin_1T",          1600.0),
        ("cav_80_20_minmod",   0.02, 0.80, "minmod", "spin_1T",          1600.0),
        ("cav_50_50_spin",     0.02, 0.50, "mc",     "spin_1T",          1600.0),
        ("cav_50_50_standing", 0.02, 0.50, "mc",     "standing_1T_pure", 1000.0),
    ]
    exp3_eigs = {}
    for cid, btot, f1_v in [("bare_euler", 0.0, 1.0), ("cav_100pct_1T", 0.02, 1.0), ("cav_80_20_dual", 0.02, 0.80), ("cav_50_50_dual", 0.02, 0.50)]:
        if btot > 0.0:
            b96, w96, x96, _ = make_dual_cavity_1d(96, btot, f1_v, xi_mult=1.0, C_d=0.65, sigma_o=0.035)
        else:
            b96, w96, x96 = 0.0, 1.0, 0.14
        ev = linear_ring_eigenspectrum(n_p=0.085, beta_local=b96, omega_H_arr=w96, xi_H_arr=x96, n_spec=96)
        exp3_eigs[cid] = ev
        ls = ev["least_stable"]
        print(f"  [Linear eigen {cid:14s}] max_alpha={ls['alpha']:+.5f} (mode {ls['dom_m']}T, w={ls['omega']:.3f})")

    exp3_summary = {}
    exp3_runs = {}
    for cid, btot, f1_v, lim, itype, tend in cav_configs:
        if btot > 0.0:
            b_loc, wH, xiH, c_nl = make_dual_cavity_1d(256, btot, f1_v, xi_mult=1.0, C_d=0.65, sigma_o=0.035)
        else:
            b_loc, wH, xiH, c_nl = 0.0, 1.0, 0.14, 0.0
        rr = simulate_euler_ring(
            n_cells=256, t_end=tend, n_p=0.085, bulk_linear_mode=False, limiter=lim,
            beta_local=b_loc, omega_H_arr=wH, xi_H_arr=xiH, c_nl_arr=c_nl,
            init_type=itype, A_eq=0.055, record_stride=24,
        )
        st = lc_stats(rr)
        exp3_summary[cid] = st
        exp3_runs[cid] = rr
        print(
            f"  [PDE {cid:18s}] t={tend:4.0f} | P={st['p_over_pbar_pct']:4.2f}% | "
            f"E_wave={st['e_wave_late']:.2e} (l/m={st['e_wave_late_over_mid']:.3f}) | "
            f"e1={st['e1_late']:.1e} e2={st['e2_late']:.1e} e3={st['e3_late']:.1e} | u_m={st['u_mean']:+.4f}"
        )

    # ------------------------------------------------------------------
    # Figure 1: Sawtooth Shock Steepening & Attractor/Saddle Mode-Selection Reversal
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.4), dpi=150)

    ax1.plot(
        r_bl_sat["theta"], r_bl_sat["P_final"] / P_BAR * 100.0, "--", color="#1f77b4", lw=2.1,
        label=f"Bulk-linear saturated spin ($P={st_bl_sat['p_over_pbar_pct']:.1f}\\%$)",
    )
    ax1.plot(
        r_eul_spin_tr["theta"], r_eul_spin_tr["P_final"] / P_BAR * 100.0, ":", color="#ff7f0e", lw=2.1,
        label=f"Euler MC spin saddle ($P={st_eul_spin_tr['p_over_pbar_pct']:.1f}\\%$, transient)",
    )
    ax1.plot(
        r_eul_att_256["theta"], r_eul_att_256["P_final"] / P_BAR * 100.0, "-", color="#d62728", lw=2.3,
        label=f"Euler MC standing attractor ($P={st_eul_att_256['p_over_pbar_pct']:.1f}\\%$)",
    )
    ax1.set_xlabel("Azimuthal angle $\\theta$ (rad)")
    ax1.set_ylabel("Pressure fluctuation $p' / \\bar{p}$ (%)")
    ax1.set_title("Saturated Bulk-Linear vs. Compressible Euler ($n_p = 0.068$)", fontweight="bold")
    ax1.grid(True, alpha=0.28)
    ax1.legend(loc="upper right", fontsize=8.1, framealpha=0.92)

    st_bl_sel = lc_stats(r_sel_st_bl)
    sp_bl_sel = lc_stats(r_sel_sp_bl)
    sp_eu_sel = lc_stats(r_sel_sp_eul)
    ax2.plot(
        r_sel_st_bl["t"], r_sel_st_bl["spin_signed"], "--", color="#1f77b4", lw=2.2,
        label=f"Bulk-linear from stand ($S \\rightarrow {st_bl_sel['spin_signed_late']:+.2f}$)",
    )
    ax2.plot(
        r_sel_sp_bl["t"], r_sel_sp_bl["spin_signed"], ":", color="#4fa3e0", lw=2.0,
        label=f"Bulk-linear from spin ($S \\rightarrow {sp_bl_sel['spin_signed_late']:+.2f}$)",
    )
    ax2.plot(
        r_eul_att_256["t"], r_eul_att_256["spin_signed"], "-", color="#d62728", lw=2.3,
        label=f"Euler MC from stand ($S \\rightarrow {st_eul_att_256['spin_signed_late']:+.2f}$)",
    )
    ax2.plot(
        r_sel_sp_eul["t"], r_sel_sp_eul["spin_signed"], "-.", color="#ff7f0e", lw=2.2,
        label=f"Euler MC from spin ($S \\rightarrow {sp_eu_sel['spin_signed_late']:+.2f}$)",
    )
    ax2.set_xlabel("Dimensionless time $t \\bar{c} / R$")
    ax2.set_ylabel("Signed spin index $S(t)$ ($0 =$ standing, $+1 =$ spinning)")
    ax2.set_ylim(-0.12, 1.12)
    ax2.set_title("Attractor/Saddle Exchange ($\\nu_{\\mathrm{res}} = 0.02$, MC $N = 256$)", fontweight="bold")
    ax2.grid(True, alpha=0.28)
    ax2.legend(loc="center right", fontsize=8.0, framealpha=0.92)

    fig.suptitle(
        "Rocket Combustion Entry 007 (Part 1): Compressible Euler Shock Capping & Mode-Selection Reversal",
        fontsize=11.6, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp007_euler_shock_steepening.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Low-Dissipation MC Cavity Absorption & 3T Instability Exposure
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.4), dpi=150)

    c_styles = [
        ("bare_euler",         "-",  "#d62728", "Bare Euler MC (shocked limit cycle)"),
        ("cav_100pct_1T",      "--", "#1f77b4", "100% $1T$ ring MC (jumps to $2T$ LC!)"),
        ("cav_80_20_mc",       "-",  "#2ca02c", "80/20 ring MC (true $3T$ regrowth, $\\alpha=+0.0019$!)"),
        ("cav_80_20_minmod",   ":",  "#2ca02c", "80/20 ring minmod $N=256$ (false num. decay!)"),
        ("cav_50_50_spin",     "-.", "#ff7f0e", "50/50 ring MC spin IC (marginal $3T$)"),
        ("cav_50_50_standing", "--", "#9467bd", "50/50 ring MC standing IC (orifice choked!)"),
    ]
    for cid, ls, col, lbl in c_styles:
        rr = exp3_runs[cid]
        ax1.semilogy(rr["t"], np.maximum(rr["e_wave"], 1e-10), ls, color=col, lw=2.1, label=lbl)
    ax1.set_xlabel("Dimensionless time $t \\bar{c} / R$")
    ax1.set_ylabel("Oscillatory wave energy $E_{\\mathrm{wave}}(t)$ (log)")
    ax1.set_ylim(1e-9, 5e-3)
    ax1.set_title("Long-Horizon Cavity Evolution ($n_p = 0.085, \\sigma_o = 3.5\\%$)", fontweight="bold")
    ax1.grid(True, which="both", alpha=0.24)
    ax1.legend(loc="lower left", fontsize=7.6, framealpha=0.92)

    # Panel 2: Final waveforms (t=1600) showing bare sawtooth, 100% 1T -> 2T, 80/20 MC -> 3T!
    for cid, ls, col, lbl in [
        ("bare_euler",    "-",  "#d62728", "Bare Euler (sawtooth)"),
        ("cav_100pct_1T", "--", "#1f77b4", "100% $1T$ ($2T$ limit cycle)"),
        ("cav_80_20_mc",  "-",  "#2ca02c", "80/20 MC (growing $3T$ wave!)"),
        ("cav_50_50_standing", "--", "#9467bd", "50/50 MC standing IC ($t=1000$)"),
    ]:
        rr = exp3_runs[cid]
        ax2.plot(rr["theta"], rr["P_final"] / P_BAR * 100.0, ls, color=col, lw=2.1, label=lbl)
    ax2.set_xlabel("Azimuthal angle $\\theta$ (rad)")
    ax2.set_ylabel("Final pressure waveform $p' / \\bar{p}$ (%)")
    ax2.set_title("Final Modes: $1T$ Shock vs. $2T$ Escape vs. $3T$ Regrowth", fontweight="bold")
    ax2.grid(True, alpha=0.28)
    ax2.legend(loc="upper right", fontsize=8.0, framealpha=0.92)

    fig.suptitle(
        "Rocket Combustion Entry 007 (Part 2): Cavity Cascade Absorption & 3T Numerical-Dissipation Audit",
        fontsize=11.6, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp007_shock_cavity_absorption.png")
    plt.close(fig)

    metrics = {
        "parameters": {
            "gamma": GAMMA, "L_a": L_A, "zeta": ZETA, "nu_res_default": ZETA / L_A,
            "omega_hp": OMEGA_HP, "tau_bar": TAU_BAR, "q_s": Q_S, "kappa": KAPPA_NL,
            "limiter_primary": "mc",
        },
        "exp1a_inviscid_conservation": r_inv["conservation"],
        "exp1b_exact_continuum_hopf": hopf_exact,
        "exp1b_numerical_damping_N256": num_diss,
        "exp1c_fay_burgers_theory": {"lsq_slope_115_175": lsq_ex, "mean_rate_115_175": mean_ex},
        "exp1c_riemann_shock_time": shock_ver,
        "exp2a_attractor_capping": {
            "bulk_linear_saturated_spin_t600": st_bl_sat,
            "euler_mc_spin_saddle_t220": st_eul_spin_tr,
            "euler_mc_standing_att_256_t450": st_eul_att_256,
            "euler_mc_standing_att_512_t450": st_eul_att_512,
            "capping_attractor_vs_attractor_pct": cap_att_pct,
            "capping_spin_saddle_vs_bl_pct": cap_tr_pct,
        },
        "exp2b_attractor_saddle_exchange": {
            "bulk_linear_from_standing_t450": st_bl_sel,
            "bulk_linear_from_spinning_t450": sp_bl_sel,
            "euler_mc_from_standing_t450": st_eul_att_256,
            "euler_mc_from_spinning_t450": sp_eu_sel,
        },
        "exp2b_nu_res_sensitivity_t600": nu_sens,
        "exp2c_subcritical_bomb_comparison": sub_summary,
        "exp3_linear_eigenspectrum": exp3_eigs,
        "exp3_shocked_cavity_absorption": exp3_summary,
    }
    with open(RESULTS_DIR / "exp007_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved hardened Entry 007 plots and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
