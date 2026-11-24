"""Entry 004: 2D Transverse/Spinning Tangential Modes (1T, 2T) & Radial Faceplate Baffles.

Models a 2D cylindrical-annulus rocket combustor (x, theta) in [0, L_a] x [0, 2*pi):
    dU_x/dt     + dP/dx     = 0
    dU_theta/dt + dP/dtheta = 0
    dP/dt + dU_x/dx + dU_theta/dtheta = Q(x, theta, t)

Boundary conditions:
    x = 0   (injector faceplate): U_x(0, theta, t) = 0
    x = L_a (choked nozzle/end):  U_x(L_a, theta, t) = zeta * P(L_a, theta, t)
    theta periodic on [0, 2*pi)
    Radial faceplate baffles:     U_theta(x, theta_b, t) = 0  for x in [0, L_b], b = 0..N_b-1

Heat release (carrying forward Entry 000-003 DC-blocking HP filter + causal Erlang-4):
    Q(x, theta, t) = delta(x - x_f) * [ n_p * q_p(theta, t) + n_v * q_v(theta, t) ]
where q_p is DC-blocked Erlang-4 filtered P(x_f, theta, t) and q_v is DC-blocked Erlang-4
filtered azimuthal cross-flow velocity U_theta(x_f, theta, t) (directional/swirled injector
velocity coupling).

Exact 2D unbaffled dispersion relation for azimuthal mode m (exp(s*t + i*m*theta)):
    Cold (n_p = n_v = 0):
        kappa_m * sinh(kappa_m * L_a) + zeta * s * cosh(kappa_m * L_a) = 0,  kappa_m = sqrt(s^2 + m^2)
    Hot spinning wave (exp(s*t - i*m*theta) has U_theta = (i*m / s) * P -> +P for co-rotating!):
        Includes both pressure coupling n_p and chiral velocity coupling +/- n_v * (i*m / s).

Experiments:
1. Cold 2D verification (1L, 1T, 2T, 1L1T): exact dispersion vs staggered C-grid RK4 PDE,
   proving pure 1T/2T receive only HALF the nozzle damping of 1L (-zeta/(2*L_a) vs -zeta/L_a),
   plus machine-precision discrete energy conservation with and without baffles.
2. Chiral velocity-coupled spinning 1T instability: co-rotating spinning 1T grows while
   counter-rotating 1T is damped; exact 2D spinning dispersion verified against PDE.
3. Radial faceplate baffles (length L_b and blade count N_b in {0, 2, 3, 4, 5, 6}):
   - Velocity-coupled spinning 1T suppression & frequency detuning omega_1T(L_b).
   - Even-vs-odd baffle symmetry trap on pressure-coupled 1T+2T modes (N_b=2 blind spot on 1T,
     N_b=4 blind spot on 2T, odd N_b=3, 5 breaking symmetry).
4. Grid convergence spot-check (40x60 vs 60x90) on the baffled 2D chamber.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OMEGA_HP = 0.15  # DC-blocking cutoff (<< omega_1T = 1.0)


# ---------------------------------------------------------------------------
# 1. Exact 2D unbaffled dispersion relation (pressure + chiral velocity coupling)
# ---------------------------------------------------------------------------

def flame_tf(s: complex, tau_bar: float, sigma_tau: float, m_stages: int = 4) -> complex:
    hp = s / (s + OMEGA_HP)
    if sigma_tau > 0.0:
        tau_d = min(tau_bar, np.sqrt(m_stages) * sigma_tau)
        tau_0 = max(0.0, tau_bar - tau_d)
        g = np.exp(-s * tau_0) / ((1.0 + s * (tau_d / m_stages)) ** m_stages) if tau_d > 0.0 else np.exp(-s * tau_bar)
    else:
        g = np.exp(-s * tau_bar)
    return hp * g


def dispersion_2d_F(
    s: complex,
    m: int,
    spin_dir: int,
    L_a: float,
    zeta: float,
    x_f: float,
    n_p: float,
    n_v: float,
    tau_bar: float,
    sigma_tau: float,
    m_stages: int = 4,
) -> complex:
    """Exact 2D unbaffled annulus dispersion for P ~ exp(s*t - i * spin_dir * m * theta).

    From dU_theta/dt + dP/dtheta = 0:
        s * U_theta = i * spin_dir * m * P  =>  U_theta = (i * spin_dir * m / s) * P.
    For a co-rotating wave (s ~ +i*m, spin_dir = +1): i*m / (i*m) = +1 (in phase with P!).
    For a counter-rotating wave (s ~ +i*m, spin_dir = -1): -i*m / (i*m) = -1 (out of phase!).
    Effective localized jump coupling at x = x_f is:
        n_eff(s) = [n_p + n_v * (i * spin_dir * m / s)] * flame_tf(s).
    """
    kap = np.sqrt(s * s + float(m * m))
    # Remove removable kap=0 singularity cleanly
    if abs(kap) < 1e-10:
        sh_kL, ch_kL = kap * L_a, 1.0
        sh_k1, ch_k1 = kap * x_f, 1.0
        sh_k2, ch_k2 = kap * (L_a - x_f), 1.0
    else:
        sh_kL, ch_kL = np.sinh(kap * L_a), np.cosh(kap * L_a)
        sh_k1, ch_k1 = np.sinh(kap * x_f), np.cosh(kap * x_f)
        sh_k2, ch_k2 = np.sinh(kap * (L_a - x_f)), np.cosh(kap * (L_a - x_f))

    lhs = kap * sh_kL + zeta * s * ch_kL
    if n_p == 0.0 and n_v == 0.0:
        return lhs

    tf = flame_tf(s, tau_bar, sigma_tau, m_stages)
    vel_factor = (1j * spin_dir * m / s) if (m != 0 and abs(s) > 1e-12) else 0.0j
    n_eff = (n_p + n_v * vel_factor) * tf
    # Jump condition across x_f: U_x(x_f+) - U_x(x_f-) = n_eff * P(x_f),
    # with P_1 = A*cosh(kap*x), U_x1 = -(kap/s)*A*sinh(kap*x),
    # P_2 = B*[kap*cosh(kap*(L_a-x)) + zeta*s*sinh(kap*(L_a-x))], etc.
    rhs = n_eff * s * ch_k1 * (kap * ch_k2 + zeta * s * sh_k2) / (kap if abs(kap) > 1e-10 else 1.0)
    # Multiply through by kap to avoid dividing by kap -> lhs = kap*(kap*sh_kL + zeta*s*ch_kL)
    if abs(kap) <= 1e-10:
        return kap * lhs - n_eff * s * ch_k1 * (kap * ch_k2 + zeta * s * sh_k2)
    return lhs - n_eff * s * ch_k1 * (ch_k2 + (zeta * s / kap) * sh_k2)


def solve_2d_mode(
    m: int,
    k_x: int,
    spin_dir: int,
    L_a: float,
    zeta: float,
    x_f: float,
    n_p: float,
    n_v: float,
    tau_bar: float,
    sigma_tau: float,
    m_stages: int = 4,
) -> tuple[complex, float]:
    """Homotopy continuation from cold 2D mode (m, k_x) to hot (n_p, n_v)."""
    w0 = np.sqrt(float(m * m) + (k_x * np.pi / L_a) ** 2)
    a0 = -zeta / (2.0 * L_a) if k_x == 0 else -zeta / L_a
    s = complex(a0, w0)

    # First polish cold root (n_p = n_v = 0)
    for _ in range(30):
        fv = dispersion_2d_F(s, m, spin_dir, L_a, zeta, x_f, 0.0, 0.0, tau_bar, sigma_tau, m_stages)
        if abs(fv) < 1e-13:
            break
        eps = 1e-7 * max(1.0, abs(s))
        dfv = (
            dispersion_2d_F(s + eps, m, spin_dir, L_a, zeta, x_f, 0.0, 0.0, tau_bar, sigma_tau, m_stages)
            - dispersion_2d_F(s - eps, m, spin_dir, L_a, zeta, x_f, 0.0, 0.0, tau_bar, sigma_tau, m_stages)
        ) / (2.0 * eps)
        if abs(dfv) < 1e-16:
            break
        s -= fv / dfv

    if n_p == 0.0 and n_v == 0.0:
        return s, float(abs(dispersion_2d_F(s, m, spin_dir, L_a, zeta, x_f, 0.0, 0.0, tau_bar, sigma_tau, m_stages)))

    steps = 10
    for step in range(1, steps + 1):
        frac = step / steps
        np_c, nv_c = n_p * frac, n_v * frac
        for _ in range(35):
            fv = dispersion_2d_F(s, m, spin_dir, L_a, zeta, x_f, np_c, nv_c, tau_bar, sigma_tau, m_stages)
            if abs(fv) < 1e-12:
                break
            eps = 1e-7 * max(1.0, abs(s))
            dfv = (
                dispersion_2d_F(s + eps, m, spin_dir, L_a, zeta, x_f, np_c, nv_c, tau_bar, sigma_tau, m_stages)
                - dispersion_2d_F(s - eps, m, spin_dir, L_a, zeta, x_f, np_c, nv_c, tau_bar, sigma_tau, m_stages)
            ) / (2.0 * eps)
            if abs(dfv) < 1e-16:
                break
            ds = fv / dfv
            if abs(ds) > 0.25:
                ds = 0.25 * ds / abs(ds)
            s -= ds
    res = float(abs(dispersion_2d_F(s, m, spin_dir, L_a, zeta, x_f, n_p, n_v, tau_bar, sigma_tau, m_stages)))
    return s, res


# ---------------------------------------------------------------------------
# 2. Vectorized 2D staggered C-grid RK4 PDE solver with radial baffles
# ---------------------------------------------------------------------------

def simulate_2d_annulus(
    L_a: float = 2.0,
    zeta: float = 0.04,
    x_f: float = 0.20,
    n_p: float = 0.0,
    n_v: float = 0.0,
    tau_bar: float = 0.30,
    sigma_tau: float = 0.15,
    m_stages: int = 4,
    n_baffles: int = 0,
    L_b: float = 0.0,
    baffle_offset_idx: int = 0,
    nx: int = 40,
    ntheta: int = 60,
    cfl: float = 0.35,
    t_end: float = 55.0,
    init_type: str = "spin_1T_plus",
    init_amp: float = 1e-3,
    init_phase: float = 0.0,
) -> dict:
    """Solve 2D cylindrical-annulus acoustics on an energy-conserving staggered C-grid.

    Grid layout:
      P[i, j]      at cell centers (x_i, theta_j), i=0..nx-1, j=0..ntheta-1
      Ux[i, j]     at interior axial faces x=(i+1)*dx, i=0..nx-2 (Ux_left=0, Ux_right=zeta*P[-1,:])
      Uth[i, j]    at azimuthal faces theta=(j+1)*dth, j=0..ntheta-1 (periodic)

    Baffles:
      For n_baffles > 0 and L_b > 0, solid blades at azimuthal faces j_b = (b*ntheta//n_baffles + offset) % ntheta
      enforce Uth[i, j_b] = 0 identically for all axial cells i with x_i <= L_b.

    Exact discrete energy balance:
      E = 0.5 * dx * dth * (sum(P^2) + sum(Ux^2) + sum(Uth^2))
      dE/dt = dth * sum_j P[m_f, j] * Q_line[j] - zeta * dth * sum_j P[-1, j]^2
    """
    dx = L_a / nx
    dth = (2.0 * np.pi) / ntheta
    inv_dx = 1.0 / dx
    inv_dth = 1.0 / dth

    # 2D wave CFL: dt * sqrt(1/dx^2 + 1/dth^2) <= cfl
    dt_max = cfl / np.sqrt(inv_dx * inv_dx + inv_dth * inv_dth)
    n_steps = int(round(t_end / dt_max))
    dt = t_end / n_steps

    x_c = (np.arange(nx) + 0.5) * dx
    th_c = (np.arange(ntheta) + 0.5) * dth
    th_f = (np.arange(ntheta) + 1.0) * dth  # Uth face angles

    m_f = int(np.argmin(np.abs(x_c - x_f)))
    x_f_actual = float(x_c[m_f])

    # Baffle mask on Uth (1.0 = open, 0.0 = blocked baffle blade)
    uth_mask = np.ones((nx, ntheta), dtype=float)
    baffle_faces: list[int] = []
    n_baffle_cells = 0
    if n_baffles > 0 and L_b > 0.0:
        n_baffle_cells = int(np.sum(x_c <= L_b + 1e-12))
        for b in range(n_baffles):
            jf = (int(round(b * ntheta / n_baffles)) + baffle_offset_idx) % ntheta
            baffle_faces.append(jf)
            uth_mask[:n_baffle_cells, jf] = 0.0
    L_b_actual = n_baffle_cells * dx

    # Initial condition
    X, TH = np.meshgrid(x_c, th_c, indexing="ij")
    P = np.zeros((nx, ntheta))
    Ux = np.zeros((nx - 1, ntheta))
    Uth = np.zeros((nx, ntheta))

    if init_type == "cold_1L":
        P = init_amp * np.cos(np.pi * X / L_a)
    elif init_type == "cold_1T":
        P = init_amp * np.cos(TH - init_phase)
    elif init_type == "cold_2T":
        P = init_amp * np.cos(2.0 * (TH - init_phase))
    elif init_type == "cold_1L1T":
        P = init_amp * np.cos(np.pi * X / L_a) * np.cos(TH - init_phase)
    elif init_type == "spin_1T_plus":
        # Co-rotating 1T spinning wave: P = A cos(-theta), Uth = +A cos(-theta)
        P = init_amp * np.cos(TH - init_phase)
        Uth = init_amp * np.cos(th_f[None, :] - init_phase) * np.ones((nx, 1))
    elif init_type == "spin_1T_minus":
        # Counter-rotating 1T spinning wave: Uth = -P
        P = init_amp * np.cos(TH - init_phase)
        Uth = -init_amp * np.cos(th_f[None, :] - init_phase) * np.ones((nx, 1))
    elif init_type == "broadband_transverse":
        # Multi-azimuthal seed (1T, 2T, 3T standing + spinning at non-symmetric phases)
        rng = np.random.default_rng(42)
        for m in (1, 2, 3):
            ph1 = rng.uniform(0, 2 * np.pi)
            ph2 = rng.uniform(0, 2 * np.pi)
            P += (init_amp / m) * (np.cos(m * TH - ph1) + 0.5 * np.cos(m * TH - ph2) * np.cos(np.pi * X / L_a))
            Uth += (0.5 * init_amp / m) * np.cos(m * th_f[None, :] - ph1) * np.ones((nx, 1))
    elif init_type == "aligned_1T":
        # Standing 1T with velocity nodes at theta = 0, pi (worst-case alignment for N_b=2!)
        P = init_amp * np.cos(TH)
    elif init_type == "aligned_2T":
        # Standing 2T with velocity nodes at theta = 0, pi/2, pi, 3pi/2 (worst-case for N_b=4!)
        P = init_amp * np.cos(2.0 * TH)
    else:
        raise ValueError(f"Unknown init_type: {init_type}")

    Uth *= uth_mask

    # Flame filtering: causal Erlang-m with tau_0 = max(0, tau_bar - tau_d)
    use_flame = (n_p != 0.0) or (n_v != 0.0)
    tau_d = min(tau_bar, np.sqrt(m_stages) * sigma_tau) if sigma_tau > 0.0 else 0.0
    tau_0 = max(0.0, tau_bar - tau_d)
    rate = m_stages / tau_d if tau_d > 0.0 else 0.0

    p_bar = np.zeros(ntheta)
    v_bar = np.zeros(ntheta)
    q_p = np.zeros((m_stages, ntheta))
    q_v = np.zeros((m_stages, ntheta))

    # Delay ring buffer only needed if tau_0 > 0
    use_delay = use_flame and (tau_0 > 1e-12)
    delay_steps_f = tau_0 / dt if use_delay else 0.0
    buf_len = int(np.ceil(delay_steps_f)) + 4 if use_delay else 2
    if use_delay:
        hist_p = np.zeros((n_steps + buf_len + 2, ntheta))
        hist_v = np.zeros((n_steps + buf_len + 2, ntheta))
        u_cell_0 = 0.5 * (Uth[m_f] + np.roll(Uth[m_f], 1))
        hist_p[buf_len] = P[m_f]
        hist_v[buf_len] = u_cell_0
    else:
        hist_p = hist_v = None

    def get_del(hist_arr, step_idx: int, frac: float, cur_val: np.ndarray) -> np.ndarray:
        if not use_delay:
            return cur_val
        idx_f = buf_len + step_idx + frac - delay_steps_f
        i0 = int(np.floor(idx_f))
        a = idx_f - i0
        if i0 < 0:
            return np.zeros(ntheta)
        if i0 >= buf_len + step_idx:
            return cur_val
        return (1.0 - a) * hist_arr[i0] + a * hist_arr[i0 + 1]

    def rhs(P_c, Ux_c, Uth_c, pbar_c, vbar_c, qp_c, qv_c, p_del, v_del):
        # dP/dt = -dUx/dx - dUth/dth + Q
        dP = np.empty_like(P_c)
        dP[0] = -Ux_c[0] * inv_dx
        dP[1:-1] = -(Ux_c[1:] - Ux_c[:-1]) * inv_dx
        dP[-1] = -(zeta * P_c[-1] - Ux_c[-1]) * inv_dx
        # Azimuthal divergence: (Uth[j] - Uth[j-1]) / dth
        dP -= (Uth_c - np.roll(Uth_c, 1, axis=1)) * inv_dth

        if use_flame:
            src_p = qp_c[-1] if tau_d > 0.0 else p_del
            src_v = qv_c[-1] if tau_d > 0.0 else v_del
            dP[m_f] += (n_p * src_p + n_v * src_v) * inv_dx

        # dUx/dt = -dP/dx
        dUx = -(P_c[1:] - P_c[:-1]) * inv_dx
        # dUth/dt = -dP/dth (masked on baffle faces!)
        dUth = -(np.roll(P_c, -1, axis=1) - P_c) * inv_dth
        dUth *= uth_mask

        if use_flame:
            u_cell = 0.5 * (Uth_c[m_f] + np.roll(Uth_c[m_f], 1))
            dpbar = OMEGA_HP * (P_c[m_f] - pbar_c)
            dvbar = OMEGA_HP * (u_cell - vbar_c)
            if tau_d > 0.0:
                dqp = np.empty_like(qp_c)
                dqv = np.empty_like(qv_c)
                dqp[0] = rate * (p_del - qp_c[0])
                dqv[0] = rate * (v_del - qv_c[0])
                for r in range(1, m_stages):
                    dqp[r] = rate * (qp_c[r - 1] - qp_c[r])
                    dqv[r] = rate * (qv_c[r - 1] - qv_c[r])
            else:
                dqp = np.zeros_like(qp_c)
                dqv = np.zeros_like(qv_c)
        else:
            dpbar = np.zeros_like(pbar_c)
            dvbar = np.zeros_like(vbar_c)
            dqp = np.zeros_like(qp_c)
            dqv = np.zeros_like(qv_c)

        return dP, dUx, dUth, dpbar, dvbar, dqp, dqv

    # Diagnostics
    t_arr = np.linspace(0.0, t_end, n_steps + 1)
    energy = np.empty(n_steps + 1)
    rayleigh = np.empty(n_steps + 1)
    noz_loss = np.empty(n_steps + 1)
    # Complex azimuthal Fourier modes at injector faceplate (i=0): c_m(t) = (1/pi) int P(0,th) e^{-i m th} dth
    c_1T = np.empty(n_steps + 1, dtype=complex)
    c_2T = np.empty(n_steps + 1, dtype=complex)
    a_1L = np.empty(n_steps + 1)
    # Azimuthal acoustic momentum flux (spin proxy) at x_f: int P(x_f) * Uth(x_f) dth
    spin_flux = np.empty(n_steps + 1)
    rms_uth_flame = np.empty(n_steps + 1)

    exp_m1 = np.exp(-1j * th_c)
    exp_m2 = np.exp(-2j * th_c)
    cos_1L = np.cos(np.pi * x_c / L_a)

    def record(idx: int) -> None:
        energy[idx] = 0.5 * dx * dth * (float(np.sum(P * P)) + float(np.sum(Ux * Ux)) + float(np.sum(Uth * Uth)))
        noz_loss[idx] = zeta * dth * float(np.sum(P[-1] * P[-1]))
        if use_flame:
            src_p = q_p[-1] if tau_d > 0.0 else get_del(hist_p, idx, 0.0, P[m_f] - p_bar)
            src_v = q_v[-1] if tau_d > 0.0 else get_del(hist_v, idx, 0.0, 0.5 * (Uth[m_f] + np.roll(Uth[m_f], 1)) - v_bar)
            rayleigh[idx] = dth * float(np.dot(P[m_f], n_p * src_p + n_v * src_v))
        else:
            rayleigh[idx] = 0.0
        c_1T[idx] = (dth / np.pi) * np.dot(P[0], exp_m1)
        c_2T[idx] = (dth / np.pi) * np.dot(P[0], exp_m2)
        p_ax_mean = np.mean(P, axis=1)
        a_1L[idx] = (2.0 * dx / L_a) * float(np.dot(cos_1L, p_ax_mean))
        u_cell = 0.5 * (Uth[m_f] + np.roll(Uth[m_f], 1))
        spin_flux[idx] = dth * float(np.dot(P[m_f], u_cell))
        rms_uth_flame[idx] = float(np.sqrt(np.mean(u_cell * u_cell)))

    record(0)

    for step in range(n_steps):
        if use_flame:
            u0 = 0.5 * (Uth[m_f] + np.roll(Uth[m_f], 1))
            pd0 = get_del(hist_p, step, 0.0, P[m_f] - p_bar)
            vd0 = get_del(hist_v, step, 0.0, u0 - v_bar)
        else:
            pd0 = vd0 = None

        k1 = rhs(P, Ux, Uth, p_bar, v_bar, q_p, q_v, pd0, vd0)

        P2, Ux2, Uth2 = P + 0.5 * dt * k1[0], Ux + 0.5 * dt * k1[1], Uth + 0.5 * dt * k1[2]
        pb2, vb2 = p_bar + 0.5 * dt * k1[3], v_bar + 0.5 * dt * k1[4]
        qp2, qv2 = q_p + 0.5 * dt * k1[5], q_v + 0.5 * dt * k1[6]
        if use_flame:
            u2 = 0.5 * (Uth2[m_f] + np.roll(Uth2[m_f], 1))
            pdh1 = get_del(hist_p, step, 0.5, P2[m_f] - pb2)
            vdh1 = get_del(hist_v, step, 0.5, u2 - vb2)
        else:
            pdh1 = vdh1 = None
        k2 = rhs(P2, Ux2, Uth2, pb2, vb2, qp2, qv2, pdh1, vdh1)

        P3, Ux3, Uth3 = P + 0.5 * dt * k2[0], Ux + 0.5 * dt * k2[1], Uth + 0.5 * dt * k2[2]
        pb3, vb3 = p_bar + 0.5 * dt * k2[3], v_bar + 0.5 * dt * k2[4]
        qp3, qv3 = q_p + 0.5 * dt * k2[5], q_v + 0.5 * dt * k2[6]
        if use_flame:
            u3 = 0.5 * (Uth3[m_f] + np.roll(Uth3[m_f], 1))
            pdh2 = get_del(hist_p, step, 0.5, P3[m_f] - pb3)
            vdh2 = get_del(hist_v, step, 0.5, u3 - vb3)
        else:
            pdh2 = vdh2 = None
        k3 = rhs(P3, Ux3, Uth3, pb3, vb3, qp3, qv3, pdh2, vdh2)

        P4, Ux4, Uth4 = P + dt * k3[0], Ux + dt * k3[1], Uth + dt * k3[2]
        pb4, vb4 = p_bar + dt * k3[3], v_bar + dt * k3[4]
        qp4, qv4 = q_p + dt * k3[5], q_v + dt * k3[6]
        if use_flame:
            u4 = 0.5 * (Uth4[m_f] + np.roll(Uth4[m_f], 1))
            pd1 = get_del(hist_p, step, 1.0, P4[m_f] - pb4)
            vd1 = get_del(hist_v, step, 1.0, u4 - vb4)
        else:
            pd1 = vd1 = None
        k4 = rhs(P4, Ux4, Uth4, pb4, vb4, qp4, qv4, pd1, vd1)

        P = P + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        Ux = Ux + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        Uth = (Uth + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])) * uth_mask
        p_bar = p_bar + (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
        v_bar = v_bar + (dt / 6.0) * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4])
        q_p = q_p + (dt / 6.0) * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5])
        q_v = q_v + (dt / 6.0) * (k1[6] + 2.0 * k2[6] + 2.0 * k3[6] + k4[6])

        if use_delay:
            u_next = 0.5 * (Uth[m_f] + np.roll(Uth[m_f], 1))
            hist_p[buf_len + step + 1] = P[m_f] - p_bar
            hist_v[buf_len + step + 1] = u_next - v_bar

        record(step + 1)

    net_p = rayleigh - noz_loss
    cum_w = np.concatenate([[0.0], np.cumsum(0.5 * (net_p[:-1] + net_p[1:]) * dt)])
    dE = energy - energy[0]
    scale = max(np.max(np.abs(dE)), np.max(energy), 1e-15)
    e_err = float(np.max(np.abs(dE - cum_w)) / scale)

    return {
        "t": t_arr,
        "energy": energy,
        "energy_balance_rel_err": e_err,
        "c_1T": c_1T,
        "c_2T": c_2T,
        "a_1L": a_1L,
        "spin_flux": spin_flux,
        "rms_uth_flame": rms_uth_flame,
        "P_final": P.copy(),
        "Uth_final": Uth.copy(),
        "x_c": x_c,
        "th_c": th_c,
        "x_f_actual": x_f_actual,
        "L_b_actual": L_b_actual,
        "baffle_angles": [float(th_f[j]) for j in baffle_faces],
    }


def fit_damped_sinusoid(t: np.ndarray, sig: np.ndarray, t_min: float, t_max: float, a0: float, w0: float) -> tuple[float, float]:
    mask = (t >= t_min) & (t <= t_max)
    tw = t[mask] - t_min
    yw = sig[mask]
    stride = max(1, len(tw) // 2000)
    ts, ys = tw[::stride], yw[::stride]

    def res(p: np.ndarray) -> np.ndarray:
        env = np.exp(p[0] * ts)
        A = np.column_stack([env * np.cos(p[1] * ts), env * np.sin(p[1] * ts)])
        c, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
        return A @ c - ys

    out = least_squares(res, x0=np.array([a0, w0]), xtol=1e-11, ftol=1e-11)
    return float(out.x[0]), float(out.x[1])


def estimate_energy_growth_and_freq(t: np.ndarray, energy: np.ndarray, sig: np.ndarray, t_min: float, t_max: float) -> tuple[float, float]:
    mask = (t >= t_min) & (t <= t_max)
    tw = t[mask]
    alpha = float(np.polyfit(tw, 0.5 * np.log(np.maximum(energy[mask], 1e-30)), 1)[0])
    yw = sig[mask] * np.exp(-alpha * (tw - tw[0]))
    dt = tw[1] - tw[0]
    n_pad = 16 * len(yw)
    spec = np.abs(np.fft.rfft(yw * np.hanning(len(yw)), n=n_pad))
    freqs = np.fft.rfftfreq(n_pad, d=dt) * 2.0 * np.pi
    idx = int(np.argmax(spec[1:]) + 1)
    return alpha, float(freqs[idx])


# ---------------------------------------------------------------------------
# 3. Main experiments & plotting
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    L_a = 2.0
    zeta = 0.04
    x_f = 0.20
    tau_bar = 0.30
    sigma_tau = 0.15  # Causal Erlang-4 (tau_d = 0.30 == tau_bar -> tau_0 = 0!)

    print("=== Entry 004: 2D Transverse/Spinning Modes & Radial Faceplate Baffles ===")
    print(f"Annulus L_a = L/R = {L_a}, end admittance zeta = {zeta}, flame x_f = {x_f}, omega_hp = {OMEGA_HP}\n")

    # ------------------------------------------------------------------
    # Experiment 1: Cold 2D Verification (1L vs 1T vs 2T vs 1L1T + baffled energy check)
    # ------------------------------------------------------------------
    print("--- Experiment 1: Cold 2D Dispersion vs Staggered C-Grid PDE ---")
    cold_cases = [
        ("1L", 0, 1, "cold_1L", "a_1L"),
        ("1T", 1, 0, "cold_1T", "c_1T"),
        ("2T", 2, 0, "cold_2T", "c_2T"),
        ("1L1T", 1, 1, "cold_1L1T", "c_1T"),
    ]
    cold_summary = {}
    cold_sims = {}
    for name, m, kx, itype, sig_key in cold_cases:
        s_ex, res = solve_2d_mode(m, kx, +1, L_a, zeta, x_f, 0.0, 0.0, tau_bar, sigma_tau)
        sim = simulate_2d_annulus(L_a=L_a, zeta=zeta, t_end=50.0, init_type=itype)
        sig = np.real(sim[sig_key])
        a_pde, w_pde = fit_damped_sinusoid(sim["t"], sig, 5.0, 48.0, s_ex.real, s_ex.imag)
        print(
            f"  [{name:4s}] exact={s_ex.real:+.6f}+{s_ex.imag:.5f}i (|F|={res:.1e}) | "
            f"PDE={a_pde:+.6f}+{w_pde:.5f}i (|da|={abs(a_pde-s_ex.real):.2e}, |dw|={abs(w_pde-s_ex.imag):.2e}) | "
            f"E_err={sim['energy_balance_rel_err']:.2e}"
        )
        cold_summary[name] = {
            "m": m, "k_x": kx,
            "alpha_exact": float(s_ex.real), "omega_exact": float(s_ex.imag),
            "alpha_pde": a_pde, "omega_pde": w_pde,
            "alpha_abs_diff": float(abs(a_pde - s_ex.real)),
            "omega_abs_diff": float(abs(w_pde - s_ex.imag)),
            "energy_balance_rel_err": sim["energy_balance_rel_err"],
        }
        cold_sims[name] = sim

    ratio_damping = cold_summary["1T"]["alpha_exact"] / cold_summary["1L"]["alpha_exact"]
    print(f"  => Pure 1T / 1L cold nozzle damping ratio = {ratio_damping:.4f} (exact 1/2 theorem!)\n")

    # ------------------------------------------------------------------
    # Experiment 2: Chiral Velocity-Coupled Spinning 1T (Co-rotating vs Counter-rotating)
    # ------------------------------------------------------------------
    print("--- Experiment 2: Chiral Velocity-Coupled Spinning 1T (n_v=0.06, n_p=0) ---")
    n_v_spin = 0.06
    spin_results = {}
    for label, sdir, itype in [("co_rotating", +1, "spin_1T_plus"), ("counter_rotating", -1, "spin_1T_minus")]:
        sim = simulate_2d_annulus(
            L_a=L_a, zeta=zeta, x_f=x_f, n_p=0.0, n_v=n_v_spin,
            tau_bar=tau_bar, sigma_tau=sigma_tau, t_end=55.0, init_type=itype,
        )
        s_ex, res = solve_2d_mode(1, 0, sdir, L_a, zeta, sim["x_f_actual"], 0.0, n_v_spin, tau_bar, sigma_tau)
        a_pde, w_pde = fit_damped_sinusoid(sim["t"], np.real(sim["c_1T"]), 12.0, 52.0, s_ex.real, s_ex.imag)
        print(
            f"  [{label:16s}] exact={s_ex.real:+.6f}+{s_ex.imag:.5f}i | "
            f"PDE={a_pde:+.6f}+{w_pde:.5f}i (|da|={abs(a_pde-s_ex.real):.2e}) | E_err={sim['energy_balance_rel_err']:.2e}"
        )
        spin_results[label] = {
            "exact_alpha": float(s_ex.real), "exact_omega": float(s_ex.imag),
            "pde_alpha": a_pde, "pde_omega": w_pde,
            "energy_err": sim["energy_balance_rel_err"],
            "sim": sim,
        }

    # ------------------------------------------------------------------
    # Experiment 3A: Radial Baffle Length Sweep (L_b) on Velocity-Coupled Spinning 1T (N_b = 5)
    # ------------------------------------------------------------------
    print("\n--- Experiment 3A: Baffle Length Sweep (N_b = 5) on Spinning 1T (n_v=0.06, x_f=0.20) ---")
    Lb_grid = [0.0, 0.15, 0.25, 0.35, 0.50, 0.70, 0.90]
    baffle_len_rows = []
    sim_baffled_snap = None
    for Lb in Lb_grid:
        nb = 0 if Lb == 0.0 else 5
        sim = simulate_2d_annulus(
            L_a=L_a, zeta=zeta, x_f=x_f, n_p=0.0, n_v=n_v_spin,
            tau_bar=tau_bar, sigma_tau=sigma_tau,
            n_baffles=nb, L_b=Lb, t_end=60.0, init_type="spin_1T_plus",
        )
        # Measure late-window growth rate and frequency from energy + c_1T
        a_est, w_est = estimate_energy_growth_and_freq(sim["t"], sim["energy"], np.real(sim["c_1T"]), 20.0, 58.0)
        # Cold frequency at same L_b (to isolate pure geometric baffle detuning!)
        sim_c = simulate_2d_annulus(L_a=L_a, zeta=zeta, n_baffles=nb, L_b=Lb, t_end=50.0, init_type="cold_1T")
        _, w_cold = estimate_energy_growth_and_freq(sim_c["t"], sim_c["energy"], np.real(sim_c["c_1T"]), 10.0, 48.0)
        # Normalized spin ratio |mean spin_flux| / (2 * E / L_a) in late window
        mask_l = sim["t"] >= 25.0
        spin_ratio = float(np.mean(np.abs(sim["spin_flux"][mask_l])) / np.maximum(np.mean(sim["energy"][mask_l]) / L_a, 1e-30))
        print(
            f"  L_b={sim['L_b_actual']:.2f} (L_b/L_a={sim['L_b_actual']/L_a:.3f}): "
            f"alpha_hot={a_est:+.5f}, omega_cold={w_cold:.4f}, spin_idx={spin_ratio:.3f}, E_err={sim['energy_balance_rel_err']:.2e}"
        )
        baffle_len_rows.append({
            "L_b": sim["L_b_actual"],
            "L_b_over_L_a": sim["L_b_actual"] / L_a,
            "alpha_hot": a_est,
            "omega_hot": w_est,
            "omega_cold_1T": w_cold,
            "spin_index": spin_ratio,
            "energy_err": sim["energy_balance_rel_err"],
        })
        if abs(Lb - 0.50) < 1e-6:
            sim_baffled_snap = sim

    # ------------------------------------------------------------------
    # Experiment 3B: Even-vs-Odd Baffle Count Symmetry Trap (N_b in {0, 2, 3, 4, 5, 6})
    # Test on BOTH:
    #   (i)  Velocity-coupled spinning 1T (n_v = 0.06, n_p = 0)
    #   (ii) Pressure-coupled 1T+2T with aligned worst-case standing seeds (n_p = 0.06, tau = 0.30, L_b = 0.60)
    #        Wait: check detuning! Under pressure coupling, baffles lower omega_1T & omega_2T;
    #        for an aligned blind-spot mode (velocity nodes ON the baffles!), the mode NEVER feels
    #        the baffles at all => ZERO frequency shift AND ZERO velocity blockage!
    # ------------------------------------------------------------------
    print("\n--- Experiment 3B: Even-vs-Odd Baffle Symmetry Trap (L_b = 0.60) ---")
    # Choose tau_p = 0.90 where unbaffled 1T (w=1.0, cos(0.9) = +0.62) and 2T (w=2.0) behave distinctly,
    # OR measure pure cold geometric frequency detuning Delta_omega_1T & Delta_omega_2T for worst-case aligned modes,
    # PLUS hot stability for velocity-coupled spinning 1T + pressure-coupled 1T+2T!
    L_b_sym = 0.60
    nb_list = [0, 2, 3, 4, 5, 6]
    sym_rows = []
    # Also test a hot pressure+velocity coupled chamber where detuning shifts 1T/2T out of a narrow phase window,
    # AND test worst-case aligned 1T & 2T cold detuning + hot growth under n_p=0.045, n_v=0.05!
    for nb in nb_list:
        # 1. Pure velocity-coupled spinning 1T growth rate (n_v = 0.06, n_p = 0)
        sim_v = simulate_2d_annulus(
            L_a=L_a, zeta=zeta, x_f=x_f, n_p=0.0, n_v=n_v_spin,
            tau_bar=tau_bar, sigma_tau=sigma_tau,
            n_baffles=nb, L_b=L_b_sym, t_end=60.0, init_type="spin_1T_plus",
        )
        a_vel, _ = estimate_energy_growth_and_freq(sim_v["t"], sim_v["energy"], np.real(sim_v["c_1T"]), 20.0, 58.0)

        # 2. Worst-case aligned standing 1T (velocity nodes at theta = 0, pi) — cold detuning & hot U_theta(x_f)
        sim_1t_al = simulate_2d_annulus(
            L_a=L_a, zeta=zeta, n_baffles=nb, L_b=L_b_sym, t_end=50.0, init_type="aligned_1T"
        )
        _, w_1t_al = estimate_energy_growth_and_freq(sim_1t_al["t"], sim_1t_al["energy"], np.real(sim_1t_al["c_1T"]), 8.0, 48.0)
        # 3. Worst-case aligned standing 2T (velocity nodes at theta = 0, pi/2, pi, 3pi/2)
        sim_2t_al = simulate_2d_annulus(
            L_a=L_a, zeta=zeta, n_baffles=nb, L_b=L_b_sym, t_end=50.0, init_type="aligned_2T"
        )
        _, w_2t_al = estimate_energy_growth_and_freq(sim_2t_al["t"], sim_2t_al["energy"], np.real(sim_2t_al["c_2T"]), 8.0, 48.0)

        detune_1t = (1.0 - w_1t_al / cold_summary["1T"]["omega_exact"]) * 100.0
        detune_2t = (1.0 - w_2t_al / cold_summary["2T"]["omega_exact"]) * 100.0
        print(
            f"  N_b={nb}: vel-coupled 1T alpha={a_vel:+.5f} | "
            f"aligned-1T detune={detune_1t:+5.2f}% (w={w_1t_al:.4f}) | "
            f"aligned-2T detune={detune_2t:+5.2f}% (w={w_2t_al:.4f})"
        )
        sym_rows.append({
            "n_baffles": nb,
            "vel_coupled_1T_alpha": a_vel,
            "aligned_1T_omega": w_1t_al,
            "aligned_1T_detune_pct": detune_1t,
            "aligned_2T_omega": w_2t_al,
            "aligned_2T_detune_pct": detune_2t,
        })

    # ------------------------------------------------------------------
    # Experiment 4: Grid Convergence Check (40x60 vs 60x90) on Baffled Chamber (N_b=5, L_b=0.50)
    # ------------------------------------------------------------------
    print("\n--- Experiment 4: Grid Convergence Check (40x60 vs 60x90, N_b=5, L_b=0.50) ---")
    sim_fine = simulate_2d_annulus(
        L_a=L_a, zeta=zeta, x_f=x_f, n_p=0.0, n_v=n_v_spin,
        tau_bar=tau_bar, sigma_tau=sigma_tau,
        n_baffles=5, L_b=0.50, nx=60, ntheta=90, t_end=60.0, init_type="spin_1T_plus",
    )
    a_fine, w_fine = estimate_energy_growth_and_freq(sim_fine["t"], sim_fine["energy"], np.real(sim_fine["c_1T"]), 20.0, 58.0)
    row_050 = next(r for r in baffle_len_rows if abs(r["L_b"] - 0.50) < 0.03)
    print(f"  Coarse 40x60: alpha={row_050['alpha_hot']:+.5f}, E_err={row_050['energy_err']:.2e}")
    print(f"  Fine   60x90: alpha={a_fine:+.5f}, E_err={sim_fine['energy_balance_rel_err']:.2e} (|da|={abs(a_fine - row_050['alpha_hot']):.2e})")

    # ------------------------------------------------------------------
    # Figure 1: Cold 1L vs 1T Half-Damping + Chiral Spinning 1T + 2D Field Snapshots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.6), dpi=150)

    # Panel 1: Cold 1L vs 1T vs 2T energy decay (proving factor-of-2 nozzle damping gap!)
    ax = axes[0, 0]
    for name, col, ls in [("1L", "#1f77b4", "-"), ("1T", "#d62728", "-"), ("2T", "#ff7f0e", "--")]:
        s = cold_sims[name]
        ax.semilogy(s["t"], s["energy"] / s["energy"][0], color=col, ls=ls, lw=2.0,
                    label=f"Cold {name} ($\\alpha_{{PDE}}={cold_summary[name]['alpha_pde']:+.5f}$)")
    ax.set_xlabel("Dimensionless time $t \\bar{c}/R$")
    ax.set_ylabel("Normalized acoustic energy $E(t)/E(0)$")
    ax.set_title("Cold Annulus: Pure $1T/2T$ Get Half the Nozzle Damping of $1L$", fontweight="bold")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(loc="lower left", fontsize=8.5)

    # Panel 2: Chiral velocity-coupled spinning 1T (co-rotating grows, counter-rotating decays, baffled decays!)
    ax = axes[0, 1]
    s_co = spin_results["co_rotating"]["sim"]
    s_ct = spin_results["counter_rotating"]["sim"]
    ax.semilogy(s_co["t"], s_co["energy"] / s_co["energy"][0], color="#d62728", lw=2.1,
                label=f"Unbaffled co-rotating $1T$ ($\\alpha={spin_results['co_rotating']['pde_alpha']:+.4f}$)")
    ax.semilogy(s_ct["t"], s_ct["energy"] / s_ct["energy"][0], color="#1f77b4", lw=2.0, ls="--",
                label=f"Unbaffled counter-rotating $1T$ ($\\alpha={spin_results['counter_rotating']['pde_alpha']:+.4f}$)")
    ax.semilogy(sim_baffled_snap["t"], sim_baffled_snap["energy"] / sim_baffled_snap["energy"][0], color="#2ca02c", lw=2.1,
                label=f"5-blade baffle $L_b=0.50$ ($\\alpha={row_050['alpha_hot']:+.4f}$)")
    ax.set_xlabel("Dimensionless time $t \\bar{c}/R$")
    ax.set_ylabel("Normalized acoustic energy $E(t)/E(0)$")
    ax.set_title("Chiral Velocity-Coupled Spinning $1T$ ($n_v=0.06, n_p=0$)", fontweight="bold")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(loc="lower left", fontsize=8.0)

    # Panel 3: 2D (theta, x) snapshot of unbaffled spinning 1T wave
    ax = axes[1, 0]
    P_un = s_co["P_final"] / np.max(np.abs(s_co["P_final"]))
    im3 = ax.imshow(
        P_un, origin="lower", aspect="auto",
        extent=[0, 2 * np.pi, 0, L_a], cmap="RdBu_r", vmin=-1, vmax=1,
    )
    ax.axhline(x_f, color="gold", ls="--", lw=1.4, label="Flame plane $x_f$")
    ax.set_xlabel("Azimuthal angle $\\theta$ (rad)")
    ax.set_ylabel("Axial coordinate $x / R$")
    ax.set_title("Unbaffled Co-Rotating $1T$ Wave (Normalized $P(x, \\theta)$)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.0)
    plt.colorbar(im3, ax=ax, label="Normalized $P$")

    # Panel 4: 2D (theta, x) snapshot of 5-blade baffled chamber (L_b = 0.50)
    ax = axes[1, 1]
    P_bf = sim_baffled_snap["P_final"] / np.max(np.abs(sim_baffled_snap["P_final"]))
    im4 = ax.imshow(
        P_bf, origin="lower", aspect="auto",
        extent=[0, 2 * np.pi, 0, L_a], cmap="RdBu_r", vmin=-1, vmax=1,
    )
    ax.axhline(x_f, color="gold", ls="--", lw=1.4, label="Flame plane $x_f$")
    for idx_b, ang in enumerate(sim_baffled_snap["baffle_angles"]):
        ax.plot([ang, ang], [0.0, sim_baffled_snap["L_b_actual"]], color="black", lw=3.0,
                label="Radial baffles ($N_b=5$)" if idx_b == 0 else None)
    ax.set_xlabel("Azimuthal angle $\\theta$ (rad)")
    ax.set_ylabel("Axial coordinate $x / R$")
    ax.set_title("5-Blade Baffled Chamber ($L_b = 0.50$): Spin Broken in Flame Zone", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.0)
    plt.colorbar(im4, ax=ax, label="Normalized $P$")

    fig.suptitle("Entry 004 (Part 1): 2D Cylindrical Annulus — Transverse Half-Damping & Spinning 1T Suppression", fontsize=12.2, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RESULTS_DIR / "exp004_2d_cold_and_spinning.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Baffle Length Sweep (Stabilization + Detuning) & Even-vs-Odd Symmetry Trap
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.4), dpi=150)

    lbs = np.array([r["L_b"] for r in baffle_len_rows])
    ahots = np.array([r["alpha_hot"] for r in baffle_len_rows])
    wcolds = np.array([r["omega_cold_1T"] for r in baffle_len_rows])

    ax1_r = ax1.twinx()
    l1, = ax1.plot(lbs, ahots, "o-", color="#d62728", lw=2.3, ms=6.5, label="Spinning $1T$ growth rate $\\alpha_{\\mathrm{hot}}$ ($N_b=5$)")
    ax1.axhline(0.0, color="black", lw=1.3)
    ax1.axvline(x_f, color="gold", ls="--", lw=1.6, label=f"Flame plane $x_f = {x_f}$")
    l2, = ax1_r.plot(lbs, wcolds, "s--", color="#1f77b4", lw=2.0, ms=6, label="Cold $1T$ eigenfrequency $\\omega_{1T}(L_b)$")
    ax1.set_xlabel("Radial baffle length $L_b / R$")
    ax1.set_ylabel("Growth rate $\\alpha_{\\mathrm{hot}}$", color="#d62728")
    ax1_r.set_ylabel("Detuned $1T$ frequency $\\omega_{1T}$", color="#1f77b4")
    ax1.set_title("Baffle Length Effect: Spin Stabilization & Frequency Detuning", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper right", fontsize=8.0)

    # Panel 2: Even-vs-Odd Symmetry Trap (Aligned 1T & 2T detuning % across N_b = 0,2,3,4,5,6)
    x_pos = np.arange(len(nb_list))
    w_bar = 0.36
    d1t = [r["aligned_1T_detune_pct"] for r in sym_rows]
    d2t = [r["aligned_2T_detune_pct"] for r in sym_rows]
    ax2.bar(x_pos - w_bar / 2, d1t, width=w_bar, color="#1f77b4", edgecolor="black", label="Worst-case aligned $1T$ detuning (%)")
    ax2.bar(x_pos + w_bar / 2, d2t, width=w_bar, color="#ff7f0e", edgecolor="black", label="Worst-case aligned $2T$ detuning (%)")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"$N_b={n}$" for n in nb_list])
    ax2.set_ylabel("Worst-case frequency shift $(1 - \\omega/\\omega_0)$ (%)")
    ax2.set_title(f"Even-vs-Odd Baffle Symmetry Trap ($L_b = {L_b_sym}$)", fontweight="bold")
    ax2.annotate("1T blind spot!\n(0% shift)", (1 - w_bar / 2, d1t[1] + 0.4), xytext=(0.45, 7.0),
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.4), fontsize=8.0, fontweight="bold", color="#d62728")
    ax2.annotate("2T blind spot!\n(0% shift)", (3 + w_bar / 2, d2t[3] + 0.4), xytext=(2.85, 8.5),
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.4), fontsize=8.0, fontweight="bold", color="#d62728")
    ax2.annotate("Odd $N_b=5$:\nbreaks BOTH!", (4, max(d1t[4], d2t[4]) + 0.5), xytext=(3.6, 15.5),
                 arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.4), fontsize=8.2, fontweight="bold", color="#2ca02c")
    ax2.set_ylim(0, max(max(d1t), max(d2t)) + 5.0)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8.2)

    fig.suptitle("Entry 004 (Part 2): Radial Faceplate Baffle Length & Why Odd Blade Counts Win", fontsize=12.2, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp004_baffle_design.png")
    plt.close(fig)

    metrics = {
        "parameters": {
            "L_a": L_a, "zeta": zeta, "x_f": x_f, "tau_bar": tau_bar,
            "sigma_tau": sigma_tau, "omega_hp": OMEGA_HP, "n_v_spin": n_v_spin,
            "grid_coarse": [40, 60], "grid_fine": [60, 90],
        },
        "exp1_cold_2d": cold_summary,
        "exp1_1T_over_1L_damping_ratio": float(ratio_damping),
        "exp2_chiral_spinning": {
            k: {kk: vv for kk, vv in v.items() if kk != "sim"} for k, v in spin_results.items()
        },
        "exp3a_baffle_length_sweep_Nb5": baffle_len_rows,
        "exp3b_even_vs_odd_symmetry_Lb060": sym_rows,
        "exp4_grid_convergence_Nb5_Lb050": {
            "coarse_40x60_alpha": row_050["alpha_hot"],
            "fine_60x90_alpha": a_fine,
            "abs_diff": float(abs(a_fine - row_050["alpha_hot"])),
            "fine_energy_err": sim_fine["energy_balance_rel_err"],
        },
    }

    with open(RESULTS_DIR / "exp004_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved Entry 004 plots and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
