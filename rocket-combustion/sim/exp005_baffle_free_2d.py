"""Entry 005: Baffle-Free 2D Transverse Stabilization — Asymmetric Cavity Rings vs. Azimuthal Lag Staggering.

Extends our verified 2D cylindrical-annulus staggered C-grid PDE (Entry 004: L_a = 2.0,
zeta = 0.04, DC-blocked omega_hp = 0.15, causal Erlang-4) with two baffle-free stabilizers
that avoid protruding high-heat-load injector baffles:

Strategy A — Faceplate Helmholtz Acoustic Cavity Ring at x = 0:
    U_x(0, theta, t) = -v_r(theta, t)
    dv_r/dt + 2*xi_H(theta)*omega_H(theta)*v_r = beta_s(theta)*omega_H(theta)^2 * (P(0, theta, t) - P_c)
    dP_c/dt = v_r / beta_s(theta)
where beta_s(theta) = beta_local(theta) * L_a so total cavity-to-chamber volume ratio is
    beta_tot = (1 / (2*pi*L_a)) * int_0^{2*pi} beta_s(theta) dtheta.

Exact 2D total energy balance (chamber + cavity ring):
    E_tot = E_chamber + 0.5 * int [ beta_s*P_c^2 + v_r^2 / (beta_s*omega_H^2) ] dtheta
    dE_tot/dt = R_flame - D_nozzle - int [ (2*xi_H / (beta_s*omega_H)) * v_r^2 ] dtheta

Exact 2D 1T Exceptional-Point Law (uniform cavity ring, beta_tot):
    xi_H^(EP, 1T)   = sqrt(beta_tot) - alpha_1T^uncoupled
    alpha_1T^(EP,0) = -zeta / (2*L_a) - 0.5 * sqrt(beta_tot)
(Notice sqrt(beta_tot) vs sqrt(2*beta_tot) in 1D 1L — exact consequence of the 1/2 pressure-energy ratio!)

Strategy B — Additive-Manufactured Non-Axisymmetric Injector Lag Staggering tau(theta):
    Discrete alternating sectors tau(theta) = tau_bar +/- Delta_tau (M = 2, 3, 5 periods around rim),
    zero added cavity volume and zero protruding hardware!
    Analytical phase-cancellation factor for non-resonant M: cos(omega * Delta_tau),
    while Bragg-resonant M = 2m creates a standing-mode orientation split (one stabilized, one destabilized!).

Experiments:
1. Verify 2D 1T Helmholtz cavity exceptional-point law (exact 2D dispersion vs 2D staggered PDE
   with exact total energy balance check) on cold and hot spinning 1T.
2. Dual-tuned azimuthally alternating faceplate cavity ring (fixed beta_tot = 2%, odd N_pair = 5)
   sweeping 1T/2T volume split f_1 in [0, 1] on a combined unstable 1T + 2T combustor.
3. Azimuthal injector time-lag staggering (M = 2, 3, 5, Delta_tau sweep): verify phase-cancellation
   stabilization AND the M = 2 standing-orientation trap vs odd M = 3, 5.
4. Grid convergence check (40x60 vs 60x90) on the optimal dual-tuned cavity ring.
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
OMEGA_HP = 0.15


# ---------------------------------------------------------------------------
# 1. Exact 2D dispersion with uniform faceplate Helmholtz cavity ring
# ---------------------------------------------------------------------------

def flame_tf(s: complex, tau_bar: float, sigma_tau: float, m_stages: int = 4) -> complex:
    hp = s / (s + OMEGA_HP)
    tau_d = min(tau_bar, np.sqrt(m_stages) * sigma_tau) if sigma_tau > 0.0 else 0.0
    tau_0 = max(0.0, tau_bar - tau_d)
    g = np.exp(-s * tau_0) / ((1.0 + s * (tau_d / m_stages)) ** m_stages) if tau_d > 0.0 else np.exp(-s * tau_bar)
    return hp * g


def sinhc(z: complex) -> complex:
    """Analytic sinh(z)/z with smooth removable limit sinhc(0) = 1."""
    if abs(z) < 1e-6:
        z2 = z * z
        return 1.0 + z2 / 6.0 + (z2 * z2) / 120.0
    return np.sinh(z) / z


def dispersion_2d_cavity_F(
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
    beta_tot: float,
    omega_H: float,
    xi_H: float,
    m_stages: int = 4,
) -> complex:
    """Exact singularity-free 2D unbaffled dispersion with uniform faceplate cavity ring.

    Uses analytic sinhc(z) = sinh(z)/z so kap = sqrt(s^2 + m^2) = 0 NEVER creates a spurious zero:
        [(kap^2 * denom + zeta * num_Y * s^2) * L_a * sinhc(kap * L_a)
         + (zeta * denom + num_Y) * s * cosh(kap * L_a)]
      = n_eff * s * [denom * cosh(kap*x_f) + num_Y * s * x_f * sinhc(kap*x_f)]
                  * [cosh(kap*(L_a-x_f)) + zeta * s * (L_a-x_f) * sinhc(kap*(L_a-x_f))]
    """
    beta_s = beta_tot * L_a
    denom = s * s + 2.0 * xi_H * omega_H * s + omega_H * omega_H
    num_Y = beta_s * (omega_H ** 2) * s if beta_tot > 0.0 else 0.0j

    kap2 = s * s + float(m * m)
    kap = np.sqrt(kap2)
    d1 = x_f
    d2 = L_a - x_f

    shc_L = sinhc(kap * L_a)
    ch_L = np.cosh(kap * L_a)
    lhs = (denom * kap2 + zeta * num_Y * s * s) * L_a * shc_L + (zeta * denom + num_Y) * s * ch_L
    if n_p == 0.0 and n_v == 0.0:
        return lhs

    tf = flame_tf(s, tau_bar, sigma_tau, m_stages)
    vel_factor = (1j * spin_dir * m / s) if (m != 0 and abs(s) > 1e-12) else 0.0j
    n_eff = (n_p + n_v * vel_factor) * tf

    p1f = denom * np.cosh(kap * d1) + num_Y * s * d1 * sinhc(kap * d1)
    p2f = np.cosh(kap * d2) + zeta * s * d2 * sinhc(kap * d2)
    return lhs - n_eff * s * p1f * p2f


def polish_2d_cavity_root(s0: complex, *args) -> tuple[complex, float]:
    s = complex(s0)
    for _ in range(45):
        fv = dispersion_2d_cavity_F(s, *args)
        if abs(fv) < 1e-12:
            return s, float(abs(fv))
        eps = 1e-7 * max(1.0, abs(s))
        dfv = (dispersion_2d_cavity_F(s + eps, *args) - dispersion_2d_cavity_F(s - eps, *args)) / (2.0 * eps)
        if abs(dfv) < 1e-16:
            break
        ds = fv / dfv
        if abs(ds) > 0.25:
            ds = 0.25 * ds / abs(ds)
        s -= ds
        if abs(s.real) > 3.0 or abs(s.imag) > 15.0:
            return s, 1e9
    return s, float(abs(dispersion_2d_cavity_F(s, *args)))


def find_coupled_1T_pair(
    L_a: float, zeta: float, x_f: float, n_p: float, n_v: float,
    tau_bar: float, sigma_tau: float, beta_tot: float, omega_H: float, xi_H: float, spin_dir: int = +1
) -> list[complex]:
    args = (1, spin_dir, L_a, zeta, x_f, n_p, n_v, tau_bar, sigma_tau, beta_tot, omega_H, xi_H)
    if beta_tot <= 0.0:
        r, _ = polish_2d_cavity_root(complex(0.0, 1.0), *args)
        return [r]
    split = 0.5 * np.sqrt(max(1e-5, beta_tot))
    seeds = [
        complex(-0.02, 1.0 - split), complex(-0.02, 1.0 + split),
        complex(-0.06, 1.0 - 0.5 * split), complex(-0.06, 1.0 + 0.5 * split),
        complex(-0.10, 1.0), complex(-xi_H * omega_H, omega_H * np.sqrt(max(0.1, 1.0 - xi_H * xi_H))),
        complex(0.02, 1.0 - 0.15), complex(0.02, 1.0 + 0.15),
        complex(-0.25, omega_H),
    ]
    roots = []
    for sd in seeds:
        r, res = polish_2d_cavity_root(sd, *args)
        if res < 1e-8 and 0.4 <= r.imag <= 1.6 and -2.0 <= r.real <= 0.5:
            if all(abs(r - e) > 1e-4 for e in roots):
                roots.append(r)
    roots.sort(key=lambda z: z.imag)
    return roots


# ---------------------------------------------------------------------------
# 2. Vectorized 2D C-grid PDE with azimuthally patterned cavities AND tau(theta)
# ---------------------------------------------------------------------------

def simulate_2d_baffle_free(
    L_a: float = 2.0,
    zeta: float = 0.04,
    x_f: float = 0.20,
    n_p: float = 0.0,
    n_v: float = 0.0,
    tau_profile: np.ndarray | float = 0.30,
    sigma_tau: float = 0.15,
    m_stages: int = 4,
    # Cavity ring arrays (length ntheta); beta_s = beta_local * L_a
    beta_local: np.ndarray | float = 0.0,
    omega_H_arr: np.ndarray | float = 1.0,
    xi_H_arr: np.ndarray | float = 0.15,
    nx: int = 40,
    ntheta: int = 60,
    cfl: float = 0.35,
    t_end: float = 65.0,
    init_type: str = "spin_1T_plus",
    init_amp: float = 1e-3,
    init_phase: float = 0.0,
) -> dict:
    """2D annular combustor PDE with azimuthally varying faceplate cavities and/or tau(theta).

    Uses pure m-stage Erlang cascades at every azimuthal station j when sigma_tau is capped
    causally (tau_d[j] = min(tau[j], sqrt(m)*sigma_tau)), plus exact per-station delay buffer
    when tau_0[j] = tau[j] - tau_d[j] > 0.
    """
    dx = L_a / nx
    dth = (2.0 * np.pi) / ntheta
    inv_dx = 1.0 / dx
    inv_dth = 1.0 / dth

    dt_max = cfl / np.sqrt(inv_dx * inv_dx + inv_dth * inv_dth)
    n_steps = int(round(t_end / dt_max))
    dt = t_end / n_steps

    x_c = (np.arange(nx) + 0.5) * dx
    th_c = (np.arange(ntheta) + 0.5) * dth
    th_f = (np.arange(ntheta) + 1.0) * dth
    m_f = int(np.argmin(np.abs(x_c - x_f)))
    x_f_actual = float(x_c[m_f])

    # Expand per-angle arrays
    tau_arr = np.full(ntheta, float(tau_profile)) if np.ndim(tau_profile) == 0 else np.asarray(tau_profile, dtype=float)
    beta_arr = np.full(ntheta, float(beta_local)) if np.ndim(beta_local) == 0 else np.asarray(beta_local, dtype=float)
    wH_arr = np.full(ntheta, float(omega_H_arr)) if np.ndim(omega_H_arr) == 0 else np.asarray(omega_H_arr, dtype=float)
    xiH_arr = np.full(ntheta, float(xi_H_arr)) if np.ndim(xi_H_arr) == 0 else np.asarray(xi_H_arr, dtype=float)

    beta_s = beta_arr * L_a  # surface compliance per radian
    has_cav = bool(np.any(beta_s > 0.0))
    cav_mask = beta_s > 0.0
    inv_beta_s = np.where(cav_mask, 1.0 / np.maximum(beta_s, 1e-30), 0.0)
    cav_k_vr = beta_s * (wH_arr ** 2)
    cav_damp = 2.0 * xiH_arr * wH_arr

    # Initial condition
    X, TH = np.meshgrid(x_c, th_c, indexing="ij")
    P = np.zeros((nx, ntheta))
    Ux = np.zeros((nx - 1, ntheta))
    Uth = np.zeros((nx, ntheta))

    if init_type == "cold_1T":
        P = init_amp * np.cos(TH - init_phase)
    elif init_type == "spin_1T_plus":
        P = init_amp * np.cos(TH - init_phase)
        Uth = init_amp * np.cos(th_f[None, :] - init_phase) * np.ones((nx, 1))
    elif init_type == "spin_2T_plus":
        P = init_amp * np.cos(2.0 * (TH - init_phase))
        Uth = init_amp * np.cos(2.0 * (th_f[None, :] - init_phase)) * np.ones((nx, 1))
    elif init_type == "standing_1T":
        P = init_amp * np.cos(TH - init_phase)
    elif init_type == "standing_2T":
        P = init_amp * np.cos(2.0 * (TH - init_phase))
    elif init_type == "combined_1T_2T":
        # Seed BOTH co-rotating 1T and co-rotating 2T + slight standing/1L perturbation
        P = init_amp * (np.cos(TH - init_phase) + 0.8 * np.cos(2.0 * TH - 0.7) + 0.3 * np.cos(np.pi * X / L_a))
        Uth = init_amp * (np.cos(th_f[None, :] - init_phase) + 0.8 * np.cos(2.0 * th_f[None, :] - 0.7)) * np.ones((nx, 1))
    else:
        raise ValueError(f"Unknown init_type: {init_type}")

    # Consistent cavity pre-pressurization at t = 0
    v_r = np.zeros(ntheta)
    p_c = np.where(cav_mask, P[0], 0.0)

    # Flame filtering setup (supports per-angle tau_arr[j]!)
    use_flame = (n_p != 0.0) or (n_v != 0.0)
    tau_d = np.minimum(tau_arr, np.sqrt(m_stages) * sigma_tau) if sigma_tau > 0.0 else np.zeros(ntheta)
    tau_0 = np.maximum(0.0, tau_arr - tau_d)
    rate = np.where(tau_d > 0.0, m_stages / np.maximum(tau_d, 1e-30), 0.0)
    has_erlang = bool(np.any(tau_d > 0.0))

    p_bar = np.zeros(ntheta)
    v_bar = np.zeros(ntheta)
    q_p = np.zeros((m_stages, ntheta))
    q_v = np.zeros((m_stages, ntheta))

    use_delay = use_flame and bool(np.any(tau_0 > 1e-12))
    delay_steps_arr = tau_0 / dt
    buf_len = int(np.ceil(float(np.max(delay_steps_arr)))) + 4 if use_delay else 2
    j_idx = np.arange(ntheta)
    if use_delay:
        hist_p = np.zeros((n_steps + buf_len + 2, ntheta))
        hist_v = np.zeros((n_steps + buf_len + 2, ntheta))
        u0_init = 0.5 * (Uth[m_f] + np.roll(Uth[m_f], 1))
        hist_p[buf_len] = P[m_f]
        hist_v[buf_len] = u0_init
    else:
        hist_p = hist_v = None

    def get_del_vec(hist_arr, step_idx: int, frac: float, cur_val: np.ndarray) -> np.ndarray:
        if not use_delay:
            return cur_val
        idx_f = buf_len + step_idx + frac - delay_steps_arr
        i0 = np.floor(idx_f).astype(int)
        a = idx_f - i0
        i0_c = np.clip(i0, 0, buf_len + step_idx)
        i1_c = np.clip(i0 + 1, 0, buf_len + step_idx)
        val = (1.0 - a) * hist_arr[i0_c, j_idx] + a * hist_arr[i1_c, j_idx]
        val = np.where(i0 < 0, 0.0, val)
        val = np.where(tau_0 <= 1e-12, cur_val, val)
        val = np.where(i0 >= buf_len + step_idx, cur_val, val)
        return val

    def rhs(P_c, Ux_c, Uth_c, vr_c, pc_c, pb_c, vb_c, qp_c, qv_c, p_del, v_del):
        dP = np.empty_like(P_c)
        dP[0] = -(Ux_c[0] + vr_c) * inv_dx
        dP[1:-1] = -(Ux_c[1:] - Ux_c[:-1]) * inv_dx
        dP[-1] = -(zeta * P_c[-1] - Ux_c[-1]) * inv_dx
        dP -= (Uth_c - np.roll(Uth_c, 1, axis=1)) * inv_dth

        if use_flame:
            src_p = np.where(tau_d > 0.0, qp_c[-1], p_del)
            src_v = np.where(tau_d > 0.0, qv_c[-1], v_del)
            dP[m_f] += (n_p * src_p + n_v * src_v) * inv_dx

        dUx = -(P_c[1:] - P_c[:-1]) * inv_dx
        dUth = -(np.roll(P_c, -1, axis=1) - P_c) * inv_dth

        if has_cav:
            dvr = np.where(cav_mask, cav_k_vr * (P_c[0] - pc_c) - cav_damp * vr_c, 0.0)
            dpc = vr_c * inv_beta_s
        else:
            dvr = np.zeros(ntheta)
            dpc = np.zeros(ntheta)

        if use_flame:
            u_cell = 0.5 * (Uth_c[m_f] + np.roll(Uth_c[m_f], 1))
            dpb = OMEGA_HP * (P_c[m_f] - pb_c)
            dvb = OMEGA_HP * (u_cell - vb_c)
            if has_erlang:
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
            dpb = np.zeros(ntheta)
            dvb = np.zeros(ntheta)
            dqp = np.zeros_like(qp_c)
            dqv = np.zeros_like(qv_c)

        return dP, dUx, dUth, dvr, dpc, dpb, dvb, dqp, dqv

    t_arr = np.linspace(0.0, t_end, n_steps + 1)
    energy_tot = np.empty(n_steps + 1)
    rayleigh = np.empty(n_steps + 1)
    loss_noz = np.empty(n_steps + 1)
    loss_cav = np.empty(n_steps + 1)
    c_1T = np.empty(n_steps + 1, dtype=complex)
    c_2T = np.empty(n_steps + 1, dtype=complex)

    exp_m1 = np.exp(-1j * th_c)
    exp_m2 = np.exp(-2j * th_c)
    cav_e_vr_coeff = np.where(cav_mask, 1.0 / np.maximum(beta_s * (wH_arr ** 2), 1e-30), 0.0)
    cav_loss_coeff = np.where(cav_mask, (2.0 * xiH_arr) / np.maximum(beta_s * wH_arr, 1e-30), 0.0)

    def record(idx: int) -> None:
        e_ch = 0.5 * dx * dth * (float(np.sum(P * P)) + float(np.sum(Ux * Ux)) + float(np.sum(Uth * Uth)))
        e_cav = 0.5 * dth * float(np.sum(beta_s * p_c * p_c + cav_e_vr_coeff * v_r * v_r)) if has_cav else 0.0
        energy_tot[idx] = e_ch + e_cav
        loss_noz[idx] = zeta * dth * float(np.sum(P[-1] * P[-1]))
        loss_cav[idx] = dth * float(np.sum(cav_loss_coeff * v_r * v_r)) if has_cav else 0.0
        if use_flame:
            u_c = 0.5 * (Uth[m_f] + np.roll(Uth[m_f], 1))
            src_p = np.where(tau_d > 0.0, q_p[-1], get_del_vec(hist_p, idx, 0.0, P[m_f] - p_bar))
            src_v = np.where(tau_d > 0.0, q_v[-1], get_del_vec(hist_v, idx, 0.0, u_c - v_bar))
            rayleigh[idx] = dth * float(np.dot(P[m_f], n_p * src_p + n_v * src_v))
        else:
            rayleigh[idx] = 0.0
        c_1T[idx] = (dth / np.pi) * np.dot(P[0], exp_m1)
        c_2T[idx] = (dth / np.pi) * np.dot(P[0], exp_m2)

    record(0)

    for step in range(n_steps):
        if use_flame:
            u0 = 0.5 * (Uth[m_f] + np.roll(Uth[m_f], 1))
            pd0 = get_del_vec(hist_p, step, 0.0, P[m_f] - p_bar)
            vd0 = get_del_vec(hist_v, step, 0.0, u0 - v_bar)
        else:
            pd0 = vd0 = None

        k1 = rhs(P, Ux, Uth, v_r, p_c, p_bar, v_bar, q_p, q_v, pd0, vd0)

        P2, Ux2, Uth2 = P + 0.5 * dt * k1[0], Ux + 0.5 * dt * k1[1], Uth + 0.5 * dt * k1[2]
        vr2, pc2 = v_r + 0.5 * dt * k1[3], p_c + 0.5 * dt * k1[4]
        pb2, vb2 = p_bar + 0.5 * dt * k1[5], v_bar + 0.5 * dt * k1[6]
        qp2, qv2 = q_p + 0.5 * dt * k1[7], q_v + 0.5 * dt * k1[8]
        if use_flame:
            u2 = 0.5 * (Uth2[m_f] + np.roll(Uth2[m_f], 1))
            pdh1 = get_del_vec(hist_p, step, 0.5, P2[m_f] - pb2)
            vdh1 = get_del_vec(hist_v, step, 0.5, u2 - vb2)
        else:
            pdh1 = vdh1 = None
        k2 = rhs(P2, Ux2, Uth2, vr2, pc2, pb2, vb2, qp2, qv2, pdh1, vdh1)

        P3, Ux3, Uth3 = P + 0.5 * dt * k2[0], Ux + 0.5 * dt * k2[1], Uth + 0.5 * dt * k2[2]
        vr3, pc3 = v_r + 0.5 * dt * k2[3], p_c + 0.5 * dt * k2[4]
        pb3, vb3 = p_bar + 0.5 * dt * k2[5], v_bar + 0.5 * dt * k2[6]
        qp3, qv3 = q_p + 0.5 * dt * k2[7], q_v + 0.5 * dt * k2[8]
        if use_flame:
            u3 = 0.5 * (Uth3[m_f] + np.roll(Uth3[m_f], 1))
            pdh2 = get_del_vec(hist_p, step, 0.5, P3[m_f] - pb3)
            vdh2 = get_del_vec(hist_v, step, 0.5, u3 - vb3)
        else:
            pdh2 = vdh2 = None
        k3 = rhs(P3, Ux3, Uth3, vr3, pc3, pb3, vb3, qp3, qv3, pdh2, vdh2)

        P4, Ux4, Uth4 = P + dt * k3[0], Ux + dt * k3[1], Uth + dt * k3[2]
        vr4, pc4 = v_r + dt * k3[3], p_c + dt * k3[4]
        pb4, vb4 = p_bar + dt * k3[5], v_bar + dt * k3[6]
        qp4, qv4 = q_p + dt * k3[7], q_v + dt * k3[8]
        if use_flame:
            u4 = 0.5 * (Uth4[m_f] + np.roll(Uth4[m_f], 1))
            pd1 = get_del_vec(hist_p, step, 1.0, P4[m_f] - pb4)
            vd1 = get_del_vec(hist_v, step, 1.0, u4 - vb4)
        else:
            pd1 = vd1 = None
        k4 = rhs(P4, Ux4, Uth4, vr4, pc4, pb4, vb4, qp4, qv4, pd1, vd1)

        P = P + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        Ux = Ux + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        Uth = Uth + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        v_r = v_r + (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
        p_c = p_c + (dt / 6.0) * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4])
        p_bar = p_bar + (dt / 6.0) * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5])
        v_bar = v_bar + (dt / 6.0) * (k1[6] + 2.0 * k2[6] + 2.0 * k3[6] + k4[6])
        q_p = q_p + (dt / 6.0) * (k1[7] + 2.0 * k2[7] + 2.0 * k3[7] + k4[7])
        q_v = q_v + (dt / 6.0) * (k1[8] + 2.0 * k2[8] + 2.0 * k3[8] + k4[8])

        if use_delay:
            u_n = 0.5 * (Uth[m_f] + np.roll(Uth[m_f], 1))
            hist_p[buf_len + step + 1] = P[m_f] - p_bar
            hist_v[buf_len + step + 1] = u_n - v_bar

        record(step + 1)

    net_p = rayleigh - loss_noz - loss_cav
    cum_w = np.concatenate([[0.0], np.cumsum(0.5 * (net_p[:-1] + net_p[1:]) * dt)])
    dE = energy_tot - energy_tot[0]
    scale = max(np.max(np.abs(dE)), np.max(energy_tot), 1e-15)
    e_err = float(np.max(np.abs(dE - cum_w)) / scale)

    return {
        "t": t_arr,
        "energy_tot": energy_tot,
        "energy_balance_rel_err": e_err,
        "c_1T": c_1T,
        "c_2T": c_2T,
        "x_f_actual": x_f_actual,
    }


def fit_energy_slope(t: np.ndarray, energy: np.ndarray, t_min: float, t_max: float) -> float:
    mask = (t >= t_min) & (t <= t_max)
    return float(np.polyfit(t[mask], 0.5 * np.log(np.maximum(energy[mask], 1e-30)), 1)[0])


def fit_2mode_lsq(t: np.ndarray, sig: np.ndarray, t_min: float, t_max: float, init_roots: list[complex]) -> list[complex]:
    mask = (t >= t_min) & (t <= t_max)
    tw = t[mask] - t_min
    yw = sig[mask]
    stride = max(1, len(tw) // 2000)
    ts, ys = tw[::stride], yw[::stride]
    m = len(init_roots)
    x0 = np.array([v for r in init_roots for v in (r.real, r.imag)], dtype=float)

    def res(p):
        cols = []
        for idx in range(m):
            env = np.exp(p[2 * idx] * ts)
            cols.append(env * np.cos(p[2 * idx + 1] * ts))
            cols.append(env * np.sin(p[2 * idx + 1] * ts))
        A = np.column_stack(cols)
        c, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
        return A @ c - ys

    out = least_squares(res, x0=x0, xtol=1e-11, ftol=1e-11)
    return sorted([complex(out.x[2 * idx], out.x[2 * idx + 1]) for idx in range(m)], key=lambda z: z.imag)


# ---------------------------------------------------------------------------
# 3. Main experiments & plotting
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    L_a = 2.0
    zeta = 0.04
    x_f = 0.20
    tau_bar = 0.30
    sigma_tau = 0.15
    beta_tot = 0.02  # 2% total cavity-to-chamber volume ratio

    print("=== Entry 005: Baffle-Free 2D Transverse Stabilization ===")

    # ------------------------------------------------------------------
    # Experiment 1: Exact 2D 1T Exceptional-Point Law (Cold & Hot Spinning 1T)
    # ------------------------------------------------------------------
    n_v_spin = 0.06
    s1_hot_unc = find_coupled_1T_pair(L_a, zeta, 0.225, 0.0, n_v_spin, tau_bar, sigma_tau, 0.0, 1.0, 0.1)[0]
    xi_ep_cold_pred = np.sqrt(beta_tot) + zeta / (2.0 * L_a)
    alpha_ep_cold_pred = -zeta / (2.0 * L_a) - 0.5 * np.sqrt(beta_tot)
    xi_ep_hot_pred = np.sqrt(beta_tot) - s1_hot_unc.real

    print("--- Experiment 1: 2D Faceplate Cavity Ring 1T Exceptional-Point Law (beta_tot = 2%) ---")
    print(f"  Analytical cold 1T EP: xi_EP = {xi_ep_cold_pred:.5f}, alpha_EP = {alpha_ep_cold_pred:+.5f}")
    print(f"  Analytical hot  1T EP: xi_EP = {xi_ep_hot_pred:.5f} (uncoupled hot 1T alpha = {s1_hot_unc.real:+.5f})")

    xi_sweep = np.linspace(0.01, 0.40, 45)
    cold_1T_amax = [max(r.real for r in find_coupled_1T_pair(L_a, zeta, 0.225, 0.0, 0.0, tau_bar, sigma_tau, beta_tot, 1.0, xi)) for xi in xi_sweep]
    hot_1T_amax = [max(r.real for r in find_coupled_1T_pair(L_a, zeta, 0.225, 0.0, n_v_spin, tau_bar, sigma_tau, beta_tot, s1_hot_unc.imag, xi)) for xi in xi_sweep]

    idx_c = int(np.argmin(cold_1T_amax))
    idx_h = int(np.argmin(hot_1T_amax))
    print(f"  Exact 2D cold optimum: xi_opt = {xi_sweep[idx_c]:.4f}, alpha_max = {cold_1T_amax[idx_c]:+.5f}")
    print(f"  Exact 2D hot  optimum: xi_opt = {xi_sweep[idx_h]:.4f}, alpha_max = {hot_1T_amax[idx_h]:+.5f} (vs 5-blade baffle L_b=0.90: -0.0035!)")

    # Direct 2D PDE verification at hot EP (xi = 0.13) and under-damped (xi = 0.01)
    pde_ep_check = {}
    for label, xi_v in [("underdamped_xi001", 0.01), ("hot_EP_xi013", 0.13)]:
        sim = simulate_2d_baffle_free(
            L_a=L_a, zeta=zeta, x_f=x_f, n_p=0.0, n_v=n_v_spin,
            tau_profile=tau_bar, sigma_tau=sigma_tau,
            beta_local=beta_tot, omega_H_arr=s1_hot_unc.imag, xi_H_arr=xi_v,
            t_end=60.0, init_type="spin_1T_plus",
        )
        r_ex = find_coupled_1T_pair(L_a, zeta, sim["x_f_actual"], 0.0, n_v_spin, tau_bar, sigma_tau, beta_tot, s1_hot_unc.imag, xi_v)
        r_fit = fit_2mode_lsq(sim["t"], np.real(sim["c_1T"]), 12.0, 56.0, r_ex[:2])
        a_ex_max = max(r.real for r in r_ex)
        a_fit_max = max(r.real for r in r_fit)
        print(
            f"  2D PDE [{label:18s}]: exact_max={a_ex_max:+.5f}, PDE_2mode_LSQ={a_fit_max:+.5f} "
            f"(|da|={abs(a_fit_max - a_ex_max):.2e}) | Energy err={sim['energy_balance_rel_err']:.2e}"
        )
        pde_ep_check[label] = {
            "xi_H": xi_v, "alpha_exact_max": float(a_ex_max), "alpha_pde_lsq": float(a_fit_max),
            "abs_diff": float(abs(a_fit_max - a_ex_max)), "energy_err": sim["energy_balance_rel_err"],
        }

    # ------------------------------------------------------------------
    # Experiment 2: Dual-Tuned Azimuthally Alternating Cavity Ring (Fixed beta_tot = 2%, N_pair = 5)
    # Combined unstable 1T + 2T combustor: n_p = 0.055, n_v = 0.05, tau = 0.30, sigma_tau = 0.10
    # ------------------------------------------------------------------
    print("\n--- Experiment 2: Dual-Tuned Cavity Ring (beta_tot=2%, odd N_pair=5) on Combined 1T + 2T ---")
    n_p_comb = 0.055
    n_v_comb = 0.050
    sig_comb = 0.10  # Both 1T (w=1) and 2T (w=2) are strongly unstable without cavities!

    # Check bare unbaffled 1T and 2T growth rates
    sim_bare_1T = simulate_2d_baffle_free(L_a, zeta, x_f, n_p_comb, n_v_comb, tau_bar, sig_comb, t_end=60.0, init_type="spin_1T_plus")
    sim_bare_2T = simulate_2d_baffle_free(L_a, zeta, x_f, n_p_comb, n_v_comb, tau_bar, sig_comb, t_end=60.0, init_type="spin_2T_plus")
    a_bare_1T = fit_energy_slope(sim_bare_1T["t"], sim_bare_1T["energy_tot"], 18.0, 56.0)
    a_bare_2T = fit_energy_slope(sim_bare_2T["t"], sim_bare_2T["energy_tot"], 18.0, 56.0)
    print(f"  Bare combined combustor: spinning 1T alpha = {a_bare_1T:+.5f}, spinning 2T alpha = {a_bare_2T:+.5f}")

    # Sweep 1T volume fraction f_1 in [0.0, 1.0] (with f_2 = 1 - f_1), alternating 5 pairs around 60 cells (6 cells per half-sector!)
    ntheta = 60
    # Pattern: 5 pairs of (6 cells tuned to 1T, 6 cells tuned to 2T) -> exact volume mean = beta_tot!
    is_zone1 = np.array([((j // 6) % 2 == 0) for j in range(ntheta)])
    f1_grid = [0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0]
    dual_rows = []
    for f1 in f1_grid:
        f2 = 1.0 - f1
        if f1 == 0.0:
            b_loc = np.full(ntheta, beta_tot)
            wH_loc = np.full(ntheta, 2.0)
            xi_loc = np.full(ntheta, 0.14)
        elif f1 == 1.0:
            b_loc = np.full(ntheta, beta_tot)
            wH_loc = np.full(ntheta, 1.0)
            xi_loc = np.full(ntheta, 0.14)
        else:
            # Each zone occupies half the perimeter, so local beta density = 2 * f_i * beta_tot -> mean = beta_tot!
            b_loc = np.where(is_zone1, 2.0 * f1 * beta_tot, 2.0 * f2 * beta_tot)
            wH_loc = np.where(is_zone1, 1.0, 2.0)
            # Tune each zone's damping near its local EP sqrt(f_i * beta_tot)
            xi1 = max(0.06, np.sqrt(f1 * beta_tot))
            xi2 = max(0.06, np.sqrt(f2 * beta_tot))
            xi_loc = np.where(is_zone1, xi1, xi2)

        s1 = simulate_2d_baffle_free(
            L_a, zeta, x_f, n_p_comb, n_v_comb, tau_bar, sig_comb,
            beta_local=b_loc, omega_H_arr=wH_loc, xi_H_arr=xi_loc, t_end=60.0, init_type="spin_1T_plus"
        )
        s2 = simulate_2d_baffle_free(
            L_a, zeta, x_f, n_p_comb, n_v_comb, tau_bar, sig_comb,
            beta_local=b_loc, omega_H_arr=wH_loc, xi_H_arr=xi_loc, t_end=60.0, init_type="spin_2T_plus"
        )
        a1 = fit_energy_slope(s1["t"], s1["energy_tot"], 18.0, 56.0)
        a2 = fit_energy_slope(s2["t"], s2["energy_tot"], 18.0, 56.0)
        amax = max(a1, a2)
        print(f"  f1={f1:.2f} (1T:{100*f1:3.0f}%, 2T:{100*f2:3.0f}%): alpha_1T={a1:+.5f}, alpha_2T={a2:+.5f} | max={amax:+.5f} (E_err={s1['energy_balance_rel_err']:.1e})")
        dual_rows.append({
            "f1": f1, "alpha_1T": a1, "alpha_2T": a2, "alpha_max": amax,
            "energy_err_1T": s1["energy_balance_rel_err"], "energy_err_2T": s2["energy_balance_rel_err"],
        })

    # Grid convergence check on optimal dual-tuned split (40x60 vs 60x90)
    best_dual = min(dual_rows, key=lambda r: r["alpha_max"])
    f1_opt = best_dual["f1"]
    is_zone1_90 = np.array([((j // 9) % 2 == 0) for j in range(90)])
    b_loc_90 = np.where(is_zone1_90, 2.0 * f1_opt * beta_tot, 2.0 * (1.0 - f1_opt) * beta_tot)
    wH_90 = np.where(is_zone1_90, 1.0, 2.0)
    xi_90 = np.where(is_zone1_90, max(0.06, np.sqrt(f1_opt * beta_tot)), max(0.06, np.sqrt((1.0 - f1_opt) * beta_tot)))
    s1_fine = simulate_2d_baffle_free(
        L_a, zeta, x_f, n_p_comb, n_v_comb, tau_bar, sig_comb,
        beta_local=b_loc_90, omega_H_arr=wH_90, xi_H_arr=xi_90, nx=60, ntheta=90, t_end=60.0, init_type="spin_1T_plus"
    )
    a1_fine = fit_energy_slope(s1_fine["t"], s1_fine["energy_tot"], 18.0, 56.0)
    print(f"  Grid check at f1={f1_opt:.2f} (1T): 40x60={best_dual['alpha_1T']:+.5f} vs 60x90={a1_fine:+.5f} (|da|={abs(a1_fine - best_dual['alpha_1T']):.2e})\n")

    # ------------------------------------------------------------------
    # Experiment 3: Zero-Hardware Azimuthal Injector Lag Staggering tau(theta) = tau_bar +/- dtau
    # Test on pressure-coupled 1T+2T (n_p = 0.055, n_v = 0, tau_bar = 1.00, sigma_tau = 0.15)
    # where cos(omega*dtau) phase cancellation acts cleanly across M = 2, 3, 5!
    # Wait: at tau_bar = 1.00, 1T (w=1.0) has cos(1.0)=+0.54 > 0 (unstable at n_p=0.075!),
    # and we can test BOTH M=2 orientation split (phase=0 vs phase=pi/2) AND odd M=3, 5!
    # ------------------------------------------------------------------
    print("--- Experiment 3: Zero-Hardware Azimuthal Injector Lag Staggering (tau_bar=1.00, n_p=0.075, n_v=0) ---")
    tau_stg_bar = 1.00
    n_p_stg = 0.075
    sig_stg = 0.15
    th_c = (np.arange(ntheta) + 0.5) * (2.0 * np.pi / ntheta)

    # First verify M=2 vs M=3 vs M=5 across Delta_tau in [0.0, 0.85] (strictly causal: tau_min >= 0.15!)
    dtau_grid = [0.0, 0.20, 0.40, 0.60, 0.75, 0.85]
    stag_rows = []
    for dtau in dtau_grid:
        row = {"dtau": dtau, "cos_w1_dtau": float(np.cos(1.0 * dtau)), "cos_w2_dtau": float(np.cos(2.0 * dtau))}
        for M in (2, 3, 5):
            # Discrete alternating square-wave sectors: +dtau where cos(M*theta) >= 0, -dtau where < 0
            tau_vec = tau_stg_bar + dtau * np.where(np.cos(M * th_c) >= 0.0, 1.0, -1.0)
            # Test BOTH orthogonal standing 1T orientations (phase=0 aligned with +dtau, phase=pi/2 aligned with -dtau)
            s_ph0 = simulate_2d_baffle_free(
                L_a, zeta, x_f, n_p_stg, 0.0, tau_vec, sig_stg, t_end=60.0, init_type="standing_1T", init_phase=0.0
            )
            s_ph90 = simulate_2d_baffle_free(
                L_a, zeta, x_f, n_p_stg, 0.0, tau_vec, sig_stg, t_end=60.0, init_type="standing_1T", init_phase=0.5 * np.pi
            )
            a_ph0 = fit_energy_slope(s_ph0["t"], s_ph0["energy_tot"], 18.0, 56.0)
            a_ph90 = fit_energy_slope(s_ph90["t"], s_ph90["energy_tot"], 18.0, 56.0)
            row[f"M{M}_1T_ph0"] = a_ph0
            row[f"M{M}_1T_ph90"] = a_ph90
            row[f"M{M}_1T_worst"] = max(a_ph0, a_ph90)
        print(
            f"  dtau={dtau:.2f}: M=2 [ph0={row['M2_1T_ph0']:+.5f}, ph90={row['M2_1T_ph90']:+.5f}, worst={row['M2_1T_worst']:+.5f}] | "
            f"M=3 worst={row['M3_1T_worst']:+.5f} | M=5 worst={row['M5_1T_worst']:+.5f}"
        )
        stag_rows.append(row)

    # Also test M=2, 3, 5 lag staggering on chiral SPINNING 1T (n_v = 0.06, n_p = 0, tau_bar = 1.00 -> cos(1.0)=0.54, use tau_bar=0.80, dtau=0.70)
    print("\n--- Experiment 3B: Azimuthal Lag Staggering on Chiral Velocity-Coupled Spinning 1T (tau_bar=0.85, n_v=0.075) ---")
    spin_stag = {}
    for M in (0, 2, 3, 5):
        tau_vec = np.full(ntheta, 0.85) if M == 0 else (0.85 + 0.70 * np.where(np.cos(M * th_c) >= 0.0, 1.0, -1.0))
        sim_ss = simulate_2d_baffle_free(L_a, zeta, x_f, 0.0, 0.075, tau_vec, 0.15, t_end=65.0, init_type="spin_1T_plus")
        a_ss = fit_energy_slope(sim_ss["t"], sim_ss["energy_tot"], 20.0, 60.0)
        print(f"  M={M} (dtau={'0.00' if M==0 else '0.70'}): spinning 1T alpha = {a_ss:+.5f}")
        spin_stag[f"M{M}"] = a_ss

    # ------------------------------------------------------------------
    # Figure 1: 2D 1T Cavity EP Law & Dual-Tuned 1T+2T Cavity Ring Split
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.4), dpi=150)

    ax1.axhline(0.0, color="black", lw=1.3)
    ax1.axhline(s1_hot_unc.real, color="#7f7f7f", ls=":", lw=1.2, label=f"Unbaffled hot spinning $1T$ ($\\alpha={s1_hot_unc.real:+.4f}$)")
    ax1.axhline(-0.00349, color="#2ca02c", ls="-.", lw=1.3, label="Entry 004 5-blade baffle $L_b=0.90$ ($\\alpha=-0.0035$)")
    ax1.plot(xi_sweep, cold_1T_amax, "--", color="#1f77b4", lw=1.8, label="Cold $1T$ $\\max\\alpha_\\pm$ ($\\beta_{\\mathrm{tot}}=2\\%$)")
    ax1.plot(xi_sweep, hot_1T_amax, "-", color="#d62728", lw=2.2, label="Hot spinning $1T$ $\\max\\alpha_\\pm$ ($\\beta_{\\mathrm{tot}}=2\\%$)")
    ax1.axvline(xi_ep_cold_pred, color="#1f77b4", ls=":", lw=1.2)
    ax1.axvline(xi_ep_hot_pred, color="#d62728", ls=":", lw=1.2)
    ax1.scatter([0.01, 0.13], [pde_ep_check["underdamped_xi001"]["alpha_pde_lsq"], pde_ep_check["hot_EP_xi013"]["alpha_pde_lsq"]],
                c=["orange", "lime"], edgecolors="black", s=65, zorder=5, label="2D PDE LSQ verification")
    ax1.set_xlabel("Cavity damping ratio $\\xi_H$")
    ax1.set_ylabel("Dominant $1T$ growth rate $\\max(\\alpha_-, \\alpha_+)$")
    ax1.set_title("2D Faceplate Cavity Ring: $1T$ Exceptional-Point Law ($\\beta_{\\mathrm{tot}}=2\\%$)", fontweight="bold")
    ax1.set_xlim(0.0, 0.40)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=7.6)

    f1s = [r["f1"] * 100.0 for r in dual_rows]
    ax2.axhline(0.0, color="black", lw=1.4)
    ax2.axhline(a_bare_1T, color="#1f77b4", ls=":", lw=1.1, alpha=0.8, label=f"Bare $1T$ ($\\alpha={a_bare_1T:+.4f}$)")
    ax2.axhline(a_bare_2T, color="#ff7f0e", ls=":", lw=1.1, alpha=0.8, label=f"Bare $2T$ ($\\alpha={a_bare_2T:+.4f}$)")
    ax2.plot(f1s, [r["alpha_1T"] for r in dual_rows], "o-", color="#1f77b4", lw=2.0, ms=6, label="Dual-ring $1T$ growth rate")
    ax2.plot(f1s, [r["alpha_2T"] for r in dual_rows], "s-", color="#ff7f0e", lw=2.0, ms=6, label="Dual-ring $2T$ growth rate")
    ax2.plot(f1s, [r["alpha_max"] for r in dual_rows], "D--", color="#d62728", lw=2.2, ms=6.5, label="Worst-case $\\max(\\alpha_{1T}, \\alpha_{2T})$")
    ax2.set_xlabel("Fraction of cavity volume tuned to $1T$ ($f_1$, %; remainder $1-f_1$ to $2T$)")
    ax2.set_ylabel("2D PDE growth rate $\\alpha$")
    ax2.set_title("Dual-Tuned Rim Cavity Ring ($\\beta_{\\mathrm{tot}}=2\\%$, Odd $N_{\\mathrm{pair}}=5$)", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper center", fontsize=7.8)

    fig.suptitle("Entry 005 (Part 1): Baffle-Free Faceplate Acoustic Cavity Rings for 2D Transverse Modes", fontsize=12.2, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp005_2d_cavity_rings.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Zero-Hardware Azimuthal Injector Lag Staggering & M=2 Orientation Trap
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.3), dpi=150)

    dt_arr = [r["dtau"] for r in stag_rows]
    ax1.axhline(0.0, color="black", lw=1.4)
    ax1.plot(dt_arr, [r["M2_1T_ph0"] for r in stag_rows], "^--", color="#d62728", lw=1.8, ms=6, label="$M=2$ aligned with $\\bar{\\tau}+\\Delta\\tau$ (stable)")
    ax1.plot(dt_arr, [r["M2_1T_ph90"] for r in stag_rows], "v-", color="#d62728", lw=2.2, ms=6.5, label="$M=2$ aligned with $\\bar{\\tau}-\\Delta\\tau$ (WORSE!)")
    ax1.plot(dt_arr, [r["M3_1T_worst"] for r in stag_rows], "s-", color="#1f77b4", lw=2.1, ms=6, label="Odd $M=3$ worst orientation")
    ax1.plot(dt_arr, [r["M5_1T_worst"] for r in stag_rows], "o-", color="#2ca02c", lw=2.1, ms=6, label="Odd $M=5$ worst orientation")
    ax1.set_xlabel("Azimuthal time-lag stagger amplitude $\\Delta\\tau$ (around $\\bar{\\tau}=1.00$)")
    ax1.set_ylabel("Standing $1T$ growth rate $\\alpha$")
    ax1.set_title("Standing $1T$: $M=2$ Orientation Split Trap vs. Odd $M=3, 5$", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", fontsize=8.0)

    # Panel 2: Chiral spinning 1T suppression by zero-hardware lag staggering M = 0, 2, 3, 5
    ms = [0, 2, 3, 5]
    vals_ss = [spin_stag[f"M{m}"] for m in ms]
    cols_ss = ["#7f7f7f", "#d62728", "#1f77b4", "#2ca02c"]
    ax2.axhline(0.0, color="black", lw=1.4)
    bars = ax2.bar([f"$M={m}$" for m in ms], vals_ss, color=cols_ss, edgecolor="black", width=0.52)
    for b, v in zip(bars, vals_ss):
        va = "bottom" if v >= 0 else "top"
        dy = 0.0004 if v >= 0 else -0.0005
        ax2.text(b.get_x() + b.get_width() / 2, v + dy, f"{v:+.4f}", ha="center", va=va, fontsize=9, fontweight="bold")
    ax2.set_ylim(0.0, max(vals_ss) * 1.18)
    ax2.set_ylabel("Spinning $1T$ growth rate $\\alpha$ ($n_v=0.075, \\bar{\\tau}=0.85, \\Delta\\tau=0.70$)")
    ax2.set_title("Zero-Hardware Azimuthal Lag Staggering on Chiral Spinning $1T$", fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Entry 005 (Part 2): Zero-Hardware Azimuthal Injector Time-Lag Staggering", fontsize=12.2, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "exp005_azimuthal_lag_staggering.png")
    plt.close(fig)

    metrics = {
        "parameters": {
            "L_a": L_a, "zeta": zeta, "x_f": x_f, "tau_bar": tau_bar, "sigma_tau": sigma_tau,
            "omega_hp": OMEGA_HP, "beta_tot": beta_tot,
        },
        "exp1_2d_1T_cavity_ep": {
            "s1_hot_uncoupled": [float(s1_hot_unc.real), float(s1_hot_unc.imag)],
            "xi_ep_cold_pred": float(xi_ep_cold_pred),
            "alpha_ep_cold_pred": float(alpha_ep_cold_pred),
            "xi_ep_hot_pred": float(xi_ep_hot_pred),
            "exact_cold_opt": {"xi_opt": float(xi_sweep[idx_c]), "alpha_max": float(cold_1T_amax[idx_c])},
            "exact_hot_opt": {"xi_opt": float(xi_sweep[idx_h]), "alpha_max": float(hot_1T_amax[idx_h])},
            "pde_verification": pde_ep_check,
        },
        "exp2_dual_tuned_cavity_ring": {
            "bare_1T_alpha": a_bare_1T,
            "bare_2T_alpha": a_bare_2T,
            "sweep_f1": dual_rows,
            "grid_check_f1_opt": {
                "f1_opt": f1_opt,
                "coarse_40x60_alpha_1T": best_dual["alpha_1T"],
                "fine_60x90_alpha_1T": a1_fine,
                "abs_diff": float(abs(a1_fine - best_dual["alpha_1T"])),
            },
        },
        "exp3_azimuthal_lag_staggering": {
            "standing_1T_sweep": stag_rows,
            "spinning_1T_M_comparison": spin_stag,
        },
    }

    with open(RESULTS_DIR / "exp005_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved Entry 005 plots and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
